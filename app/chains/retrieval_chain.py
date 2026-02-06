"""
Retrieval Chain - LangChain 1.x Production Pattern

Implements a hybrid retrieval chain with:
- Vector similarity search
- BM25 keyword search
- Reciprocal Rank Fusion
- Cross-encoder reranking
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

from ..config import settings
from ..logger import logger


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval chain."""
    
    # Retrieval settings (optimized for better recall - was k=10, final_k=5)
    k: int = Field(default=15, description="Number of documents to retrieve initially")
    final_k: int = Field(default=10, description="Number of documents after reranking")
    score_threshold: float = Field(default=0.25, description="Minimum similarity score")
    
    # Hybrid search
    enable_bm25: bool = Field(default=True, description="Enable BM25 keyword search")
    bm25_weight: float = Field(default=0.3, description="Weight for BM25 in fusion")
    vector_weight: float = Field(default=0.7, description="Weight for vector search")
    
    # Reranking
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking"
    )
    
    # RRF settings
    rrf_k: int = Field(default=60, description="RRF constant")


class RetrievalChain:
    """
    Production-grade hybrid retrieval chain.
    
    Uses LangChain 1.x patterns with ChromaDB and optional reranking.
    
    Example:
        ```python
        from app.chains import RetrievalChain
        
        chain = RetrievalChain()
        await chain.initialize()
        
        docs = await chain.retrieve("What is the protein requirement?")
        ```
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.vectorstore = None
        self.bm25_retriever = None
        self.reranker = None
        self._initialized = False
    
    async def initialize(self, documents: Optional[List[Document]] = None):
        """Initialize the retrieval chain with documents."""
        
        # Import LangChain components
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        # Initialize embeddings
        if settings.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"trust_remote_code": True}
            )
        
        # Initialize or load vector store
        persist_dir = str(settings.chromadb_dir)
        
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        
        # Add documents if provided
        if documents:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vectorstore.add_documents(documents)
        
        # Initialize BM25 if enabled
        if self.config.enable_bm25:
            await self._initialize_bm25()
        
        # Initialize reranker if enabled
        if self.config.enable_reranking:
            self._initialize_reranker()
        
        self._initialized = True
        logger.info("RetrievalChain initialized successfully")
    
    async def _initialize_bm25(self):
        """Initialize BM25 retriever from existing documents."""
        try:
            from langchain_community.retrievers import BM25Retriever
            
            # Get all documents from vector store
            collection = self.vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])
            
            if result and result.get("documents"):
                docs = [
                    Document(
                        page_content=content,
                        metadata=meta or {}
                    )
                    for content, meta in zip(
                        result["documents"],
                        result.get("metadatas", [{}] * len(result["documents"]))
                    )
                ]
                
                self.bm25_retriever = BM25Retriever.from_documents(
                    docs, k=self.config.k * 2
                )
                logger.info(f"BM25 initialized with {len(docs)} documents")
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
            self.bm25_retriever = None
    
    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.config.reranker_model)
            logger.info(f"Reranker initialized: {self.config.reranker_model}")
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
            self.reranker = None
    
    async def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Pipeline:
        1. Vector similarity search
        2. BM25 keyword search (if enabled)
        3. Reciprocal Rank Fusion
        4. Cross-encoder reranking (if enabled)
        """
        if not self._initialized:
            await self.initialize()
        
        # Step 1: Vector search
        vector_docs = self.vectorstore.similarity_search(
            query, k=self.config.k * 2
        )
        
        # Step 2: BM25 search
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
        
        # Step 3: Combine with RRF
        if bm25_docs:
            combined = self._reciprocal_rank_fusion([vector_docs, bm25_docs])
        else:
            combined = vector_docs
        
        # Step 4: Rerank
        if self.reranker and combined:
            combined = self._rerank(query, combined)
        
        # Return top k
        return combined[:self.config.final_k]
    
    def _reciprocal_rank_fusion(
        self, 
        doc_lists: List[List[Document]]
    ) -> List[Document]:
        """Combine multiple ranked lists using RRF."""
        k = self.config.rrf_k
        doc_scores: Dict[str, tuple] = {}
        
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list, 1):
                doc_key = hash(doc.page_content)
                rrf_score = 1.0 / (k + rank)
                
                if doc_key in doc_scores:
                    doc_scores[doc_key] = (
                        doc_scores[doc_key][0] + rrf_score,
                        doc_scores[doc_key][1]
                    )
                else:
                    doc_scores[doc_key] = (rrf_score, doc)
        
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs]
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not self.reranker or not documents:
            return documents
        
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.reranker.predict(pairs)
            
            scored = list(zip(scores, documents))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for _, doc in scored]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval chain statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            count = self.vectorstore._collection.count()
        except:
            count = 0
        
        return {
            "status": "initialized",
            "total_documents": count,
            "bm25_enabled": self.bm25_retriever is not None,
            "reranker_enabled": self.reranker is not None,
            "config": self.config.model_dump(),
        }


def create_retrieval_chain(
    k: int = 10,
    enable_bm25: bool = True,
    enable_reranking: bool = True,
) -> RetrievalChain:
    """
    Factory function to create a configured retrieval chain.
    
    Example:
        ```python
        chain = create_retrieval_chain(k=10, enable_reranking=True)
        await chain.initialize()
        docs = await chain.retrieve("amino acid requirements")
        ```
    """
    config = RetrievalConfig(
        k=k,
        enable_bm25=enable_bm25,
        enable_reranking=enable_reranking,
    )
    return RetrievalChain(config=config)
