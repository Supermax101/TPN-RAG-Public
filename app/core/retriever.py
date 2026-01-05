"""
Production-grade hybrid retriever with LangChain.

Implements:
- Vector similarity search (ChromaDB)
- BM25 keyword search
- Cross-encoder reranking
- Reciprocal Rank Fusion (RRF)
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field, ConfigDict

from ..config import settings
from ..logger import logger


class RerankerType(str, Enum):
    """Supported reranker types."""
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"


@dataclass
class RetrieverConfig:
    """Configuration for the hybrid retriever.
    
    Tuning Guide:
    - For high precision: Lower k values, enable reranking
    - For high recall: Higher k values, enable BM25 hybrid
    - For clinical QA: k=10-20 for retrieval, rerank to top 5
    """
    
    # Vector store settings
    collection_name: str = "documents"
    persist_directory: Optional[str] = None
    
    # Retrieval settings
    k: int = 10  # Number of documents to retrieve
    score_threshold: Optional[float] = None  # Minimum score threshold
    
    # Hybrid search settings
    enable_bm25: bool = True
    bm25_weight: float = 0.3  # Weight for BM25 vs vector (0-1)
    
    # Reranking settings
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5  # Final number after reranking
    
    # RRF settings
    rrf_k: int = 60  # RRF constant (higher = more weight to lower ranks)
    
    # Search filters
    default_filters: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever(BaseRetriever):
    """Production-grade hybrid retriever combining vector and keyword search.
    
    Features:
    - Vector similarity search via ChromaDB
    - BM25 keyword search for exact matches
    - Cross-encoder reranking for precision
    - Reciprocal Rank Fusion for combining results
    - Configurable thresholds and weights
    
    Example:
        ```python
        from app.core.retriever import HybridRetriever, RetrieverConfig
        from app.core.embeddings import EmbeddingManager
        
        embeddings = EmbeddingManager.from_preset("best_open_source")
        config = RetrieverConfig(k=10, enable_bm25=True)
        
        retriever = HybridRetriever(
            embedding_manager=embeddings,
            config=config
        )
        
        docs = retriever.invoke("What is the amino acid requirement for preterm infants?")
        ```
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Required attributes
    embedding_manager: Any  # EmbeddingManager instance
    config: RetrieverConfig = Field(default_factory=RetrieverConfig)
    
    # Internal state (initialized lazily)
    _vectorstore: Any = None
    _bm25_retriever: Any = None
    _reranker: Any = None
    _all_docs: List[Document] = []
    
    def __init__(self, embedding_manager: Any, config: Optional[RetrieverConfig] = None, **kwargs):
        super().__init__(
            embedding_manager=embedding_manager,
            config=config or RetrieverConfig(),
            **kwargs
        )
        self._vectorstore = None
        self._bm25_retriever = None
        self._reranker = None
        self._all_docs = []
    
    @property
    def vectorstore(self):
        """Lazy initialization of vector store."""
        if self._vectorstore is None:
            self._vectorstore = self._create_vectorstore()
        return self._vectorstore
    
    @property
    def reranker(self):
        """Lazy initialization of reranker."""
        if self._reranker is None and self.config.reranker_type != RerankerType.NONE:
            self._reranker = self._create_reranker()
        return self._reranker
    
    def _create_vectorstore(self):
        """Create ChromaDB vector store."""
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        
        persist_dir = self.config.persist_directory or str(settings.chromadb_dir)
        
        return Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_manager.embeddings,
            persist_directory=persist_dir,
        )
    
    def _create_reranker(self):
        """Create cross-encoder reranker."""
        if self.config.reranker_type == RerankerType.CROSS_ENCODER:
            try:
                from sentence_transformers import CrossEncoder
                return CrossEncoder(self.config.reranker_model)
            except ImportError:
                logger.warning("sentence-transformers not installed. Reranking disabled.")
                return None
        return None
    
    def _create_bm25_retriever(self, documents: List[Document]):
        """Create BM25 retriever from documents."""
        try:
            from langchain_community.retrievers import BM25Retriever
            return BM25Retriever.from_documents(documents, k=self.config.k * 2)
        except ImportError:
            logger.warning("BM25Retriever not available. Using vector-only search.")
            return None
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Main retrieval method - required by BaseRetriever."""
        
        # 1. Vector search
        vector_docs = self._vector_search(query)
        
        # 2. BM25 search (if enabled)
        bm25_docs = []
        if self.config.enable_bm25 and self._bm25_retriever:
            bm25_docs = self._bm25_search(query)
        
        # 3. Combine with RRF
        if bm25_docs:
            combined_docs = self._reciprocal_rank_fusion([vector_docs, bm25_docs])
        else:
            combined_docs = vector_docs
        
        # 4. Rerank (if enabled)
        if self.reranker and combined_docs:
            combined_docs = self._rerank(query, combined_docs)
        
        # 5. Apply final limit
        final_docs = combined_docs[:self.config.rerank_top_k]
        
        logger.debug(f"Retrieved {len(final_docs)} documents for query: {query[:50]}...")
        return final_docs
    
    def _vector_search(self, query: str) -> List[Document]:
        """Perform vector similarity search."""
        try:
            if self.config.score_threshold:
                docs_and_scores = self.vectorstore.similarity_search_with_score(
                    query, k=self.config.k * 2
                )
                # Filter by threshold and convert
                docs = [
                    doc for doc, score in docs_and_scores
                    if self._score_passes_threshold(score)
                ]
            else:
                docs = self.vectorstore.similarity_search(query, k=self.config.k * 2)
            
            return docs
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _score_passes_threshold(self, score: float) -> bool:
        """Check if score passes threshold (accounting for distance vs similarity)."""
        # ChromaDB returns distance (lower = better)
        # Convert to similarity: 1 - (distance / 2) for cosine
        similarity = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        return similarity >= (self.config.score_threshold or 0.0)
    
    def _bm25_search(self, query: str) -> List[Document]:
        """Perform BM25 keyword search."""
        try:
            if self._bm25_retriever:
                return self._bm25_retriever.invoke(query)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        return []
    
    def _reciprocal_rank_fusion(
        self,
        doc_lists: List[List[Document]],
    ) -> List[Document]:
        """Combine multiple ranked lists using RRF.
        
        RRF Score = sum(1 / (k + rank)) for each list
        Higher k = more weight to lower ranks
        """
        k = self.config.rrf_k
        doc_scores: Dict[str, Tuple[float, Document]] = {}
        
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list, 1):
                # Use page_content hash as key
                doc_key = hash(doc.page_content)
                rrf_score = 1.0 / (k + rank)
                
                if doc_key in doc_scores:
                    doc_scores[doc_key] = (
                        doc_scores[doc_key][0] + rrf_score,
                        doc_scores[doc_key][1]
                    )
                else:
                    doc_scores[doc_key] = (rrf_score, doc)
        
        # Sort by RRF score (descending)
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs]
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not self.reranker or not documents:
            return documents
        
        try:
            # Create query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get scores
            scores = self.reranker.predict(pairs)
            
            # Sort by score (descending)
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            logger.debug(f"Reranked {len(documents)} docs. Top score: {scored_docs[0][0]:.3f}")
            return [doc for score, doc in scored_docs]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store and BM25 index."""
        # Add to vector store
        ids = self.vectorstore.add_documents(documents)
        
        # Update BM25 index
        self._all_docs.extend(documents)
        if self.config.enable_bm25:
            self._bm25_retriever = self._create_bm25_retriever(self._all_docs)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self._all_docs)}")
        return ids
    
    def initialize_bm25(self):
        """Initialize BM25 from existing vector store documents.
        
        Call this after loading an existing vector store to enable BM25 hybrid search.
        """
        if not self.config.enable_bm25:
            return
        
        try:
            # Get all documents from vector store
            result = self.vectorstore._collection.get(include=["documents", "metadatas"])
            
            if result and result.get("documents"):
                self._all_docs = [
                    Document(
                        page_content=content,
                        metadata=meta or {}
                    )
                    for content, meta in zip(result["documents"], result.get("metadatas", [{}] * len(result["documents"])))
                ]
                
                self._bm25_retriever = self._create_bm25_retriever(self._all_docs)
                logger.info(f"Initialized BM25 with {len(self._all_docs)} documents")
        except Exception as e:
            logger.warning(f"Could not initialize BM25 from vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            count = self.vectorstore._collection.count()
        except:
            count = 0
        
        return {
            "collection_name": self.config.collection_name,
            "total_documents": count,
            "bm25_enabled": self.config.enable_bm25,
            "bm25_docs": len(self._all_docs),
            "reranker_enabled": self.reranker is not None,
            "reranker_type": self.config.reranker_type.value,
        }


def create_retriever(
    embedding_manager: Any,
    collection_name: str = "documents",
    k: int = 10,
    enable_bm25: bool = True,
    enable_reranking: bool = True,
    **kwargs
) -> HybridRetriever:
    """Factory function to create a configured retriever.
    
    Args:
        embedding_manager: EmbeddingManager instance
        collection_name: ChromaDB collection name
        k: Number of documents to retrieve
        enable_bm25: Enable hybrid BM25 search
        enable_reranking: Enable cross-encoder reranking
        **kwargs: Additional RetrieverConfig options
    
    Returns:
        Configured HybridRetriever instance
    
    Example:
        ```python
        embeddings = EmbeddingManager.from_preset("best_open_source")
        retriever = create_retriever(embeddings, k=10, enable_reranking=True)
        docs = retriever.invoke("amino acid requirements")
        ```
    """
    config = RetrieverConfig(
        collection_name=collection_name,
        k=k,
        enable_bm25=enable_bm25,
        reranker_type=RerankerType.CROSS_ENCODER if enable_reranking else RerankerType.NONE,
        **kwargs
    )
    
    return HybridRetriever(embedding_manager=embedding_manager, config=config)
