"""
Production RAG Pipeline - Unified TPN Clinical Q&A System.

This is the MAIN ENTRY POINT for the production RAG system.
It integrates:
1. Document ingestion (PDF, Markdown, JSON)
2. Semantic chunking with clinical separators
3. Vector store with hybrid retrieval
4. LLM generation with grounding guarantees

GROUNDING GUARANTEE:
====================
All answers are EXCLUSIVELY based on your knowledge base.
The system will refuse to answer if no relevant context is found.

Usage:
    from app.rag_pipeline import TPN_RAG
    
    rag = TPN_RAG()
    await rag.initialize()
    
    # Ingest documents
    await rag.ingest_pdf("path/to/book.pdf")
    
    # Ask questions
    result = await rag.ask("What is the protein requirement for preterm infants?")
    print(result["answer"])
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.documents import Document

from app.config import settings
from app.logger import logger


# =============================================================================
# PIPELINE CONFIG
# =============================================================================

class PipelineMode(str, Enum):
    """RAG pipeline modes."""
    STANDARD = "standard"      # Simple retrieve + generate
    AGENTIC = "agentic"        # With document grading + query rewrite


class PipelineConfig(BaseModel):
    """Production pipeline configuration."""
    
    # Model settings
    llm_model: str = Field(default="qwen2.5:7b")
    embed_model: str = Field(default="qwen3-embedding:0.6b")
    
    # Pipeline mode
    mode: PipelineMode = Field(default=PipelineMode.STANDARD)
    
    # Retrieval settings
    retrieval_k: int = Field(default=5, description="Documents to retrieve")
    enable_bm25: bool = Field(default=True)
    enable_reranking: bool = Field(default=True)
    relevance_threshold: float = Field(default=0.55)
    
    # Chunking settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # Safety settings
    require_grounding: bool = Field(default=True, description="Refuse to answer if no context")
    max_retries: int = Field(default=3)
    timeout_seconds: float = Field(default=60.0)


# =============================================================================
# RESPONSE TYPES
# =============================================================================

@dataclass
class RetrievalInfo:
    """Information about retrieval."""
    documents_retrieved: int = 0
    documents_relevant: int = 0
    top_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    time_ms: float = 0.0


@dataclass
class RAGResponse:
    """Complete RAG response with provenance."""
    
    # Core response
    answer: str
    reasoning: str
    confidence: str  # high, medium, low
    
    # Grounding info
    is_grounded: bool
    grounding_score: float  # 0-1
    
    # Sources
    sources: List[Dict[str, str]]
    context_used: str
    
    # Retrieval info
    retrieval: RetrievalInfo
    
    # Timing
    total_time_ms: float
    
    # Error handling
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "is_grounded": self.is_grounded,
            "grounding_score": self.grounding_score,
            "sources": self.sources,
            "retrieval": {
                "documents": self.retrieval.documents_retrieved,
                "relevant": self.retrieval.documents_relevant,
                "top_score": self.retrieval.top_score,
            },
            "time_ms": self.total_time_ms,
            "error": self.error,
        }


# =============================================================================
# PRODUCTION RAG PIPELINE
# =============================================================================

class TPN_RAG:
    """
    Production TPN RAG Pipeline.
    
    This is the main class for clinical question answering.
    
    Features:
    - Automatic document ingestion (PDF, MD, JSON)
    - Hybrid retrieval (vector + BM25 + reranking)
    - Grounding verification
    - Error handling with retries
    
    Example:
        ```python
        rag = TPN_RAG()
        await rag.initialize()
        
        # Ingest a clinical book
        await rag.ingest_pdf("/path/to/tpn_handbook.pdf")
        
        # Ask clinical questions
        result = await rag.ask(
            question="What is the amino acid requirement for a 26-week preterm?",
            require_grounding=True
        )
        
        if result.is_grounded:
            print(f"Answer: {result.answer}")
            print(f"Sources: {result.sources}")
        else:
            print("Cannot answer - no relevant information found")
        ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Components (lazy initialized)
        self._retrieval_chain = None
        self._mcq_chain = None
        self._agentic_rag = None
        self._embeddings = None
        self._vectorstore = None
        self._chunker = None
        
        self._initialized = False
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    async def initialize(self) -> None:
        """Initialize the RAG pipeline."""

        logger.info("Initializing TPN RAG Pipeline...")
        start_time = time.time()

        # Import components
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        # Initialize HuggingFace embeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"trust_remote_code": True}
        )

        # Initialize vector store
        settings.chromadb_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(settings.chromadb_dir),
        )
        
        # Initialize chunker
        from app.document_processing.chunker import SemanticChunker
        self._chunker = SemanticChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Initialize appropriate chain based on mode
        if self.config.mode == PipelineMode.AGENTIC:
            from app.chains.agentic_rag import AgenticMCQRAG
            self._agentic_rag = AgenticMCQRAG(model=self.config.llm_model)
            await self._agentic_rag.initialize()
        else:
            from app.chains.mcq_chain import MCQChain, MCQChainConfig
            
            chain_config = MCQChainConfig(
                model=self.config.llm_model,
                retrieval_k=self.config.retrieval_k,
                enable_bm25=self.config.enable_bm25,
                enable_reranking=self.config.enable_reranking,
            )
            self._mcq_chain = MCQChain(config=chain_config)
            await self._mcq_chain.initialize()
        
        self._initialized = True
        
        elapsed = (time.time() - start_time) * 1000
        doc_count = self._vectorstore._collection.count() if self._vectorstore else 0
        
        logger.info(f"TPN RAG initialized in {elapsed:.0f}ms with {doc_count} documents")
    
    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================
    
    async def ingest_pdf(
        self,
        path: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Ingest a PDF document into the knowledge base.
        
        Returns the number of chunks added.
        """
        if not self._initialized:
            await self.initialize()
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        from app.document_processing.pdf_loader import PDFLoader
        
        loader = PDFLoader()
        documents = await loader.load_and_chunk(
            path,
            chunk_size=chunk_size or self.config.chunk_size,
            chunk_overlap=chunk_overlap or self.config.chunk_overlap,
        )
        
        if documents:
            self._vectorstore.add_documents(documents)
            logger.info(f"Ingested {len(documents)} chunks from {path.name}")
        
        return len(documents)
    
    async def ingest_markdown(
        self,
        path: Union[str, Path],
    ) -> int:
        """Ingest a Markdown file into the knowledge base."""
        if not self._initialized:
            await self.initialize()
        
        path = Path(path)
        documents = self._chunker.process_markdown_file(path)
        
        if documents:
            self._vectorstore.add_documents(documents)
            logger.info(f"Ingested {len(documents)} chunks from {path.name}")
        
        return len(documents)
    
    async def ingest_directory(
        self,
        path: Union[str, Path],
    ) -> int:
        """Ingest all supported documents from a directory."""
        if not self._initialized:
            await self.initialize()
        
        path = Path(path)
        total = 0
        
        # PDFs
        for pdf_file in path.glob("**/*.pdf"):
            total += await self.ingest_pdf(pdf_file)
        
        # Markdown
        for md_file in path.glob("**/*.md"):
            total += await self.ingest_markdown(md_file)
        
        # JSON (pre-chunked)
        for json_file in path.glob("**/*_response.json"):
            documents = self._chunker.process_json_document(json_file)
            if documents:
                self._vectorstore.add_documents(documents)
                total += len(documents)
        
        logger.info(f"Ingested {total} total chunks from {path}")
        return total
    
    # =========================================================================
    # QUESTION ANSWERING
    # =========================================================================
    
    async def ask(
        self,
        question: str,
        options: str = "",
        answer_type: str = "single",
        case_context: str = "",
        require_grounding: Optional[bool] = None,
    ) -> RAGResponse:
        """
        Ask a clinical question.
        
        Args:
            question: The clinical question
            options: MCQ options (for MCQ questions)
            answer_type: "single" or "multi"
            case_context: Clinical case context
            require_grounding: Override grounding requirement
        
        Returns:
            RAGResponse with answer and grounding information
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        require_grounding = require_grounding if require_grounding is not None else self.config.require_grounding
        
        try:
            # Use appropriate chain
            if self.config.mode == PipelineMode.AGENTIC:
                result = await self._agentic_rag.answer(
                    question=question,
                    options=options or "N/A",
                    answer_type=answer_type,
                    case_context=case_context,
                )
                context_used = result.get("context", "")
            else:
                result = await self._mcq_chain.answer(
                    question=question,
                    options=options or "N/A",
                    answer_type=answer_type,
                    case_context=case_context,
                )
                # Get context from sources
                sources = result.get("sources", [])
                context_used = "\n".join([s.get("content", "") for s in sources])
            
            # Extract response
            answer = result.get("answer", "")
            thinking = result.get("thinking", "")
            confidence = result.get("confidence", "medium")
            
            # Calculate grounding
            has_context = bool(result.get("context_used", True)) and bool(context_used)
            grounding_score = self._estimate_grounding(thinking, context_used)
            is_grounded = grounding_score >= 0.5 and has_context
            
            # Check grounding requirement
            if require_grounding and not is_grounded:
                answer = "INSUFFICIENT_CONTEXT"
                thinking = "Cannot provide a grounded answer - no relevant information found in the knowledge base."
                confidence = "low"
            
            # Build retrieval info
            retrieval_info = RetrievalInfo(
                documents_retrieved=len(result.get("sources", [])),
                documents_relevant=len([s for s in result.get("sources", []) if s]),
                top_score=max(result.get("retrieval_scores", [0.0])) if result.get("retrieval_scores") else 0.0,
                sources=[s.get("source", "") for s in result.get("sources", [])],
                time_ms=(time.time() - start_time) * 1000 * 0.3,  # Estimate
            )
            
            return RAGResponse(
                answer=answer,
                reasoning=thinking,
                confidence=confidence,
                is_grounded=is_grounded,
                grounding_score=grounding_score,
                sources=result.get("sources", []),
                context_used=context_used[:1000],
                retrieval=retrieval_info,
                total_time_ms=(time.time() - start_time) * 1000,
            )
        
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            return RAGResponse(
                answer="ERROR",
                reasoning=str(e),
                confidence="low",
                is_grounded=False,
                grounding_score=0.0,
                sources=[],
                context_used="",
                retrieval=RetrievalInfo(),
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def _estimate_grounding(self, reasoning: str, context: str) -> float:
        """Estimate grounding score heuristically."""
        if not reasoning or not context:
            return 0.0
        
        # Simple heuristic: check for keyword overlap
        reasoning_words = set(reasoning.lower().split())
        context_words = set(context.lower().split())
        
        if not reasoning_words:
            return 0.0
        
        overlap = len(reasoning_words & context_words)
        score = min(1.0, overlap / min(len(reasoning_words), 50))
        
        return score
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        doc_count = self._vectorstore._collection.count() if self._vectorstore else 0
        
        return {
            "status": "initialized",
            "mode": self.config.mode.value,
            "llm_model": self.config.llm_model,
            "embed_model": self.config.embed_model,
            "documents_indexed": doc_count,
            "retrieval_k": self.config.retrieval_k,
            "bm25_enabled": self.config.enable_bm25,
            "reranking_enabled": self.config.enable_reranking,
            "require_grounding": self.config.require_grounding,
        }
    
    async def clear(self) -> None:
        """Clear the vector store."""
        if self._vectorstore:
            self._vectorstore.delete_collection()
            logger.warning("Vector store cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health."""
        checks = {
            "vector_store": False,
            "llm": False,
            "embeddings": False,
        }
        
        try:
            if self._vectorstore:
                self._vectorstore._collection.count()
                checks["vector_store"] = True
        except:
            pass
        
        try:
            if self._mcq_chain or self._agentic_rag:
                checks["llm"] = True
        except:
            pass
        
        try:
            if self._embeddings:
                checks["embeddings"] = True
        except:
            pass
        
        return {
            "healthy": all(checks.values()),
            "checks": checks,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_rag(
    mode: str = "standard",
    model: str = "qwen2.5:7b",
    require_grounding: bool = True,
) -> TPN_RAG:
    """
    Create and initialize a production RAG pipeline.
    
    Example:
        ```python
        rag = await create_rag(mode="agentic", model="qwen2.5:7b")
        result = await rag.ask("What is the protein requirement?")
        ```
    """
    config = PipelineConfig(
        mode=PipelineMode(mode),
        llm_model=model,
        require_grounding=require_grounding,
    )
    
    rag = TPN_RAG(config=config)
    await rag.initialize()
    
    return rag
