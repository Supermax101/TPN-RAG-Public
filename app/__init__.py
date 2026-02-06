"""
RAG application package.
Provides document retrieval and question answering using vector search and LLMs.

PRODUCTION USAGE:
=================
```python
from app import TPN_RAG, create_rag

# Quick start
rag = await create_rag()
await rag.ingest_pdf("path/to/book.pdf")
result = await rag.ask("What is the protein requirement?")
print(result.answer)
```

MODULAR USAGE (no LangChain dependencies):
==========================================
```python
# Use specific modules without importing everything
from app.retrieval import RetrievalPipeline, HybridRetriever
from app.ingestion import IngestionPipeline, SemanticChunker
from app.evaluation import BenchmarkRunner
```

LangChain 1.x Pipeline:
- rag_pipeline: Main production entry point (TPN_RAG)
- document_processing: Semantic chunking and document preparation
- chains: LCEL chains for retrieval and MCQ answering
- parsers: Structured output models and parsing
"""
__version__ = "2.1.0"

# Lazy imports to avoid loading LangChain dependencies unless needed
# This allows submodules (retrieval, ingestion, evaluation) to work independently

__all__ = [
    # Primary interface (requires LangChain)
    "TPN_RAG",
    "create_rag",
    "RAGResponse",
    "PipelineConfig",
    "PipelineMode",
    # Chains (requires LangChain)
    "MCQChain",
    "create_mcq_chain",
    "RetrievalChain",
    "create_retrieval_chain",
    "AgenticMCQRAG",
    "create_agentic_mcq_rag",
    # Parsers
    "MCQAnswer",
    "MCQMultiAnswer",
    "parse_mcq_response",
]


def __getattr__(name: str):
    """Lazy import to avoid loading LangChain unless needed."""
    # Primary interface
    if name in ("TPN_RAG", "create_rag", "RAGResponse", "PipelineConfig", "PipelineMode"):
        from .rag_pipeline import TPN_RAG, create_rag, RAGResponse, PipelineConfig, PipelineMode
        return locals()[name]

    # Chains
    if name in ("MCQChain", "create_mcq_chain", "RetrievalChain", "create_retrieval_chain"):
        from .chains import MCQChain, create_mcq_chain, RetrievalChain, create_retrieval_chain
        return locals()[name]
    if name in ("AgenticMCQRAG", "create_agentic_mcq_rag"):
        from .chains import AgenticMCQRAG, create_agentic_mcq_rag
        return locals()[name]

    # Parsers
    if name in ("MCQAnswer", "MCQMultiAnswer", "parse_mcq_response"):
        from .parsers import MCQAnswer, MCQMultiAnswer, parse_mcq_response
        return locals()[name]

    raise AttributeError(f"module 'app' has no attribute '{name}'")
