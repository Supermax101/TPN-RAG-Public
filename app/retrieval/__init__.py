"""
Retrieval module for TPN RAG system.

This module provides advanced retrieval capabilities:
- HybridRetriever: Vector + BM25 with Reciprocal Rank Fusion
- HyDERetriever: Hypothetical Document Embeddings
- MultiQueryRetriever: Query expansion for better recall
- CrossEncoderReranker: Reranking with cross-encoder models
- RetrievalPipeline: Unified pipeline combining all techniques
- CitationGrounder: Fix hallucinated citations from fine-tuned models

Example usage:
    >>> from app.retrieval import RetrievalPipeline
    >>> pipeline = RetrievalPipeline(vector_store, bm25_index)
    >>> results = pipeline.retrieve("protein requirements for preterm infants")

    >>> # For fine-tuned models with citation hallucination
    >>> from app.retrieval import CitationGrounder
    >>> grounder = CitationGrounder()
    >>> grounded = grounder.ground_citations(model_output, retrieved_chunks)
"""

from .hybrid import HybridRetriever, RRFConfig
from .hyde import HyDERetriever
from .multi_query import MultiQueryRetriever
from .reranker import CrossEncoderReranker
from .pipeline import RetrievalPipeline, RetrievalConfig
from .citation_grounding import CitationGrounder, GroundingResult

__all__ = [
    "HybridRetriever",
    "RRFConfig",
    "HyDERetriever",
    "MultiQueryRetriever",
    "CrossEncoderReranker",
    "RetrievalPipeline",
    "RetrievalConfig",
    # Citation grounding for fine-tuned models
    "CitationGrounder",
    "GroundingResult",
]
