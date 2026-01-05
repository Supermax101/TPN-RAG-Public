"""
Ingestion module for TPN RAG system.

This module provides document processing pipeline:
- DocumentCleaner: Remove DPT2 OCR artifacts
- SemanticChunker: Split documents with clinical-aware boundaries
- IngestionPipeline: Orchestrate the full ingestion workflow

Example usage:
    >>> from app.ingestion import IngestionPipeline
    >>> pipeline = IngestionPipeline(docs_dir="/path/to/docs", persist_dir="./data")
    >>> stats = pipeline.run()
"""

from .cleaner import DocumentCleaner, CleaningStats
from .chunker import SemanticChunker, Chunk, ChunkingStats
from .pipeline import IngestionPipeline, IngestionStats

__all__ = [
    "DocumentCleaner",
    "CleaningStats",
    "SemanticChunker",
    "Chunk",
    "ChunkingStats",
    "IngestionPipeline",
    "IngestionStats",
]
