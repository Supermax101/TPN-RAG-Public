"""
Data models for document processing.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    
    source_file: str = Field(description="Original source filename")
    section_header: Optional[str] = Field(default=None, description="Section heading")
    page_number: Optional[int] = Field(default=None, description="Page number in source")
    chunk_index: int = Field(description="Index of chunk within document")
    total_chunks: int = Field(description="Total chunks in document")
    chunk_type: str = Field(default="text", description="Type: text, table, figure")
    start_char: Optional[int] = Field(default=None, description="Start character position")
    end_char: Optional[int] = Field(default=None, description="End character position")
    
    # Clinical-specific metadata
    clinical_topic: Optional[str] = Field(default=None, description="TPN topic category")
    contains_dosing: bool = Field(default=False, description="Contains dosing information")
    contains_calculation: bool = Field(default=False, description="Contains calculation/formula")


class ProcessedDocument(BaseModel):
    """A fully processed document ready for vectorization."""
    
    doc_id: str = Field(description="Unique document identifier")
    doc_name: str = Field(description="Human-readable document name")
    content: str = Field(description="Full document content")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Processed chunks")
    total_chunks: int = Field(default=0, description="Number of chunks")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    
    class Config:
        extra = "allow"
