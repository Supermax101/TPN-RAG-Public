"""
Data models for document chunks, search queries, and RAG responses.
Used throughout the application for type safety and validation.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with associated metadata."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str = "text"
    page_num: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class SearchResult(BaseModel):
    """A single search result containing a chunk and its relevance score."""
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0)
    document_name: str
    
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
    
    @property
    def content(self) -> str:
        return self.chunk.content
    
    class Config:
        frozen = True


class SearchQuery(BaseModel):
    """Parameters for a vector similarity search."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True


class SearchResponse(BaseModel):
    """Response containing search results and timing information."""
    query: SearchQuery
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = Field(ge=0)
    search_time_ms: float = Field(ge=0)
    model_used: Optional[str] = None
    
    class Config:
        frozen = True


class RAGQuery(BaseModel):
    """Parameters for a RAG query combining retrieval and generation."""
    question: str = Field(..., min_length=1)
    search_limit: int = Field(default=5, ge=1, le=20)
    model: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    
    class Config:
        frozen = True


class RAGResponse(BaseModel):
    """Response containing the generated answer and source documents."""
    question: str
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    search_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
    model_used: str
    
    class Config:
        frozen = True


@dataclass
class DocumentEmbeddings:
    """Container for document chunks and their vector embeddings."""
    doc_id: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    model_used: str = "unknown"
