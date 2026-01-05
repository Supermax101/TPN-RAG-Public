"""
API request and response schemas.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    filters: Dict[str, Any] = Field(default_factory=dict)


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    search_limit: int = Field(default=5, ge=1, le=10)
    model: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0"
    services: Dict[str, bool] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    collection_name: str
    embedding_model: str
