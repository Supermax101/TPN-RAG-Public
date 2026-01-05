"""
Abstract base classes defining interfaces for providers.
All concrete implementations must inherit from these.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..data_models import DocumentChunk


class EmbeddingProvider(ABC):
    """Interface for text embedding generation."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class VectorStore(ABC):
    """Interface for vector storage and similarity search."""
    
    @abstractmethod
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_name: str
    ) -> None:
        pass
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass


class LLMProvider(ABC):
    """Interface for large language model text generation."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        pass
