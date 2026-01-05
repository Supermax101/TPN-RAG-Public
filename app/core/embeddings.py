"""
Production-grade embedding management with support for multiple providers.

Supports:
- Qwen3-Embedding (via Ollama) - MTEB #1, best open-source
- BGE-M3 (via HuggingFace) - Excellent hybrid retrieval
- OpenAI text-embedding-3-large - Best commercial option
- Sentence Transformers (local) - Flexible local deployment
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import numpy as np

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from ..config import settings
from ..logger import logger


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.
    
    Recommended models by use case:
    - Best overall: Qwen/Qwen3-Embedding-8B (via Ollama as qwen3-embedding:8b)
    - Best hybrid: BAAI/bge-m3 (supports dense + sparse + multi-vector)
    - Best commercial: text-embedding-3-large (OpenAI)
    - Best lightweight: nomic-embed-text (768d, fast)
    """
    
    provider: EmbeddingProvider = EmbeddingProvider.OLLAMA
    model_name: str = "qwen3-embedding:8b"
    
    # Dimension of embeddings (auto-detected if None)
    dimension: Optional[int] = None
    
    # Batch settings
    batch_size: int = 32
    max_concurrent: int = 10
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Provider-specific settings
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    
    # HuggingFace settings
    hf_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    hf_encode_kwargs: Dict[str, Any] = field(default_factory=lambda: {"normalize_embeddings": True})
    
    # Instruction prefix for instruction-following models
    query_instruction: Optional[str] = None
    document_instruction: Optional[str] = None


class EmbeddingManager:
    """Manages embedding generation with multiple provider support.
    
    This is the main interface for embeddings. It wraps LangChain embeddings
    with additional features:
    - Automatic batching with concurrency control
    - Retry logic with exponential backoff
    - Dimension validation
    - Provider-agnostic interface
    """
    
    # Recommended models for clinical/medical RAG
    RECOMMENDED_MODELS = {
        "best_open_source": {
            "provider": EmbeddingProvider.OLLAMA,
            "model": "qwen3-embedding:8b",
            "description": "MTEB #1, 8192 context, Apache 2.0"
        },
        "best_hybrid": {
            "provider": EmbeddingProvider.HUGGINGFACE,
            "model": "BAAI/bge-m3",
            "description": "Dense + sparse + multi-vector, 8192 context"
        },
        "best_commercial": {
            "provider": EmbeddingProvider.OPENAI,
            "model": "text-embedding-3-large",
            "description": "3072d, best quality, pay-per-use"
        },
        "best_lightweight": {
            "provider": EmbeddingProvider.OLLAMA,
            "model": "nomic-embed-text",
            "description": "768d, fast, good for development"
        },
    }
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._embeddings: Optional[Embeddings] = None
        self._dimension: Optional[int] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        logger.info(f"Initializing EmbeddingManager with {self.config.provider.value}/{self.config.model_name}")
    
    @property
    def embeddings(self) -> Embeddings:
        """Lazy initialization of the embedding model."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (auto-detected on first use)."""
        if self._dimension is None:
            # Generate a test embedding to detect dimension
            test_embedding = self.embed_query("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name
    
    def _create_embeddings(self) -> Embeddings:
        """Create the appropriate embedding instance based on provider."""
        
        if self.config.provider == EmbeddingProvider.OLLAMA:
            return self._create_ollama_embeddings()
        
        elif self.config.provider == EmbeddingProvider.OPENAI:
            return self._create_openai_embeddings()
        
        elif self.config.provider == EmbeddingProvider.HUGGINGFACE:
            return self._create_huggingface_embeddings()
        
        elif self.config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self._create_sentence_transformers_embeddings()
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _create_ollama_embeddings(self) -> Embeddings:
        """Create Ollama embeddings (for Qwen3-Embedding, nomic-embed-text, etc.)."""
        try:
            from langchain_ollama import OllamaEmbeddings
            
            return OllamaEmbeddings(
                model=self.config.model_name,
                base_url=self.config.ollama_base_url,
            )
        except ImportError:
            # Fallback to community version
            from langchain_community.embeddings import OllamaEmbeddings
            
            return OllamaEmbeddings(
                model=self.config.model_name,
                base_url=self.config.ollama_base_url,
            )
    
    def _create_openai_embeddings(self) -> Embeddings:
        """Create OpenAI embeddings."""
        from langchain_openai import OpenAIEmbeddings
        
        api_key = self.config.openai_api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        
        return OpenAIEmbeddings(
            model=self.config.model_name,
            openai_api_key=api_key,
        )
    
    def _create_huggingface_embeddings(self) -> Embeddings:
        """Create HuggingFace embeddings (for BGE-M3, etc.)."""
        from langchain_huggingface import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs=self.config.hf_model_kwargs,
            encode_kwargs=self.config.hf_encode_kwargs,
        )
    
    def _create_sentence_transformers_embeddings(self) -> Embeddings:
        """Create Sentence Transformers embeddings."""
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        return SentenceTransformerEmbeddings(
            model_name=self.config.model_name,
        )
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text synchronously."""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents synchronously."""
        return self.embeddings.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query text asynchronously with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:
                    # Use asyncio.to_thread for sync embeddings
                    return await asyncio.to_thread(self.embed_query, text)
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Embedding failed after {self.config.max_retries} attempts: {e}")
                    raise
    
    async def aembed_documents(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed multiple documents asynchronously with batching and progress."""
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (total + self.config.batch_size - 1) // self.config.batch_size
            
            if show_progress and (batch_num == 1 or batch_num % 5 == 0 or batch_num == total_batches):
                logger.info(f"Embedding batch {batch_num}/{total_batches} ({i + len(batch)}/{total} texts)")
            
            # Process batch with retry
            for attempt in range(self.config.max_retries):
                try:
                    async with self._semaphore:
                        batch_embeddings = await asyncio.to_thread(
                            self.embed_documents, batch
                        )
                        all_embeddings.extend(batch_embeddings)
                        break
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Batch {batch_num} attempt {attempt + 1} failed: {e}. Retrying...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Batch {batch_num} failed after {self.config.max_retries} attempts")
                        raise
        
        return all_embeddings
    
    @classmethod
    def from_preset(cls, preset: str) -> "EmbeddingManager":
        """Create an EmbeddingManager from a recommended preset.
        
        Available presets:
        - "best_open_source": Qwen3-Embedding-8B (MTEB #1)
        - "best_hybrid": BGE-M3 (dense + sparse)
        - "best_commercial": OpenAI text-embedding-3-large
        - "best_lightweight": nomic-embed-text (fast, development)
        """
        if preset not in cls.RECOMMENDED_MODELS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.RECOMMENDED_MODELS.keys())}")
        
        preset_config = cls.RECOMMENDED_MODELS[preset]
        
        config = EmbeddingConfig(
            provider=preset_config["provider"],
            model_name=preset_config["model"],
        )
        
        logger.info(f"Using embedding preset '{preset}': {preset_config['description']}")
        return cls(config)
    
    @staticmethod
    def list_available_ollama_models() -> List[str]:
        """List embedding models available in Ollama."""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                embed_models = [m["name"] for m in models if "embed" in m["name"].lower()]
                return embed_models
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
        return []


# Factory function for easy creation
def create_embeddings(
    provider: str = "ollama",
    model: str = "qwen3-embedding:8b",
    **kwargs
) -> EmbeddingManager:
    """Create an EmbeddingManager with the specified configuration.
    
    Args:
        provider: One of "ollama", "openai", "huggingface", "sentence_transformers"
        model: Model name (e.g., "qwen3-embedding:8b", "text-embedding-3-large")
        **kwargs: Additional configuration options
    
    Returns:
        Configured EmbeddingManager instance
    
    Example:
        # Best open source (MTEB #1)
        embeddings = create_embeddings("ollama", "qwen3-embedding:8b")
        
        # OpenAI commercial
        embeddings = create_embeddings("openai", "text-embedding-3-large")
        
        # HuggingFace BGE-M3
        embeddings = create_embeddings("huggingface", "BAAI/bge-m3")
    """
    config = EmbeddingConfig(
        provider=EmbeddingProvider(provider),
        model_name=model,
        **kwargs
    )
    return EmbeddingManager(config)
