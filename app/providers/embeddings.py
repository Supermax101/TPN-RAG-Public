"""
Embedding provider implementations.
Converts text into vector representations for similarity search.
Supports OpenAI (text-embedding-3-large) and HuggingFace models.
"""
import asyncio
import os
from typing import List, Optional
from .base import EmbeddingProvider as EmbeddingProviderBase
from ..config import settings


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """
    Generates embeddings using OpenAI's text-embedding-3-large API.

    This is the default and recommended provider for benchmark accuracy.
    Dimension: 3072 (text-embedding-3-large).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        self._model_name = model_name or settings.embedding_model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._client = None
        # Known dimensions for OpenAI embedding models
        self._dimension = 3072 if "large" in self._model_name else 1536

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> List[float]:
        result = await self.embed_texts([query])
        return result[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddingProvider(EmbeddingProviderBase):
    """
    Generates embeddings using HuggingFace models via sentence-transformers.

    Use for local/offline embedding when OpenAI API is not available.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        self._model_name = model_name or settings.embedding_model
        self._device = device
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                # Auto-detect device
                if self._device is None:
                    if torch.cuda.is_available():
                        self._device = "cuda"
                    elif torch.backends.mps.is_available():
                        self._device = "mps"
                    else:
                        self._device = "cpu"

                print(f"Loading embedding model: {self._model_name} on {self._device}")

                self._model = SentenceTransformer(
                    self._model_name,
                    trust_remote_code=True,
                    device=self._device,
                    model_kwargs={"torch_dtype": torch.bfloat16 if self._device != "cpu" else torch.float32}
                )

                # Get dimension from a test embedding
                test_emb = self._model.encode(["test"], show_progress_bar=False)
                self._dimension = len(test_emb[0])
                print(f"Embedding dimension: {self._dimension}")

            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                texts,
                show_progress_bar=len(texts) > 100,
                batch_size=32,
            )

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        model = self._load_model()
        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                [query],
                show_progress_bar=False,
            )

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings[0].tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension
