"""
Embedding provider implementations.
Converts text into vector representations for similarity search.
Uses HuggingFace models via sentence-transformers.
"""
import asyncio
from typing import List, Optional
from .base import EmbeddingProvider
from ..config import settings


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Generates embeddings using HuggingFace models via sentence-transformers.

    Recommended models for clinical/medical domain:
    - Qwen/Qwen3-Embedding-8B (best quality, larger)
    - BAAI/bge-large-en-v1.5 (good balance)
    - sentence-transformers/all-mpnet-base-v2 (fast, general)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        """
        Initialize HuggingFace embedding provider.

        Args:
            model_name: HuggingFace model ID (default from settings)
            device: Device to run on ("cuda", "mps", "cpu", or None for auto)
            max_concurrent: Max concurrent embedding operations
        """
        self._model_name = model_name or settings.hf_embedding_model
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
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        model = self._load_model()

        # Use sync encoding since sentence-transformers handles batching well
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                texts,
                prompt_name="document",  # For Qwen3 instruction-aware embeddings
                show_progress_bar=len(texts) > 100,
                batch_size=32,
            )

        embeddings = await loop.run_in_executor(None, _encode)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        model = self._load_model()

        loop = asyncio.get_event_loop()

        def _encode():
            return model.encode(
                [query],
                prompt_name="query",  # For Qwen3 instruction-aware embeddings
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
            self._load_model()  # This will set _dimension
        return self._dimension


# Alias for backwards compatibility
EmbeddingProvider = HuggingFaceEmbeddingProvider
