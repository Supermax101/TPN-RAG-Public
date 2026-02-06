"""
FastAPI dependency injection.
Uses HuggingFace for embeddings and LLMs.
"""
import asyncio
from functools import lru_cache

from ..services.rag import RAGService
from ..providers import HuggingFaceEmbeddingProvider, ChromaVectorStore
from ..config import settings


class AsyncModelProviderAdapter:
    """Adapter that bridges sync model providers to async RAGService contract."""

    def __init__(self, provider):
        self.provider = provider

    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        # The legacy model providers are synchronous and use question/context style API.
        # For compatibility, pass the fully formatted prompt as the question.
        response = await asyncio.to_thread(
            self.provider.generate,
            question=prompt,
            context=None,
            use_rag=False,
        )
        return response.answer if hasattr(response, "answer") else str(response)

    @property
    def available_models(self):
        return [getattr(self.provider, "model_name", "default")]


@lru_cache()
def get_rag_service() -> RAGService:
    """Creates and caches RAG service instance."""
    embedding_provider = HuggingFaceEmbeddingProvider()
    vector_store = ChromaVectorStore()

    from ..providers.huggingface import HuggingFaceProvider
    llm_provider = AsyncModelProviderAdapter(
        HuggingFaceProvider(model_name=settings.hf_llm_model)
    )

    return RAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider
    )


async def check_services_health() -> dict:
    """Checks health of dependent services."""
    health = {
        "chromadb": False,
        "huggingface": True  # HuggingFace is always available
    }

    try:
        vector_store = ChromaVectorStore()
        await vector_store.get_stats()
        health["chromadb"] = True
    except Exception:
        pass

    return health
