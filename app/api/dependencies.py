"""
FastAPI dependency injection.
Uses HuggingFace for embeddings and LLMs.
"""
from functools import lru_cache

from ..services.rag import RAGService
from ..providers import HuggingFaceEmbeddingProvider, ChromaVectorStore
from ..config import settings


@lru_cache()
def get_rag_service() -> RAGService:
    """Creates and caches RAG service instance."""
    embedding_provider = HuggingFaceEmbeddingProvider()
    vector_store = ChromaVectorStore()

    # Use HuggingFace provider from app.models
    from ..models import HuggingFaceProvider
    llm_provider = HuggingFaceProvider(model_name=settings.hf_llm_model)

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
