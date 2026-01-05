"""
FastAPI dependency injection.
"""
from functools import lru_cache
import httpx

from ..services.rag import RAGService
from ..providers import OllamaEmbeddingProvider, ChromaVectorStore, OllamaLLMProvider
from ..config import settings


@lru_cache()
def get_rag_service() -> RAGService:
    """Creates and caches RAG service instance."""
    embedding_provider = OllamaEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = OllamaLLMProvider(default_model=settings.ollama_llm_model)
    
    return RAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider
    )


async def check_services_health() -> dict:
    """Checks health of dependent services."""
    health = {
        "chromadb": False,
        "ollama": False
    }
    
    try:
        vector_store = ChromaVectorStore()
        await vector_store.get_stats()
        health["chromadb"] = True
    except Exception:
        pass
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/version")
            health["ollama"] = response.status_code == 200
    except Exception:
        pass
    
    return health
