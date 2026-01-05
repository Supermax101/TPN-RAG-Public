"""
Embedding provider implementations.
Converts text into vector representations for similarity search.
"""
import asyncio
import httpx
from typing import List, Optional
from .base import EmbeddingProvider
from ..config import settings


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Ollama's local embedding models."""
    
    def __init__(self, model: str = None, base_url: str = None, max_concurrent: int = 10):
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self._dimension = None
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Use provided model, or from settings, or auto-detect from Ollama
        if model:
            self.model = model
            print(f"Ollama Embedding: {model} (specified)")
        elif settings.ollama_embed_model:
            self.model = settings.ollama_embed_model
            print(f"Ollama Embedding: {settings.ollama_embed_model} (from .env)")
        else:
            self.model = self._auto_select_embedding_model()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=180.0) as client:
            tasks = [self._embed_with_semaphore(client, text, idx, len(texts)) 
                     for idx, text in enumerate(texts)]
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed = []
            failed = 0
            for emb in embeddings:
                if isinstance(emb, Exception):
                    failed += 1
                    processed.append([0.0] * self._dimension if self._dimension else None)
                else:
                    processed.append(emb)
            
            if self._dimension:
                processed = [e if e is not None else [0.0] * self._dimension for e in processed]
            
            if failed > 0:
                print(f"Warning: {failed}/{len(texts)} embeddings failed")
            
            return processed
    
    async def embed_query(self, query: str) -> List[float]:
        async with httpx.AsyncClient(timeout=180.0) as client:
            return await self._embed_with_retry(client, query, max_retries=3)
    
    async def _embed_with_semaphore(self, client: httpx.AsyncClient, text: str, 
                                     idx: int, total: int) -> List[float]:
        async with self.semaphore:
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"Embedding {idx+1}/{total}")
            return await self._embed_with_retry(client, text, max_retries=3)
    
    async def _embed_with_retry(self, client: httpx.AsyncClient, text: str, 
                                 max_retries: int = 3) -> List[float]:
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self._embed_single(client, text)
            except httpx.ReadTimeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
    
    async def _embed_single(self, client: httpx.AsyncClient, text: str) -> List[float]:
        response = await client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding", [])
        
        if self._dimension is None and embedding:
            self._dimension = len(embedding)
            
        return embedding
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Dimension unknown - generate at least one embedding first")
        return self._dimension
    
    def _auto_select_embedding_model(self) -> str:
        """Auto-detect available embedding models from Ollama server."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                all_models = [m["name"] for m in response.json().get("models", [])]
                
                # Filter for embedding models
                embed_models = [m for m in all_models if "embed" in m.lower()]
                
                if embed_models:
                    print(f"\nAvailable Ollama Embedding models ({len(embed_models)}):")
                    for i, model in enumerate(embed_models, 1):
                        print(f"  {i}. {model}")
                    print(f"\nUsing: {embed_models[0]}")
                    print("(Set OLLAMA_EMBED_MODEL in .env to change default)\n")
                    return embed_models[0]
                else:
                    print("No embedding models found in Ollama.")
                    print("Run: ollama pull nomic-embed-text (or qwen3-embedding:8b)")
        except Exception as e:
            print(f"Could not connect to Ollama: {e}")
        
        return settings.ollama_embed_model or "nomic-embed-text"
