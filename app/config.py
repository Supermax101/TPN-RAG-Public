"""
Application configuration.
Loads settings from environment variables and .env file.
"""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    
    # Embedding model - Qwen3 recommended for clinical/medical domain
    # Use instruction-aware embeddings with "search_query:" / "search_document:" prefixes
    ollama_embed_model: Optional[str] = Field(default="qwen3-embedding:0.6b", alias="OLLAMA_EMBED_MODEL")
    ollama_llm_model: Optional[str] = Field(default="qwen2.5:7b", alias="OLLAMA_LLM_MODEL")
    
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", alias="GEMINI_BASE_URL")
    
    xai_api_key: Optional[str] = Field(default=None, alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    
    kimi_api_key: Optional[str] = Field(default=None, alias="KIMI_API_KEY")
    kimi_base_url: str = Field(default="https://api.moonshot.ai/v1", alias="KIMI_BASE_URL")
    
    # Vector store settings
    chroma_collection_name: str = Field(default="tpn_documents", alias="CHROMA_COLLECTION_NAME")
    default_search_limit: int = Field(default=10, alias="DEFAULT_SEARCH_LIMIT")
    
    # Chunking settings - optimized for clinical documents
    # 1000 chars ~ 200-250 tokens, good for semantic coherence
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # Reranker settings
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    max_concurrent_requests: int = Field(default=10, alias="MAX_CONCURRENT_REQUESTS")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    @property
    def project_root(self) -> Path:
        return Path(__file__).parents[1]
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"
    
    @property
    def chromadb_dir(self) -> Path:
        return self.data_dir / "chromadb"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
