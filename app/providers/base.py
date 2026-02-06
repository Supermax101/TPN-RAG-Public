"""
Abstract base classes defining interfaces for providers.
All concrete implementations must inherit from these.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..data_models import DocumentChunk


# ---------------------------------------------------------------------------
# Shared data classes (migrated from app.models.base)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for model inference."""
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    system_prompt: Optional[str] = None
    include_thinking: bool = True
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    answer: str
    thinking: Optional[str] = None
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "thinking": self.thinking,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Async provider interfaces (used by app.providers.openai, gemini, etc.)
# ---------------------------------------------------------------------------

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
    """Interface for large language model text generation (async)."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        pass

    @abstractmethod
    async def get_available_models(self) -> List[str]:
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        pass


# ---------------------------------------------------------------------------
# Sync provider base (used by HuggingFaceProvider)
# ---------------------------------------------------------------------------

# Prompt templates for sync providers
RAG_PROMPT_TEMPLATE = """Answer the following question about Total Parenteral Nutrition (TPN) using the provided context.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, accurate answer. If the context doesn't contain enough information, say so."""

BASELINE_PROMPT_TEMPLATE = """Answer the following question about Total Parenteral Nutrition (TPN) based on your knowledge.

QUESTION: {question}

Provide a clear, accurate answer. If you're uncertain, indicate your level of confidence."""

DEFAULT_SYSTEM_PROMPT = """You are a clinical nutrition expert specializing in Total Parenteral Nutrition (TPN).
Answer questions accurately and concisely based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite specific values, dosages, and guidelines when available."""


class SyncLLMProvider(ABC):
    """
    Synchronous LLM provider base class.
    Used by HuggingFaceProvider and any future sync providers.
    """

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        self.model_name = model_name
        self.config = config or ModelConfig()
        self._initialized = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    def _initialize(self) -> bool:
        ...

    @abstractmethod
    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        ...

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        use_rag: bool = True,
    ) -> LLMResponse:
        if not self._initialized:
            if not self._initialize():
                return LLMResponse(
                    answer="Error: Failed to initialize model provider",
                    model=self.model_name,
                    provider=self.provider_name,
                )

        if use_rag and context:
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        else:
            prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)

        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        try:
            response = self._generate_impl(prompt, system_prompt)
            response.model = self.model_name
            response.provider = self.provider_name
            return response
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Generation failed: {e}")
            return LLMResponse(
                answer=f"Error: {str(e)}",
                model=self.model_name,
                provider=self.provider_name,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"
