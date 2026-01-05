"""
Base classes and protocols for LLM providers.

Defines the common interface that all providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model inference."""

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0

    # Prompt settings
    system_prompt: Optional[str] = None

    # RAG-specific
    include_thinking: bool = True

    # Provider-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    # Generated content
    answer: str
    thinking: Optional[str] = None

    # Metadata
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0

    # Raw response for debugging
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


# Default system prompt for TPN Q&A
DEFAULT_SYSTEM_PROMPT = """You are a clinical nutrition expert specializing in Total Parenteral Nutrition (TPN).
Answer questions accurately and concisely based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always cite specific values, dosages, and guidelines when available."""

# Citation-aware system prompt (for fine-tuned models that need verifiable citations)
CITATION_SYSTEM_PROMPT = """You are a clinical nutrition expert specializing in Total Parenteral Nutrition (TPN).
Answer questions accurately based on the provided context.

CITATION REQUIREMENTS:
1. ALWAYS cite your sources using this format: [Document Name, p.XX]
2. Include page numbers when available from the context
3. Only state facts that are supported by the provided context
4. If making a claim, immediately follow it with the citation

Example format:
"Protein requirements for preterm infants are 3-4 g/kg/day [ASPEN Guidelines, p.44]."

If the context doesn't contain relevant information, say so clearly."""

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the following question about Total Parenteral Nutrition (TPN) using the provided context.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, accurate answer. If the context doesn't contain enough information, say so."""

# RAG prompt template with citation requirements
RAG_CITATION_TEMPLATE = """Answer the following question about Total Parenteral Nutrition (TPN) using ONLY the provided context.

CONTEXT (with source information):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Cite your sources using [Document Name, p.XX] format
3. If multiple sources support a fact, cite all of them
4. If the context doesn't contain the answer, say "The provided context does not contain information about this topic."

Your answer:"""

# No-RAG prompt template (baseline)
BASELINE_PROMPT_TEMPLATE = """Answer the following question about Total Parenteral Nutrition (TPN) based on your knowledge.

QUESTION: {question}

Provide a clear, accurate answer. If you're uncertain, indicate your level of confidence."""


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - generate(): Generate a response for a question with optional context
    - generate_batch(): Generate responses for multiple questions (optional optimization)
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the provider.

        Args:
            model_name: Model identifier (e.g., "qwen3:8b", "gpt-4o")
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config or ModelConfig()
        self._initialized = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'ollama', 'openai')."""
        ...

    @abstractmethod
    def _initialize(self) -> bool:
        """
        Initialize the provider (lazy loading).

        Returns:
            True if initialization successful
        """
        ...

    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Internal generation implementation.

        Args:
            prompt: Full prompt to send to the model
            system_prompt: System prompt override

        Returns:
            LLMResponse with generated content
        """
        ...

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        use_rag: bool = True,
    ) -> LLMResponse:
        """
        Generate a response for a question.

        Args:
            question: User question
            context: Retrieved context (None for baseline)
            use_rag: Whether to use RAG prompt template

        Returns:
            LLMResponse with answer
        """
        # Initialize if needed
        if not self._initialized:
            if not self._initialize():
                return LLMResponse(
                    answer="Error: Failed to initialize model provider",
                    model=self.model_name,
                    provider=self.provider_name,
                )

        # Build prompt
        if use_rag and context:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )
        else:
            prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)

        # Get system prompt
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Generate
        try:
            response = self._generate_impl(prompt, system_prompt)
            response.model = self.model_name
            response.provider = self.provider_name
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return LLMResponse(
                answer=f"Error: {str(e)}",
                model=self.model_name,
                provider=self.provider_name,
            )

    def generate_batch(
        self,
        questions: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        use_rag: bool = True,
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple questions.

        Default implementation calls generate() sequentially.
        Providers can override for batch optimization.

        Args:
            questions: List of questions
            contexts: List of contexts (same length as questions, or None)
            use_rag: Whether to use RAG prompt template

        Returns:
            List of LLMResponse objects
        """
        if contexts is None:
            contexts = [None] * len(questions)

        results = []
        for q, c in zip(questions, contexts):
            results.append(self.generate(q, c, use_rag))

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"
