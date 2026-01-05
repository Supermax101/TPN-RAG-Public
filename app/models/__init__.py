"""
Model providers for TPN RAG system.

Provides unified interface for multiple LLM providers:
- HuggingFace: Open models (Qwen, Llama, Mistral, Gemma, etc.) - DYNAMIC
- OpenAI: GPT-4, GPT-4o
- Anthropic: Claude 3.5, Claude 4

Example usage:
    >>> from app.models import HuggingFaceProvider, create_model
    >>>
    >>> # Direct instantiation with any HuggingFace model
    >>> model = HuggingFaceProvider(model_name="Qwen/Qwen2.5-7B-Instruct")
    >>> response = model.generate("What is TPN?", context="...")
    >>>
    >>> # Factory function
    >>> model = create_model("hf", "Qwen/Qwen2.5-7B-Instruct")
    >>>
    >>> # Discover available models dynamically
    >>> from app.models import search_hf_models
    >>> models = search_hf_models("Qwen instruct", limit=10)
"""

from .base import LLMProvider, LLMResponse, ModelConfig
from .huggingface_provider import (
    HuggingFaceProvider,
    search_models as search_hf_models,
    list_trending_models,
    get_model_info,
    validate_model_id,
)
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ModelConfig",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_model",
    "list_available_models",
    "search_hf_models",
    "list_trending_models",
    "get_model_info",
    "validate_model_id",
]


def create_model(
    provider: str,
    model_name: str,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create a model provider.

    Args:
        provider: Provider name ("huggingface", "hf", "openai", "anthropic")
        model_name: Model identifier (full HuggingFace model ID for open models)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMProvider instance

    Examples:
        >>> create_model("hf", "Qwen/Qwen2.5-7B-Instruct")
        >>> create_model("huggingface", "meta-llama/Llama-3.1-8B-Instruct")
        >>> create_model("openai", "gpt-4o")
        >>> create_model("anthropic", "claude-sonnet-4-20250514")
    """
    providers = {
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,  # Alias
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    if provider not in providers:
        available = list(set(providers.keys()) - {"hf"})  # Don't show alias
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    return providers[provider](model_name=model_name, **kwargs)


def list_available_models(fetch_hf: bool = True, hf_limit: int = 10) -> dict:
    """
    List available models by provider.

    For HuggingFace, fetches trending models dynamically from the Hub.
    For OpenAI and Anthropic, returns commonly used model IDs.

    Args:
        fetch_hf: Whether to fetch HuggingFace models from Hub (requires internet)
        hf_limit: Number of HuggingFace models to fetch

    Returns:
        Dict mapping provider names to lists of model IDs
    """
    result = {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1-preview",
            "o1-mini",
        ],
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ],
    }

    # Fetch HuggingFace models dynamically
    if fetch_hf:
        try:
            hf_models = list_trending_models(limit=hf_limit)
            result["huggingface"] = hf_models if hf_models else ["(fetch failed - check internet)"]
        except Exception:
            result["huggingface"] = ["(fetch failed - check internet)"]
    else:
        result["huggingface"] = ["(use search_hf_models() to discover)"]

    return result
