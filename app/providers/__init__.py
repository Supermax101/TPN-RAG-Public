"""
External service providers for embeddings, LLMs, and vector storage.

LLM provider classes are lazily imported so that a missing SDK (e.g.
google-genai) does not break unrelated commands that never touch that
provider.
"""
from .base import ModelConfig, LLMResponse, SyncLLMProvider

__all__ = [
    "ModelConfig",
    "LLMResponse",
    "SyncLLMProvider",
    "HuggingFaceEmbeddingProvider",
    "ChromaVectorStore",
    "OpenAILLMProvider",
    "GeminiLLMProvider",
    "XAILLMProvider",
    "KimiLLMProvider",
    "AnthropicLLMProvider",
    "HuggingFaceProvider",
    "search_hf_models",
    "list_trending_models",
    "get_model_info",
    "validate_model_id",
]

# Lazy import map: attribute name -> (module, real_name)
_LAZY_IMPORTS = {
    "HuggingFaceEmbeddingProvider": (".embeddings", "HuggingFaceEmbeddingProvider"),
    "ChromaVectorStore": (".vectorstore", "ChromaVectorStore"),
    "OpenAILLMProvider": (".openai", "OpenAILLMProvider"),
    "GeminiLLMProvider": (".gemini", "GeminiLLMProvider"),
    "XAILLMProvider": (".xai", "XAILLMProvider"),
    "KimiLLMProvider": (".kimi", "KimiLLMProvider"),
    "AnthropicLLMProvider": (".anthropic", "AnthropicLLMProvider"),
    "HuggingFaceProvider": (".huggingface", "HuggingFaceProvider"),
    "search_hf_models": (".huggingface", "search_models"),
    "list_trending_models": (".huggingface", "list_trending_models"),
    "get_model_info": (".huggingface", "get_model_info"),
    "validate_model_id": (".huggingface", "validate_model_id"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, real_name = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __package__)
        attr = getattr(mod, real_name)
        # Cache on the module so __getattr__ isn't called again
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
