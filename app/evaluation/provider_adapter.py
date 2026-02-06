"""
Unified async provider adapters for benchmark experiments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    tokens_used: int = 0


class ProviderAdapter(Protocol):
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        ...


# Per-provider temperature rules (hard-won from prior evaluation runs)
# GPT-5+: API errors if you send temperature at all → None (skip param)
# Gemini 3: lower values cause issues → force 1.0
# Kimi K2.5: thinking mode requires 1.0
_TEMPERATURE_OVERRIDES = {
    "gpt-5": None,   # matched by substring
    "gemini-3": 1.0,
    "kimi-k2.5": 1.0,
    "kimi-k2-5": 1.0,
}

# Per-provider concurrency / delay hints
PROVIDER_RATE_LIMITS = {
    "openai":    {"max_concurrent": 5, "delay": 0.2},
    "anthropic": {"max_concurrent": 5, "delay": 0.5},
    "google":    {"max_concurrent": 2, "delay": 2.0},
    "gemini":    {"max_concurrent": 2, "delay": 2.0},
    "xai":       {"max_concurrent": 5, "delay": 0.2},
    "kimi":      {"max_concurrent": 3, "delay": 1.0},
}


def _apply_temperature_override(model_name: str, temperature: float) -> Optional[float]:
    """Return the effective temperature for a model.

    Returns None if the model must NOT receive a temperature parameter.
    """
    lower = model_name.lower()
    for pattern, override in _TEMPERATURE_OVERRIDES.items():
        if pattern in lower:
            return override
    return temperature


class AsyncProviderWrapper:
    """Wrap app.providers.* async providers with a unified benchmark signature."""

    def __init__(self, provider, default_model: str):
        self.provider = provider
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        effective_model = model_id or self.default_model
        effective_temp = _apply_temperature_override(effective_model, temperature)

        start = time.time()
        kwargs = dict(
            prompt=prompt,
            system_prompt=system,
            model=effective_model,
            max_tokens=max_tokens,
            seed=seed,
        )
        # Only pass temperature when the model accepts it
        if effective_temp is not None:
            kwargs["temperature"] = effective_temp
        text = await self.provider.generate(**kwargs)
        return GenerationResult(text=text, latency_ms=(time.time() - start) * 1000)

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Try provider's native structured output; fall back to text + JSON parse.
        """
        try:
            return await self.provider.generate_structured(
                prompt=prompt,
                schema=schema,
                model=model_id or self.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system,
            )
        except NotImplementedError:
            pass
        except Exception as e:
            logger.debug("Native structured output failed, falling back to text: %s", e)

        # Fallback: ask for JSON in the prompt, parse the response
        json_prompt = f"{prompt}\n\nRespond with ONLY valid JSON matching this schema: {json.dumps(schema)}"
        text = await self.provider.generate(
            prompt=json_prompt,
            system_prompt=system,
            model=model_id or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Try to extract JSON from the response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


class SyncModelWrapper:
    """Wrap app.providers.huggingface sync providers with async interface."""

    def __init__(self, provider):
        self.provider = provider
        self._initialized = False

    def _ensure_init(self):
        """Lazily initialize the underlying provider (model download, GPU load, etc.)."""
        if not self._initialized:
            if not self.provider._initialized:
                if not self.provider._initialize():
                    raise RuntimeError(f"Failed to initialize {self.provider.model_name}")
            self._initialized = True

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        self._ensure_init()
        start = time.time()
        response = await asyncio.to_thread(
            self.provider._generate_impl,
            prompt=prompt,
            system_prompt=system,
        )
        text = response.answer if hasattr(response, "answer") else str(response)
        tokens = getattr(response, "tokens_used", 0)
        latency = getattr(response, "latency_ms", (time.time() - start) * 1000)
        return GenerationResult(text=text, latency_ms=latency, tokens_used=tokens)


def create_provider_adapter(provider: str, model_name: str, api_key_env: Optional[str] = None):
    """
    Create a unified async provider adapter from model spec.

    Supported providers:
    - openai, gemini, kimi, xai, anthropic (app.providers async)
    - huggingface (app.models sync wrapped as async)
    """
    provider = provider.lower()

    if provider == "openai":
        from ..providers.openai import OpenAILLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.openai_api_key
        return AsyncProviderWrapper(OpenAILLMProvider(api_key=key, default_model=model_name), model_name)

    if provider == "gemini":
        from ..providers.gemini import GeminiLLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.gemini_api_key
        return AsyncProviderWrapper(GeminiLLMProvider(api_key=key, default_model=model_name), model_name)

    if provider == "kimi":
        from ..providers.kimi import KimiLLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.kimi_api_key
        return AsyncProviderWrapper(KimiLLMProvider(api_key=key, default_model=model_name), model_name)

    if provider == "xai":
        from ..providers.xai import XAILLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.xai_api_key
        return AsyncProviderWrapper(XAILLMProvider(api_key=key, default_model=model_name), model_name)

    if provider in {"huggingface", "hf"}:
        from ..providers.huggingface import HuggingFaceProvider

        # Auto-detect: 100B+ models need 4-bit quantization on single GPU
        param_hint = model_name.lower()
        needs_quant = any(s in param_hint for s in ["120b", "100b", "70b", "72b"])
        return SyncModelWrapper(
            HuggingFaceProvider(
                model_name=model_name,
                use_local=True,
                device="cuda",
                quantize_4bit=needs_quant,
            )
        )

    if provider == "anthropic":
        from ..providers.anthropic import AnthropicLLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.anthropic_api_key
        return AsyncProviderWrapper(AnthropicLLMProvider(api_key=key, default_model=model_name), model_name)

    raise ValueError(f"Unsupported provider: {provider}")
