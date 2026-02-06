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
        start = time.time()
        text = await self.provider.generate(
            prompt=prompt,
            system_prompt=system,
            model=model_id or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
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
    """Wrap app.models.* sync providers with async interface."""

    def __init__(self, provider):
        self.provider = provider

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
        start = time.time()
        if system:
            self.provider.config.system_prompt = system
        response = await asyncio.to_thread(
            self.provider.generate,
            question=prompt,
            context=None,
            use_rag=False,
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

        return SyncModelWrapper(HuggingFaceProvider(model_name=model_name))

    if provider == "anthropic":
        from ..providers.anthropic import AnthropicLLMProvider

        key = os.getenv(api_key_env) if api_key_env else settings.anthropic_api_key
        return AsyncProviderWrapper(AnthropicLLMProvider(api_key=key, default_model=model_name), model_name)

    raise ValueError(f"Unsupported provider: {provider}")
