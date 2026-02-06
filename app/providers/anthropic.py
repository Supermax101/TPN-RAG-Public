"""
Anthropic (Claude) LLM provider implementation.

Uses the official ``anthropic`` SDK (AsyncAnthropic) for native
system-prompt support, automatic retries, and structured error types.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from .base import LLMProvider
from ..config import settings

logger = logging.getLogger(__name__)


class AnthropicLLMProvider(LLMProvider):
    """Anthropic-based LLM provider (Claude models)."""

    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-sonnet-4-5-20250514"):
        self.api_key = api_key or settings.anthropic_api_key
        self.default_model = default_model
        self._available_models = None

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text response using Anthropic Claude."""
        model_name = model or self.default_model

        # Normalize model name (handle both formats)
        if model_name == "claude-sonnet-4-5":
            model_name = "claude-sonnet-4-5-20250514"

        try:
            kwargs: dict = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            # Only add temperature for non-thinking models
            if temperature > 0:
                kwargs["temperature"] = temperature

            message = await self.client.messages.create(**kwargs)

            for block in message.content:
                if block.type == "text":
                    return block.text.strip()

            raise RuntimeError(f"Unexpected Anthropic response format: {message}")

        except Exception as e:
            if "anthropic" in type(e).__module__:
                raise RuntimeError(f"Failed to generate text with Anthropic: {e}")
            raise RuntimeError(f"Failed to generate text with Anthropic: {e}")

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate structured output using Anthropic tool_use with schema as tool input."""
        model_name = model or self.default_model
        if model_name == "claude-sonnet-4-5":
            model_name = "claude-sonnet-4-5-20250514"

        try:
            kwargs: dict = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [
                    {
                        "name": "mcq_answer",
                        "description": "Provide a structured MCQ answer",
                        "input_schema": schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "mcq_answer"},
            }

            if system_prompt:
                kwargs["system"] = system_prompt
            if temperature > 0:
                kwargs["temperature"] = temperature

            message = await self.client.messages.create(**kwargs)

            for block in message.content:
                if block.type == "tool_use" and block.name == "mcq_answer":
                    return block.input

            raise RuntimeError(f"No tool_use block in Anthropic response: {message}")

        except Exception as e:
            raise RuntimeError(f"Structured output failed with Anthropic: {e}")

    async def get_available_models(self) -> List[str]:
        """Return list of available Anthropic models."""
        if self._available_models is not None:
            return self._available_models

        self._available_models = [
            "claude-sonnet-4-5-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
        return self._available_models

    async def check_health(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            await self.generate("Hello", max_tokens=5)
            return True
        except Exception:
            return False
