"""
Anthropic (Claude) LLM provider implementation.
"""
from typing import List, Optional
import httpx
from .base import LLMProvider
from ..config import settings


class AnthropicLLMProvider(LLMProvider):
    """Anthropic-based LLM provider (Claude models)."""

    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-sonnet-4-5-20250514"):
        self.api_key = api_key or settings.anthropic_api_key
        self.default_model = default_model
        self.base_url = "https://api.anthropic.com/v1"
        self._available_models = None

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using Anthropic Claude."""
        model_name = model or self.default_model

        # Normalize model name (handle both formats)
        if model_name == "claude-sonnet-4-5":
            model_name = "claude-sonnet-4-5-20250514"

        try:
            url = f"{self.base_url}/messages"

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            payload = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

            # Only add temperature for non-thinking models
            # Anthropic reasoning models may not support temperature
            if temperature > 0:
                payload["temperature"] = temperature

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()

                if "content" in result and len(result["content"]) > 0:
                    for block in result["content"]:
                        if block.get("type") == "text":
                            return block.get("text", "").strip()

                raise RuntimeError(f"Unexpected Anthropic response format: {result}")

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
            raise RuntimeError(f"Failed to generate text with Anthropic (HTTP {e.response.status_code}): {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Anthropic: {e}")

    async def get_available_models(self) -> List[str]:
        """Return list of available Anthropic models."""
        if self._available_models is not None:
            return self._available_models

        # Anthropic doesn't have a models list endpoint, return known models
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
            # Try a minimal request
            await self.generate("Hello", max_tokens=5)
            return True
        except Exception:
            return False
