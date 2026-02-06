"""
Google Gemini LLM provider implementation.

Uses the official ``google-genai`` SDK for native system-instruction
support, async generation, and seed-based reproducibility.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .base import LLMProvider
from ..config import settings

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    """Google Gemini-based LLM provider (Gemini 2.5 Pro, Flash, etc.)."""

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-2.5-flash"):
        self.api_key = api_key or settings.gemini_api_key
        self.default_model = default_model

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self._available_models = None

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text response using Gemini."""
        model_name = model or self.default_model

        if model_name.startswith("models/"):
            model_name = model_name[7:]

        try:
            config_kwargs: dict = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
            }

            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            if seed is not None:
                config_kwargs["seed"] = seed

            config = types.GenerateContentConfig(**config_kwargs)

            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            if response.text:
                return response.text.strip()

            # Check for blocked/empty responses
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason and str(finish_reason) in ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"]:
                    return ""

            raise RuntimeError(f"Unexpected Gemini response format: {response}")

        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Gemini: {e}")

    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate structured JSON output using Gemini's response_mime_type."""
        model_name = model or self.default_model
        if model_name.startswith("models/"):
            model_name = model_name[7:]

        try:
            config_kwargs: dict = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
                "response_schema": schema,
            }

            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt

            config = types.GenerateContentConfig(**config_kwargs)

            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            if response.text:
                return json.loads(response.text)

            raise RuntimeError(f"Empty Gemini structured response: {response}")

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini returned invalid JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Structured output failed with Gemini: {e}")

    async def get_available_models(self) -> List[str]:
        """Return list of available Gemini models from API."""
        if self._available_models is not None:
            return self._available_models

        try:
            chat_models = []
            async for model in self.client.aio.models.list():
                model_name = model.name or ""
                if model_name.startswith("models/"):
                    model_id = model_name[7:]
                else:
                    model_id = model_name

                if model_id in ["gemini-2.5-pro", "gemini-2.5-flash"]:
                    supported = getattr(model, "supported_generation_methods", []) or []
                    if "generateContent" in supported:
                        chat_models.append(model_id)

            chat_models.sort(key=lambda x: 0 if "pro" in x else 1)
            self._available_models = chat_models
            return self._available_models

        except Exception as e:
            logger.warning("Could not fetch Gemini models: %s", e)
            return ["gemini-2.5-pro", "gemini-2.5-flash"]

    async def check_health(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            models = []
            async for m in self.client.aio.models.list():
                models.append(m)
                break
            return True
        except Exception:
            return False
