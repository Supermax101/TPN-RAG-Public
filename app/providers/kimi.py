"""
Kimi K2 (Moonshot AI) LLM provider implementation.
Uses OpenAI-compatible API from Moonshot AI platform.
"""
from typing import List, Optional
import asyncio
from openai import AsyncOpenAI
from .base import LLMProvider
from ..config import settings


class KimiLLMProvider(LLMProvider):
    """Kimi K2 (Moonshot AI) LLM provider using OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None, default_model: str = "kimi-k2-0905-preview"):
        self.api_key = api_key or settings.kimi_api_key
        self.default_model = default_model
        self.base_url = settings.kimi_base_url
        self._available_models = None

        if not self.api_key:
            raise ValueError(
                "Kimi API key not found. Set KIMI_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://platform.moonshot.cn"
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using Kimi K2 with retry logic for rate limits."""
        model_name = model or self.default_model
        
        max_retries = 5
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + (attempt * 2)
                        print(f"Kimi rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"Failed to generate text with Kimi K2 after {max_retries} retries: {e}")
                else:
                    raise RuntimeError(f"Failed to generate text with Kimi K2: {e}")
        
        raise RuntimeError(f"Failed to generate text with Kimi K2 after {max_retries} retries")

    @property
    async def available_models(self) -> List[str]:
        """Return list of available Kimi models."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            models_response = await self.client.models.list()
            models = [model.id for model in models_response.data]
            
            kimi_models = [m for m in models if 'kimi' in m.lower() or 'moonshot' in m.lower()]
            
            if kimi_models:
                self._available_models = kimi_models
                return self._available_models
                
        except Exception as e:
            print(f"Warning: Could not fetch Kimi models: {e}")
        
        self._available_models = [
            "kimi-k2-0905-preview",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k"
        ]
        return self._available_models

    async def check_health(self) -> bool:
        """Check if Kimi API is accessible."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
