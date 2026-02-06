"""
xAI (Grok) LLM provider implementation.
"""
import json
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from .base import LLMProvider
from ..config import settings


class XAILLMProvider(LLMProvider):
    """xAI-based LLM provider (Grok models)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "grok-4-fast-reasoning"):
        self.api_key = api_key or settings.xai_api_key
        self.default_model = default_model
        
        if not self.api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
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
        """Generate text response using xAI."""
        model_name = model or self.default_model

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 60.0,
            }

            if seed is not None:
                kwargs["seed"] = seed

            response = await self.client.chat.completions.create(**kwargs)

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to generate text with xAI: {e}")
    
    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate structured JSON output using OpenAI-compatible JSON mode."""
        model_name = model or self.default_model

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": f"{prompt}\n\nRespond with ONLY valid JSON matching this schema: {json.dumps(schema)}",
            })

            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=60.0,
            )

            content = response.choices[0].message.content or ""
            return json.loads(content)

        except Exception as e:
            raise RuntimeError(f"Structured output failed with xAI: {e}")

    async def get_available_models(self) -> List[str]:
        """Return list of available xAI models from API."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            models_response = await self.client.models.list()
            
            models = [
                model.id for model in models_response.data
                if model.id == "grok-4-fast-reasoning"
            ]
            
            self._available_models = models
            return self._available_models
            
        except Exception as e:
            print(f"Warning: Could not fetch xAI models: {e}")
            return []
    
    async def check_health(self) -> bool:
        """Check if xAI API is accessible."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
