"""
xAI (Grok) LLM provider implementation.
"""
from typing import List, Optional
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
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using xAI."""
        model_name = model or self.default_model
        
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 60.0
            }
            
            if seed is not None:
                kwargs["seed"] = seed
            
            response = await self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with xAI: {e}")
    
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
