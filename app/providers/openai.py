"""
OpenAI LLM provider implementation.
"""
from typing import List, Optional
from openai import AsyncOpenAI
from .base import LLMProvider
from ..config import settings


class OpenAILLMProvider(LLMProvider):
    """OpenAI-based LLM provider (GPT-4, GPT-5, O1, O3 reasoning models, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o"):
        self.api_key = api_key or settings.openai_api_key
        self.default_model = default_model
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self._available_models = None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using OpenAI."""
        model_name = model or self.default_model
        
        try:
            # Detect reasoning models (GPT-5, O1, O3)
            is_reasoning_model = any(x in model_name.lower() for x in ['gpt-5', 'o1', 'o3'])
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "timeout": 120.0,
            }
            
            if is_reasoning_model:
                kwargs["max_completion_tokens"] = max(max_tokens, 16000)
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
                kwargs["frequency_penalty"] = 0.3
                kwargs["presence_penalty"] = 0.1
            
            if seed is not None:
                kwargs["seed"] = seed
            
            response = await self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with OpenAI: {e}")
    
    async def get_available_models(self) -> List[str]:
        """Return list of available OpenAI models from API."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            models_response = await self.client.models.list()
            
            chat_models = [
                model.id for model in models_response.data
                if (
                    model.id.lower() in ['gpt-5', 'gpt-5-2025-08-07', 'gpt-5-chat-latest'] or
                    ('gpt-5' in model.id.lower() and 'mini' in model.id.lower()) or
                    'gpt-4o' in model.id.lower()
                )
            ]
            
            self._available_models = sorted(chat_models)
            return self._available_models
            
        except Exception as e:
            print(f"Warning: Could not fetch OpenAI models: {e}")
            return []
    
    async def check_health(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
