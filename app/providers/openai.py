"""
OpenAI LLM provider implementation.
"""
import json
from typing import Any, Dict, List, Optional
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
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text response using OpenAI."""
        model_name = model or self.default_model

        try:
            # Detect reasoning models (GPT-5, O1, O3)
            is_reasoning_model = any(x in model_name.lower() for x in ['gpt-5', 'o1', 'o3'])
            # GPT-5+ models reject explicit temperature parameter
            is_gpt5 = 'gpt-5' in model_name.lower()

            messages = []
            if system_prompt:
                # Use "developer" role for reasoning models, "system" for standard
                role = "developer" if is_reasoning_model else "system"
                messages.append({"role": role, "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": model_name,
                "messages": messages,
                "timeout": 120.0,
            }

            if is_reasoning_model:
                # Reasoning models use max_completion_tokens (not max_tokens).
                # Keep it aligned with the benchmark runner's per-strategy budgets
                # to control cost and latency for MCQ evaluation.
                kwargs["max_completion_tokens"] = max_tokens
                # GPT-5 does not accept temperature at all; other reasoning
                # models (o1/o3) also skip it
                if not is_gpt5:
                    pass  # o1/o3 also skip temperature
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
    
    async def generate_structured(
        self,
        prompt: str,
        schema: dict,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate structured JSON output using OpenAI's json_schema response format."""
        model_name = model or self.default_model

        try:
            is_reasoning_model = any(x in model_name.lower() for x in ['gpt-5', 'o1', 'o3'])
            is_gpt5 = 'gpt-5' in model_name.lower()
            messages = []
            if system_prompt:
                role = "developer" if is_reasoning_model else "system"
                messages.append({"role": role, "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": model_name,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mcq_answer",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "timeout": 120.0,
            }
            if is_reasoning_model:
                kwargs["max_completion_tokens"] = max_tokens
                # GPT-5/o1/o3 reject temperature; omit it.
                if not is_gpt5:
                    pass
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            return json.loads(content)

        except Exception as e:
            raise RuntimeError(f"Structured output failed with OpenAI: {e}")

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
