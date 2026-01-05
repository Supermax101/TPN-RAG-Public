"""
Google Gemini LLM provider implementation.
"""
from typing import List, Optional
import httpx
from .base import LLMProvider
from ..config import settings


class GeminiLLMProvider(LLMProvider):
    """Google Gemini-based LLM provider (Gemini 2.5 Pro, Flash, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-2.5-flash"):
        self.api_key = api_key or settings.gemini_api_key
        self.default_model = default_model
        self.base_url = settings.gemini_base_url
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
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
        """Generate text response using Gemini."""
        model_name = model or self.default_model
        
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        
        try:
            url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            if seed is not None:
                payload["generationConfig"]["seed"] = seed
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            return parts[0]["text"].strip()
                    
                    finish_reason = candidate.get("finishReason", "")
                    if finish_reason in ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"]:
                        return ""
                
                raise RuntimeError(f"Unexpected Gemini response format: {result}")
                
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
            raise RuntimeError(f"Failed to generate text with Gemini (HTTP {e.response.status_code}): {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text with Gemini: {e}")
    
    @property
    async def available_models(self) -> List[str]:
        """Return list of available Gemini models from API."""
        if self._available_models is not None:
            return self._available_models
        
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                result = response.json()
                
                chat_models = []
                if "models" in result:
                    for model in result["models"]:
                        model_name = model.get("name", "")
                        if model_name.startswith("models/"):
                            model_id = model_name[7:]
                        else:
                            model_id = model_name
                        
                        if model_id in ["gemini-2.5-pro", "gemini-2.5-flash"]:
                            supported_methods = model.get("supportedGenerationMethods", [])
                            if "generateContent" in supported_methods:
                                chat_models.append(model_id)
                
                chat_models.sort(key=lambda x: 0 if "pro" in x else 1)
                
                self._available_models = chat_models
                return self._available_models
                
        except Exception as e:
            print(f"Warning: Could not fetch Gemini models: {e}")
            return ["gemini-2.5-pro", "gemini-2.5-flash"]
    
    async def check_health(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                return response.status_code == 200
                
        except Exception:
            return False
