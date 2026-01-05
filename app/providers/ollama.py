"""
Ollama LLM provider implementation.
"""
import httpx
from typing import List, Optional
from .base import LLMProvider
from ..config import settings


class OllamaLLMProvider(LLMProvider):
    """Ollama-based LLM provider."""
    
    def __init__(self, base_url: str = None, default_model: str = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self._available_models = None
        
        # Use provided model, or from settings, or auto-detect from Ollama
        if default_model:
            self.default_model = default_model
            print(f"Ollama LLM: {default_model} (specified)")
        elif settings.ollama_llm_model:
            self.default_model = settings.ollama_llm_model
            print(f"Ollama LLM: {settings.ollama_llm_model} (from .env)")
        else:
            self.default_model = self._auto_detect_model()
    
    def _auto_detect_model(self) -> str:
        """Auto-detect available LLM models from Ollama server."""
        import httpx
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # Filter out embedding models
                llm_models = [
                    m["name"] for m in models 
                    if not any(kw in m["name"].lower() for kw in ["embed", "nomic-embed"])
                ]
                
                if llm_models:
                    print(f"\nAvailable Ollama LLM models ({len(llm_models)}):")
                    for i, model in enumerate(llm_models, 1):
                        print(f"  {i}. {model}")
                    print(f"\nUsing: {llm_models[0]}")
                    print("(Set OLLAMA_LLM_MODEL in .env to change default)\n")
                    return llm_models[0]
                else:
                    print("No LLM models found in Ollama. Run: ollama pull <model>")
        except Exception as e:
            print(f"Could not connect to Ollama: {e}")
        
        return None
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        seed: Optional[int] = None
    ) -> str:
        """Generate text response using Ollama."""
        model_name = model or self.default_model
        
        if not model_name:
            raise RuntimeError(
                "No Ollama model specified. Either:\n"
                "  1. Set OLLAMA_LLM_MODEL in .env\n"
                "  2. Pass model parameter\n"
                "  3. Ensure Ollama has models downloaded (ollama pull <model>)"
            )
        
        # Increase max_tokens for thinking models (GPT-OSS, DeepSeek, etc.)
        if "gpt-oss" in model_name.lower() or "deepseek" in model_name.lower():
            max_tokens = max(max_tokens, 1000)
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                options = {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.3,
                    "num_ctx": 8192
                }
                
                if seed is not None:
                    options["seed"] = seed
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": options
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                generated_text = result.get("response", "").strip()
                
                # Handle thinking models
                if not generated_text:
                    thinking_text = result.get("thinking", "").strip()
                    
                    if thinking_text:
                        import re
                        json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', thinking_text)
                        if json_match:
                            generated_text = thinking_text
                        else:
                            generated_text = thinking_text
                    else:
                        raise RuntimeError("Empty response from Ollama")
                
                return generated_text
                
            except httpx.TimeoutException:
                raise RuntimeError("Request to Ollama timed out")
            except Exception as e:
                raise RuntimeError(f"Failed to generate text with Ollama: {e}")
    
    @property
    async def available_models(self) -> List[str]:
        """Return list of available Ollama models."""
        if self._available_models is not None:
            return self._available_models
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("models", []):
                    name = model.get("name", "")
                    if name:
                        models.append(name)
                
                self._available_models = models
                return models
                
            except Exception:
                return [self.default_model] if self.default_model else []
    
    async def check_health(self) -> bool:
        """Check if Ollama service is healthy."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
            except Exception:
                return False
