"""
Ollama provider for open-source models.

Supports models like Qwen3, Llama3, Mistral, Gemma, etc.
Requires Ollama to be running locally.

Example:
    >>> provider = OllamaProvider("qwen3:8b")
    >>> response = provider.generate("What is TPN?", context="...")
"""

import json
import logging
import time
from typing import Optional, Dict, Any

from .base import LLMProvider, LLMResponse, ModelConfig

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama provider for local open-source models.

    Recommended models for TPN Q&A:
    - qwen3:8b - Good balance of speed and quality
    - qwen3:14b - Better quality, slower
    - llama3.3:70b - Best quality, requires GPU
    - gemma2:9b - Fast, good for testing
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        config: Optional[ModelConfig] = None,
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama provider.

        Args:
            model_name: Ollama model name (e.g., "qwen3:8b")
            config: Model configuration
            base_url: Ollama API base URL
        """
        super().__init__(model_name, config)
        self.base_url = base_url
        self._client = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _initialize(self) -> bool:
        """Initialize httpx client and verify Ollama is running."""
        try:
            import httpx

            self._client = httpx.Client(base_url=self.base_url, timeout=120.0)

            # Verify connection
            response = self._client.get("/api/tags")
            if response.status_code != 200:
                logger.error(f"Ollama not responding: {response.status_code}")
                return False

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check for exact match or partial match (e.g., "qwen3:8b" in "qwen3:8b-instruct")
            model_found = any(
                self.model_name in name or name.startswith(self.model_name.split(":")[0])
                for name in model_names
            )

            if not model_found:
                logger.warning(
                    f"Model '{self.model_name}' not found locally. "
                    f"Available: {model_names[:5]}... "
                    f"Will attempt to pull on first use."
                )

            self._initialized = True
            return True

        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return False

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using Ollama API."""
        start_time = time.time()

        # Build request
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }

        if system_prompt:
            request_data["system"] = system_prompt

        # Add any extra options
        if self.config.extra_options:
            request_data["options"].update(self.config.extra_options)

        try:
            response = self._client.post(
                "/api/generate",
                json=request_data,
            )
            response.raise_for_status()
            data = response.json()

            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            answer = data.get("response", "")

            # Extract thinking if present (for models that support it)
            thinking = None
            if "<think>" in answer and "</think>" in answer:
                think_start = answer.find("<think>") + 7
                think_end = answer.find("</think>")
                thinking = answer[think_start:think_end].strip()
                answer = answer[think_end + 8:].strip()

            return LLMResponse(
                answer=answer,
                thinking=thinking,
                tokens_used=data.get("eval_count", 0),
                latency_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def list_local_models(self) -> list:
        """List models available locally in Ollama."""
        if not self._initialized:
            self._initialize()

        try:
            response = self._client.get("/api/tags")
            models = response.json().get("models", [])
            return [m.get("name") for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """Pull a model from Ollama registry."""
        model = model_name or self.model_name

        if not self._initialized:
            self._initialize()

        try:
            logger.info(f"Pulling model: {model}")
            response = self._client.post(
                "/api/pull",
                json={"name": model},
                timeout=600.0,  # Models can be large
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


def demo_ollama():
    """Demo function to test Ollama provider."""
    print("=" * 60)
    print("OLLAMA PROVIDER DEMO")
    print("=" * 60)

    # Create provider
    provider = OllamaProvider(model_name="qwen3:8b")

    # List available models
    print("\nLocal models:")
    models = provider.list_local_models()
    for m in models[:5]:
        print(f"  - {m}")

    # Test generation
    print("\n--- Testing RAG Generation ---")
    context = """Protein requirements for preterm infants are 3-4 g/kg/day according to ASPEN guidelines.
    For term infants, the recommendation is 2.5-3 g/kg/day.
    Protein should be initiated early to prevent catabolism."""

    response = provider.generate(
        question="What is the protein requirement for preterm infants?",
        context=context,
        use_rag=True,
    )

    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"Tokens: {response.tokens_used}")
    print(f"\nAnswer: {response.answer[:300]}...")

    # Test baseline (no RAG)
    print("\n--- Testing Baseline (No RAG) ---")
    response = provider.generate(
        question="What is the protein requirement for preterm infants?",
        context=None,
        use_rag=False,
    )
    print(f"Answer: {response.answer[:300]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_ollama()
