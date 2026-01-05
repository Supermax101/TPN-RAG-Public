"""
OpenAI provider for GPT models.

Supports GPT-4o, GPT-4 Turbo, o1 series, etc.
Requires OPENAI_API_KEY environment variable.

Example:
    >>> provider = OpenAIProvider("gpt-4o")
    >>> response = provider.generate("What is TPN?", context="...")
"""

import logging
import os
import time
from typing import Optional

from .base import LLMProvider, LLMResponse, ModelConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider for GPT models.

    Recommended models:
    - gpt-4o - Best for accuracy, fast
    - gpt-4o-mini - Good balance of cost and quality
    - gpt-4-turbo - Good quality, slower
    - o1-preview - Reasoning model (best for complex questions)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        config: Optional[ModelConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model_name: OpenAI model name
            config: Model configuration
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _initialize(self) -> bool:
        """Initialize OpenAI client."""
        if not self.api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return False

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
            self._initialized = True
            return True

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # Handle o1 models differently (no system prompt, no temperature)
            if self.model_name.startswith("o1"):
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}],
                    max_completion_tokens=self.config.max_tokens,
                )
            else:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                )

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract response
            answer = response.choices[0].message.content or ""

            # Calculate tokens
            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.total_tokens

            return LLMResponse(
                answer=answer,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost in USD for a given number of tokens."""
        # Approximate costs per 1M tokens (as of Dec 2024)
        costs = {
            "gpt-4o": 5.0,  # $5/1M input, ~$15/1M output
            "gpt-4o-mini": 0.15,
            "gpt-4-turbo": 10.0,
            "o1-preview": 15.0,
            "o1-mini": 3.0,
        }
        rate = costs.get(self.model_name, 5.0)
        return (tokens / 1_000_000) * rate


def demo_openai():
    """Demo function to test OpenAI provider."""
    print("=" * 60)
    print("OPENAI PROVIDER DEMO")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Create provider
    provider = OpenAIProvider(model_name="gpt-4o-mini")

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
    print(f"Est. cost: ${provider.estimate_cost(response.tokens_used):.4f}")
    print(f"\nAnswer: {response.answer}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_openai()
