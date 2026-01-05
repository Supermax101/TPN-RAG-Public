"""
Anthropic provider for Claude models.

Supports Claude 3.5 Sonnet, Claude 3 Opus, etc.
Requires ANTHROPIC_API_KEY environment variable.

Example:
    >>> provider = AnthropicProvider("claude-sonnet-4-20250514")
    >>> response = provider.generate("What is TPN?", context="...")
"""

import logging
import os
import time
from typing import Optional

from .base import LLMProvider, LLMResponse, ModelConfig

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider for Claude models.

    Recommended models:
    - claude-sonnet-4-20250514 - Latest, best balance
    - claude-3-5-sonnet-20241022 - Great quality
    - claude-3-opus-20240229 - Best quality, expensive
    - claude-3-haiku-20240307 - Fast, cheap
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        config: Optional[ModelConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model_name: Claude model name
            config: Model configuration
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _initialize(self) -> bool:
        """Initialize Anthropic client."""
        if not self.api_key:
            logger.error("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            return False

        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._initialized = True
            return True

        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()

        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract response
            answer = ""
            if response.content:
                answer = response.content[0].text

            # Calculate tokens
            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Extract thinking if present
            thinking = None
            if "<thinking>" in answer and "</thinking>" in answer:
                think_start = answer.find("<thinking>") + 10
                think_end = answer.find("</thinking>")
                thinking = answer[think_start:think_end].strip()
                answer = answer[think_end + 11:].strip()

            return LLMResponse(
                answer=answer,
                thinking=thinking,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
                raw_response={
                    "id": response.id,
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                },
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost in USD for a given number of tokens."""
        # Approximate costs per 1M tokens (as of Dec 2024)
        costs = {
            "claude-sonnet-4-20250514": 3.0,
            "claude-3-5-sonnet-20241022": 3.0,
            "claude-3-opus-20240229": 15.0,
            "claude-3-haiku-20240307": 0.25,
        }
        rate = costs.get(self.model_name, 3.0)
        return (tokens / 1_000_000) * rate


def demo_anthropic():
    """Demo function to test Anthropic provider."""
    print("=" * 60)
    print("ANTHROPIC PROVIDER DEMO")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    # Create provider
    provider = AnthropicProvider(model_name="claude-3-haiku-20240307")

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
    demo_anthropic()
