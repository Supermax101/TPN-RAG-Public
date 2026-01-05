"""
HuggingFace provider for open-source models.

Supports models from HuggingFace Hub via:
1. HuggingFace Inference API (cloud, requires HF_TOKEN)
2. Local transformers (requires GPU/CPU resources)

Recommended models for TPN Q&A:
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- google/gemma-2-9b-it

Example:
    >>> provider = HuggingFaceProvider("Qwen/Qwen2.5-7B-Instruct")
    >>> response = provider.generate("What is TPN?", context="...")
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List

from .base import LLMProvider, LLMResponse, ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """
    HuggingFace provider for open-source models.

    Supports two modes:
    1. Inference API (default): Uses HuggingFace serverless inference
    2. Local: Loads model locally with transformers

    For Inference API, set HF_TOKEN environment variable.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[ModelConfig] = None,
        use_local: bool = False,
        device: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        """
        Initialize HuggingFace provider.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
            config: Model configuration
            use_local: If True, load model locally instead of using API
            device: Device for local inference ("cuda", "mps", "cpu", or None for auto)
            api_token: HuggingFace API token (or use HF_TOKEN env var)
        """
        super().__init__(model_name, config)
        self.use_local = use_local
        self.device = device
        self.api_token = api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        self._client = None
        self._tokenizer = None
        self._model = None

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _initialize(self) -> bool:
        """Initialize HuggingFace client or local model."""
        if self.use_local:
            return self._initialize_local()
        else:
            return self._initialize_api()

    def _initialize_api(self) -> bool:
        """Initialize HuggingFace Inference API client."""
        try:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(
                model=self.model_name,
                token=self.api_token,
            )

            self._initialized = True
            logger.info(f"Initialized HuggingFace API client for {self.model_name}")
            return True

        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            return False

    def _initialize_local(self) -> bool:
        """Initialize local model with transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if self.device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            logger.info(f"Loading {self.model_name} on {self.device}...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )

            self._initialized = True
            logger.info(f"Model loaded successfully on {self.device}")
            return True

        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using HuggingFace."""
        if self.use_local:
            return self._generate_local(prompt, system_prompt)
        else:
            return self._generate_api(prompt, system_prompt)

    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using HuggingFace Inference API."""
        start_time = time.time()

        # Build messages for chat models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else 0.01,
                top_p=self.config.top_p,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract response
            answer = response.choices[0].message.content or ""

            # Extract thinking if present
            thinking = None
            if "<think>" in answer and "</think>" in answer:
                think_start = answer.find("<think>") + 7
                think_end = answer.find("</think>")
                thinking = answer[think_start:think_end].strip()
                answer = answer[think_end + 8:].strip()

            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens

            return LLMResponse(
                answer=answer,
                thinking=thinking,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
                raw_response={"id": response.id} if hasattr(response, 'id') else {},
            )

        except Exception as e:
            logger.error(f"HuggingFace API generation failed: {e}")
            raise

    def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate using local model."""
        import torch

        start_time = time.time()

        # Build chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            answer = self._tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            tokens_used = outputs.shape[1]

            # Extract thinking if present
            thinking = None
            if "<think>" in answer and "</think>" in answer:
                think_start = answer.find("<think>") + 7
                think_end = answer.find("</think>")
                thinking = answer[think_start:think_end].strip()
                answer = answer[think_end + 8:].strip()

            return LLMResponse(
                answer=answer,
                thinking=thinking,
                tokens_used=tokens_used,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise


def search_models(
    query: str = "instruct",
    pipeline_tag: str = "text-generation",
    limit: int = 20,
    sort: str = "downloads",
) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub for available models.

    Args:
        query: Search query (e.g., "Qwen", "Llama", "instruct")
        pipeline_tag: Pipeline filter (e.g., "text-generation", "text2text-generation")
        limit: Maximum number of results
        sort: Sort by ("downloads", "likes", "lastModified")

    Returns:
        List of model info dicts with id, downloads, likes, etc.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(
            search=query,
            filter=pipeline_tag,
            sort=sort,
            direction=-1,
            limit=limit,
        )

        return [
            {
                "id": m.id,
                "downloads": m.downloads,
                "likes": m.likes,
                "pipeline_tag": m.pipeline_tag,
            }
            for m in models
        ]

    except ImportError:
        logger.error("huggingface_hub not installed")
        return []
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return []


def list_trending_models(limit: int = 10) -> List[str]:
    """
    List trending text-generation models from HuggingFace Hub.

    Returns:
        List of model IDs sorted by downloads
    """
    models = search_models(
        query="instruct",
        pipeline_tag="text-generation",
        limit=limit,
        sort="downloads",
    )
    return [m["id"] for m in models]


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific model.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        Model info dict or None if not found
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        model = api.model_info(model_id)

        return {
            "id": model.id,
            "downloads": model.downloads,
            "likes": model.likes,
            "pipeline_tag": model.pipeline_tag,
            "tags": model.tags,
            "library_name": model.library_name,
        }

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def validate_model_id(model_id: str) -> bool:
    """Check if a model ID exists on HuggingFace Hub."""
    return get_model_info(model_id) is not None


def demo_huggingface():
    """Demo function to test HuggingFace provider."""
    print("=" * 60)
    print("HUGGINGFACE PROVIDER DEMO")
    print("=" * 60)

    # Fetch trending models dynamically
    print("\nFetching trending instruction-tuned models from HuggingFace Hub...")
    trending = list_trending_models(limit=10)

    if trending:
        print("\nTop 10 Instruction Models (by downloads):")
        for i, model_id in enumerate(trending, 1):
            print(f"  {i:2}. {model_id}")
    else:
        print("Could not fetch models. Check your internet connection.")

    # Search for specific models
    print("\n--- Searching for Qwen models ---")
    qwen_models = search_models(query="Qwen instruct", limit=5)
    for m in qwen_models:
        print(f"  {m['id']} (downloads: {m['downloads']:,})")

    # Check for API token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("\nWARNING: HF_TOKEN not set. Some models may not be accessible.")
        print("Set with: export HF_TOKEN=your_token")

    # Try to create provider
    print("\n--- Testing API Mode ---")
    try:
        # Use first trending model or default
        model_id = trending[0] if trending else "Qwen/Qwen2.5-7B-Instruct"
        print(f"Using model: {model_id}")

        provider = HuggingFaceProvider(
            model_name=model_id,
            use_local=False,
        )

        context = """Protein requirements for preterm infants are 3-4 g/kg/day according to ASPEN guidelines.
        For term infants, the recommendation is 2.5-3 g/kg/day."""

        response = provider.generate(
            question="What is the protein requirement for preterm infants?",
            context=context,
            use_rag=True,
        )

        print(f"Model: {response.model}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        print(f"Tokens: {response.tokens_used}")
        print(f"\nAnswer: {response.answer[:300]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTip: Ensure HF_TOKEN is set for gated models")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_huggingface()
