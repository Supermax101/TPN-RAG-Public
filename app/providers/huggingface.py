"""
HuggingFace provider for open-source models.

Supports models from HuggingFace Hub via:
1. HuggingFace Inference API (cloud, requires HF_TOKEN)
2. Local transformers (requires GPU/CPU resources)

Example:
    >>> from app.providers.huggingface import HuggingFaceProvider
    >>> provider = HuggingFaceProvider("Qwen/Qwen2.5-7B-Instruct")
    >>> response = provider.generate("What is TPN?", context="...")
"""

import logging
import os
import time
from typing import Optional, Dict, Any, List

from .base import SyncLLMProvider, LLMResponse, ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceProvider(SyncLLMProvider):
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
        quantize_4bit: bool = False,
    ):
        super().__init__(model_name, config)
        self.use_local = use_local
        self.device = device
        self.api_token = api_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.quantize_4bit = quantize_4bit

        self._client = None
        self._tokenizer = None
        self._model = None

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _initialize(self) -> bool:
        if self.use_local:
            return self._initialize_local()
        else:
            return self._initialize_api()

    def _initialize_api(self) -> bool:
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
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            logger.info(f"Loading {self.model_name} on {self.device} (4bit={self.quantize_4bit})...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )

            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",  # let accelerate handle multi-device placement
            }
            if self.quantize_4bit:
                # Requires: pip install bitsandbytes accelerate
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16  # H200 native BF16
            else:
                # Use "auto" to respect each model's native dtype (BF16 for Gemma/Qwen)
                load_kwargs["torch_dtype"] = "auto"

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs,
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
        self, prompt: str, system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if self.use_local:
            return self._generate_local(prompt, system_prompt)
        else:
            return self._generate_api(prompt, system_prompt)

    def _generate_api(
        self, prompt: str, system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        start_time = time.time()

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
            answer = response.choices[0].message.content or ""

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
        self, prompt: str, system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        import torch

        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Qwen3 thinking models: disable thinking to control latency/tokens
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "qwen3" in self.model_name.lower():
                template_kwargs["enable_thinking"] = False

            text = self._tokenizer.apply_chat_template(
                messages, **template_kwargs,
            )
            # Use model's actual device (respects device_map="auto" placement)
            target_device = next(self._model.parameters()).device
            inputs = self._tokenizer(text, return_tensors="pt").to(target_device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            answer = self._tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            tokens_used = outputs.shape[1]

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
    """Search HuggingFace Hub for available models."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = api.list_models(
            search=query, filter=pipeline_tag, sort=sort, direction=-1, limit=limit,
        )
        return [
            {"id": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline_tag": m.pipeline_tag}
            for m in models
        ]
    except ImportError:
        logger.error("huggingface_hub not installed")
        return []
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return []


def list_trending_models(limit: int = 10) -> List[str]:
    """List trending text-generation models from HuggingFace Hub."""
    models = search_models(query="instruct", pipeline_tag="text-generation", limit=limit, sort="downloads")
    return [m["id"] for m in models]


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model = api.model_info(model_id)
        return {
            "id": model.id, "downloads": model.downloads, "likes": model.likes,
            "pipeline_tag": model.pipeline_tag, "tags": model.tags, "library_name": model.library_name,
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def validate_model_id(model_id: str) -> bool:
    """Check if a model ID exists on HuggingFace Hub."""
    return get_model_info(model_id) is not None
