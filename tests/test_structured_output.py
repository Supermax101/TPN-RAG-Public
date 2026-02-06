"""Tests for structured output in provider adapter (JSON + fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.evaluation.provider_adapter import AsyncProviderWrapper


@dataclass
class _FakeProvider:
    """Minimal provider mock for testing generate_structured."""

    supports_structured: bool = True

    async def generate(self, prompt, model=None, temperature=0.0, max_tokens=500, system_prompt=None, seed=None):
        return json.dumps({"thinking": "test", "answer": "B", "confidence": "high"})

    async def generate_structured(self, prompt, schema, model=None, temperature=0.0, max_tokens=500, system_prompt=None):
        if not self.supports_structured:
            raise NotImplementedError
        return {"thinking": "structured reasoning", "answer": "A", "confidence": "high"}


_MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["thinking", "answer", "confidence"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_native_structured_output():
    """When provider supports structured output, use it directly."""
    provider = _FakeProvider(supports_structured=True)
    wrapper = AsyncProviderWrapper(provider, "test-model")

    result = await wrapper.generate_structured(
        prompt="What is the answer?",
        schema=_MCQ_SCHEMA,
        system="You are a TPN specialist.",
    )

    assert result["answer"] == "A"
    assert result["confidence"] == "high"


@pytest.mark.asyncio
async def test_fallback_to_text_json():
    """When provider doesn't support structured output, fall back to text + JSON parse."""
    provider = _FakeProvider(supports_structured=False)
    wrapper = AsyncProviderWrapper(provider, "test-model")

    result = await wrapper.generate_structured(
        prompt="What is the answer?",
        schema=_MCQ_SCHEMA,
    )

    assert result["answer"] == "B"
    assert result["thinking"] == "test"


@pytest.mark.asyncio
async def test_fallback_on_native_error():
    """When native structured output raises an error, fall back to text."""
    provider = _FakeProvider(supports_structured=True)
    # Override to raise
    original = provider.generate_structured

    async def fail(*args, **kwargs):
        raise RuntimeError("API error")

    provider.generate_structured = fail
    wrapper = AsyncProviderWrapper(provider, "test-model")

    result = await wrapper.generate_structured(
        prompt="What is the answer?",
        schema=_MCQ_SCHEMA,
    )

    assert result["answer"] == "B"


@pytest.mark.asyncio
async def test_invalid_json_fallback_extraction():
    """When text response has JSON embedded in other text, extract it."""
    provider = _FakeProvider(supports_structured=False)

    async def generate_with_prefix(prompt, model=None, temperature=0.0, max_tokens=500, system_prompt=None, seed=None):
        return 'Here is the answer:\n{"thinking": "clinical reasoning", "answer": "C", "confidence": "medium"}\nDone.'

    provider.generate = generate_with_prefix
    wrapper = AsyncProviderWrapper(provider, "test-model")

    result = await wrapper.generate_structured(
        prompt="What is the answer?",
        schema=_MCQ_SCHEMA,
    )

    assert result["answer"] == "C"
