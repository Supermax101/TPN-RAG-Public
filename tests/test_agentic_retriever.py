"""Tests for AgenticRetrieverAdapter with mocked LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.evaluation.agentic_retriever import AgenticRetrieverAdapter
from app.evaluation.benchmark_types import (
    NormalizedChunk,
    RetrievalDiagnostics,
    RetrievalSnapshot,
)


def _make_snapshot(n_chunks: int = 4) -> RetrievalSnapshot:
    chunks = [
        NormalizedChunk(
            doc_id=f"doc_{i}",
            source=f"source_{i}.md",
            content=f"Chunk content about TPN topic {i}",
            score=1.0 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n_chunks)
    ]
    diag = RetrievalDiagnostics(
        query="protein requirement preterm",
        retrieval_time_ms=100.0,
        returned_count=n_chunks,
        query_plan=["protein requirement preterm"],
    )
    return RetrievalSnapshot(
        query_id="q1",
        run_id="r1",
        top_k=n_chunks,
        context_hash="abc123",
        context_text="test context",
        chunks=chunks,
        diagnostics=diag,
    )


@dataclass
class _FakeResult:
    text: str
    latency_ms: float = 10.0
    tokens_used: int = 20


class TestAgenticRetriever:
    def _make_adapter(self, base_retriever, judge_responses):
        """Create adapter with mocked judge provider."""
        call_idx = {"i": 0}

        async def fake_generate(**kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(judge_responses):
                return _FakeResult(text=judge_responses[idx])
            return _FakeResult(text='{"score": "yes"}')

        with patch(
            "app.evaluation.agentic_retriever.create_provider_adapter"
        ) as mock_create:
            mock_adapter = MagicMock()
            mock_adapter.generate = AsyncMock(side_effect=fake_generate)
            mock_create.return_value = mock_adapter
            adapter = AgenticRetrieverAdapter(
                base_retriever=base_retriever,
                judge_provider="openai",
                judge_model="gpt-4o-mini",
            )
        adapter._judge_adapter = mock_adapter
        return adapter

    def test_all_relevant_keeps_all(self):
        """When all chunks are relevant, all are kept."""
        base = MagicMock()
        snapshot = _make_snapshot(4)
        base.retrieve.return_value = snapshot
        base.max_context_chars = 12000

        responses = ['{"score": "yes"}'] * 4
        adapter = self._make_adapter(base, responses)

        result = adapter.retrieve("protein?", "q1", "r1")
        assert len(result.chunks) == 4

    def test_filters_irrelevant(self):
        """When some chunks are irrelevant, they are filtered out."""
        base = MagicMock()
        snapshot = _make_snapshot(4)
        base.retrieve.return_value = snapshot
        base.max_context_chars = 12000

        # 3 relevant, 1 irrelevant — above 50% threshold, no rewrite
        responses = ['{"score": "yes"}', '{"score": "yes"}', '{"score": "no"}', '{"score": "yes"}']
        adapter = self._make_adapter(base, responses)

        result = adapter.retrieve("protein?", "q1", "r1")
        assert len(result.chunks) == 3

    def test_rewrite_triggered_when_mostly_irrelevant(self):
        """When <50% relevant, query rewrite + re-retrieve happens."""
        base = MagicMock()
        snapshot1 = _make_snapshot(4)
        snapshot2 = _make_snapshot(2)  # Rewrite retrieval
        base.retrieve.side_effect = [snapshot1, snapshot2]
        base.max_context_chars = 12000

        # 1 relevant out of 4 = 25% < 50% threshold → rewrite
        responses = [
            '{"score": "no"}',
            '{"score": "yes"}',
            '{"score": "no"}',
            '{"score": "no"}',
            "protein requirement ASPEN preterm infant amino acids",  # rewrite response
        ]
        adapter = self._make_adapter(base, responses)

        result = adapter.retrieve("protein?", "q1", "r1")
        # Should have called retrieve twice (original + rewrite)
        assert base.retrieve.call_count == 2
        assert len(result.chunks) > 0
