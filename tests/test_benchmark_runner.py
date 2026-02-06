import asyncio
import json
from pathlib import Path

from app.evaluation.benchmark_runner import BenchmarkRunner
from app.evaluation.benchmark_types import (
    DatasetTrack,
    ExperimentConfig,
    ModelSpec,
    ModelTier,
    PromptStrategy,
    RetrievalDiagnostics,
    RetrievalSnapshot,
)


class _FakeAdapter:
    async def generate(self, **kwargs):
        class R:
            text = "Reasoning: test\nAnswer: A"
            latency_ms = 10.0
            tokens_used = 5

        return R()


class _FakeRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, query_id, run_id, force_refresh=False):
        self.calls += 1
        return RetrievalSnapshot(
            query_id=query_id,
            run_id=run_id,
            top_k=5,
            context_hash="ctxhash",
            context_text="[Source: doc]\nctx",
            chunks=[],
            diagnostics=RetrievalDiagnostics(query=query),
        )


def test_benchmark_runner_fair_shared_context(tmp_path, monkeypatch):
    dataset = tmp_path / "mcq.jsonl"
    rec = {
        "sample_id": "q1",
        "track": DatasetTrack.MCQ.value,
        "split": "holdout",
        "question": "Q?",
        "options": ["opt1", "opt2", "opt3", "opt4"],
        "answer_key": "A",
    }
    dataset.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    models = [
        ModelSpec(model_id="m1", provider="openai", model_name="gpt-4o", tier=ModelTier.SOTA),
        ModelSpec(model_id="m2", provider="openai", model_name="gpt-4o-mini", tier=ModelTier.SOTA),
    ]

    config = ExperimentConfig(
        models=models,
        mcq_dataset_path=str(dataset),
        repeats=2,
        prompt_strategies=[PromptStrategy.ZS],
        include_no_rag=False,
        include_rag=True,
        output_dir=str(tmp_path / "out"),
    )

    import app.evaluation.benchmark_runner as br

    monkeypatch.setattr(br, "create_provider_adapter", lambda *a, **k: _FakeAdapter())

    retriever = _FakeRetriever()
    runner = BenchmarkRunner(config=config, retriever=retriever)
    result = asyncio.run(runner.run())

    # One sample x one strategy => one shared retrieval snapshot call.
    assert retriever.calls == 1
    assert Path(result["records_path"]).exists()
    assert Path(result["summary_path"]).exists()
