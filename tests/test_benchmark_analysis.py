from app.evaluation.benchmark_analysis import compute_rag_lift
from app.evaluation.benchmark_types import DatasetTrack, ModelTier, PromptStrategy, RunRecord


def test_compute_rag_lift_basic():
    records = [
        RunRecord(
            run_id="1",
            sample_id="s1",
            track=DatasetTrack.MCQ,
            model_id="m1",
            model_name="x",
            provider="openai",
            model_tier=ModelTier.OPEN,
            prompt_strategy=PromptStrategy.ZS,
            rag_enabled=False,
            repeat_index=0,
            question="q",
            prompt="p",
            response_text="a",
            parsed_answer="A",
            correct=False,
        ),
        RunRecord(
            run_id="2",
            sample_id="s1",
            track=DatasetTrack.MCQ,
            model_id="m1",
            model_name="x",
            provider="openai",
            model_tier=ModelTier.OPEN,
            prompt_strategy=PromptStrategy.ZS,
            rag_enabled=True,
            repeat_index=0,
            question="q",
            prompt="p",
            response_text="a",
            parsed_answer="A",
            correct=True,
        ),
    ]
    rows = compute_rag_lift(records)
    assert len(rows) == 1
    assert rows[0]["delta"] >= 0
