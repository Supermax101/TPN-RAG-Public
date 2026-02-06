from app.config import settings
from app.evaluation.benchmark_types import (
    DatasetSchema,
    DatasetSplit,
    DatasetTrack,
    ExperimentConfig,
    ModelSpec,
    ModelTier,
    PromptStrategy,
)
from app.evaluation.prompting import render_prompt


def test_settings_imports_and_embedding_config():
    assert settings.embedding_model
    assert settings.embedding_provider in {"openai", "huggingface", "hf"}


def test_render_prompt_all_strategies():
    q = "What is neonatal GIR start?"
    options = ["4 mg/kg/min", "6 mg/kg/min", "10 mg/kg/min"]
    for strategy in PromptStrategy:
        prompt = render_prompt(
            strategy=strategy,
            question=q,
            options=options,
            context="retrieved context",
        )
        assert q in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 10


def test_dataset_schema_mcq_validation():
    record = DatasetSchema(
        sample_id="mcq-1",
        track=DatasetTrack.MCQ,
        split=DatasetSplit.HOLDOUT,
        question="Q",
        options=["A", "B"],
        answer_key="A",
    )
    assert record.track == DatasetTrack.MCQ


def test_experiment_config_requires_models():
    cfg = ExperimentConfig(
        models=[ModelSpec(model_id="m1", provider="openai", model_name="gpt-4o", tier=ModelTier.SOTA)],
        mcq_dataset_path="x.jsonl",
        include_no_rag=True,
        include_rag=True,
    )
    assert cfg.repeats >= 1
