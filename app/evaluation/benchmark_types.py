"""
Core schemas for the publishable benchmark runner.

These types define:
- experiment configuration
- dataset validation schema
- normalized retrieval outputs
- per-run records for full reproducibility
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PromptStrategy(str, Enum):
    """Prompt strategies evaluated in benchmark conditions."""

    ZS = "ZS"
    FEW_SHOT = "FEW_SHOT"
    COT = "COT"
    COT_SC = "COT_SC"
    RAP = "RAP"


class DatasetTrack(str, Enum):
    """Supported evaluation tracks."""

    MCQ = "mcq"
    OPEN_ENDED = "open_ended"


class DatasetSplit(str, Enum):
    """Strict split labels for leakage-safe evaluation."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    HOLDOUT = "holdout"


class ModelTier(str, Enum):
    """Model grouping for parity analysis."""

    OPEN = "open"
    SOTA = "sota"


class ModelSpec(BaseModel):
    """Single model entry in the benchmark matrix."""

    model_id: str = Field(..., description="Stable experiment identifier")
    provider: str = Field(..., description="Provider key (openai, anthropic, gemini, kimi, xai, huggingface)")
    model_name: str = Field(..., description="Exact API model name used at runtime")
    tier: ModelTier = Field(..., description="Open or SOTA for aggregate comparisons")
    api_base: Optional[str] = Field(default=None, description="Optional override base URL")
    api_key_env: Optional[str] = Field(default=None, description="Optional env var to read API key from")
    enabled: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Top-level benchmark experiment configuration."""

    name: str = "tpn_publishable_benchmark"
    seed: int = 42
    repeats: int = Field(default=5, ge=1)
    top_k: int = Field(default=6, ge=1, le=50)
    retrieval_candidate_k: int = Field(default=40, ge=1, le=200)
    iterative_retrieval: bool = True
    retrieval_iterations: int = Field(default=2, ge=1, le=4)
    max_query_decompositions: int = Field(default=3, ge=1, le=8)
    max_context_chars: int = Field(default=6000, ge=1000)
    fair_shared_context: bool = True
    include_no_rag: bool = True
    include_rag: bool = True
    prompt_strategies: List[PromptStrategy] = Field(
        default_factory=lambda: [
            PromptStrategy.ZS,
            PromptStrategy.FEW_SHOT,
            PromptStrategy.COT,
            PromptStrategy.COT_SC,
            PromptStrategy.RAP,
        ]
    )
    models: List[ModelSpec] = Field(default_factory=list)
    mcq_dataset_path: Optional[str] = None
    open_dataset_path: Optional[str] = None
    require_holdout_only: bool = True
    max_concurrent: int = Field(default=5, ge=1, le=50, description="Max concurrent API calls")
    output_dir: str = "eval/results/benchmark"
    agentic_retrieval: bool = Field(default=False, description="Enable LLM relevance judging on retrieved chunks")
    agentic_judge_provider: str = Field(default="openai", description="Provider for agentic relevance judge")
    agentic_judge_model: str = Field(default="gpt-4o-mini", description="Model for agentic relevance judge")
    dynamic_few_shot: bool = Field(default=False, description="Enable embedding-based few-shot example selection")

    @model_validator(mode="after")
    def validate_modes(self) -> "ExperimentConfig":
        if not (self.include_no_rag or self.include_rag):
            raise ValueError("At least one of include_no_rag/include_rag must be true.")
        if not self.models:
            raise ValueError("At least one model is required.")
        if self.require_holdout_only and not (self.mcq_dataset_path or self.open_dataset_path):
            raise ValueError("At least one dataset path is required when holdout-only mode is enabled.")
        return self


class DatasetSchema(BaseModel):
    """Strict dataset record schema for both tracks."""

    sample_id: str
    track: DatasetTrack
    split: DatasetSplit
    question: str
    options: Optional[List[str]] = None
    answer_key: Optional[str] = None
    reference_answer: Optional[str] = None
    domain: Optional[str] = None
    proficiency: Optional[str] = None
    source_doc: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_track_requirements(self) -> "DatasetSchema":
        if self.track == DatasetTrack.MCQ:
            if not self.options or len(self.options) < 2:
                raise ValueError("MCQ records require at least 2 options.")
            if not self.answer_key:
                raise ValueError("MCQ records require answer_key.")
        else:
            if not self.reference_answer:
                raise ValueError("Open-ended records require reference_answer.")
        return self


class NormalizedChunk(BaseModel):
    """Normalized retrieval result for consistent downstream handling."""

    doc_id: str = ""
    source: str = "unknown"
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str = ""
    content: str
    score: float = 0.0
    rank: int = 0
    retrieval_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RetrievalDiagnostics(BaseModel):
    """Per-query retrieval diagnostics for observability."""

    query: str
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    source_diversity: int = 0
    context_chars: int = 0
    candidate_count: int = 0
    returned_count: int = 0
    iteration_count: int = 1
    refinement_used: bool = False
    query_plan: List[str] = Field(default_factory=list)


class RetrievalSnapshot(BaseModel):
    """Deterministic context snapshot reused across models for fairness."""

    query_id: str
    run_id: str
    top_k: int
    context_hash: str
    context_text: str
    chunks: List[NormalizedChunk]
    diagnostics: RetrievalDiagnostics


class RunRecord(BaseModel):
    """Single generation record (one model x prompt x repeat x sample)."""

    run_id: str
    sample_id: str
    track: DatasetTrack
    model_id: str
    model_name: str
    provider: str
    model_tier: ModelTier
    prompt_strategy: PromptStrategy
    rag_enabled: bool
    repeat_index: int
    question: str
    prompt: str
    response_text: str
    parsed_answer: Optional[str] = None
    correct: Optional[bool] = None
    retrieval_context_hash: Optional[str] = None
    retrieval_snapshot_id: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    structured_output_used: bool = False
    error: Optional[str] = None


def stable_text_hash(text: str) -> str:
    """Stable SHA256 hash for fairness/reproducibility records."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
