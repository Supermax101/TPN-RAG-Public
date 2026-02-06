"""
Evaluation module for TPN RAG system.

Publishable benchmark framework:
- BenchmarkRunner: Run full model x prompt-strategy matrix
- RetrieverAdapter: Bridge between benchmark and retrieval pipeline
- PromptStrategy / ExperimentConfig / RunRecord: Configuration types
- Statistical tools: Cohen's kappa, Fleiss' kappa, McNemar-Bowker, etc.

Citation evaluation:
- CitationEvaluator: Evaluate citation quality for fine-tuned models
"""

from .citation_metrics import CitationEvaluator, CitationResult, RetrievedChunk
from .benchmark_types import (
    PromptStrategy,
    DatasetTrack,
    DatasetSchema,
    ModelSpec,
    ModelTier,
    ExperimentConfig,
    RunRecord,
)
from .retriever_adapter import RetrieverAdapter
from .benchmark_runner import BenchmarkRunner, run_benchmark
from .statistics import (
    cohen_kappa,
    fleiss_kappa,
    mcnemar_bowker,
    paired_bootstrap_ci,
    cohen_d_paired,
    holm_bonferroni,
)
from .benchmark_analysis import (
    load_run_records,
    summarize_accuracy,
    compute_intra_rater_fleiss,
    compute_inter_rater,
    compute_rag_lift,
    build_analysis_report,
)
from .data_leakage import check_data_leakage

__all__ = [
    # Citation evaluation
    "CitationEvaluator",
    "CitationResult",
    "RetrievedChunk",
    # Publishable benchmark framework
    "PromptStrategy",
    "DatasetTrack",
    "DatasetSchema",
    "ModelSpec",
    "ModelTier",
    "ExperimentConfig",
    "RunRecord",
    "RetrieverAdapter",
    "BenchmarkRunner",
    "run_benchmark",
    # Statistical tools
    "cohen_kappa",
    "fleiss_kappa",
    "mcnemar_bowker",
    "paired_bootstrap_ci",
    "cohen_d_paired",
    "holm_bonferroni",
    "load_run_records",
    "summarize_accuracy",
    "compute_intra_rater_fleiss",
    "compute_inter_rater",
    "compute_rag_lift",
    "build_analysis_report",
    "check_data_leakage",
]
