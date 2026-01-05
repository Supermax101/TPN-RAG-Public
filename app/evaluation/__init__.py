"""
Evaluation module for TPN RAG system.

This module provides:
- EvaluationDataset: Load and parse grounded Q&A pairs
- RetrievalMetrics: Measure retrieval quality (source matching)
- AnswerMetrics: Measure answer quality (semantic similarity)
- EvaluationHarness: Run full evaluation pipeline
- ModelComparison: Compare multiple models with/without RAG
- CitationEvaluator: Evaluate citation quality for fine-tuned models

Example usage:
    >>> from app.evaluation import EvaluationHarness, ModelComparison
    >>> harness = EvaluationHarness(rag_pipeline, dataset_path)
    >>> results = harness.run(sample_size=100)
    >>> print(f"Accuracy: {results.accuracy:.1%}")

    >>> # For fine-tuned models with citation grounding
    >>> from app.evaluation import CitationEvaluator
    >>> evaluator = CitationEvaluator()
    >>> result = evaluator.evaluate(question, answer, chunks, ground_truth_source)
    >>> print(f"Citation accuracy: {result.source_accuracy:.1%}")
"""

from .dataset import EvaluationDataset, QAPair
from .metrics import RetrievalMetrics, AnswerMetrics, EvaluationResult
from .harness import EvaluationHarness
from .comparison import ModelComparison, ComparisonResult, statistical_significance
from .citation_metrics import CitationEvaluator, CitationResult, RetrievedChunk

__all__ = [
    "EvaluationDataset",
    "QAPair",
    "RetrievalMetrics",
    "AnswerMetrics",
    "EvaluationResult",
    "EvaluationHarness",
    "ModelComparison",
    "ComparisonResult",
    "statistical_significance",
    # Citation evaluation for fine-tuned models
    "CitationEvaluator",
    "CitationResult",
    "RetrievedChunk",
]
