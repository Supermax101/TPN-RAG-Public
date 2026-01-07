"""
Model Comparison Framework for TPN RAG System.

Runs controlled experiments comparing:
- Multiple LLM models
- RAG vs No-RAG (baseline)
- Different retrieval configurations

Produces statistical analysis and visualizations.

Example:
    >>> comparison = ModelComparison(
    ...     dataset_path="test.jsonl",
    ...     retriever=my_retriever,
    ... )
    >>> comparison.add_model("huggingface", "Qwen/Qwen2.5-7B-Instruct")
    >>> comparison.add_model("openai", "gpt-4o-mini")
    >>> results = comparison.run(sample_size=100)
    >>> comparison.generate_report()
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

from .dataset import EvaluationDataset, QAPair
from .metrics import AnswerMetrics, AnswerResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results for a single model in the comparison."""

    model_name: str
    provider: str
    use_rag: bool

    # Metrics
    f1_scores: List[float] = field(default_factory=list)
    key_phrase_scores: List[float] = field(default_factory=list)
    exact_matches: List[bool] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    tokens_used: List[int] = field(default_factory=list)

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def mean_f1(self) -> float:
        return statistics.mean(self.f1_scores) if self.f1_scores else 0.0

    @property
    def std_f1(self) -> float:
        return statistics.stdev(self.f1_scores) if len(self.f1_scores) > 1 else 0.0

    @property
    def mean_key_phrase(self) -> float:
        return statistics.mean(self.key_phrase_scores) if self.key_phrase_scores else 0.0

    @property
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_used)

    @property
    def exact_match_rate(self) -> float:
        return sum(self.exact_matches) / len(self.exact_matches) if self.exact_matches else 0.0

    @property
    def error_rate(self) -> float:
        total = len(self.f1_scores) + len(self.errors)
        return len(self.errors) / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "provider": self.provider,
            "use_rag": self.use_rag,
            "samples": len(self.f1_scores),
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "mean_key_phrase": self.mean_key_phrase,
            "exact_match_rate": self.exact_match_rate,
            "mean_latency_ms": self.mean_latency,
            "total_tokens": self.total_tokens,
            "error_rate": self.error_rate,
        }


@dataclass
class ComparisonResult:
    """Results from a full comparison run."""

    timestamp: str
    sample_size: int
    models: List[ModelResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def get_rag_lift(self, model_name: str) -> Optional[float]:
        """
        Calculate RAG lift for a model.

        RAG lift = (RAG F1 - Baseline F1) / Baseline F1
        """
        rag_result = None
        baseline_result = None

        for m in self.models:
            if m.model_name == model_name:
                if m.use_rag:
                    rag_result = m
                else:
                    baseline_result = m

        if rag_result and baseline_result and baseline_result.mean_f1 > 0:
            return (rag_result.mean_f1 - baseline_result.mean_f1) / baseline_result.mean_f1
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sample_size": self.sample_size,
            "config": self.config,
            "models": [m.to_dict() for m in self.models],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Model Comparison Report",
            f"\n**Timestamp:** {self.timestamp}",
            f"**Sample Size:** {self.sample_size}",
            "",
            "## Results Summary",
            "",
            "| Model | Provider | RAG | F1 | Std | Key Phrase | Latency (ms) |",
            "|-------|----------|-----|-----|-----|------------|--------------|",
        ]

        for m in sorted(self.models, key=lambda x: x.mean_f1, reverse=True):
            rag_str = "Yes" if m.use_rag else "No"
            lines.append(
                f"| {m.model_name} | {m.provider} | {rag_str} | "
                f"{m.mean_f1:.3f} | {m.std_f1:.3f} | {m.mean_key_phrase:.3f} | "
                f"{m.mean_latency:.0f} |"
            )

        lines.extend([
            "",
            "## RAG Lift Analysis",
            "",
        ])

        # Calculate RAG lift for each model
        model_names = set(m.model_name for m in self.models)
        for name in model_names:
            lift = self.get_rag_lift(name)
            if lift is not None:
                lines.append(f"- **{name}**: {lift:+.1%} improvement with RAG")

        return "\n".join(lines)


class ModelComparison:
    """
    Framework for comparing multiple models with and without RAG.

    Usage:
        >>> comparison = ModelComparison(dataset_path, retriever)
        >>> comparison.add_model("huggingface", "Qwen/Qwen2.5-7B-Instruct")
        >>> comparison.add_model("openai", "gpt-4o-mini")
        >>> results = comparison.run(sample_size=100)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        retriever: Optional[Any] = None,
        top_k: int = 5,
    ):
        """
        Initialize comparison framework.

        Args:
            dataset_path: Path to evaluation dataset
            retriever: Retriever for RAG mode (optional)
            top_k: Number of documents to retrieve
        """
        self.dataset = EvaluationDataset(dataset_path)
        self.retriever = retriever
        self.top_k = top_k
        self.answer_metrics = AnswerMetrics()

        # Models to compare: (provider, model_name, config)
        self.models: List[Tuple[str, str, Dict]] = []

    def add_model(
        self,
        provider: str,
        model_name: str,
        **config,
    ) -> "ModelComparison":
        """
        Add a model to the comparison.

        Args:
            provider: Provider name ("huggingface", "openai", "anthropic")
            model_name: Model identifier
            **config: Additional model configuration

        Returns:
            self for chaining
        """
        self.models.append((provider, model_name, config))
        logger.info(f"Added model: {provider}/{model_name}")
        return self

    def run(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42,
        include_baseline: bool = True,
        save_results: bool = True,
        output_dir: str = "./comparison_results",
    ) -> ComparisonResult:
        """
        Run the comparison experiment.

        Args:
            sample_size: Number of samples (None = full dataset)
            seed: Random seed for sampling
            include_baseline: Include no-RAG baseline for each model
            save_results: Save results to JSON
            output_dir: Output directory for results

        Returns:
            ComparisonResult with all model results
        """
        # Import model factory
        from app.models import create_model

        # Get samples
        if sample_size:
            samples = self.dataset.sample(sample_size, seed=seed)
        else:
            samples = list(self.dataset)

        result = ComparisonResult(
            timestamp=datetime.now().isoformat(),
            sample_size=len(samples),
            config={
                "top_k": self.top_k,
                "include_baseline": include_baseline,
                "models": [(p, m) for p, m, _ in self.models],
            },
        )

        logger.info(f"Starting comparison with {len(samples)} samples")
        logger.info(f"Models: {[f'{p}/{m}' for p, m, _ in self.models]}")

        # Run each model
        for provider, model_name, config in self.models:
            logger.info(f"\nEvaluating {provider}/{model_name}...")

            try:
                model = create_model(provider, model_name, **config)
            except Exception as e:
                logger.error(f"Failed to create model: {e}")
                continue

            # Run with RAG
            if self.retriever:
                rag_result = self._evaluate_model(
                    model=model,
                    samples=samples,
                    use_rag=True,
                )
                result.models.append(rag_result)

            # Run baseline (no RAG)
            if include_baseline:
                baseline_result = self._evaluate_model(
                    model=model,
                    samples=samples,
                    use_rag=False,
                )
                result.models.append(baseline_result)

        # Save results
        if save_results:
            self._save_results(result, output_dir)

        return result

    def _evaluate_model(
        self,
        model: Any,
        samples: List[QAPair],
        use_rag: bool,
    ) -> ModelResult:
        """Evaluate a single model configuration."""
        result = ModelResult(
            model_name=model.model_name,
            provider=model.provider_name,
            use_rag=use_rag,
        )

        mode = "RAG" if use_rag else "Baseline"
        logger.info(f"  Running {mode} mode...")

        for i, qa in enumerate(samples):
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)}")

            try:
                # Get context if using RAG
                context = None
                if use_rag and self.retriever:
                    retrieved = self.retriever.retrieve(qa.question, top_k=self.top_k)
                    context = self._build_context(retrieved)

                # Generate answer
                response = model.generate(
                    question=qa.question,
                    context=context,
                    use_rag=use_rag,
                )

                # Evaluate answer
                answer_eval = self.answer_metrics.evaluate_single(
                    question=qa.question,
                    generated=response.answer,
                    reference=qa.answer,
                    source_doc=qa.source_doc,
                    page_num=qa.page_num,
                )

                # Record results
                result.f1_scores.append(answer_eval.f1_score)
                result.key_phrase_scores.append(answer_eval.key_phrase_overlap)
                result.exact_matches.append(answer_eval.exact_match)
                result.latencies_ms.append(response.latency_ms)
                result.tokens_used.append(response.tokens_used)

            except Exception as e:
                result.errors.append(str(e))
                logger.warning(f"Error on sample {i}: {e}")

        return result

    def _build_context(self, retrieved: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        parts = []
        for doc in retrieved:
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Unknown")
            parts.append(f"[Source: {source}]\n{content}")
        return "\n\n---\n\n".join(parts)

    def _save_results(self, result: ComparisonResult, output_dir: str) -> None:
        """Save comparison results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = output_path / f"comparison_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {json_path}")

        # Save markdown report
        md_path = output_path / f"comparison_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(result.to_markdown())
        logger.info(f"Report saved to {md_path}")


def statistical_significance(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """
    Calculate statistical significance between two score distributions.

    Uses paired t-test and Wilcoxon signed-rank test.

    Returns:
        Dict with t-statistic, p-value, and effect size (Cohen's d)
    """
    from scipy import stats

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    n = len(scores_a)
    if n < 2:
        return {"error": "Not enough samples"}

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(scores_a, scores_b)

    # Wilcoxon (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        w_stat, w_pvalue = None, None

    # Effect size (Cohen's d)
    diff = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = statistics.mean(diff)
    std_diff = statistics.stdev(diff) if len(diff) > 1 else 1.0
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pvalue,
        "cohens_d": cohens_d,
        "mean_diff": mean_diff,
        "significant_005": t_pvalue < 0.05 if t_pvalue else False,
        "significant_001": t_pvalue < 0.01 if t_pvalue else False,
    }


def demo_comparison():
    """Demo function to test comparison framework."""
    print("=" * 60)
    print("MODEL COMPARISON DEMO")
    print("=" * 60)

    # This is a demo showing the framework structure
    # Actual usage requires models and retriever

    print("\nUsage example:")
    print("""
    from app.evaluation.comparison import ModelComparison
    from app.retrieval import RetrievalPipeline

    # Setup
    retriever = RetrievalPipeline.from_persisted("./data")
    comparison = ModelComparison(
        dataset_path="test.jsonl",
        retriever=retriever,
    )

    # Add models
    comparison.add_model("huggingface", "Qwen/Qwen2.5-7B-Instruct")
    comparison.add_model("huggingface", "meta-llama/Llama-3.1-8B-Instruct")
    comparison.add_model("openai", "gpt-4o-mini")

    # Run comparison
    results = comparison.run(
        sample_size=100,
        include_baseline=True,  # Also test without RAG
    )

    # View results
    print(results.to_markdown())

    # Check RAG lift
    lift = results.get_rag_lift("Qwen/Qwen2.5-7B-Instruct")
    print(f"RAG lift for Qwen2.5-7B: {lift:+.1%}")
    """)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_comparison()
