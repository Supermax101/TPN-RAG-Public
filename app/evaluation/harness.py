"""
Evaluation Harness for TPN RAG System.

Orchestrates the full evaluation pipeline:
1. Load evaluation dataset
2. Run queries through RAG retrieval
3. Generate answers with LLM
4. Compare against ground truth
5. Aggregate metrics

Supports multiple evaluation modes:
- Retrieval-only: Test just the retrieval component
- End-to-end: Test full RAG pipeline with LLM generation
- Baseline: Test LLM without RAG (to measure RAG value-add)

Example:
    >>> harness = EvaluationHarness(
    ...     dataset_path="/path/to/test.jsonl",
    ...     retriever=my_retriever,
    ...     llm=my_llm
    ... )
    >>> results = harness.run(sample_size=100)
    >>> print(results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union,  List, Optional, Dict, Any, Callable, Protocol
from datetime import datetime

from .dataset import EvaluationDataset, QAPair
from .metrics import (
    RetrievalMetrics,
    AnswerMetrics,
    EvaluationResult,
    RetrievalResult,
    AnswerResult,
)

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Protocol for retriever implementations."""

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        ...


class LLM(Protocol):
    """Protocol for LLM implementations."""

    def generate(self, prompt: str, context: str) -> str:
        """
        Generate an answer given a prompt and context.

        Returns:
            Generated answer string
        """
        ...


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""

    # Sampling
    sample_size: Optional[int] = None  # None = full dataset
    random_seed: int = 42

    # Retrieval settings
    top_k: int = 5
    retrieval_only: bool = False  # Skip LLM generation

    # Answer generation settings
    include_thinking: bool = True
    max_tokens: int = 500

    # Output settings
    save_results: bool = True
    output_dir: str = "./eval_results"
    verbose: bool = True

    # Progress callback
    progress_callback: Optional[Callable[[int, int], None]] = None


class EvaluationHarness:
    """
    Main evaluation harness for TPN RAG system.

    Runs evaluation on the grounded Q&A dataset and produces
    comprehensive metrics for retrieval and answer quality.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLM] = None,
    ):
        """
        Initialize the evaluation harness.

        Args:
            dataset_path: Path to the JSONL evaluation dataset
            retriever: Retriever implementation for document retrieval
            llm: LLM implementation for answer generation
        """
        self.dataset = EvaluationDataset(dataset_path)
        self.retriever = retriever
        self.llm = llm

        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()

    def run(self, config: Optional[EvaluationConfig] = None) -> EvaluationResult:
        """
        Run the full evaluation.

        Args:
            config: Evaluation configuration (uses defaults if None)

        Returns:
            EvaluationResult with aggregated metrics
        """
        config = config or EvaluationConfig()
        result = EvaluationResult()

        # Get samples
        if config.sample_size:
            samples = self.dataset.sample(config.sample_size, seed=config.random_seed)
        else:
            samples = list(self.dataset)

        result.total_samples = len(samples)

        logger.info(f"Starting evaluation on {len(samples)} samples")
        start_time = time.time()

        # Evaluate each sample
        for i, qa in enumerate(samples):
            try:
                # Progress callback
                if config.progress_callback:
                    config.progress_callback(i + 1, len(samples))

                if config.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(samples)}")

                # Run retrieval if retriever is available
                retrieval_result = None
                retrieved_docs = []

                if self.retriever and qa.source_doc:
                    retrieved = self.retriever.retrieve(qa.question, top_k=config.top_k)
                    retrieved_sources = [
                        doc.get("metadata", {}).get("source", "")
                        for doc in retrieved
                    ]
                    retrieved_docs = retrieved

                    retrieval_result = self.retrieval_metrics.evaluate_single(
                        qa.question,
                        retrieved_sources,
                        qa.source_doc,
                    )
                    result.retrieval_results.append(retrieval_result)

                # Skip LLM generation if retrieval-only mode
                if config.retrieval_only:
                    result.samples_evaluated += 1
                    continue

                # Generate answer if LLM is available
                answer_result = None
                if self.llm:
                    # Build context from retrieved documents
                    context = self._build_context(retrieved_docs)

                    # Generate answer
                    generated = self.llm.generate(qa.question, context)

                    # Evaluate answer
                    answer_result = self.answer_metrics.evaluate_single(
                        qa.question,
                        generated,
                        qa.answer,
                        qa.source_doc,
                        qa.page_num,
                    )
                    result.answer_results.append(answer_result)

                result.samples_evaluated += 1

            except Exception as e:
                error_msg = f"Error on sample {i}: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg)

        # Aggregate metrics
        self._aggregate_metrics(result)

        elapsed = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed:.1f}s")

        # Save results if configured
        if config.save_results:
            self._save_results(result, config)

        return result

    def run_retrieval_only(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42
    ) -> EvaluationResult:
        """
        Run retrieval-only evaluation (no LLM generation).

        Faster evaluation for testing retrieval quality.
        """
        config = EvaluationConfig(
            sample_size=sample_size,
            random_seed=seed,
            retrieval_only=True,
        )
        return self.run(config)

    def run_baseline(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42
    ) -> EvaluationResult:
        """
        Run baseline evaluation (LLM without RAG context).

        Measures what the LLM knows without retrieval.
        """
        # Temporarily disable retriever
        original_retriever = self.retriever
        self.retriever = None

        config = EvaluationConfig(
            sample_size=sample_size,
            random_seed=seed,
        )

        try:
            result = self.run(config)
        finally:
            self.retriever = original_retriever

        return result

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return ""

        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Unknown")
            context_parts.append(f"[Source: {source}]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _aggregate_metrics(self, result: EvaluationResult) -> None:
        """Calculate aggregate metrics from individual results."""
        # Retrieval metrics
        if result.retrieval_results:
            n = len(result.retrieval_results)
            result.retrieval_hit_at_1 = sum(r.hit_at_1 for r in result.retrieval_results) / n
            result.retrieval_hit_at_3 = sum(r.hit_at_3 for r in result.retrieval_results) / n
            result.retrieval_hit_at_5 = sum(r.hit_at_5 for r in result.retrieval_results) / n
            result.retrieval_mrr = sum(r.reciprocal_rank for r in result.retrieval_results) / n

        # Answer metrics
        if result.answer_results:
            n = len(result.answer_results)
            result.answer_exact_match = sum(r.exact_match for r in result.answer_results) / n
            result.answer_f1 = sum(r.f1_score for r in result.answer_results) / n
            result.answer_key_phrase_overlap = sum(r.key_phrase_overlap for r in result.answer_results) / n
            result.answer_citation_accuracy = sum(1 for r in result.answer_results if r.citation_match) / n

            # Semantic similarity if available
            sem_scores = [r.semantic_similarity for r in result.answer_results if r.semantic_similarity is not None]
            if sem_scores:
                result.answer_semantic_similarity = sum(sem_scores) / len(sem_scores)

    def _save_results(self, result: EvaluationResult, config: EvaluationConfig) -> None:
        """Save evaluation results to JSON file."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{timestamp}.json"

        output_data = {
            "timestamp": timestamp,
            "config": {
                "sample_size": config.sample_size,
                "top_k": config.top_k,
                "retrieval_only": config.retrieval_only,
            },
            "summary": result.to_dict(),
            "detailed_results": {
                "retrieval": [
                    {
                        "query": r.query[:100],
                        "ground_truth": r.ground_truth_source,
                        "hit_at_1": r.hit_at_1,
                        "hit_at_5": r.hit_at_5,
                        "rr": r.reciprocal_rank,
                    }
                    for r in result.retrieval_results[:100]  # Limit to first 100 for file size
                ],
                "answer": [
                    {
                        "question": r.question[:100],
                        "f1": r.f1_score,
                        "key_phrase": r.key_phrase_overlap,
                        "citation": r.citation_match,
                    }
                    for r in result.answer_results[:100]
                ],
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")


class MockRetriever:
    """Mock retriever for testing the evaluation harness."""

    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return mock documents (for testing)."""
        return self.docs[:top_k]


class MockLLM:
    """Mock LLM for testing the evaluation harness."""

    def generate(self, prompt: str, context: str) -> str:
        """Return mock answer based on context."""
        if context:
            return f"Based on the provided context: {context[:200]}..."
        return "I don't have enough information to answer this question."


def demo_harness():
    """Demo function to test the evaluation harness."""
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/chandra/Desktop/TPN2.OFinetuning/data/final/test.jsonl"

    print("=" * 60)
    print("EVALUATION HARNESS DEMO")
    print("=" * 60)

    try:
        # Create mock retriever and LLM
        mock_docs = [
            {
                "content": "Protein requirements for preterm infants are 3-4 g/kg/day.",
                "metadata": {"source": "ASPEN Pediatric Handbook.md"},
            },
            {
                "content": "Dextrose should be started at 6-8 mg/kg/min.",
                "metadata": {"source": "NICU Guidelines.md"},
            },
        ]

        retriever = MockRetriever(mock_docs)
        llm = MockLLM()

        # Create harness
        harness = EvaluationHarness(
            dataset_path=dataset_path,
            retriever=retriever,
            llm=llm,
        )

        print(f"\nDataset loaded: {len(harness.dataset)} Q&A pairs")
        print(f"Citation coverage: {harness.dataset.citation_coverage:.1%}")

        # Run small sample evaluation
        print("\nRunning evaluation on 10 samples...")
        config = EvaluationConfig(
            sample_size=10,
            random_seed=42,
            save_results=False,
            verbose=False,
        )

        result = harness.run(config)
        print(result)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python harness.py [dataset_path]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_harness()
