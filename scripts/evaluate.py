#!/usr/bin/env python3
"""
Evaluation Script for TPN RAG System.

Runs evaluation on the grounded Q&A dataset and produces metrics.

Usage:
    python scripts/evaluate.py [--dataset PATH] [--sample-size N] [options]

Examples:
    # Full evaluation
    python scripts/evaluate.py

    # Quick test with 50 samples
    python scripts/evaluate.py --sample-size 50

    # Retrieval-only evaluation
    python scripts/evaluate.py --retrieval-only

    # Save results to custom directory
    python scripts/evaluate.py --output-dir ./my_results
"""

import argparse
import json
import logging
import re
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QAPair:
    """A question-answer pair with ground truth and source citation."""
    question: str
    answer: str
    source_doc: Optional[str] = None
    page_num: Optional[int] = None
    raw_citation: Optional[str] = None
    index: int = 0

    @property
    def has_citation(self) -> bool:
        return self.source_doc is not None and self.page_num is not None


@dataclass
class RetrievalResult:
    """Result from a single retrieval evaluation."""
    query: str
    ground_truth_source: str
    retrieved_sources: List[str]
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    reciprocal_rank: float = 0.0


@dataclass
class AnswerResult:
    """Result from a single answer evaluation."""
    question: str
    ground_truth: str
    generated: str
    exact_match: float = 0.0
    f1_score: float = 0.0
    key_phrase_overlap: float = 0.0
    citation_match: bool = False


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    total_samples: int = 0
    samples_evaluated: int = 0
    retrieval_hit_at_1: float = 0.0
    retrieval_hit_at_3: float = 0.0
    retrieval_hit_at_5: float = 0.0
    retrieval_mrr: float = 0.0
    answer_exact_match: float = 0.0
    answer_f1: float = 0.0
    answer_key_phrase_overlap: float = 0.0
    answer_citation_accuracy: float = 0.0
    errors: List[str] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    answer_results: List[AnswerResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "samples_evaluated": self.samples_evaluated,
            "retrieval": {
                "hit_at_1": round(self.retrieval_hit_at_1, 4),
                "hit_at_3": round(self.retrieval_hit_at_3, 4),
                "hit_at_5": round(self.retrieval_hit_at_5, 4),
                "mrr": round(self.retrieval_mrr, 4),
            },
            "answer": {
                "exact_match": round(self.answer_exact_match, 4),
                "f1": round(self.answer_f1, 4),
                "key_phrase_overlap": round(self.answer_key_phrase_overlap, 4),
                "citation_accuracy": round(self.answer_citation_accuracy, 4),
            },
            "errors_count": len(self.errors),
        }


# =============================================================================
# DATASET LOADER
# =============================================================================

class EvaluationDataset:
    """Load and manage the grounded Q&A evaluation dataset."""

    CITATION_PATTERNS = [
        re.compile(r'According to (?:the )?([^(]+?)\s*\(p\.(\d+)\)', re.IGNORECASE),
        re.compile(r'According to ([^(]+?)\s*\(p\.(\d+)\)', re.IGNORECASE),
        re.compile(r'\[([^\],]+?),?\s*p\.(\d+):', re.IGNORECASE),
    ]

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self._qa_pairs: List[QAPair] = []
        self._load()

    def _load(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                qa_pair = self._parse_entry(data, idx)
                self._qa_pairs.append(qa_pair)

    def _parse_entry(self, data: Dict[str, Any], index: int) -> QAPair:
        messages = data.get("messages", [])
        question = ""
        answer = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content

        source_doc, page_num, raw_citation = self._extract_citation(answer)

        return QAPair(
            question=question,
            answer=answer,
            source_doc=source_doc,
            page_num=page_num,
            raw_citation=raw_citation,
            index=index,
        )

    def _extract_citation(self, answer: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
        for pattern in self.CITATION_PATTERNS:
            match = pattern.search(answer)
            if match:
                return match.group(1).strip(), int(match.group(2)), match.group(0)
        return None, None, None

    def __len__(self) -> int:
        return len(self._qa_pairs)

    def __iter__(self):
        return iter(self._qa_pairs)

    def sample(self, n: int, seed: Optional[int] = None) -> List[QAPair]:
        if seed is not None:
            random.seed(seed)
        n = min(n, len(self._qa_pairs))
        return random.sample(self._qa_pairs, n)

    @property
    def citation_coverage(self) -> float:
        if not self._qa_pairs:
            return 0.0
        cited = sum(1 for qa in self._qa_pairs if qa.has_citation)
        return cited / len(self._qa_pairs)

    def get_source_distribution(self) -> Dict[str, int]:
        sources = [qa.source_doc for qa in self._qa_pairs if qa.source_doc]
        return dict(Counter(sources).most_common())


# =============================================================================
# METRICS
# =============================================================================

class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    def _normalize(self, source: str) -> str:
        if not source:
            return ""
        s = source.lower()
        for prefix in ["the ", "a "]:
            if s.startswith(prefix):
                s = s[len(prefix):]
        s = re.sub(r'\.(md|json|pdf|txt)$', '', s)
        s = s.replace("_", " ")
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return s.strip()

    def source_matches(self, source1: str, source2: str) -> bool:
        norm1 = self._normalize(source1)
        norm2 = self._normalize(source2)
        if norm1 == norm2:
            return True
        if norm1 in norm2 or norm2 in norm1:
            return True
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if len(words1 & words2) >= min(len(words1), len(words2)) * 0.5:
            return True
        return False

    def hit_at_k(self, retrieved: List[str], ground_truth: str, k: int = 5) -> bool:
        for source in retrieved[:k]:
            if self.source_matches(source, ground_truth):
                return True
        return False

    def reciprocal_rank(self, retrieved: List[str], ground_truth: str) -> float:
        for i, source in enumerate(retrieved):
            if self.source_matches(source, ground_truth):
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_single(self, query: str, retrieved: List[str], ground_truth: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            ground_truth_source=ground_truth,
            retrieved_sources=retrieved[:5],
            hit_at_1=self.hit_at_k(retrieved, ground_truth, k=1),
            hit_at_3=self.hit_at_k(retrieved, ground_truth, k=3),
            hit_at_5=self.hit_at_k(retrieved, ground_truth, k=5),
            reciprocal_rank=self.reciprocal_rank(retrieved, ground_truth),
        )


class AnswerMetrics:
    """Metrics for evaluating answer quality."""

    CLINICAL_PATTERNS = [
        r'\d+\.?\d*\s*(?:mg|g|kg|ml|l|mcg|iu|mmol|meq)/(?:kg|day|hr|min|l)',
        r'\d+\s*(?:to|-)\s*\d+\s*(?:mg|g|kg|ml|%)',
        r'\d+\s*(?:hours?|days?|weeks?|months?)',
    ]

    def __init__(self):
        self._clinical_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLINICAL_PATTERNS]

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s./%-]', ' ', text)
        return [t for t in text.split() if t]

    def f1_score(self, generated: str, ground_truth: str) -> float:
        gen_tokens = Counter(self._tokenize(generated))
        gt_tokens = Counter(self._tokenize(ground_truth))
        common = gen_tokens & gt_tokens
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / sum(gen_tokens.values())
        recall = num_common / sum(gt_tokens.values())
        return 2 * precision * recall / (precision + recall)

    def extract_key_phrases(self, text: str) -> Set[str]:
        phrases = set()
        for pattern in self._clinical_patterns:
            for match in pattern.finditer(text):
                phrases.add(match.group(0).lower().strip())
        return phrases

    def key_phrase_overlap(self, generated: str, ground_truth: str) -> float:
        gen_phrases = self.extract_key_phrases(generated)
        gt_phrases = self.extract_key_phrases(ground_truth)
        if not gt_phrases:
            return 1.0
        return len(gen_phrases & gt_phrases) / len(gt_phrases)

    def evaluate_single(self, question: str, generated: str, ground_truth: str) -> AnswerResult:
        gen_tokens = self._tokenize(generated)
        gt_tokens = self._tokenize(ground_truth)
        return AnswerResult(
            question=question,
            ground_truth=ground_truth,
            generated=generated,
            exact_match=1.0 if gen_tokens == gt_tokens else 0.0,
            f1_score=self.f1_score(generated, ground_truth),
            key_phrase_overlap=self.key_phrase_overlap(generated, ground_truth),
        )


# =============================================================================
# EVALUATION HARNESS
# =============================================================================

class EvaluationHarness:
    """Main evaluation harness for TPN RAG system."""

    def __init__(
        self,
        dataset_path: str | Path,
        retriever=None,
        llm=None,
    ):
        self.dataset = EvaluationDataset(dataset_path)
        self.retriever = retriever
        self.llm = llm
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()

    def run(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42,
        retrieval_only: bool = False,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        result = EvaluationResult()

        # Get samples
        if sample_size:
            samples = self.dataset.sample(sample_size, seed=seed)
        else:
            samples = list(self.dataset)

        result.total_samples = len(samples)
        start_time = time.time()

        if verbose:
            logger.info(f"Starting evaluation on {len(samples)} samples")

        # Evaluate each sample
        for i, qa in enumerate(samples):
            try:
                if verbose and (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(samples)}")

                # Retrieval evaluation
                if self.retriever and qa.source_doc:
                    retrieved = self.retriever.retrieve(qa.question, top_k=5)
                    retrieved_sources = [
                        doc.get("metadata", {}).get("source", "")
                        for doc in retrieved
                    ]
                    ret_result = self.retrieval_metrics.evaluate_single(
                        qa.question, retrieved_sources, qa.source_doc
                    )
                    result.retrieval_results.append(ret_result)

                # Answer evaluation (if not retrieval-only)
                if not retrieval_only and self.llm:
                    context = self._build_context(retrieved if self.retriever else [])
                    generated = self.llm.generate(qa.question, context)
                    ans_result = self.answer_metrics.evaluate_single(
                        qa.question, generated, qa.answer
                    )
                    result.answer_results.append(ans_result)

                result.samples_evaluated += 1

            except Exception as e:
                result.errors.append(f"Sample {i}: {e}")

        # Aggregate metrics
        self._aggregate_metrics(result)

        elapsed = time.time() - start_time
        if verbose:
            logger.info(f"Evaluation completed in {elapsed:.1f}s")

        # Save results
        if output_dir:
            self._save_results(result, output_dir)

        return result

    def _build_context(self, docs: List[Dict]) -> str:
        if not docs:
            return ""
        return "\n\n".join(doc.get("content", "") for doc in docs)

    def _aggregate_metrics(self, result: EvaluationResult) -> None:
        if result.retrieval_results:
            n = len(result.retrieval_results)
            result.retrieval_hit_at_1 = sum(r.hit_at_1 for r in result.retrieval_results) / n
            result.retrieval_hit_at_3 = sum(r.hit_at_3 for r in result.retrieval_results) / n
            result.retrieval_hit_at_5 = sum(r.hit_at_5 for r in result.retrieval_results) / n
            result.retrieval_mrr = sum(r.reciprocal_rank for r in result.retrieval_results) / n

        if result.answer_results:
            n = len(result.answer_results)
            result.answer_exact_match = sum(r.exact_match for r in result.answer_results) / n
            result.answer_f1 = sum(r.f1_score for r in result.answer_results) / n
            result.answer_key_phrase_overlap = sum(r.key_phrase_overlap for r in result.answer_results) / n

    def _save_results(self, result: EvaluationResult, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"eval_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TPN RAG system on grounded Q&A dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        default="/Users/chandra/Desktop/TPN2.OFinetuning/data/final/test.jsonl",
        help="Path to evaluation dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval (skip LLM generation)",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_results",
        help="Directory to save results (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze dataset (don't run evaluation)",
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    print("=" * 60)
    print("TPN RAG EVALUATION")
    print("=" * 60)

    # Load and analyze dataset
    try:
        dataset = EvaluationDataset(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nDataset: {args.dataset}")
    print(f"Total Q&A pairs: {len(dataset)}")
    print(f"Citation coverage: {dataset.citation_coverage:.1%}")

    # Show source distribution
    dist = dataset.get_source_distribution()
    print(f"\nTop 5 sources:")
    for source, count in list(dist.items())[:5]:
        print(f"  {count:3d}x - {source[:50]}")

    if args.analyze_only:
        return 0

    print("\n" + "-" * 60)

    # Note about running actual evaluation
    print("\nNOTE: This script currently runs without a retriever or LLM.")
    print("To run full evaluation, integrate with your RAG pipeline:")
    print()
    print("  from scripts.evaluate import EvaluationHarness")
    print("  harness = EvaluationHarness(dataset_path, retriever=my_retriever, llm=my_llm)")
    print("  results = harness.run(sample_size=100)")
    print()

    # Demo with mock data
    if args.sample_size:
        print(f"\nRunning demo evaluation on {args.sample_size} samples...")
        harness = EvaluationHarness(args.dataset)
        result = harness.run(
            sample_size=args.sample_size,
            seed=args.seed,
            retrieval_only=True,
            output_dir=args.output_dir if not args.quiet else None,
            verbose=not args.quiet,
        )

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (No retriever - baseline analysis)")
        print("=" * 60)
        print(f"\nSamples evaluated: {result.samples_evaluated}/{result.total_samples}")
        print(f"Errors: {len(result.errors)}")

        if result.errors:
            print("\nFirst 3 errors:")
            for e in result.errors[:3]:
                print(f"  - {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
