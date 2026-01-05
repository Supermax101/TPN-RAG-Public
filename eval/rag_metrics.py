"""
Industry-Standard RAG Evaluation Metrics
Based on RAGAS, TruLens, and academic literature.

Metrics Categories:
1. Retrieval Quality - How good are the retrieved documents?
2. Generation Quality - How good is the generated answer?
3. End-to-End Quality - Overall system performance
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import json
from collections import Counter


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    # Basic counts
    num_retrieved: int = 0
    num_relevant: int = 0  # Based on score threshold
    
    # Score-based metrics
    avg_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0
    score_variance: float = 0.0
    
    # Threshold-based (score > 0.5 = relevant)
    precision_at_k: float = 0.0  # relevant / retrieved
    
    # Ranking quality
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Quality indicators
    high_quality_docs: int = 0   # score > 0.7
    medium_quality_docs: int = 0  # 0.3 < score < 0.7
    low_quality_docs: int = 0    # score < 0.3
    
    # Time
    retrieval_time_ms: float = 0.0
    
    def quality_grade(self) -> str:
        """Grade retrieval quality A-F."""
        if self.avg_score >= 0.7:
            return "A"
        elif self.avg_score >= 0.5:
            return "B"
        elif self.avg_score >= 0.3:
            return "C"
        elif self.avg_score >= 0.2:
            return "D"
        else:
            return "F"


@dataclass
class GenerationMetrics:
    """Metrics for evaluating generation quality."""
    
    # Answer extraction
    answer_extracted: bool = True
    answer_confidence: str = "unknown"
    
    # Faithfulness indicators (heuristic-based)
    uses_context_phrases: int = 0  # How many context phrases appear in answer
    potential_hallucination: bool = False
    
    # Response quality
    response_length: int = 0
    generation_time_ms: float = 0.0
    
    # For MCQ
    answer_format_correct: bool = True  # Did it output valid letter(s)?


@dataclass
class EndToEndMetrics:
    """Combined end-to-end metrics."""
    
    # Correctness
    is_correct: bool = False
    partial_correct: bool = False  # For multi-answer
    
    # Retrieval-Generation gap analysis
    retrieval_quality: str = "unknown"  # A-F grade
    generation_used_context: bool = True
    
    # Error classification
    error_type: Optional[str] = None
    # Types: "retrieval_failure", "generation_failure", "multi_answer_partial", "format_error"
    
    # Timing
    total_time_ms: float = 0.0


class RAGMetricsCalculator:
    """Calculate comprehensive RAG metrics."""
    
    # NOTE: Threshold 0.55 is appropriate for cosine similarity.
    # Previous threshold (0.3) was too low and inflated precision metrics.
    def __init__(self, relevance_threshold: float = 0.55):
        self.relevance_threshold = relevance_threshold
    
    def calculate_retrieval_metrics(
        self, 
        scores: List[float], 
        retrieval_time_ms: float = 0.0
    ) -> RetrievalMetrics:
        """Calculate retrieval quality metrics from similarity scores."""
        
        metrics = RetrievalMetrics()
        
        if not scores:
            return metrics
        
        metrics.num_retrieved = len(scores)
        metrics.retrieval_time_ms = retrieval_time_ms
        
        # Basic stats
        metrics.avg_score = sum(scores) / len(scores)
        metrics.max_score = max(scores)
        metrics.min_score = min(scores)
        
        # Variance
        mean = metrics.avg_score
        metrics.score_variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        
        # Quality buckets
        for score in scores:
            if score >= 0.7:
                metrics.high_quality_docs += 1
            elif score >= 0.3:
                metrics.medium_quality_docs += 1
            else:
                metrics.low_quality_docs += 1
        
        # Relevant docs (above threshold)
        metrics.num_relevant = sum(1 for s in scores if s >= self.relevance_threshold)
        
        # Precision@K
        if metrics.num_retrieved > 0:
            metrics.precision_at_k = metrics.num_relevant / metrics.num_retrieved
        
        # MRR - reciprocal rank of first relevant doc
        for i, score in enumerate(scores):
            if score >= self.relevance_threshold:
                metrics.mrr = 1.0 / (i + 1)
                break
        
        return metrics
    
    def calculate_generation_metrics(
        self,
        answer: str,
        context: str,
        raw_response: str,
        generation_time_ms: float = 0.0
    ) -> GenerationMetrics:
        """Calculate generation quality metrics."""
        
        metrics = GenerationMetrics()
        metrics.generation_time_ms = generation_time_ms
        metrics.response_length = len(raw_response)
        
        # Check answer extraction
        if not answer or answer in ["UNKNOWN", "ERROR", "PARSE_ERROR"]:
            metrics.answer_extracted = False
            metrics.answer_format_correct = False
        
        # Check if valid MCQ format
        valid_letters = set("ABCDEF")
        answer_letters = set(answer.replace(",", "").replace(" ", "").upper())
        metrics.answer_format_correct = answer_letters.issubset(valid_letters) and len(answer_letters) > 0
        
        # Heuristic faithfulness check
        # Count how many context key phrases appear in the answer
        if context and raw_response:
            # Extract key phrases from context (simple approach)
            context_words = set(w.lower() for w in re.findall(r'\b\w{5,}\b', context))
            answer_words = set(w.lower() for w in re.findall(r'\b\w{5,}\b', raw_response))
            
            overlap = context_words & answer_words
            metrics.uses_context_phrases = len(overlap)
            
            # If answer is long but uses no context words, might be hallucination
            if len(raw_response) > 200 and metrics.uses_context_phrases < 3:
                metrics.potential_hallucination = True
        
        return metrics
    
    def calculate_e2e_metrics(
        self,
        predicted: str,
        expected: str,
        retrieval_metrics: RetrievalMetrics,
        generation_metrics: GenerationMetrics
    ) -> EndToEndMetrics:
        """Calculate end-to-end metrics with error analysis."""
        
        metrics = EndToEndMetrics()
        metrics.retrieval_quality = retrieval_metrics.quality_grade()
        metrics.generation_used_context = generation_metrics.uses_context_phrases > 2
        metrics.total_time_ms = retrieval_metrics.retrieval_time_ms + generation_metrics.generation_time_ms
        
        # Normalize answers
        pred_norm = self._normalize_answer(predicted)
        exp_norm = self._normalize_answer(expected)
        
        # Exact match
        metrics.is_correct = pred_norm == exp_norm
        
        # Partial match for multi-answer
        if not metrics.is_correct and "," in exp_norm:
            expected_set = set(exp_norm.split(","))
            predicted_set = set(pred_norm.split(","))
            
            if predicted_set & expected_set:
                metrics.partial_correct = True
        
        # Error classification
        if not metrics.is_correct:
            if retrieval_metrics.avg_score < 0.3:
                metrics.error_type = "retrieval_failure"
            elif not generation_metrics.answer_format_correct:
                metrics.error_type = "format_error"
            elif metrics.partial_correct:
                metrics.error_type = "multi_answer_partial"
            elif generation_metrics.potential_hallucination:
                metrics.error_type = "hallucination"
            else:
                metrics.error_type = "generation_failure"
        
        return metrics
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.strip().upper()
        letters = re.findall(r'\b([A-F])\b', answer)
        return ",".join(sorted(set(letters))) if letters else answer


@dataclass
class EvaluationSummary:
    """Summary of all evaluation metrics."""
    
    total_questions: int = 0
    correct: int = 0
    partial_correct: int = 0
    
    # Accuracy metrics
    accuracy: float = 0.0
    accuracy_with_partial: float = 0.0
    
    # Retrieval summary
    avg_retrieval_score: float = 0.0
    retrieval_grade_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Error analysis
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    avg_total_time_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    
    def print_summary(self):
        """Print formatted summary with clean tables."""
        from rich.console import Console
        from rich.table import Table
        from rich import box
        
        console = Console()
        
        print("\n" + "=" * 75)
        print("                    RAG EVALUATION SUMMARY")
        print("=" * 75)
        
        # Core metrics table
        main_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        main_table.add_column("Metric", width=28)
        main_table.add_column("Value", justify="right", width=12)
        main_table.add_column("Details", width=28)
        
        main_table.add_row(
            "Accuracy (Exact)", 
            f"{self.accuracy:.1f}%",
            f"{self.correct}/{self.total_questions} correct"
        )
        main_table.add_row(
            "Accuracy (with Partial)", 
            f"{self.accuracy_with_partial:.1f}%",
            f"+{self.partial_correct} partial matches"
        )
        main_table.add_row(
            "Avg Retrieval Score",
            f"{self.avg_retrieval_score:.3f}",
            self._score_quality_label(self.avg_retrieval_score)
        )
        main_table.add_row("Avg Retrieval Time", f"{self.avg_retrieval_time_ms:.0f} ms", "")
        main_table.add_row("Avg Generation Time", f"{self.avg_generation_time_ms:.0f} ms", "")
        main_table.add_row("Avg Total Time", f"{self.avg_total_time_ms:.0f} ms", "")
        
        console.print(main_table)
        
        # Retrieval grade distribution
        print("\n" + "-" * 75)
        print("Retrieval Quality Distribution")
        print("-" * 75)
        
        grade_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        grade_table.add_column("Grade", width=8)
        grade_table.add_column("Count", justify="right", width=8)
        grade_table.add_column("Distribution", width=35)
        grade_table.add_column("Score Range", width=15)
        
        grade_thresholds = {
            "A": ">= 0.70", "B": "0.50-0.69", "C": "0.30-0.49", 
            "D": "0.20-0.29", "F": "< 0.20"
        }
        
        for grade in ["A", "B", "C", "D", "F"]:
            count = self.retrieval_grade_distribution.get(grade, 0)
            bar = "#" * min(count * 2, 35)
            grade_table.add_row(grade, str(count), bar, grade_thresholds[grade])
        
        console.print(grade_table)
        
        # Error analysis
        if self.error_distribution:
            print("\n" + "-" * 75)
            print("Error Analysis")
            print("-" * 75)
            
            error_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
            error_table.add_column("Error Type", width=25)
            error_table.add_column("Count", justify="right", width=8)
            error_table.add_column("Percentage", justify="right", width=12)
            error_table.add_column("Root Cause", width=25)
            
            total_errors = sum(self.error_distribution.values())
            
            error_causes = {
                "retrieval_failure": "Documents missing/poor match",
                "generation_failure": "LLM reasoning error",
                "multi_answer_partial": "Multi-select not handled",
                "hallucination": "Answer not from context",
                "format_error": "Invalid output format"
            }
            
            for error_type, count in sorted(self.error_distribution.items(), key=lambda x: -x[1]):
                pct = (count / total_errors * 100) if total_errors > 0 else 0
                cause = error_causes.get(error_type, "Unknown")
                error_table.add_row(error_type, str(count), f"{pct:.0f}%", cause)
            
            console.print(error_table)
        
        print("=" * 75)
    
    def _score_quality_label(self, score: float) -> str:
        """Get quality label for a score."""
        if score >= 0.7:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.3:
            return "Moderate"
        else:
            return "Poor"


def aggregate_metrics(results: List[Dict[str, Any]]) -> EvaluationSummary:
    """Aggregate individual results into summary."""
    
    summary = EvaluationSummary()
    summary.total_questions = len(results)
    
    if not results:
        return summary
    
    retrieval_scores = []
    retrieval_grades = []
    error_types = []
    total_times = []
    retrieval_times = []
    generation_times = []
    
    for r in results:
        if r.get("correct"):
            summary.correct += 1
        elif r.get("partial_correct"):
            summary.partial_correct += 1
        
        if "retrieval_metrics" in r:
            rm = r["retrieval_metrics"]
            retrieval_scores.append(rm.get("avg_score", 0))
            retrieval_times.append(rm.get("retrieval_ms", 0))
        
        if "retrieval_grade" in r:
            retrieval_grades.append(r["retrieval_grade"])
        
        if "error_type" in r and r["error_type"]:
            error_types.append(r["error_type"])
        
        if "generation_time_ms" in r:
            generation_times.append(r["generation_time_ms"])
        
        if "total_time_ms" in r:
            total_times.append(r["total_time_ms"])
    
    # Calculate averages
    summary.accuracy = (summary.correct / summary.total_questions) * 100
    summary.accuracy_with_partial = ((summary.correct + summary.partial_correct) / summary.total_questions) * 100
    
    if retrieval_scores:
        summary.avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
    
    # Grade distribution
    summary.retrieval_grade_distribution = dict(Counter(retrieval_grades))
    
    # Error distribution
    summary.error_distribution = dict(Counter(error_types))
    
    # Timing
    if total_times:
        summary.avg_total_time_ms = sum(total_times) / len(total_times)
    if retrieval_times:
        summary.avg_retrieval_time_ms = sum(retrieval_times) / len(retrieval_times)
    if generation_times:
        summary.avg_generation_time_ms = sum(generation_times) / len(generation_times)
    
    return summary


# Improvement strategies based on metrics
IMPROVEMENT_STRATEGIES = {
    "retrieval_failure": [
        "Add more domain-specific documents to the knowledge base",
        "Use a domain-specific embedding model (e.g., medical embeddings)",
        "Improve document chunking - try smaller chunks with more overlap",
        "Add document summaries and key terms as metadata"
    ],
    "multi_answer_partial": [
        "Update prompt: 'Select ALL correct answers, not just one'",
        "Add few-shot examples with multi-answer questions",
        "Train/fine-tune on multi-answer MCQ format"
    ],
    "generation_failure": [
        "Try a different/larger LLM model",
        "Improve prompt with chain-of-thought reasoning",
        "Add more few-shot examples in the prompt",
        "Increase temperature slightly for more exploration"
    ],
    "hallucination": [
        "Add instruction: 'Only use information from the provided context'",
        "Add instruction: 'If unsure, say so rather than guessing'",
        "Reduce temperature to 0 for more deterministic outputs",
        "Add source citations requirement"
    ],
    "format_error": [
        "Simplify output format requirements",
        "Add more output format examples",
        "Use structured output/JSON mode if available"
    ]
}


def print_recommendations(summary: EvaluationSummary):
    """Print improvement recommendations as a clean table."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    
    console = Console()
    
    if not summary.error_distribution:
        print("No errors to analyze.")
        return
    
    print("\n" + "-" * 75)
    print("Improvement Recommendations")
    print("-" * 75)
    
    rec_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    rec_table.add_column("Priority", width=8, justify="center")
    rec_table.add_column("Issue", width=22)
    rec_table.add_column("Count", justify="center", width=8)
    rec_table.add_column("Recommended Fix", width=40)
    
    sorted_errors = sorted(summary.error_distribution.items(), key=lambda x: -x[1])
    
    for priority, (error_type, count) in enumerate(sorted_errors[:5], 1):
        if error_type in IMPROVEMENT_STRATEGIES:
            fix = IMPROVEMENT_STRATEGIES[error_type][0]
            rec_table.add_row(f"#{priority}", error_type, str(count), fix)
    
    console.print(rec_table)


def get_improvement_recommendations(summary: EvaluationSummary) -> List[str]:
    """Get prioritized improvement recommendations (legacy text format)."""
    
    recommendations = []
    
    # Sort errors by frequency
    sorted_errors = sorted(summary.error_distribution.items(), key=lambda x: -x[1])
    
    for error_type, count in sorted_errors[:3]:  # Top 3 error types
        if error_type in IMPROVEMENT_STRATEGIES:
            recommendations.append(f"\n🔧 For {error_type} ({count} occurrences):")
            for strategy in IMPROVEMENT_STRATEGIES[error_type][:2]:
                recommendations.append(f"   • {strategy}")
    
    # General recommendations based on metrics
    if summary.avg_retrieval_score < 0.4:
        recommendations.append("\n🔧 Low overall retrieval quality:")
        recommendations.append("   • Consider using hybrid search (BM25 + vector)")
        recommendations.append("   • Review and expand your document corpus")
    
    return recommendations
