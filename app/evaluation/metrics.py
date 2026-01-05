"""
Evaluation Metrics for TPN RAG System.

Provides metrics for evaluating:
1. Retrieval Quality: Did we find the right source documents?
2. Answer Quality: Is the generated answer correct and complete?

Key metrics:
- Retrieval: Hit@K, MRR, Source Match Rate
- Answer: Exact Match, F1 Score, Semantic Similarity, Citation Accuracy

Example:
    >>> retrieval_metrics = RetrievalMetrics()
    >>> score = retrieval_metrics.hit_at_k(retrieved_docs, ground_truth_source, k=5)

    >>> answer_metrics = AnswerMetrics()
    >>> score = answer_metrics.f1_score(generated_answer, ground_truth_answer)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import Counter


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
    semantic_similarity: Optional[float] = None


@dataclass
class EvaluationResult:
    """
    Aggregated results from running evaluation on a dataset.

    Contains both retrieval and answer quality metrics.
    """
    # Dataset info
    total_samples: int = 0
    samples_evaluated: int = 0

    # Retrieval metrics
    retrieval_hit_at_1: float = 0.0
    retrieval_hit_at_3: float = 0.0
    retrieval_hit_at_5: float = 0.0
    retrieval_mrr: float = 0.0  # Mean Reciprocal Rank

    # Answer metrics
    answer_exact_match: float = 0.0
    answer_f1: float = 0.0
    answer_key_phrase_overlap: float = 0.0
    answer_citation_accuracy: float = 0.0
    answer_semantic_similarity: Optional[float] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)

    # Detailed results
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    answer_results: List[AnswerResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
                "semantic_similarity": round(self.answer_semantic_similarity, 4) if self.answer_semantic_similarity else None,
            },
            "errors_count": len(self.errors),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Samples: {self.samples_evaluated}/{self.total_samples}",
            "",
            "RETRIEVAL METRICS:",
            f"  Hit@1: {self.retrieval_hit_at_1:.1%}",
            f"  Hit@3: {self.retrieval_hit_at_3:.1%}",
            f"  Hit@5: {self.retrieval_hit_at_5:.1%}",
            f"  MRR:   {self.retrieval_mrr:.3f}",
            "",
            "ANSWER METRICS:",
            f"  Exact Match:       {self.answer_exact_match:.1%}",
            f"  F1 Score:          {self.answer_f1:.1%}",
            f"  Key Phrase Overlap: {self.answer_key_phrase_overlap:.1%}",
            f"  Citation Accuracy: {self.answer_citation_accuracy:.1%}",
        ]
        if self.answer_semantic_similarity is not None:
            lines.append(f"  Semantic Similarity: {self.answer_semantic_similarity:.1%}")
        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
        lines.append("=" * 50)
        return "\n".join(lines)


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval quality.

    Measures whether the RAG system retrieved the correct source documents.
    """

    def __init__(self, source_normalizer: Optional[callable] = None):
        """
        Initialize retrieval metrics.

        Args:
            source_normalizer: Optional function to normalize source names
                              for fuzzy matching
        """
        self.source_normalizer = source_normalizer or self._default_normalizer

    def _default_normalizer(self, source: str) -> str:
        """
        Normalize source name for matching.

        Handles variations like:
        - "the ASPEN Handbook" vs "ASPEN Handbook"
        - "ASPEN_Handbook.md" vs "ASPEN Handbook"
        """
        if not source:
            return ""

        # Convert to lowercase
        s = source.lower()

        # Remove common prefixes
        prefixes = ["the ", "a "]
        for prefix in prefixes:
            if s.startswith(prefix):
                s = s[len(prefix):]

        # Remove file extensions
        s = re.sub(r'\.(md|json|pdf|txt)$', '', s)

        # Remove underscores and extra spaces
        s = s.replace("_", " ")
        s = re.sub(r'\s+', ' ', s)

        # Remove special characters
        s = re.sub(r'[^\w\s]', '', s)

        return s.strip()

    def source_matches(self, source1: str, source2: str) -> bool:
        """
        Check if two source names refer to the same document.

        Uses fuzzy matching with normalization.
        """
        norm1 = self.source_normalizer(source1)
        norm2 = self.source_normalizer(source2)

        # Exact match after normalization
        if norm1 == norm2:
            return True

        # Substring match (one contains the other)
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Check if key words match
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # If significant overlap in key words
        if len(words1 & words2) >= min(len(words1), len(words2)) * 0.5:
            return True

        return False

    def hit_at_k(
        self,
        retrieved_sources: List[str],
        ground_truth_source: str,
        k: int = 5
    ) -> bool:
        """
        Check if ground truth source is in top-k retrieved sources.

        Args:
            retrieved_sources: List of retrieved source document names
            ground_truth_source: The correct source document
            k: Number of top results to consider

        Returns:
            True if ground truth is in top-k results
        """
        for source in retrieved_sources[:k]:
            if self.source_matches(source, ground_truth_source):
                return True
        return False

    def reciprocal_rank(
        self,
        retrieved_sources: List[str],
        ground_truth_source: str
    ) -> float:
        """
        Calculate reciprocal rank (1/position of first correct result).

        Args:
            retrieved_sources: List of retrieved source document names
            ground_truth_source: The correct source document

        Returns:
            Reciprocal rank (1.0 for rank 1, 0.5 for rank 2, etc.)
        """
        for i, source in enumerate(retrieved_sources):
            if self.source_matches(source, ground_truth_source):
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_single(
        self,
        query: str,
        retrieved_sources: List[str],
        ground_truth_source: str
    ) -> RetrievalResult:
        """
        Evaluate retrieval for a single query.

        Returns:
            RetrievalResult with all metrics
        """
        return RetrievalResult(
            query=query,
            ground_truth_source=ground_truth_source,
            retrieved_sources=retrieved_sources[:5],
            hit_at_1=self.hit_at_k(retrieved_sources, ground_truth_source, k=1),
            hit_at_3=self.hit_at_k(retrieved_sources, ground_truth_source, k=3),
            hit_at_5=self.hit_at_k(retrieved_sources, ground_truth_source, k=5),
            reciprocal_rank=self.reciprocal_rank(retrieved_sources, ground_truth_source),
        )


class AnswerMetrics:
    """
    Metrics for evaluating answer quality.

    Measures how well the generated answer matches the ground truth.
    """

    # Clinical key phrases that should be preserved
    CLINICAL_PATTERNS = [
        r'\d+\.?\d*\s*(?:mg|g|kg|ml|l|mcg|iu|mmol|meq)/(?:kg|day|hr|min|l)',  # Dosing
        r'\d+\s*(?:to|-)\s*\d+\s*(?:mg|g|kg|ml|%)',  # Ranges
        r'\d+\s*(?:hours?|days?|weeks?|months?)',  # Time durations
        r'(?:sodium|potassium|calcium|phosphorus|magnesium|chloride)',  # Electrolytes
        r'(?:glucose|dextrose|protein|amino acids?|lipids?)',  # Nutrients
    ]

    def __init__(self):
        """Initialize answer metrics."""
        self._clinical_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLINICAL_PATTERNS]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for comparison.

        Normalizes and splits into words.
        """
        # Lowercase and remove punctuation (except for numbers/units)
        text = text.lower()
        text = re.sub(r'[^\w\s./%-]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t]

    def exact_match(self, generated: str, ground_truth: str) -> float:
        """
        Calculate exact match score.

        Returns 1.0 if answers match exactly (after normalization), 0.0 otherwise.
        """
        gen_tokens = self._tokenize(generated)
        gt_tokens = self._tokenize(ground_truth)

        return 1.0 if gen_tokens == gt_tokens else 0.0

    def f1_score(self, generated: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.

        Measures overlap between generated and ground truth tokens.
        """
        gen_tokens = Counter(self._tokenize(generated))
        gt_tokens = Counter(self._tokenize(ground_truth))

        # Calculate intersection
        common = gen_tokens & gt_tokens
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        # Precision and recall
        precision = num_common / sum(gen_tokens.values())
        recall = num_common / sum(gt_tokens.values())

        # F1
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def extract_key_phrases(self, text: str) -> Set[str]:
        """
        Extract clinically important key phrases from text.

        Focuses on dosing, measurements, and clinical terms.
        """
        phrases = set()

        for pattern in self._clinical_patterns:
            for match in pattern.finditer(text):
                phrase = match.group(0).lower().strip()
                phrases.add(phrase)

        return phrases

    def key_phrase_overlap(self, generated: str, ground_truth: str) -> float:
        """
        Calculate overlap of clinical key phrases.

        Focuses on important clinical information like dosing and values.
        """
        gen_phrases = self.extract_key_phrases(generated)
        gt_phrases = self.extract_key_phrases(ground_truth)

        if not gt_phrases:
            return 1.0  # No key phrases to match

        overlap = len(gen_phrases & gt_phrases)
        return overlap / len(gt_phrases)

    def citation_match(
        self,
        generated: str,
        ground_truth_source: str,
        ground_truth_page: Optional[int] = None
    ) -> bool:
        """
        Check if generated answer cites the correct source.

        Args:
            generated: Generated answer text
            ground_truth_source: Expected source document
            ground_truth_page: Expected page number (optional)

        Returns:
            True if citation matches
        """
        # Look for source mention in generated text
        generated_lower = generated.lower()
        source_lower = ground_truth_source.lower() if ground_truth_source else ""

        # Check if source is mentioned
        source_mentioned = False
        source_words = source_lower.replace("the ", "").split()

        # Check for key words from source
        for word in source_words:
            if len(word) > 3 and word in generated_lower:
                source_mentioned = True
                break

        # If page number provided, check for it too
        if ground_truth_page and source_mentioned:
            page_pattern = rf'p\.?\s*{ground_truth_page}|page\s*{ground_truth_page}'
            if re.search(page_pattern, generated_lower):
                return True

        return source_mentioned

    def evaluate_single(
        self,
        question: str,
        generated: str,
        ground_truth: str,
        ground_truth_source: Optional[str] = None,
        ground_truth_page: Optional[int] = None
    ) -> AnswerResult:
        """
        Evaluate answer quality for a single question.

        Returns:
            AnswerResult with all metrics
        """
        return AnswerResult(
            question=question,
            ground_truth=ground_truth,
            generated=generated,
            exact_match=self.exact_match(generated, ground_truth),
            f1_score=self.f1_score(generated, ground_truth),
            key_phrase_overlap=self.key_phrase_overlap(generated, ground_truth),
            citation_match=self.citation_match(generated, ground_truth_source, ground_truth_page),
        )


def demo_metrics():
    """Demo function to test the metrics."""
    print("=" * 60)
    print("EVALUATION METRICS DEMO")
    print("=" * 60)

    # Test retrieval metrics
    print("\n--- Retrieval Metrics ---")
    retrieval = RetrievalMetrics()

    retrieved = [
        "ASPEN Guidelines 2020.md",
        "NICU Nutrition Protocol.md",
        "TPN Handbook.md",
    ]
    ground_truth = "the ASPEN Guidelines"

    result = retrieval.evaluate_single("test query", retrieved, ground_truth)
    print(f"Ground truth: {ground_truth}")
    print(f"Retrieved: {retrieved}")
    print(f"Hit@1: {result.hit_at_1}")
    print(f"Hit@3: {result.hit_at_3}")
    print(f"RR: {result.reciprocal_rank}")

    # Test answer metrics
    print("\n--- Answer Metrics ---")
    answer = AnswerMetrics()

    gt = "Protein requirements for preterm infants are 3-4 g/kg/day according to ASPEN guidelines."
    gen = "According to ASPEN, preterm infants need 3-4 g/kg/day of protein for adequate growth."

    result = answer.evaluate_single("What are protein requirements?", gen, gt, "ASPEN guidelines")
    print(f"Ground truth: {gt}")
    print(f"Generated: {gen}")
    print(f"F1 Score: {result.f1_score:.3f}")
    print(f"Key Phrase Overlap: {result.key_phrase_overlap:.3f}")
    print(f"Citation Match: {result.citation_match}")

    # Extract key phrases
    print(f"\nKey phrases in GT: {answer.extract_key_phrases(gt)}")
    print(f"Key phrases in Gen: {answer.extract_key_phrases(gen)}")


if __name__ == "__main__":
    demo_metrics()
