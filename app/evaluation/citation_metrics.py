"""
Citation Quality Metrics for TPN RAG System.

Evaluates how well the RAG system provides verifiable citations,
especially important when using fine-tuned models that may already
"know" the answers but need to prove their sources.

Key Metrics:
1. Source Attribution Accuracy - Did we cite the correct document?
2. Page-Level Precision - Did we cite the correct page?
3. Citation Faithfulness - Does the answer actually come from the cited context?
4. Citation Completeness - Did we cite all relevant sources?

Use Case: Fine-Tuned Model + RAG
- Model may already know the answer from training
- RAG provides VERIFIABILITY by showing where the answer comes from
- These metrics measure the quality of that attribution

Example:
    >>> from app.evaluation.citation_metrics import CitationEvaluator
    >>> evaluator = CitationEvaluator()
    >>> result = evaluator.evaluate(
    ...     question="What is the protein requirement?",
    ...     generated_answer="3-4 g/kg/day [ASPEN Handbook, p.44]",
    ...     retrieved_chunks=retrieved_chunks,
    ...     ground_truth_source="ASPEN Handbook",
    ...     ground_truth_page=44
    ... )
    >>> print(f"Source accuracy: {result.source_accuracy}")
    >>> print(f"Faithfulness: {result.faithfulness_score}")
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""
    content: str
    source_doc: str
    page_num: Optional[int] = None
    chunk_id: Optional[str] = None
    score: float = 0.0

    @property
    def citation_string(self) -> str:
        """Generate citation string for this chunk."""
        if self.page_num:
            return f"{self.source_doc} (p.{self.page_num})"
        return self.source_doc


@dataclass
class CitationResult:
    """
    Result from citation quality evaluation.

    Attributes:
        source_accuracy: Did the citation match the ground truth source? (0-1)
        page_precision: Did we cite the exact page? (0-1)
        faithfulness_score: How much of the answer comes from cited context? (0-1)
        completeness: Did we cite all relevant chunks? (0-1)
        hallucination_risk: Estimated risk of hallucinated content (0-1)
        extracted_citations: Citations found in the generated answer
        matched_citations: Citations that match retrieved chunks
    """
    source_accuracy: float = 0.0
    page_precision: float = 0.0
    faithfulness_score: float = 0.0
    completeness: float = 0.0
    hallucination_risk: float = 0.0

    # Detailed breakdown
    extracted_citations: List[Dict[str, Any]] = field(default_factory=list)
    matched_citations: List[Dict[str, Any]] = field(default_factory=list)

    # Ground truth comparison
    ground_truth_source_found: bool = False
    ground_truth_page_found: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_accuracy": round(self.source_accuracy, 4),
            "page_precision": round(self.page_precision, 4),
            "faithfulness_score": round(self.faithfulness_score, 4),
            "completeness": round(self.completeness, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "ground_truth_source_found": self.ground_truth_source_found,
            "ground_truth_page_found": self.ground_truth_page_found,
            "num_citations_extracted": len(self.extracted_citations),
            "num_citations_matched": len(self.matched_citations),
        }

    @property
    def overall_score(self) -> float:
        """Combined citation quality score."""
        # Weighted combination
        return (
            0.3 * self.source_accuracy +
            0.2 * self.page_precision +
            0.3 * self.faithfulness_score +
            0.1 * self.completeness +
            0.1 * (1 - self.hallucination_risk)
        )


class CitationEvaluator:
    """
    Evaluates citation quality for RAG-generated answers.

    Particularly useful for fine-tuned models where:
    - Model may already know the answer
    - RAG provides verifiable citations
    - We need to measure attribution quality
    """

    # Patterns to extract citations from generated answers
    CITATION_PATTERNS = [
        # "[ASPEN Handbook, p.44]" or "[ASPEN Handbook, page 44]"
        re.compile(r'\[([^\]]+?),?\s*(?:p\.?|page)\s*(\d+)\]', re.IGNORECASE),
        # "(ASPEN Handbook, p.44)" or "(Source: ASPEN Handbook, p.44)"
        re.compile(r'\((?:Source:\s*)?([^)]+?),?\s*(?:p\.?|page)\s*(\d+)\)', re.IGNORECASE),
        # "According to ASPEN Handbook (p.44)"
        re.compile(r'According to (?:the )?([^(]+?)\s*\((?:p\.?|page)\s*(\d+)\)', re.IGNORECASE),
        # "ASPEN Handbook states (p.44)"
        re.compile(r'([^,]+?)\s+(?:states?|indicates?|recommends?)\s*\((?:p\.?|page)\s*(\d+)\)', re.IGNORECASE),
        # Just "[Source Name]" without page
        re.compile(r'\[([^\]]+?)\](?!\()', re.IGNORECASE),
    ]

    # Clinical key terms that should appear in both answer and context
    CLINICAL_TERMS = [
        r'\d+\.?\d*\s*(?:mg|g|kg|ml|mcg|iu|mmol|meq)/(?:kg|day|hr|l)',
        r'\d+\s*(?:to|-)\s*\d+\s*(?:mg|g|kg|ml|%)',
        r'(?:preterm|term|neonate|infant|pediatric)',
        r'(?:protein|amino acid|lipid|glucose|dextrose)',
        r'(?:calcium|phosphorus|sodium|potassium|magnesium)',
    ]

    def __init__(self):
        """Initialize the citation evaluator."""
        self._clinical_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLINICAL_TERMS]

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from generated answer text.

        Returns list of dicts with:
        - source: Document name
        - page: Page number (if found)
        - raw: Original citation text
        """
        citations = []
        seen = set()

        for pattern in self.CITATION_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                source = groups[0].strip() if groups[0] else None
                page = int(groups[1]) if len(groups) > 1 and groups[1] else None
                raw = match.group(0)

                # Avoid duplicates
                key = (source.lower() if source else "", page)
                if key not in seen:
                    seen.add(key)
                    citations.append({
                        "source": source,
                        "page": page,
                        "raw": raw,
                    })

        return citations

    def normalize_source(self, source: str) -> str:
        """Normalize source name for comparison."""
        if not source:
            return ""

        s = source.lower()
        # Remove common prefixes
        for prefix in ["the ", "a "]:
            if s.startswith(prefix):
                s = s[len(prefix):]
        # Remove file extensions
        s = re.sub(r'\.(md|json|pdf|txt)$', '', s)
        # Normalize whitespace
        s = re.sub(r'[_\-]+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def sources_match(self, source1: str, source2: str) -> bool:
        """Check if two source names refer to the same document."""
        norm1 = self.normalize_source(source1)
        norm2 = self.normalize_source(source2)

        if not norm1 or not norm2:
            return False

        # Exact match
        if norm1 == norm2:
            return True

        # One contains the other
        if norm1 in norm2 or norm2 in norm1:
            return True

        # Significant word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))

        if min_words > 0 and overlap / min_words >= 0.5:
            return True

        return False

    def compute_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Compute how much of the answer is supported by the context.

        This measures whether the answer content actually comes from
        the cited sources, rather than being hallucinated.

        Returns:
            Score 0-1, where 1 means fully supported by context
        """
        if not answer or not context:
            return 0.0

        answer_lower = answer.lower()
        context_lower = context.lower()

        # Method 1: Clinical term overlap
        answer_terms = set()
        context_terms = set()

        for pattern in self._clinical_patterns:
            for match in pattern.finditer(answer_lower):
                answer_terms.add(match.group(0))
            for match in pattern.finditer(context_lower):
                context_terms.add(match.group(0))

        # If answer has clinical terms, check if they're in context
        term_faithfulness = 1.0
        if answer_terms:
            overlap = len(answer_terms & context_terms)
            term_faithfulness = overlap / len(answer_terms)

        # Method 2: N-gram overlap (3-grams for phrases)
        def get_ngrams(text: str, n: int = 3) -> Set[str]:
            words = text.split()
            return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

        answer_ngrams = get_ngrams(answer_lower)
        context_ngrams = get_ngrams(context_lower)

        ngram_faithfulness = 0.0
        if answer_ngrams:
            overlap = len(answer_ngrams & context_ngrams)
            ngram_faithfulness = overlap / len(answer_ngrams)

        # Method 3: Key factual statements
        # Check if numbers/values in answer appear in context
        numbers_in_answer = set(re.findall(r'\d+\.?\d*', answer))
        numbers_in_context = set(re.findall(r'\d+\.?\d*', context))

        number_faithfulness = 1.0
        if numbers_in_answer:
            overlap = len(numbers_in_answer & numbers_in_context)
            number_faithfulness = overlap / len(numbers_in_answer)

        # Weighted combination
        faithfulness = (
            0.4 * term_faithfulness +
            0.3 * ngram_faithfulness +
            0.3 * number_faithfulness
        )

        return min(1.0, faithfulness)

    def compute_hallucination_risk(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Estimate risk that the answer contains hallucinated content.

        High risk indicators:
        - Specific numbers not in context
        - Definitive claims without context support
        - Named entities not in context

        Returns:
            Risk score 0-1, where 1 is high risk
        """
        if not answer:
            return 0.0
        if not context:
            return 1.0  # No context = high risk

        answer_lower = answer.lower()
        context_lower = context.lower()

        risk_factors = []

        # Factor 1: Unsupported numbers
        numbers_in_answer = set(re.findall(r'\d+\.?\d*', answer))
        numbers_in_context = set(re.findall(r'\d+\.?\d*', context))

        if numbers_in_answer:
            unsupported = numbers_in_answer - numbers_in_context
            risk_factors.append(len(unsupported) / len(numbers_in_answer))

        # Factor 2: Absolute claims without support
        absolute_patterns = [
            r'\balways\b', r'\bnever\b', r'\bmust\b', r'\bexactly\b',
            r'\bdefinitely\b', r'\brequired\b', r'\bmandatory\b'
        ]

        absolute_claims = 0
        supported_claims = 0
        for pattern in absolute_patterns:
            if re.search(pattern, answer_lower):
                absolute_claims += 1
                if re.search(pattern, context_lower):
                    supported_claims += 1

        if absolute_claims > 0:
            risk_factors.append(1 - (supported_claims / absolute_claims))

        # Factor 3: Drug/dosing mentions not in context
        drug_pattern = r'\b(?:mg|mcg|g|ml)/(?:kg|day|hr)\b'
        dosing_in_answer = set(re.findall(drug_pattern, answer_lower))
        dosing_in_context = set(re.findall(drug_pattern, context_lower))

        if dosing_in_answer:
            unsupported = dosing_in_answer - dosing_in_context
            if unsupported:
                risk_factors.append(0.8)  # High risk for unsupported dosing

        # Average risk factors
        if risk_factors:
            return sum(risk_factors) / len(risk_factors)
        return 0.2  # Default low risk

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        retrieved_chunks: List[RetrievedChunk],
        ground_truth_source: Optional[str] = None,
        ground_truth_page: Optional[int] = None,
    ) -> CitationResult:
        """
        Evaluate citation quality for a single question-answer pair.

        Args:
            question: The original question
            generated_answer: The model's answer (should contain citations)
            retrieved_chunks: List of chunks retrieved by RAG
            ground_truth_source: Expected source document (from evaluation set)
            ground_truth_page: Expected page number (from evaluation set)

        Returns:
            CitationResult with all metrics
        """
        result = CitationResult()

        # Extract citations from generated answer
        extracted = self.extract_citations(generated_answer)
        result.extracted_citations = extracted

        # Combine all retrieved context
        all_context = "\n".join(chunk.content for chunk in retrieved_chunks)

        # Match extracted citations to retrieved chunks
        matched = []
        for citation in extracted:
            for chunk in retrieved_chunks:
                if self.sources_match(citation["source"], chunk.source_doc):
                    # Check page match if both have page numbers
                    page_match = True
                    if citation["page"] and chunk.page_num:
                        page_match = citation["page"] == chunk.page_num

                    matched.append({
                        "citation": citation,
                        "chunk": chunk.citation_string,
                        "page_match": page_match,
                    })
                    break

        result.matched_citations = matched

        # Compute source accuracy
        if extracted:
            result.source_accuracy = len(matched) / len(extracted)
        else:
            result.source_accuracy = 0.0  # No citations = 0 accuracy

        # Compute page precision
        if matched:
            page_matches = sum(1 for m in matched if m.get("page_match", False))
            result.page_precision = page_matches / len(matched)

        # Check ground truth source
        if ground_truth_source:
            for citation in extracted:
                if self.sources_match(citation["source"], ground_truth_source):
                    result.ground_truth_source_found = True
                    if ground_truth_page and citation["page"] == ground_truth_page:
                        result.ground_truth_page_found = True
                    break

            # Also check retrieved chunks
            for chunk in retrieved_chunks:
                if self.sources_match(chunk.source_doc, ground_truth_source):
                    result.ground_truth_source_found = True
                    if ground_truth_page and chunk.page_num == ground_truth_page:
                        result.ground_truth_page_found = True
                    break

        # Compute faithfulness (does answer come from context?)
        result.faithfulness_score = self.compute_faithfulness(
            generated_answer, all_context
        )

        # Compute hallucination risk
        result.hallucination_risk = self.compute_hallucination_risk(
            generated_answer, all_context
        )

        # Compute completeness (did we use all relevant chunks?)
        if retrieved_chunks and extracted:
            cited_sources = {self.normalize_source(c["source"]) for c in extracted}
            retrieved_sources = {self.normalize_source(c.source_doc) for c in retrieved_chunks[:3]}

            if retrieved_sources:
                cited_relevant = sum(
                    1 for s in retrieved_sources
                    if any(self.sources_match(s, cs) for cs in cited_sources)
                )
                result.completeness = cited_relevant / len(retrieved_sources)

        return result

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate citation quality across multiple results.

        Args:
            results: List of dicts with:
                - question, generated_answer, retrieved_chunks
                - ground_truth_source, ground_truth_page (optional)

        Returns:
            Aggregated metrics
        """
        all_results = []

        for item in results:
            # Convert chunk dicts to RetrievedChunk objects if needed
            chunks = item.get("retrieved_chunks", [])
            if chunks and isinstance(chunks[0], dict):
                chunks = [
                    RetrievedChunk(
                        content=c.get("content", ""),
                        source_doc=c.get("metadata", {}).get("source", c.get("source_doc", "")),
                        page_num=c.get("metadata", {}).get("page", c.get("page_num")),
                        score=c.get("score", 0.0),
                    )
                    for c in chunks
                ]

            result = self.evaluate(
                question=item.get("question", ""),
                generated_answer=item.get("generated_answer", ""),
                retrieved_chunks=chunks,
                ground_truth_source=item.get("ground_truth_source"),
                ground_truth_page=item.get("ground_truth_page"),
            )
            all_results.append(result)

        # Aggregate
        n = len(all_results)
        if n == 0:
            return {}

        return {
            "source_accuracy": sum(r.source_accuracy for r in all_results) / n,
            "page_precision": sum(r.page_precision for r in all_results) / n,
            "faithfulness_score": sum(r.faithfulness_score for r in all_results) / n,
            "completeness": sum(r.completeness for r in all_results) / n,
            "hallucination_risk": sum(r.hallucination_risk for r in all_results) / n,
            "ground_truth_found_rate": sum(1 for r in all_results if r.ground_truth_source_found) / n,
            "overall_citation_score": sum(r.overall_score for r in all_results) / n,
        }


def demo_citation_metrics():
    """Demo the citation evaluation metrics."""
    print("=" * 60)
    print("CITATION METRICS DEMO")
    print("=" * 60)

    evaluator = CitationEvaluator()

    # Example with good citations
    print("\n--- Good Citation Example ---")
    good_answer = """
    According to the ASPEN Guidelines (p.44), protein requirements for preterm infants
    are 3-4 g/kg/day. This is supported by multiple studies showing improved growth
    outcomes with adequate protein intake [ASPEN Guidelines, p.45].
    """

    chunks = [
        RetrievedChunk(
            content="Protein requirements: Preterm infants require 3-4 g/kg/day of protein for optimal growth. Studies show this range supports neurodevelopment.",
            source_doc="ASPEN Guidelines",
            page_num=44,
            score=0.95,
        ),
        RetrievedChunk(
            content="Research supports protein intake of 3-4 g/kg/day for preterm populations.",
            source_doc="ASPEN Guidelines",
            page_num=45,
            score=0.88,
        ),
    ]

    result = evaluator.evaluate(
        question="What is the protein requirement for preterm infants?",
        generated_answer=good_answer,
        retrieved_chunks=chunks,
        ground_truth_source="ASPEN Guidelines",
        ground_truth_page=44,
    )

    print(f"Source Accuracy: {result.source_accuracy:.1%}")
    print(f"Page Precision: {result.page_precision:.1%}")
    print(f"Faithfulness: {result.faithfulness_score:.1%}")
    print(f"Hallucination Risk: {result.hallucination_risk:.1%}")
    print(f"Ground Truth Found: {result.ground_truth_source_found}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Extracted Citations: {len(result.extracted_citations)}")

    # Example with poor citations (hallucination)
    print("\n--- Poor Citation Example (Potential Hallucination) ---")
    poor_answer = """
    The protein requirement is exactly 5.5 g/kg/day according to recent guidelines.
    This must always be administered within the first 24 hours.
    """

    result = evaluator.evaluate(
        question="What is the protein requirement for preterm infants?",
        generated_answer=poor_answer,
        retrieved_chunks=chunks,
        ground_truth_source="ASPEN Guidelines",
        ground_truth_page=44,
    )

    print(f"Source Accuracy: {result.source_accuracy:.1%}")
    print(f"Faithfulness: {result.faithfulness_score:.1%}")
    print(f"Hallucination Risk: {result.hallucination_risk:.1%}")
    print(f"Overall Score: {result.overall_score:.1%}")
    print(f"Extracted Citations: {len(result.extracted_citations)}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT FOR FINE-TUNED MODELS:")
    print("=" * 60)
    print("""
    A fine-tuned model might "know" the answer is 3-4 g/kg/day from training.
    But RAG adds VERIFIABILITY:

    1. Source Accuracy: Did we cite the right document?
    2. Page Precision: Can the user verify on that exact page?
    3. Faithfulness: Does the answer match what's in the citation?
    4. Hallucination Risk: Is the model making things up?

    Even if the model's answer is correct, proper citation proves it.
    """)


if __name__ == "__main__":
    demo_citation_metrics()
