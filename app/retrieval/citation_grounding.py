"""
Citation Grounding Module for TPN RAG.

Fixes hallucinated citations from fine-tuned models by:
1. Stripping fake/hallucinated citations from model output
2. Matching answer content to real retrieved documents
3. Injecting verified citations with actual source and page numbers

Use Case:
- Fine-tuned TPN model gives CORRECT answers
- But HALLUCINATES citations (makes up document names, page numbers)
- RAG retrieval provides REAL documents
- This module grounds the citations to reality

Example:
    >>> from app.retrieval.citation_grounding import CitationGrounder
    >>> grounder = CitationGrounder()
    >>>
    >>> # Fine-tuned model output (correct answer, fake citations)
    >>> model_output = "Protein requirement is 3-4 g/kg/day [Fake Handbook, p.999]"
    >>>
    >>> # Real chunks from RAG retrieval
    >>> retrieved_chunks = [
    ...     {"content": "Preterm infants require 3-4 g/kg/day protein...",
    ...      "metadata": {"source": "ASPEN Guidelines.md", "page": 44}}
    ... ]
    >>>
    >>> # Ground citations to reality
    >>> grounded = grounder.ground_citations(model_output, retrieved_chunks)
    >>> print(grounded)
    "Protein requirement is 3-4 g/kg/day [ASPEN Guidelines, p.44]"
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroundedChunk:
    """A retrieved chunk with grounding information."""
    content: str
    source_doc: str
    page_num: Optional[int] = None
    relevance_score: float = 0.0
    matched_spans: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Generate citation string."""
        if self.page_num:
            return f"[{self.source_doc}, p.{self.page_num}]"
        return f"[{self.source_doc}]"


@dataclass
class GroundingResult:
    """Result from citation grounding."""
    original_text: str
    grounded_text: str
    citations_removed: int
    citations_added: int
    matched_chunks: List[GroundedChunk]
    confidence: float  # How confident are we in the grounding?

    @property
    def improvement(self) -> str:
        """Summary of what was fixed."""
        return f"Removed {self.citations_removed} hallucinated citation(s), added {self.citations_added} verified citation(s)"


class CitationGrounder:
    """
    Grounds hallucinated citations to real retrieved documents.

    The key insight: A fine-tuned model may generate correct answers
    but hallucinate the sources. RAG provides real documents.
    We match the answer content to real chunks and inject verified citations.
    """

    # Patterns to detect and remove citations
    CITATION_PATTERNS = [
        # [Document Name, p.XX] or [Document Name, page XX]
        re.compile(r'\s*\[([^\]]+?),?\s*(?:p\.?|page)\s*\d+\]', re.IGNORECASE),
        # [Document Name] without page
        re.compile(r'\s*\[([^\]]+?)\](?!\()', re.IGNORECASE),
        # (Document Name, p.XX) or (Source: Document, p.XX)
        re.compile(r'\s*\((?:Source:\s*)?([^)]+?),?\s*(?:p\.?|page)\s*\d+\)', re.IGNORECASE),
        # According to Document (p.XX)
        re.compile(r'\s*According to (?:the )?[^(]+?\s*\((?:p\.?|page)\s*\d+\)', re.IGNORECASE),
    ]

    # Clinical terms/patterns that help match content to chunks
    CLINICAL_PATTERNS = [
        r'\d+\.?\d*\s*(?:mg|g|kg|ml|mcg|iu|mmol|meq)/(?:kg|day|hr|l)',  # Dosing
        r'\d+\s*(?:to|-)\s*\d+\s*(?:mg|g|kg|ml|%)',  # Ranges
        r'(?:preterm|term|neonate|infant|pediatric)',  # Populations
        r'(?:protein|amino acid|lipid|glucose|dextrose|fat)',  # Nutrients
        r'(?:calcium|phosphorus|sodium|potassium|magnesium|zinc)',  # Electrolytes
        r'(?:ASPEN|ESPGHAN|AAP)',  # Guidelines
    ]

    def __init__(self, min_match_threshold: float = 0.3):
        """
        Initialize the citation grounder.

        Args:
            min_match_threshold: Minimum similarity score to consider a match (0-1)
        """
        self.min_match_threshold = min_match_threshold
        self._clinical_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLINICAL_PATTERNS]

    def _clean_source_name(self, source: str) -> str:
        """Clean source name for display in citations."""
        if not source:
            return "Unknown Source"

        # Remove file extensions
        source = re.sub(r'\.(md|json|pdf|txt)$', '', source, flags=re.IGNORECASE)
        # Remove underscores
        source = source.replace("_", " ")
        # Clean extra whitespace
        source = re.sub(r'\s+', ' ', source).strip()

        return source

    def _extract_clinical_terms(self, text: str) -> Set[str]:
        """Extract clinical terms from text for matching."""
        terms = set()
        text_lower = text.lower()

        for pattern in self._clinical_patterns:
            for match in pattern.finditer(text_lower):
                terms.add(match.group(0).strip())

        return terms

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract numbers and values from text."""
        # Match numbers with optional units
        pattern = r'\d+\.?\d*\s*(?:mg|g|kg|ml|mcg|%|days?|hours?|weeks?)?'
        return set(re.findall(pattern, text.lower()))

    def _calculate_match_score(self, answer_segment: str, chunk_content: str) -> float:
        """
        Calculate how well an answer segment matches a chunk.

        Uses multiple signals:
        - Clinical term overlap
        - Number/value overlap
        - N-gram overlap
        """
        if not answer_segment or not chunk_content:
            return 0.0

        answer_lower = answer_segment.lower()
        chunk_lower = chunk_content.lower()

        scores = []

        # 1. Clinical term overlap (high weight - these are specific)
        answer_terms = self._extract_clinical_terms(answer_segment)
        chunk_terms = self._extract_clinical_terms(chunk_content)

        if answer_terms:
            overlap = len(answer_terms & chunk_terms)
            scores.append(overlap / len(answer_terms))

        # 2. Number overlap (very important for clinical accuracy)
        answer_nums = self._extract_numbers(answer_segment)
        chunk_nums = self._extract_numbers(chunk_content)

        if answer_nums:
            overlap = len(answer_nums & chunk_nums)
            scores.append(overlap / len(answer_nums))

        # 3. Word overlap (general matching)
        answer_words = set(answer_lower.split())
        chunk_words = set(chunk_lower.split())

        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'and',
                      'or', 'but', 'if', 'then', 'than', 'that', 'this', 'to',
                      'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from'}

        answer_words -= stop_words
        chunk_words -= stop_words

        if answer_words:
            overlap = len(answer_words & chunk_words)
            scores.append(overlap / len(answer_words))

        # Return weighted average
        if not scores:
            return 0.0

        # Weight clinical terms and numbers higher
        if len(scores) >= 3:
            return 0.4 * scores[0] + 0.4 * scores[1] + 0.2 * scores[2]
        elif len(scores) == 2:
            return 0.5 * scores[0] + 0.5 * scores[1]
        return scores[0]

    def _remove_citations(self, text: str) -> Tuple[str, int]:
        """
        Remove all citations from text.

        Returns:
            Tuple of (cleaned_text, count_removed)
        """
        cleaned = text
        count = 0

        for pattern in self.CITATION_PATTERNS:
            matches = pattern.findall(cleaned)
            count += len(matches)
            cleaned = pattern.sub('', cleaned)

        # Clean up any double spaces left behind
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:])', r'\1', cleaned)  # Fix space before punctuation

        return cleaned.strip(), count

    def _find_sentence_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find sentence boundaries in text.

        Returns:
            List of (start, end, sentence) tuples
        """
        # Split on sentence-ending punctuation, but NOT decimal points or abbreviations
        # Negative lookbehind for: digits (3.5), common abbreviations (p., etc.)
        sentences = []
        pattern = r'(?<!\d)(?<!p)(?<!e\.g)(?<!i\.e)(?<!vs)(?<!Dr)(?<!Mr)(?<!Mrs)[.!?]+\s+'

        start = 0
        for match in re.finditer(pattern, text):
            end = match.end()
            sentence = text[start:end].strip()
            # Only split if we have a substantial sentence (not just a fragment)
            if sentence and len(sentence) > 20:
                sentences.append((start, end, sentence))
                start = end

        # Add remaining text if any
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                sentences.append((start, len(text), remaining))

        # If no good splits found, treat the whole text as one sentence
        if not sentences:
            sentences = [(0, len(text), text.strip())]

        return sentences

    def _match_sentence_to_chunks(
        self,
        sentence: str,
        chunks: List[Dict[str, Any]],
    ) -> Optional[GroundedChunk]:
        """
        Find the best matching chunk for a sentence.

        Returns:
            GroundedChunk if match found, None otherwise
        """
        best_match = None
        best_score = 0.0

        for chunk in chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            score = self._calculate_match_score(sentence, content)

            if score > best_score and score >= self.min_match_threshold:
                best_score = score
                best_match = GroundedChunk(
                    content=content[:200],  # Truncate for storage
                    source_doc=self._clean_source_name(
                        metadata.get("source", metadata.get("source_doc", "Unknown"))
                    ),
                    page_num=metadata.get("page", metadata.get("page_num")),
                    relevance_score=score,
                    matched_spans=[sentence[:100]],
                )

        return best_match

    def ground_citations(
        self,
        model_output: str,
        retrieved_chunks: List[Dict[str, Any]],
        add_inline_citations: bool = True,
        add_references_section: bool = True,
    ) -> GroundingResult:
        """
        Ground hallucinated citations to real retrieved documents.

        This is the main method. It:
        1. Removes all existing (potentially hallucinated) citations
        2. Matches answer content to real retrieved chunks
        3. Injects verified citations

        Args:
            model_output: The fine-tuned model's answer (with hallucinated citations)
            retrieved_chunks: Real chunks from RAG retrieval
            add_inline_citations: Add citations inline after relevant sentences
            add_references_section: Add a References section at the end

        Returns:
            GroundingResult with corrected text and metadata
        """
        if not model_output:
            return GroundingResult(
                original_text="",
                grounded_text="",
                citations_removed=0,
                citations_added=0,
                matched_chunks=[],
                confidence=0.0,
            )

        # Step 1: Remove existing (hallucinated) citations
        cleaned_text, citations_removed = self._remove_citations(model_output)

        # Step 2: Find sentence boundaries
        sentences = self._find_sentence_boundaries(cleaned_text)

        # Step 3: Match sentences to chunks
        matched_chunks: List[GroundedChunk] = []
        sentence_citations: Dict[int, GroundedChunk] = {}

        for idx, (start, end, sentence) in enumerate(sentences):
            match = self._match_sentence_to_chunks(sentence, retrieved_chunks)
            if match:
                matched_chunks.append(match)
                sentence_citations[idx] = match

        # Deduplicate matched chunks by source
        unique_sources: Dict[str, GroundedChunk] = {}
        for chunk in matched_chunks:
            key = f"{chunk.source_doc}_{chunk.page_num}"
            if key not in unique_sources or chunk.relevance_score > unique_sources[key].relevance_score:
                unique_sources[key] = chunk

        # Step 4: Build grounded text
        grounded_parts = []
        citations_added = 0

        if add_inline_citations:
            for idx, (start, end, sentence) in enumerate(sentences):
                grounded_parts.append(sentence)

                # Add citation if we matched this sentence
                if idx in sentence_citations:
                    chunk = sentence_citations[idx]
                    grounded_parts.append(f" {chunk.citation}")
                    citations_added += 1
        else:
            grounded_parts = [cleaned_text]

        grounded_text = " ".join(grounded_parts)

        # Clean up spacing
        grounded_text = re.sub(r'\s+', ' ', grounded_text)
        grounded_text = re.sub(r'\s+([.,;:!?])', r'\1', grounded_text)

        # Step 5: Add references section if requested
        if add_references_section and unique_sources:
            grounded_text += "\n\nReferences:\n"
            for i, (key, chunk) in enumerate(unique_sources.items(), 1):
                ref = f"{i}. {chunk.source_doc}"
                if chunk.page_num:
                    ref += f", page {chunk.page_num}"
                grounded_text += ref + "\n"

        # Calculate confidence based on match quality
        if matched_chunks:
            avg_score = sum(c.relevance_score for c in matched_chunks) / len(matched_chunks)
            coverage = len(sentence_citations) / len(sentences) if sentences else 0
            confidence = 0.6 * avg_score + 0.4 * coverage
        else:
            confidence = 0.0

        return GroundingResult(
            original_text=model_output,
            grounded_text=grounded_text.strip(),
            citations_removed=citations_removed,
            citations_added=citations_added,
            matched_chunks=list(unique_sources.values()),
            confidence=confidence,
        )

    def ground_with_context_format(
        self,
        model_output: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Simpler grounding that just adds source info at the end.

        For cases where inline citations would be too disruptive.
        """
        # Remove hallucinated citations
        cleaned, _ = self._remove_citations(model_output)

        # Find best matching chunks (top 3)
        matches = []
        for chunk in retrieved_chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            score = self._calculate_match_score(cleaned, content)

            if score >= self.min_match_threshold:
                matches.append({
                    "source": self._clean_source_name(
                        metadata.get("source", metadata.get("source_doc", "Unknown"))
                    ),
                    "page": metadata.get("page", metadata.get("page_num")),
                    "score": score,
                })

        # Sort by score and take top 3
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:3]

        if not top_matches:
            return cleaned

        # Add source line
        sources = []
        for m in top_matches:
            if m["page"]:
                sources.append(f"{m['source']} (p.{m['page']})")
            else:
                sources.append(m["source"])

        return f"{cleaned}\n\nSource(s): {', '.join(sources)}"


def demo_citation_grounding():
    """Demo the citation grounding functionality."""
    print("=" * 70)
    print("CITATION GROUNDING DEMO")
    print("=" * 70)
    print("\nProblem: Fine-tuned model gives correct answers but hallucinates citations")
    print("Solution: RAG provides real documents to ground citations\n")

    grounder = CitationGrounder()

    # Simulate fine-tuned model output (correct answer, FAKE citations)
    model_output = """
    Protein requirements for preterm infants are 3-4 g/kg/day [Fake TPN Manual, p.999].
    This should be started within the first 24-48 hours of life [Made Up Guidelines, p.123].
    For term infants, the requirement is lower at 2.5-3 g/kg/day (Imaginary Handbook, p.456).
    Amino acids should be initiated early to prevent catabolism [Nonexistent Protocol, p.789].
    """

    print("--- FINE-TUNED MODEL OUTPUT (with hallucinated citations) ---")
    print(model_output)

    # Simulate real RAG retrieval results
    retrieved_chunks = [
        {
            "content": "Preterm infants require protein at 3-4 g/kg/day for optimal growth and neurodevelopment. Early amino acid administration within 24-48 hours prevents catabolism.",
            "metadata": {"source": "ASPEN_Guidelines_2020.md", "page": 44},
        },
        {
            "content": "Term infants have lower protein requirements of 2.5-3 g/kg/day compared to preterm populations.",
            "metadata": {"source": "ASPEN_Guidelines_2020.md", "page": 52},
        },
        {
            "content": "Early initiation of parenteral amino acids is recommended to prevent negative nitrogen balance.",
            "metadata": {"source": "NICU_Nutrition_Protocol.md", "page": 12},
        },
    ]

    print("\n--- REAL RETRIEVED CHUNKS (from RAG) ---")
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        print(f"{i}. [{source}, p.{page}]: {chunk['content'][:80]}...")

    # Ground the citations
    result = grounder.ground_citations(
        model_output,
        retrieved_chunks,
        add_inline_citations=True,
        add_references_section=True,
    )

    print("\n--- GROUNDED OUTPUT (with verified citations) ---")
    print(result.grounded_text)

    print("\n--- GROUNDING STATS ---")
    print(f"Citations removed: {result.citations_removed}")
    print(f"Citations added: {result.citations_added}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Matched chunks: {len(result.matched_chunks)}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
    The fine-tuned model's ANSWER was correct (3-4 g/kg/day, 24-48 hours, etc.)
    But the CITATIONS were hallucinated (Fake TPN Manual, Made Up Guidelines)

    RAG + Citation Grounding:
    1. Removes hallucinated citations
    2. Matches content to REAL retrieved documents
    3. Injects VERIFIED citations (ASPEN Guidelines p.44, p.52, etc.)

    Result: Correct answer + Verifiable citations = Trustworthy output
    """)


if __name__ == "__main__":
    demo_citation_grounding()
