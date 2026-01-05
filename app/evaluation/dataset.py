"""
Evaluation Dataset Loader for TPN RAG.

Loads and parses the grounded Q&A pairs from JSONL format.
Each Q&A pair includes:
- Question from user
- Ground truth answer with source citation
- Extracted source document name and page number

Example:
    >>> dataset = EvaluationDataset("/path/to/test.jsonl")
    >>> print(f"Loaded {len(dataset)} Q&A pairs")
    >>> for qa in dataset.sample(10):
    ...     print(f"Q: {qa.question[:50]}...")
    ...     print(f"Source: {qa.source_doc} (p.{qa.page_num})")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional, Iterator, Dict, Any


@dataclass
class QAPair:
    """
    A question-answer pair with ground truth and source citation.

    Attributes:
        question: The user's question
        answer: Ground truth answer from the dataset
        source_doc: Name of the source document cited
        page_num: Page number in the source document
        raw_citation: The full citation string extracted
        index: Position in the original dataset
    """

    question: str
    answer: str
    source_doc: Optional[str] = None
    page_num: Optional[int] = None
    raw_citation: Optional[str] = None
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "source_doc": self.source_doc,
            "page_num": self.page_num,
            "raw_citation": self.raw_citation,
            "index": self.index,
        }

    @property
    def has_citation(self) -> bool:
        """Check if this Q&A has a valid source citation."""
        return self.source_doc is not None and self.page_num is not None


class EvaluationDataset:
    """
    Load and manage the grounded Q&A evaluation dataset.

    This class handles:
    - Loading JSONL format data
    - Parsing source citations from answers
    - Providing iteration and sampling methods
    - Filtering by source document

    Example:
        >>> dataset = EvaluationDataset("test.jsonl")
        >>> print(f"Total: {len(dataset)} pairs")
        >>> print(f"With citations: {dataset.citation_coverage:.1%}")

        >>> # Sample 100 random Q&A pairs
        >>> sample = dataset.sample(100)

        >>> # Filter by source
        >>> aspen_qs = dataset.filter_by_source("ASPEN")
    """

    # Regex patterns for extracting citations
    CITATION_PATTERNS = [
        # "According to the ASPEN Handbook (p.44)"
        re.compile(r'According to (?:the )?([^(]+?)\s*\(p\.(\d+)\)', re.IGNORECASE),
        # "According to ASPEN Handbook (p.44)"
        re.compile(r'According to ([^(]+?)\s*\(p\.(\d+)\)', re.IGNORECASE),
        # "[ASPEN Handbook, p.44:"
        re.compile(r'\[([^\],]+?),?\s*p\.(\d+):', re.IGNORECASE),
        # "Source: ASPEN Handbook (p.44)"
        re.compile(r'Source:\s*([^(]+?)\s*\(p\.(\d+)\)', re.IGNORECASE),
    ]

    def __init__(self, dataset_path: str | Path):
        """
        Initialize the evaluation dataset.

        Args:
            dataset_path: Path to the JSONL file containing Q&A pairs
        """
        self.dataset_path = Path(dataset_path)
        self._qa_pairs: List[QAPair] = []
        self._source_index: Dict[str, List[int]] = {}

        self._load()

    def _load(self) -> None:
        """Load and parse the dataset from JSONL file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                data = json.loads(line)
                qa_pair = self._parse_entry(data, idx)
                self._qa_pairs.append(qa_pair)

                # Index by source document
                if qa_pair.source_doc:
                    source_key = qa_pair.source_doc.lower()
                    if source_key not in self._source_index:
                        self._source_index[source_key] = []
                    self._source_index[source_key].append(idx)

    def _parse_entry(self, data: Dict[str, Any], index: int) -> QAPair:
        """Parse a single JSONL entry into a QAPair."""
        messages = data.get("messages", [])

        # Extract question (user message) and answer (assistant message)
        question = ""
        answer = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                question = content
            elif role == "assistant":
                answer = content

        # Extract citation from answer
        source_doc, page_num, raw_citation = self._extract_citation(answer)

        return QAPair(
            question=question,
            answer=answer,
            source_doc=source_doc,
            page_num=page_num,
            raw_citation=raw_citation,
            index=index,
        )

    def _extract_citation(self, answer: str) -> tuple:
        """
        Extract source citation from the answer text.

        Returns:
            Tuple of (source_doc, page_num, raw_citation)
        """
        for pattern in self.CITATION_PATTERNS:
            match = pattern.search(answer)
            if match:
                source_doc = match.group(1).strip()
                page_num = int(match.group(2))
                raw_citation = match.group(0)
                return source_doc, page_num, raw_citation

        return None, None, None

    def __len__(self) -> int:
        """Return the number of Q&A pairs in the dataset."""
        return len(self._qa_pairs)

    def __iter__(self) -> Iterator[QAPair]:
        """Iterate over all Q&A pairs."""
        return iter(self._qa_pairs)

    def __getitem__(self, index: int) -> QAPair:
        """Get a Q&A pair by index."""
        return self._qa_pairs[index]

    def sample(self, n: int, seed: Optional[int] = None) -> List[QAPair]:
        """
        Get a random sample of Q&A pairs.

        Args:
            n: Number of samples to return
            seed: Random seed for reproducibility

        Returns:
            List of randomly sampled QAPair objects
        """
        if seed is not None:
            random.seed(seed)

        n = min(n, len(self._qa_pairs))
        return random.sample(self._qa_pairs, n)

    def filter_by_source(self, source_pattern: str) -> List[QAPair]:
        """
        Filter Q&A pairs by source document name pattern.

        Args:
            source_pattern: Substring to match in source document names

        Returns:
            List of matching QAPair objects
        """
        pattern_lower = source_pattern.lower()
        results = []

        for qa in self._qa_pairs:
            if qa.source_doc and pattern_lower in qa.source_doc.lower():
                results.append(qa)

        return results

    @property
    def citation_coverage(self) -> float:
        """Percentage of Q&A pairs with valid citations."""
        if not self._qa_pairs:
            return 0.0
        cited = sum(1 for qa in self._qa_pairs if qa.has_citation)
        return cited / len(self._qa_pairs)

    @property
    def unique_sources(self) -> List[str]:
        """List of unique source documents in the dataset."""
        sources = set()
        for qa in self._qa_pairs:
            if qa.source_doc:
                sources.add(qa.source_doc)
        return sorted(sources)

    def get_source_distribution(self) -> Dict[str, int]:
        """Get count of Q&A pairs per source document."""
        from collections import Counter
        sources = [qa.source_doc for qa in self._qa_pairs if qa.source_doc]
        return dict(Counter(sources).most_common())

    def to_dataframe(self):
        """Convert to pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required. Run: pip install pandas")

        data = [qa.to_dict() for qa in self._qa_pairs]
        return pd.DataFrame(data)

    def save_sample(self, path: str | Path, n: int, seed: int = 42) -> None:
        """Save a sample of the dataset to a new JSONL file."""
        sample = self.sample(n, seed=seed)
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            for qa in sample:
                entry = {
                    "question": qa.question,
                    "ground_truth": qa.answer,
                    "source_doc": qa.source_doc,
                    "page_num": qa.page_num,
                }
                f.write(json.dumps(entry) + "\n")


def demo_dataset():
    """Demo function to test the dataset loader."""
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/chandra/Desktop/TPN2.OFinetuning/data/final/test.jsonl"

    print("=" * 60)
    print("EVALUATION DATASET DEMO")
    print("=" * 60)

    try:
        dataset = EvaluationDataset(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"\nLoaded {len(dataset)} Q&A pairs")
    print(f"Citation coverage: {dataset.citation_coverage:.1%}")
    print(f"Unique sources: {len(dataset.unique_sources)}")

    # Show source distribution
    print("\n--- Top 10 Sources ---")
    dist = dataset.get_source_distribution()
    for source, count in list(dist.items())[:10]:
        print(f"  {count:3d}x - {source[:50]}")

    # Show sample Q&A
    print("\n--- Sample Q&A Pairs ---")
    for qa in dataset.sample(3, seed=42):
        print(f"\nQ: {qa.question[:100]}...")
        print(f"Source: {qa.source_doc} (p.{qa.page_num})")
        print(f"A: {qa.answer[:150]}...")


if __name__ == "__main__":
    demo_dataset()
