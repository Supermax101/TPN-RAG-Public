"""
Answer evaluation metrics for open-ended TPN questions.

Provides token-level F1, exact match, clinical key-phrase overlap,
and source citation matching — the standard QA evaluation metrics
expected by BenchmarkRunner for the open-ended track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Set


# Clinical patterns reused from citation_grounding logic
_CLINICAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\d+\.?\d*\s*(?:mg|g|kg|ml|mcg|iu|mmol|meq)/(?:kg|day|hr|l)",
        r"\d+\s*(?:to|-)\s*\d+\s*(?:mg|g|kg|ml|%)",
        r"(?:preterm|term|neonate|infant|pediatric)",
        r"(?:protein|amino acid|lipid|glucose|dextrose|fat)",
        r"(?:calcium|phosphorus|sodium|potassium|magnesium|zinc)",
        r"(?:ASPEN|ESPGHAN|AAP)",
    ]
]

_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will "
    "would could should may might must and or but if then than that this to "
    "of in on at for with by from".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split tokens with stop-word removal."""
    return [t for t in text.lower().split() if t not in _STOP_WORDS]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _extract_clinical_terms(text: str) -> Set[str]:
    terms: Set[str] = set()
    for pattern in _CLINICAL_PATTERNS:
        for match in pattern.finditer(text):
            terms.add(match.group(0).strip().lower())
    return terms


@dataclass
class EvalResult:
    """Result from evaluating a single open-ended answer."""

    f1_score: float
    exact_match: float
    key_phrase_overlap: float
    citation_match: bool


class AnswerMetrics:
    """Evaluate generated answers against ground-truth references."""

    def evaluate_single(
        self,
        question: str,
        generated: str,
        ground_truth: str,
        ground_truth_source: Optional[str] = None,
        ground_truth_page: Optional[int] = None,
    ) -> EvalResult:
        """
        Compute token-level F1, exact match, key-phrase overlap,
        and source citation match for one (generated, ground_truth) pair.
        """
        gen_norm = _normalize(generated)
        ref_norm = _normalize(ground_truth)

        # --- Exact match ---
        exact_match = 1.0 if gen_norm == ref_norm else 0.0

        # --- Token-level F1 ---
        gen_tokens = _tokenize(generated)
        ref_tokens = _tokenize(ground_truth)

        if not ref_tokens:
            f1 = 1.0 if not gen_tokens else 0.0
        elif not gen_tokens:
            f1 = 0.0
        else:
            gen_set = set(gen_tokens)
            ref_set = set(ref_tokens)
            common = gen_set & ref_set
            precision = len(common) / len(gen_set)
            recall = len(common) / len(ref_set)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # --- Clinical key-phrase overlap ---
        gen_terms = _extract_clinical_terms(generated)
        ref_terms = _extract_clinical_terms(ground_truth)

        if ref_terms:
            key_phrase_overlap = len(gen_terms & ref_terms) / len(ref_terms)
        else:
            key_phrase_overlap = 1.0 if not gen_terms else 0.0

        # --- Citation / source match ---
        citation_match = False
        if ground_truth_source:
            source_clean = re.sub(r"\.(md|json|pdf|txt)$", "", ground_truth_source, flags=re.IGNORECASE)
            source_clean = source_clean.replace("_", " ").lower().strip()
            if source_clean and source_clean in generated.lower():
                citation_match = True
            if ground_truth_page is not None and f"p.{ground_truth_page}" in generated:
                citation_match = True

        return EvalResult(
            f1_score=round(f1, 4),
            exact_match=exact_match,
            key_phrase_overlap=round(key_phrase_overlap, 4),
            citation_match=citation_match,
        )
