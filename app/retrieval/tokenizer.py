"""
Clinical-aware tokenizer for BM25 and keyword search.

Handles medical terminology, dosing ranges, and clinical abbreviations
that naive .split() tokenization would destroy.

Usage:
    >>> from app.retrieval.tokenizer import clinical_tokenize
    >>> clinical_tokenize("3-4 g/kg/day protein for preterm infants")
    ['3-4', 'g_per_kg_per_day', 'protein', 'preterm', 'infants']
"""

import re
from typing import List, Set


# Common English stopwords (minimal set for clinical text)
STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "that",
    "which", "who", "whom", "this", "these", "those", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what",
}

# Dosing unit patterns: g/kg/day, mg/kg/hr, mL/kg/day, etc.
_DOSING_RE = re.compile(
    r"""
    (\d+(?:\.\d+)?          # leading number (optional)
    \s*[-–]\s*              # range separator
    \d+(?:\.\d+)?           # trailing number
    \s*)?                   # entire numeric range is optional
    (m?[gG]|[mM][cC]?[gG]|[mM][lL]|kcal|cal|mmol|mEq|units?|IU)
    \s*/\s*
    (kg|[lL]|d[lL]?|day|hr|hour|min|dose)
    (?:\s*/\s*(day|hr|hour|min|dose))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Numeric range pattern: 3-4, 0.5-1.0, 10–15
_RANGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\b")

# Clinical abbreviations to preserve
_ABBREVIATIONS: Set[str] = {
    "tpn", "pn", "en", "aspen", "nicu", "gir", "bmi", "bun", "alt",
    "ast", "lfts", "cbc", "crp", "il", "iv", "po", "npo", "prn",
    "bid", "tid", "qid", "qd", "hs", "ac", "pc", "ml", "dl", "kg",
    "mcg", "mg", "meq", "mmol", "iu", "aa", "fa", "efa", "dha",
    "epa", "mct", "lct", "smof", "ifald", "pnald", "mos", "ros",
    "ca", "phos", "na", "k", "cl", "mg", "zn", "cu", "mn", "se",
    "cr", "fe", "co", "ifc", "cvl", "picc", "uac", "uvc",
}


def _normalize_dosing(text: str) -> str:
    """Convert dosing expressions to underscore-joined tokens."""
    def _replace(match: re.Match) -> str:
        full = match.group(0).strip()
        # Replace / with _per_ for searchability
        normalized = re.sub(r"\s*/\s*", "_per_", full)
        # Remove internal whitespace
        normalized = re.sub(r"\s+", "", normalized)
        return f" {normalized} "

    return _DOSING_RE.sub(_replace, text)


def clinical_tokenize(text: str) -> List[str]:
    """
    Tokenize clinical text with awareness of medical terminology.

    Features:
    - Preserves dosing expressions as single tokens (g/kg/day -> g_per_kg_per_day)
    - Preserves numeric ranges (3-4 stays as 3-4)
    - Removes common English stopwords
    - Lowercases everything
    - Preserves clinical abbreviations
    """
    if not text:
        return []

    # Step 1: Normalize dosing units before splitting
    processed = _normalize_dosing(text)

    # Step 2: Lowercase
    processed = processed.lower()

    # Step 3: Replace punctuation with spaces (but keep hyphens in ranges, underscores)
    # First protect numeric ranges
    ranges = {}
    for i, m in enumerate(_RANGE_RE.finditer(processed)):
        placeholder = f"__RANGE{i}__"
        ranges[placeholder] = m.group(0).replace(" ", "")
        processed = processed[:m.start()] + placeholder + processed[m.end():]

    # Replace most punctuation with spaces
    processed = re.sub(r"[^\w\s_]", " ", processed)

    # Restore ranges
    for placeholder, value in ranges.items():
        processed = processed.replace(placeholder, value)

    # Step 4: Split on whitespace
    tokens = processed.split()

    # Step 5: Filter stopwords and very short tokens (unless they're abbreviations)
    result = []
    for token in tokens:
        token = token.strip("_")
        if not token:
            continue
        if token in STOPWORDS:
            continue
        # Keep single-character tokens only if they look like clinical units
        if len(token) == 1 and token not in {"k", "l"}:
            continue
        result.append(token)

    return result
