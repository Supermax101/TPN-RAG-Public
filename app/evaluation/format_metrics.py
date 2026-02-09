"""
Deterministic format validation for open-ended benchmark outputs.

Open-ended evaluation (QandA20, Takeoff41-200) is easiest to score when each
model produces a clean, comparable output. For paper-grade runs we enforce a
strict contract:

- Response must start with `Final answer:`
- No citations/sources or document names
- No "work"/reasoning/analysis/thinking sections

This module provides a lightweight validator used by the benchmark runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


_FINAL_PREFIX_RE = re.compile(r"^\s*final\s*answer\s*:\s*", re.IGNORECASE)
_BANNED_SECTION_HEADER_RE = re.compile(
    r"(?im)^\s*(work|reasoning|analysis|thinking|citations?|sources?)\s*:\s*"
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[[^\]]+\]")


@dataclass(frozen=True)
class FormatCheckResult:
    ok: bool
    reasons: List[str]

    @property
    def reason(self) -> str:
        return ";".join([r for r in self.reasons if r]) if self.reasons else ""


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line
    return ""


def validate_open_final_answer(text: str) -> FormatCheckResult:
    """
    Validate strict "Final answer only" output contract for open-ended runs.

    Returns:
      FormatCheckResult(ok=..., reasons=[...])
    """
    raw = (text or "").strip()
    reasons: List[str] = []

    if not raw:
        return FormatCheckResult(ok=False, reasons=["empty_output"])

    first = _first_nonempty_line(raw)
    if not _FINAL_PREFIX_RE.match(first):
        # Common failure mode: model starts with an "analysis"/"reasoning" block.
        if re.match(r"(?i)^\s*(analysis|reasoning|thinking)\b", first or ""):
            reasons.append("starts_with_reasoning")
        reasons.append("missing_final_answer_prefix")

    if _THINK_TAG_RE.search(raw):
        reasons.append("contains_think_tags")

    if _BANNED_SECTION_HEADER_RE.search(raw):
        reasons.append("contains_banned_section_header")

    # Explicit mention of citations/sources anywhere is disallowed.
    if re.search(r"(?i)\b(citation|citations|source|sources)\b", raw):
        reasons.append("mentions_citations_or_sources")

    # Bracket citations like "[TPN Considerations]" are disallowed.
    # Avoid false positives on short bracket usage by requiring a space and a
    # minimum length inside the brackets.
    for m in _BRACKET_RE.finditer(raw):
        inner = (m.group(0) or "")[1:-1].strip()
        if len(inner) >= 6 and " " in inner:
            reasons.append("contains_bracket_citation")
            break

    ok = not reasons
    return FormatCheckResult(ok=ok, reasons=reasons)

