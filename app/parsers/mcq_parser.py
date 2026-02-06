"""
MCQ Answer Parsing and Structured Output Models.

7-priority extraction system with negation handling.

Priority Order:
  1   │ \\boxed{X}  (LaTeX boxed)                        │ \\boxed{C}
  2   │ correct/best answer is X                         │ correct answer is B
  3   │ final answer is X                                │ final answer is C
  3b  │ Answer: X  or  Ans: X anywhere                   │ Ans: B
  4a  │ therefore/thus/hence X                           │ therefore C
  4b  │ select/choose X                                  │ select B
  4c  │ I would choose/select X                          │ I would choose A
  4d  │ it is X                                          │ it is C
  4e  │ X is correct                                     │ B is correct
  5   │ Last line is standalone letter(s)                │ ...reasoning\\nC
  6   │ First standalone letter in response              │ C. Because ...

Safety features:
  - Negation detection — rejects "The answer is not A"
  - Multi-select support — A,C,D or "A and C"
  - Precision-first — returns UNKNOWN rather than risk false positives
  - Word boundaries — won't match A inside words like "ASPEN"
"""

import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Valid answer letters
# ---------------------------------------------------------------------------
_VALID_LETTERS = frozenset("ABCDEF")

# ---------------------------------------------------------------------------
# Negation window: words before a letter that invalidate the match
# ---------------------------------------------------------------------------
_NEGATION_PATTERN = re.compile(
    r'\b(?:not|n\'t|isn\'t|isnt|cannot|never|incorrect|wrong|eliminate|except|exclude|ruling out)\b',
    re.IGNORECASE,
)

# How many characters before the match to scan for negation
_NEGATION_WINDOW = 30


# ---------------------------------------------------------------------------
# Multi-letter extraction  (handles "A, C, D" / "A and C" / "A,B")
# ---------------------------------------------------------------------------
_MULTI_LETTER = re.compile(
    r'([A-F])(?:\s*[,&]\s*|\s+and\s+)([A-F])(?:(?:\s*[,&]\s*|\s+and\s+)([A-F]))?'
    r'(?:(?:\s*[,&]\s*|\s+and\s+)([A-F]))?',
    re.IGNORECASE,
)

_SINGLE_LETTER = re.compile(r'\b([A-F])\b')


def _extract_letters(text: str) -> str:
    """Pull one or more answer letters from a short text span."""
    text = text.strip().upper()
    # Try multi first
    m = _MULTI_LETTER.search(text)
    if m:
        letters = sorted({g.upper() for g in m.groups() if g})
        return ",".join(letters)
    # Single
    m = _SINGLE_LETTER.search(text)
    if m:
        return m.group(1).upper()
    return ""


def _is_negated(text: str, match_start: int) -> bool:
    """Check if a match is preceded by negation words within the window.

    Only looks within the same clause — stops at sentence boundaries
    (period, semicolon, exclamation, question mark) to avoid rejecting
    a valid match because of negation in a *prior* sentence.
    """
    window_start = max(0, match_start - _NEGATION_WINDOW)
    window = text[window_start:match_start]
    # Trim to the last sentence boundary so negation in prior sentence
    # doesn't bleed through
    boundary = re.search(r'[.;!?]\s+', window)
    if boundary:
        window = window[boundary.end():]
    return bool(_NEGATION_PATTERN.search(window))


def _standalone_letter_on_line(line: str) -> str:
    """Return a letter only if the line is essentially just letter(s)."""
    stripped = line.strip()
    # "C" or "C." or "C)" or "(C)"
    m = re.match(r'^\(?([A-F])\)?\s*[.:]?\s*$', stripped, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Multi: "A, C" or "A,C,D"
    m = re.match(
        r'^([A-F](?:\s*[,&]\s*[A-F]|\s+and\s+[A-F])+)\s*[.:]?\s*$',
        stripped, re.IGNORECASE,
    )
    if m:
        return _extract_letters(m.group(1))
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic structured output models (unchanged API)
# ═══════════════════════════════════════════════════════════════════════════

class MCQAnswer(BaseModel):
    """Structured output for single-answer MCQ questions."""

    thinking: str = Field(
        description="Step-by-step clinical reasoning explaining the answer choice"
    )
    answer: str = Field(
        description="Single letter answer: A, B, C, D, E, or F"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the answer"
    )
    source_used: bool = Field(
        default=True,
        description="Whether the provided context was used to answer"
    )

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _VALID_LETTERS:
            match = _SINGLE_LETTER.search(v)
            if match:
                return match.group(1)
            raise ValueError(f"Answer must be A-F. Got: {v}")
        return v


class MCQMultiAnswer(BaseModel):
    """Structured output for multi-answer MCQ questions."""

    thinking: str = Field(
        description="Step-by-step clinical reasoning for each selected answer"
    )
    answers: List[str] = Field(
        description="List of correct answer letters, e.g., ['A', 'B', 'D']"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the answers"
    )
    source_used: bool = Field(
        default=True,
        description="Whether the provided context was used to answer"
    )

    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v: List[str]) -> List[str]:
        valid = []
        for ans in v:
            ans = ans.strip().upper()
            if ans in _VALID_LETTERS:
                valid.append(ans)
        if not valid:
            raise ValueError("At least one valid answer (A-F) required")
        return sorted(set(valid))

    @property
    def answer_string(self) -> str:
        return ",".join(sorted(self.answers))


# ═══════════════════════════════════════════════════════════════════════════
# 7-Priority parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_mcq_response(
    raw_response: str,
    is_multi_answer: bool = False,
) -> tuple[str, str, str]:
    """
    7-priority answer extraction with negation handling.

    Returns:
        (answer, thinking, confidence)
        answer is "UNKNOWN" when nothing matches (precision-first).
    """
    text = raw_response.strip()

    # Strip <think>...</think> tags (DeepSeek/Qwen reasoning wrappers)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    thinking = ""
    answer = ""
    confidence = "medium"

    # ------------------------------------------------------------------
    # Priority 1: \boxed{X}
    # ------------------------------------------------------------------
    m = re.search(r'\\boxed\{([A-F](?:\s*,\s*[A-F])*)\}', text, re.IGNORECASE)
    if m and not _is_negated(text, m.start()):
        answer = _extract_letters(m.group(1))
        thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 2: "correct answer is X" / "best answer is X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\b(?:correct|best|right)\s+answer\s+is\s+([A-F](?:\s*[,&]\s*[A-F]|\s+and\s+[A-F])*)\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = _extract_letters(m.group(1))
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 3: "final answer is X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\bfinal\s+answer\s+is\s+([A-F](?:\s*[,&]\s*[A-F]|\s+and\s+[A-F])*)\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = _extract_letters(m.group(1))
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 3b: "Answer: X" or "Ans: X" anywhere
    #   Use LAST match to handle self-corrections like
    #   "Answer: B. Actually, Answer: C" → C
    # ------------------------------------------------------------------
    if not answer:
        all_matches = list(re.finditer(
            r'\b(?:answer|ans)\s*[:\-]\s*([A-F](?:\s*[,&]\s*[A-F]|\s+and\s+[A-F])*)\b',
            text, re.IGNORECASE,
        ))
        # Walk backwards to find the last non-negated match
        for m in reversed(all_matches):
            if not _is_negated(text, m.start()):
                answer = _extract_letters(m.group(1))
                thinking = text[:m.start()].strip()
                break

    # ------------------------------------------------------------------
    # Priority 4a: "therefore X" / "thus X" / "hence X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\b(?:therefore|thus|hence|so)\s*,?\s+(?:the\s+answer\s+is\s+)?([A-F])\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 4b: "select X" / "choose X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\b(?:select|choose|pick)\s+([A-F])\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 4c: "I would choose X" / "I will select X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\bI\s+(?:would|will|shall)\s+(?:choose|select|go\s+with|pick)\s+([A-F])\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 4d: "it is X"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\bit\s+is\s+([A-F])\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 4e: "X is correct" / "X is the correct answer"
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(
            r'\b([A-F])\s+is\s+(?:the\s+)?(?:correct|right|best)\b',
            text, re.IGNORECASE,
        )
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 5: Last non-empty line is standalone letter(s)
    # ------------------------------------------------------------------
    if not answer:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            last = _standalone_letter_on_line(lines[-1])
            if last:
                answer = last
                thinking = "\n".join(lines[:-1]).strip()

    # ------------------------------------------------------------------
    # Priority 5b: Bold markdown **X** — extract letter from bold
    # ------------------------------------------------------------------
    if not answer:
        m = re.search(r'\*\*([A-F])\*\*', text, re.IGNORECASE)
        if m and not _is_negated(text, m.start()):
            answer = m.group(1).upper()
            thinking = text[:m.start()].strip()

    # ------------------------------------------------------------------
    # Priority 6: First standalone letter in the response
    #   (only if response is short — ≤3 lines — to avoid false positives)
    # ------------------------------------------------------------------
    if not answer:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 3 and lines:
            first = _standalone_letter_on_line(lines[0])
            if first:
                answer = first
                thinking = "\n".join(lines[1:]).strip()

    # ------------------------------------------------------------------
    # Precision-first: UNKNOWN instead of guessing
    # ------------------------------------------------------------------
    if not answer:
        answer = "UNKNOWN"

    # ------------------------------------------------------------------
    # Confidence extraction
    # ------------------------------------------------------------------
    if re.search(r'\b(?:high|highly)\s+confiden', text, re.IGNORECASE):
        confidence = "high"
    elif re.search(r'\b(?:low|uncertain|unsure|not\s+sure)', text, re.IGNORECASE):
        confidence = "low"

    if not thinking and answer not in ("UNKNOWN",):
        thinking = text

    return answer, thinking, confidence


# ═══════════════════════════════════════════════════════════════════════════
# Normalization & matching (unchanged API)
# ═══════════════════════════════════════════════════════════════════════════

def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    "A" -> "A"  |  "A,B,C" -> "A,B,C"  |  "a, b" -> "A,B"
    """
    answer = answer.strip().upper()

    if answer in ("ALL", "ALL OF THE ABOVE"):
        return "ALL"
    if answer in ("NONE", "NONE OF THE ABOVE"):
        return "NONE"

    letters = re.findall(r'\b([A-F])\b', answer)
    if letters:
        return ",".join(sorted(set(letters)))

    return answer


def answers_match(predicted: str, expected: str) -> tuple[bool, bool]:
    """
    Compare predicted and expected answers.

    Returns:
        (exact_match, partial_match)
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    exact_match = pred_norm == exp_norm

    partial_match = False
    if not exact_match and "," in exp_norm:
        pred_set = set(pred_norm.split(","))
        exp_set = set(exp_norm.split(","))
        if pred_set & exp_set:
            partial_match = True

    return exact_match, partial_match
