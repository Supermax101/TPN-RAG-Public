"""
Deterministic calculation + citation metrics for open-ended clinical TPN Q&A.

Goal: quantify numeric/unit correctness without an LLM judge.
This complements DeepEval GEval/Faithfulness by providing stable scoring
for calculations (GIR, mEq/kg/day, mL/hr, etc.).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_VALUE_RE = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

# Matches single quantities like "4.175 mg/kg/min" or "0.2 mL/hr" or "42 µg/L".
_QUANTITY_RE = re.compile(
    rf"(?P<value>{_VALUE_RE})\s*(?P<unit>mcg|µg|ug|mg|g|iu|mmol|mEq|meq|kcal|mOsm|mosm|mL|ml|L|l|dL|dl|%)"
    r"(?:\s*/\s*(?P<per1>kg|day|d|hr|h|min|ml|m|L|l|dL|dl))?"
    r"(?:\s*/\s*(?P<per2>kg|day|d|hr|h|min|ml|m|L|l|dL|dl))?",
    re.IGNORECASE,
)

# Matches ranges like "0.35 to 0.5 mg/kg/day" or "100–120 mL/kg/day".
_RANGE_RE = re.compile(
    rf"(?P<v1>{_VALUE_RE})\s*(?:to|–|-)\s*(?P<v2>{_VALUE_RE})\s*"
    r"(?P<unit>mcg|µg|ug|mg|g|iu|mmol|mEq|meq|kcal|mOsm|mosm|mL|ml|L|l|dL|dl|%)"
    r"(?:\s*/\s*(?P<per1>kg|day|d|hr|h|min|ml|m|L|l|dL|dl))?"
    r"(?:\s*/\s*(?P<per2>kg|day|d|hr|h|min|ml|m|L|l|dL|dl))?",
    re.IGNORECASE,
)

_PER_ORDER = {"kg": 0, "day": 1, "hr": 2, "min": 3, "ml": 4, "l": 5, "dl": 6}

_FINAL_ANSWER_RE = re.compile(
    r"(?is)final\s*answer\s*:\s*(?P<final>.+?)(?:\n\s*(?:work|citations|sources)\s*:|$)"
)


def _normalize_unit(unit: str) -> str:
    u = (unit or "").strip()
    if not u:
        return ""
    u = u.replace("µ", "u").lower()
    if u == "ug":
        return "mcg"
    if u == "meq":
        return "meq"
    if u == "mosm":
        return "mosm"
    if u == "ml":
        return "ml"
    if u == "l":
        return "l"
    if u == "dl":
        return "dl"
    return u


def _normalize_per(per: Optional[str]) -> str:
    p = (per or "").strip().lower()
    if not p:
        return ""
    if p == "h":
        return "hr"
    if p == "d":
        return "day"
    if p == "m":
        return "min"
    if p == "ml":
        return "ml"
    if p == "l":
        return "l"
    if p == "dl":
        return "dl"
    return p


def _parse_value(value: str) -> float:
    return float((value or "0").replace(",", "").strip())


def _canonical_per_units(per_units: Sequence[str]) -> Tuple[str, ...]:
    items = [p for p in per_units if p]
    return tuple(sorted(items, key=lambda x: (_PER_ORDER.get(x, 99), x)))


def _unit_family(unit: str) -> str:
    u = _normalize_unit(unit)
    if u in {"mcg", "mg", "g"}:
        return "mass_mg"
    if u in {"ml", "l", "dl"}:
        return "vol_ml"
    return u


def _to_base(value: float, unit: str) -> Tuple[float, str]:
    """
    Convert to a canonical base for safe comparisons within a unit family.

    Base units:
      - mass_mg: mg
      - vol_ml: ml
      - others: identity
    """
    u = _normalize_unit(unit)
    family = _unit_family(u)
    if family == "mass_mg":
        if u == "g":
            return value * 1000.0, "mg"
        if u == "mcg":
            return value / 1000.0, "mg"
        return value, "mg"
    if family == "vol_ml":
        if u == "l":
            return value * 1000.0, "ml"
        if u == "dl":
            return value * 100.0, "ml"
        return value, "ml"
    return value, u


@dataclass(frozen=True)
class Quantity:
    raw_value: float
    unit: str
    per_units: Tuple[str, ...]
    value_base: float
    unit_base: str
    family: str

    @property
    def key(self) -> Tuple[str, Tuple[str, ...]]:
        return (self.family, self.per_units)


def extract_quantities(text: str) -> List[Quantity]:
    """
    Extract numeric quantities with units from text.

    Focuses on quantities like:
      - 4.175 mg/kg/min
      - 0.2 mL/hr
      - 42 µg/L
      - 100–120 mL/kg/day (range -> two quantities)
    """
    text = text or ""
    quantities: List[Quantity] = []

    # 1) Ranges first, track spans to avoid double-counting.
    used_spans: List[Tuple[int, int]] = []
    for m in _RANGE_RE.finditer(text):
        unit = _normalize_unit(m.group("unit"))
        per1 = _normalize_per(m.group("per1"))
        per2 = _normalize_per(m.group("per2"))
        per_units = _canonical_per_units([per1, per2])
        for raw in (m.group("v1"), m.group("v2")):
            v = _parse_value(raw)
            base_v, base_u = _to_base(v, unit)
            quantities.append(
                Quantity(
                    raw_value=v,
                    unit=unit,
                    per_units=per_units,
                    value_base=base_v,
                    unit_base=base_u,
                    family=_unit_family(unit),
                )
            )
        used_spans.append(m.span())

    def _overlaps_used(span: Tuple[int, int]) -> bool:
        s0, s1 = span
        for u0, u1 in used_spans:
            if s0 < u1 and u0 < s1:
                return True
        return False

    # 2) Singles (skip anything inside a range match span).
    for m in _QUANTITY_RE.finditer(text):
        if _overlaps_used(m.span()):
            continue
        unit = _normalize_unit(m.group("unit"))
        per1 = _normalize_per(m.group("per1"))
        per2 = _normalize_per(m.group("per2"))
        per_units = _canonical_per_units([per1, per2])
        v = _parse_value(m.group("value"))
        base_v, base_u = _to_base(v, unit)
        quantities.append(
            Quantity(
                raw_value=v,
                unit=unit,
                per_units=per_units,
                value_base=base_v,
                unit_base=base_u,
                family=_unit_family(unit),
            )
        )

    return quantities


@dataclass
class CalcMetricResult:
    quantity_recall: float
    quantity_precision: float
    quantity_f1: float
    key_recall: float
    key_precision: float
    key_f1: float
    unit_mismatch_count: int
    expected_quantity_count: int
    output_quantity_count: int
    matched_quantity_count: int
    expected_key_count: int
    output_key_count: int
    matched_key_count: int
    rel_error_mean: Optional[float]
    rel_error_p50: Optional[float]
    rel_error_p95: Optional[float]


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    if q <= 0:
        return min(xs)
    if q >= 1:
        return max(xs)
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def _within_tolerance(expected: float, actual: float, rel_tol: float, abs_floor: float) -> bool:
    tol = max(abs_floor, rel_tol * abs(expected))
    return abs(actual - expected) <= tol


def extract_final_answer_text(text: str) -> str:
    """
    Extract the portion of an open-ended response intended to be the final answer.

    This avoids unfairly penalizing CoT "Work:" intermediate numbers.
    If no "Final answer:" header is present, the full text is returned.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    m = _FINAL_ANSWER_RE.search(raw)
    if not m:
        return raw
    final = (m.group("final") or "").strip()
    return final or raw


@dataclass
class ReferenceTargetingResult:
    expected_quantity_count: int
    expected_key_count: int
    multi_value_key_count: int
    is_single_target: bool


def analyze_reference_targets(
    expected_answer: str,
    rel_tol: float = 0.02,
    abs_floor: float = 0.01,
) -> ReferenceTargetingResult:
    """
    Characterize whether a reference answer is "single-target" (one value per dimension key).

    For silver labels, multi-target references (e.g., unrounded + rounded values, stepwise titration)
    are harder to score fairly; we report metrics on all samples and on the single-target subset.
    """
    expected = extract_quantities(expected_answer)
    grouped: Dict[Tuple[str, Tuple[str, ...]], List[Quantity]] = {}
    for q in expected:
        grouped.setdefault(q.key, []).append(q)

    multi_value_keys = 0
    for qs in grouped.values():
        reps: List[float] = []
        for q in qs:
            v = float(q.value_base)
            if any(_within_tolerance(r, v, rel_tol=rel_tol, abs_floor=abs_floor) for r in reps):
                continue
            reps.append(v)
        if len(reps) > 1:
            multi_value_keys += 1

    return ReferenceTargetingResult(
        expected_quantity_count=len(expected),
        expected_key_count=len(grouped),
        multi_value_key_count=int(multi_value_keys),
        is_single_target=(multi_value_keys == 0),
    )


def evaluate_calc_metrics(
    expected_answer: str,
    output_answer: str,
    rel_tol: float = 0.02,
    abs_floor: float = 0.01,
) -> CalcMetricResult:
    expected = extract_quantities(expected_answer)
    output = extract_quantities(output_answer)

    expected_n = len(expected)
    output_n = len(output)

    if expected_n == 0 and output_n == 0:
        return CalcMetricResult(
            quantity_recall=1.0,
            quantity_precision=1.0,
            quantity_f1=1.0,
            key_recall=1.0,
            key_precision=1.0,
            key_f1=1.0,
            unit_mismatch_count=0,
            expected_quantity_count=0,
            output_quantity_count=0,
            matched_quantity_count=0,
            expected_key_count=0,
            output_key_count=0,
            matched_key_count=0,
            rel_error_mean=None,
            rel_error_p50=None,
            rel_error_p95=None,
        )

    # Greedy best-match per expected quantity (stable and deterministic).
    used_output = set()
    matched_expected = 0
    matched_output = set()
    rel_errors: List[float] = []

    for ei, e in enumerate(expected):
        best = None  # (error, output_idx)
        for oi, a in enumerate(output):
            if oi in used_output:
                continue
            if a.key != e.key:
                continue
            # Candidate: compute relative error in base units.
            denom = abs(e.value_base) if abs(e.value_base) > 1e-12 else 1.0
            err = abs(a.value_base - e.value_base) / denom
            if best is None or err < best[0]:
                best = (err, oi)
        if best is None:
            continue
        err, oi = best
        a = output[oi]
        if _within_tolerance(e.value_base, a.value_base, rel_tol=rel_tol, abs_floor=abs_floor):
            matched_expected += 1
            used_output.add(oi)
            matched_output.add(oi)
            rel_errors.append(err)

    precision = (len(matched_output) / output_n) if output_n else (1.0 if expected_n == 0 else 0.0)
    recall = (matched_expected / expected_n) if expected_n else (1.0 if output_n == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Unit mismatch: output quantities with a known family that exists in expected,
    # but that never matched any expected due to unit/per-unit mismatch.
    expected_families = {e.family for e in expected}
    expected_keys = {e.key for e in expected}
    unit_mismatch = 0
    for oi, a in enumerate(output):
        if oi in matched_output:
            continue
        if a.family not in expected_families:
            continue
        if a.key in expected_keys:
            # Same dimension key but value mismatch -> not a unit mismatch.
            continue
        unit_mismatch += 1

    stats = None
    if rel_errors:
        stats = {
            "mean": mean(rel_errors),
            "p50": _percentile(rel_errors, 0.50),
            "p95": _percentile(rel_errors, 0.95),
        }

    # Key-level scoring: treat each (unit-family, per-units) as a single target dimension.
    expected_by_key: Dict[Tuple[str, Tuple[str, ...]], List[Quantity]] = {}
    for e in expected:
        expected_by_key.setdefault(e.key, []).append(e)
    output_by_key: Dict[Tuple[str, Tuple[str, ...]], List[Quantity]] = {}
    for a in output:
        output_by_key.setdefault(a.key, []).append(a)

    matched_keys = 0
    for k, exp_qs in expected_by_key.items():
        out_qs = output_by_key.get(k) or []
        ok = False
        for a in out_qs:
            for e in exp_qs:
                if _within_tolerance(e.value_base, a.value_base, rel_tol=rel_tol, abs_floor=abs_floor):
                    ok = True
                    break
            if ok:
                break
        if ok:
            matched_keys += 1

    expected_key_count = len(expected_by_key)
    output_key_count = len(output_by_key)
    key_precision = (matched_keys / output_key_count) if output_key_count else (1.0 if expected_key_count == 0 else 0.0)
    key_recall = (matched_keys / expected_key_count) if expected_key_count else (1.0 if output_key_count == 0 else 0.0)
    key_f1 = (2 * key_precision * key_recall / (key_precision + key_recall)) if (key_precision + key_recall) > 0 else 0.0

    return CalcMetricResult(
        quantity_recall=round(recall, 4),
        quantity_precision=round(precision, 4),
        quantity_f1=round(f1, 4),
        key_recall=round(key_recall, 4),
        key_precision=round(key_precision, 4),
        key_f1=round(key_f1, 4),
        unit_mismatch_count=int(unit_mismatch),
        expected_quantity_count=int(expected_n),
        output_quantity_count=int(output_n),
        matched_quantity_count=int(matched_expected),
        expected_key_count=int(expected_key_count),
        output_key_count=int(output_key_count),
        matched_key_count=int(matched_keys),
        rel_error_mean=round(stats["mean"], 6) if stats else None,
        rel_error_p50=round(stats["p50"], 6) if stats else None,
        rel_error_p95=round(stats["p95"], 6) if stats else None,
    )


# ---------------------------------------------------------------------------
# Doc-level citation metrics
# ---------------------------------------------------------------------------


_BRACKET_CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")


def normalize_doc_name(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"\.(md|json|pdf|txt)$", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_doc_citations(text: str) -> List[str]:
    citations: List[str] = []
    seen = set()
    for m in _BRACKET_CITATION_RE.finditer(text or ""):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        # Support "[Doc A; Doc B]" / "[Doc A, Doc B]".
        parts = re.split(r"[;,]\s*", raw)
        for part in parts:
            c = part.strip()
            if not c:
                continue
            if not re.search(r"[A-Za-z]", c):
                continue
            key = normalize_doc_name(c)
            if key and key not in seen:
                seen.add(key)
                citations.append(c)
    return citations


@dataclass
class CitationMetricResult:
    citation_present: bool
    cited_doc_count: int
    cites_gold_source_doc: Optional[bool]
    cited_doc_in_retrieved_context: bool


def evaluate_doc_citations(
    output_answer: str,
    gold_source_doc: Optional[str],
    retrieved_sources: Optional[Iterable[str]] = None,
) -> CitationMetricResult:
    cited_docs = extract_doc_citations(output_answer)
    cited_norm = {normalize_doc_name(x) for x in cited_docs}
    retrieved_norm = {normalize_doc_name(x) for x in (retrieved_sources or []) if x}

    gold_norm = normalize_doc_name(gold_source_doc or "")
    cites_gold = None
    if gold_norm:
        cites_gold = gold_norm in cited_norm

    cited_in_context = bool(cited_norm & retrieved_norm) if retrieved_norm else False

    return CitationMetricResult(
        citation_present=bool(cited_docs),
        cited_doc_count=len(cited_docs),
        cites_gold_source_doc=cites_gold,
        cited_doc_in_retrieved_context=cited_in_context,
    )
