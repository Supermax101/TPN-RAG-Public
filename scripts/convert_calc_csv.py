#!/usr/bin/env python3
"""
Convert the KB-generated TPN calculation Q&A CSV into strict benchmark JSONL datasets.

Inputs:
  - data/eval/TPN_Calculation_QA_200.csv

Outputs:
  - eval/data/calc_200_holdout.jsonl
  - eval/data/calc_50_holdout.jsonl
  - eval/data/calc_50_manifest.json
  - eval/data/calc_conversion_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ConversionStats:
    total_rows: int = 0
    kept_rows: int = 0
    skipped_rows: int = 0


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


_CALC_INTENT_RE = re.compile(
    r"\b("
    r"calculate|infusion\s*rate|gir|dose|meq|mmol|mosm|m\s*l\s*/\s*hr|"
    r"g\s*/\s*kg\s*/\s*day|mg\s*/\s*kg\s*/\s*min"
    r")\b",
    re.IGNORECASE,
)

# Number+unit tokens only (avoid plain integers like "day 3").
_QUANTITY_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mcg|µg|ug|mg|g|iu|mmol|meq|mEq|kcal|mosm|mOsm|ml|mL|l|L|dl|dL|%)"
    r"(?:\s*/\s*(?P<per1>kg|day|d|hr|h|min|ml|m|l|L|dl|dL))?"
    r"(?:\s*/\s*(?P<per2>kg|day|d|hr|h|min|ml|m|l|L|dl|dL))?",
    re.IGNORECASE,
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
    return p


def _extract_quantities(text: str) -> List[Tuple[float, str, Tuple[str, ...]]]:
    """
    Extract (value, unit, per_units) tuples. Used only for selection ranking.
    """
    items: List[Tuple[float, str, Tuple[str, ...]]] = []
    for m in _QUANTITY_RE.finditer(text or ""):
        value = _safe_float(m.group("value"))
        unit = _normalize_unit(m.group("unit"))
        per1 = _normalize_per(m.group("per1"))
        per2 = _normalize_per(m.group("per2"))
        per_units = tuple([p for p in (per1, per2) if p])
        if unit:
            items.append((value, unit, per_units))
    return items


def _numeric_density(question: str, answer: str) -> int:
    # Count unique quantity signatures to avoid overcounting repeated mentions.
    sigs = set(_extract_quantities(question or "") + _extract_quantities(answer or ""))
    return len(sigs)


def _answer_has_quantity(answer: str) -> bool:
    return bool(_QUANTITY_RE.search(answer or ""))


def read_calc_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({str(k): _normalize_text(v) for k, v in row.items()})
        return rows


def convert_calc_rows(rows: Iterable[Dict[str, str]], split: str = "holdout") -> Tuple[List[Dict[str, Any]], ConversionStats]:
    converted: List[Dict[str, Any]] = []
    stats = ConversionStats()

    for row in rows:
        stats.total_rows += 1
        sno = _safe_int(row.get("S.No", "0"), default=0)
        question = _normalize_text(row.get("Question", ""))
        answer = _normalize_text(row.get("Answer", ""))
        provider = _normalize_text(row.get("Provider", ""))
        complexity = _normalize_text(row.get("Complexity", ""))
        source_doc = _normalize_text(row.get("Source Document", ""))
        verification_conf = _safe_float(row.get("Verification Confidence", "0.0"), default=0.0)

        if not sno or not question or not answer:
            stats.skipped_rows += 1
            continue

        converted.append(
            {
                "sample_id": f"calc_{sno:03d}",
                "track": "open_ended",
                "split": split,
                "question": question,
                "reference_answer": answer,
                "domain": "tpn",
                "proficiency": None,
                "source_doc": source_doc or None,
                "page": None,
                "metadata": {
                    "raw_sno": sno,
                    "provider": provider,
                    "complexity": complexity,
                    "verification_confidence": verification_conf,
                    "source_doc": source_doc,
                },
            }
        )
        stats.kept_rows += 1

    return converted, stats


def select_calc_50(
    records: List[Dict[str, Any]],
    *,
    mode: str = "max_calcness",
    total_n: int = 50,
    high_n: int = 25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    def complexity_weight(value: str) -> int:
        v = (value or "").strip().lower()
        if v == "advanced":
            return 3
        if v == "intermediate":
            return 2
        if v == "basic":
            return 1
        return 0

    mode_norm = (mode or "").strip().lower()
    if mode_norm not in {"max_calcness", "stratified"}:
        raise ValueError(f"Unknown calc50 mode: {mode}. Use max_calcness or stratified.")

    total_n = int(total_n)
    high_n = int(high_n)
    if total_n <= 0:
        raise ValueError("--calc50-total must be >= 1")
    if high_n < 0:
        raise ValueError("--calc50-high must be >= 0")
    if high_n > total_n:
        high_n = total_n

    scored = []
    for rec in records:
        q = str(rec.get("question") or "")
        a = str(rec.get("reference_answer") or "")
        md = dict(rec.get("metadata") or {})
        if not _CALC_INTENT_RE.search(q):
            continue
        if not _answer_has_quantity(a):
            continue

        comp = str(md.get("complexity") or "")
        vc = float(md.get("verification_confidence") or 0.0)
        nd = _numeric_density(q, a)
        scored.append((complexity_weight(comp), vc, nd, str(rec.get("sample_id") or ""), rec))

    scored_sorted = sorted(scored, key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)

    selected_scored = []
    if mode_norm == "max_calcness" or len(scored_sorted) <= total_n:
        selected_scored = scored_sorted[:total_n]
    else:
        # Stratified: combine 1) max-calc-ness and 2) "messier" multi-step items.
        high = scored_sorted[:high_n]
        remaining = scored_sorted[high_n:]

        messy_word_re = re.compile(
            r"\b(max(?:imum)?|min(?:imum)?|initial|advance|titrate|target|limit|osm(?:ol|olar)?)\b",
            re.IGNORECASE,
        )
        range_hint_re = re.compile(r"\d+\s*(?:to|–|-)\s*\d+", re.IGNORECASE)

        def messy_score(rec: Dict[str, Any]) -> Tuple[int, int, int]:
            q = str(rec.get("question") or "")
            a = str(rec.get("reference_answer") or "")
            md = dict(rec.get("metadata") or {})
            comp_w = complexity_weight(str(md.get("complexity") or ""))
            nd = _numeric_density(q, a)

            # Dimensional variety and multi-target keys are good proxies for multi-step/constraints.
            qs = _extract_quantities(q)
            ans = _extract_quantities(a)
            keys = [(u, per) for (_, u, per) in (qs + ans)]
            key_set = set(keys)
            key_count = len(key_set)

            values_by_key: Dict[Tuple[str, Tuple[str, ...]], set] = {}
            for v, u, per in ans:
                values_by_key.setdefault((u, per), set()).add(round(float(v), 6))
            multi_value_key_count = sum(1 for vs in values_by_key.values() if len(vs) > 1)

            word_hits = len(messy_word_re.findall(q + " " + a))
            range_hits = len(range_hint_re.findall(a))

            # Weighted integer score (deterministic, no floats).
            score = comp_w * 10 + min(key_count, 6) * 2 + min(multi_value_key_count, 4) * 3 + min(range_hits, 3) * 3 + min(word_hits, 6) + min(nd, 12)
            return (score, key_count, multi_value_key_count)

        messy_ranked = []
        for t in remaining:
            rec = t[-1]
            ms, key_count, mvk = messy_score(rec)
            messy_ranked.append((ms, key_count, mvk, t[0], t[1], t[2], t[3], rec))

        # Sort by messy score, then preserve calc-ness features, then sample_id tie-break.
        messy_sorted = sorted(messy_ranked, key=lambda x: (x[0], x[3], x[4], x[5], x[6]), reverse=True)

        need = total_n - len(high)
        selected_scored = high + [(x[3], x[4], x[5], x[6], x[7]) for x in messy_sorted[:need]]

    selected = [t[-1] for t in selected_scored]

    manifest = {
        "selection": {
            "mode": mode_norm,
            "total_n": total_n,
            "high_n": high_n if mode_norm == "stratified" else None,
            "filter": {
                "question_regex": _CALC_INTENT_RE.pattern,
                "require_answer_quantity_regex": _QUANTITY_RE.pattern,
            },
            "ranking": [
                "complexity_weight (advanced>intermediate>basic)",
                "verification_confidence (desc)",
                "numeric_density (# unique quantities in question+answer, desc)",
                "sample_id (desc tie-breaker for determinism)",
            ],
        },
        "counts": {
            "input_records": len(records),
            "eligible_records": len(scored_sorted),
            "selected_records": len(selected),
        },
        "selected": [
            {
                "sample_id": t[3],
                "raw_sno": int((t[4].get("metadata") or {}).get("raw_sno") or 0),
                "complexity": str((t[4].get("metadata") or {}).get("complexity") or ""),
                "verification_confidence": float((t[4].get("metadata") or {}).get("verification_confidence") or 0.0),
                "numeric_density": int(t[2]),
                "source_doc": str(t[4].get("source_doc") or ""),
                "question_preview": (str(t[4].get("question") or "")[:200]),
            }
            for t in selected_scored
        ],
    }
    return selected, manifest


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert calc CSV into strict benchmark JSONL datasets")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data/eval/TPN_Calculation_QA_200.csv",
        help="Path to input calculation Q&A CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval/data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="holdout",
        choices=["train", "valid", "test", "holdout"],
        help="Dataset split label to assign to converted rows",
    )
    parser.add_argument("--no-calc50", action="store_true", help="Do not create calc_50 outputs")
    parser.add_argument(
        "--calc50-mode",
        type=str,
        default="max_calcness",
        choices=["max_calcness", "stratified"],
        help="How to select calc_50 from calc_200 (default: max_calcness).",
    )
    parser.add_argument(
        "--calc50-total",
        type=int,
        default=50,
        help="Total number of examples in calc_50 (default: 50).",
    )
    parser.add_argument(
        "--calc50-high",
        type=int,
        default=25,
        help="For stratified mode: number of max-calc-ness items (default: 25).",
    )
    parser.add_argument("--validate", action="store_true", help="Validate records against DatasetSchema")
    return parser.parse_args()


def _validate_records(records: List[Dict[str, Any]]) -> None:
    from app.evaluation.benchmark_types import DatasetSchema, DatasetTrack

    for rec in records:
        # Track is forced by load_dataset anyway; we validate schema shape here.
        sample = dict(rec)
        sample["track"] = DatasetTrack.OPEN_ENDED.value
        DatasetSchema.model_validate(sample)


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    rows = read_calc_csv(args.csv)
    records, stats = convert_calc_rows(rows, split=args.split)

    out_dir: Path = args.out_dir
    calc_200_path = out_dir / "calc_200_holdout.jsonl"
    write_jsonl(records, calc_200_path)

    selected = []
    manifest: Dict[str, Any] = {}
    calc_50_path = out_dir / "calc_50_holdout.jsonl"
    calc_50_manifest_path = out_dir / "calc_50_manifest.json"
    if not args.no_calc50:
        selected, manifest = select_calc_50(
            records,
            mode=args.calc50_mode,
            total_n=args.calc50_total,
            high_n=args.calc50_high,
        )
        write_jsonl(selected, calc_50_path)
        calc_50_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.validate:
        _validate_records(records)
        if selected:
            _validate_records(selected)

    conversion_manifest = {
        "input_csv": str(args.csv),
        "split": args.split,
        "outputs": {
            "calc_200": str(calc_200_path),
            "calc_50": str(calc_50_path) if selected else None,
            "calc_50_manifest": str(calc_50_manifest_path) if selected else None,
        },
        "stats": {
            "input_rows": stats.total_rows,
            "kept_rows": stats.kept_rows,
            "skipped_rows": stats.skipped_rows,
        },
    }
    (out_dir / "calc_conversion_manifest.json").write_text(
        json.dumps(conversion_manifest, indent=2), encoding="utf-8"
    )

    print("Conversion complete:")
    print(f"  calc_200: {calc_200_path} ({stats.kept_rows}/{stats.total_rows} rows)")
    if selected:
        print(f"  calc_50:  {calc_50_path} ({len(selected)}/{len(records)} eligible subset)")
        print(f"  manifest: {calc_50_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
