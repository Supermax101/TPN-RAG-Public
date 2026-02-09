#!/usr/bin/env python3
"""
Convert the Takeoff41 nutrition QA JSONL into the strict benchmark JSONL schema.

Input (committed copy):
  - data/eval/takeoff41_200_tpn_nutrition_qa.jsonl

Outputs (canonical for paper runs):
  - eval/data/benchmark_2026-02-05/takeoff41_200_holdout.jsonl
  - eval/data/benchmark_2026-02-05/takeoff41_200_conversion_manifest.json

Record mapping:
  - sample_id: takeoff41_{id:03d}
  - track: open_ended
  - split: holdout (default)
  - question: original question
  - reference_answer: original answer
  - domain: "tpn"
  - source_doc: null (do not attempt source-file mapping yet)
  - metadata: question_type, book_title, source_file, chunk_index, relevance_score, raw_id
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e}") from e
    return rows


def _coerce_int(value: object, *, field: str, sample_hint: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Invalid int for '{field}' ({sample_hint}): {value!r}") from e


def _build_record(row: dict, split: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_id = _coerce_int(row.get("id"), field="id", sample_hint="takeoff41 row")
    question = _normalize_text(row.get("question"))
    answer = _normalize_text(row.get("answer"))
    if not question:
        raise ValueError(f"Row id={raw_id} missing 'question'")
    if not answer:
        raise ValueError(f"Row id={raw_id} missing 'answer'")

    md = {
        "question_type": _normalize_text(row.get("question_type")) or None,
        "book_title": _normalize_text(row.get("book_title")) or None,
        "source_file": _normalize_text(row.get("source_file")) or None,
        "chunk_index": row.get("chunk_index"),
        "relevance_score": row.get("relevance_score"),
        "raw_id": raw_id,
    }

    record: Dict[str, Any] = {
        "sample_id": f"takeoff41_{raw_id:03d}",
        "track": "open_ended",
        "split": split,
        "question": question,
        "reference_answer": answer,
        "domain": "tpn",
        "proficiency": None,
        "source_doc": None,
        "page": None,
        "metadata": md,
    }
    return record, md


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Takeoff41 nutrition QA JSONL to strict benchmark JSONL")
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("data/eval/takeoff41_200_tpn_nutrition_qa.jsonl"),
        help="Input Takeoff41 JSONL path",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/data/benchmark_2026-02-05"),
        help="Output directory",
    )
    p.add_argument(
        "--split",
        type=str,
        default="holdout",
        choices=["train", "valid", "test", "holdout"],
        help="Split label to assign to converted rows",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = args.in_path
    if not in_path.is_absolute():
        in_path = (PROJECT_ROOT / in_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_jsonl = out_dir / "takeoff41_200_holdout.jsonl"
    out_manifest = out_dir / "takeoff41_200_conversion_manifest.json"

    rows = _load_jsonl(in_path)
    if not rows:
        raise SystemExit(f"No rows found in {in_path}")

    # Determinism: stable ordering by raw id.
    ids = [_coerce_int(r.get("id"), field="id", sample_hint="takeoff41 row") for r in rows]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise SystemExit(f"Duplicate ids in input: {dupes[:10]}")

    by_id = {int(r["id"]): r for r in rows}
    records: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    for raw_id in sorted(by_id.keys()):
        rec, md = _build_record(by_id[raw_id], split=args.split)
        records.append(rec)
        meta_rows.append(md)

    write_jsonl(records, out_jsonl)

    # Build a deterministic, conversion-provenance manifest.
    question_type_counts = Counter([m.get("question_type") or "" for m in meta_rows])
    book_title_counts = Counter([m.get("book_title") or "" for m in meta_rows])
    source_file_counts = Counter([m.get("source_file") or "" for m in meta_rows])

    manifest = {
        "input": {
            "path": str(in_path),
            "sha256": _sha256_file(in_path),
            "row_count": len(rows),
        },
        "output": {
            "jsonl_path": str(out_jsonl),
            "jsonl_sha256": _sha256_file(out_jsonl),
            "split": args.split,
            "track": "open_ended",
            "domain": "tpn",
        },
        "sample_id_scheme": "takeoff41_{id:03d}",
        "distributions": {
            "question_type": dict(sorted(question_type_counts.items(), key=lambda kv: kv[0])),
            "book_title": dict(sorted(book_title_counts.items(), key=lambda kv: kv[0])),
            "source_file": dict(sorted(source_file_counts.items(), key=lambda kv: kv[0])),
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Conversion complete:")
    print(f"  Input:   {in_path} ({len(rows)} rows)")
    print(f"  Output:  {out_jsonl} ({len(records)} rows)")
    print(f"  Manifest:{out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

