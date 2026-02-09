#!/usr/bin/env python3
"""
Convert evaluation XLSX files into strict benchmark JSONL datasets.

This script avoids optional XLSX dependencies (openpyxl) by parsing the
underlying XLSX XML directly, so it can run in constrained environments.

Outputs:
- mcq_holdout.jsonl
- open_ended_holdout.jsonl
- conversion_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class ConversionStats:
    total_rows: int = 0
    kept_rows: int = 0
    skipped_rows: int = 0


def _col_to_index(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    val = 0
    for ch in col:
        val = val * 26 + (ord(ch.upper()) - ord("A") + 1)
    return val - 1


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def _normalize_header(value: str) -> str:
    value = _normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _parse_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared: List[str] = []
    for si in root.findall("a:si", NS):
        parts: List[str] = []
        for t in si.findall(".//a:t", NS):
            parts.append(t.text or "")
        shared.append("".join(parts))
    return shared


def _cell_value(cell: ET.Element, shared: List[str]) -> str:
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        is_node = cell.find("a:is", NS)
        if is_node is None:
            return ""
        parts = [(node.text or "") for node in is_node.findall(".//a:t", NS)]
        return "".join(parts)

    v_node = cell.find("a:v", NS)
    if v_node is None or v_node.text is None:
        return ""

    if cell_type == "s":
        index = int(v_node.text)
        if 0 <= index < len(shared):
            return shared[index]
        return ""

    return v_node.text


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    """
    Read first worksheet from XLSX and return list of row dicts keyed by header.
    """
    with zipfile.ZipFile(path) as zf:
        shared = _parse_shared_strings(zf)
        sheet_xml = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    raw_rows: List[List[str]] = []
    for row in sheet_xml.findall(".//a:sheetData/a:row", NS):
        values: Dict[int, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.get("r", "")
            col_idx = _col_to_index(ref)
            values[col_idx] = _cell_value(cell, shared)

        if not values:
            continue
        max_idx = max(values.keys())
        ordered = [""] * (max_idx + 1)
        for idx, val in values.items():
            ordered[idx] = _normalize_text(val)
        raw_rows.append(ordered)

    if not raw_rows:
        return []

    headers = [_normalize_header(x) for x in raw_rows[0]]
    records: List[Dict[str, str]] = []
    for values in raw_rows[1:]:
        row: Dict[str, str] = {}
        for idx, header in enumerate(headers):
            if not header:
                continue
            row[header] = values[idx] if idx < len(values) else ""
        if any(v for v in row.values()):
            records.append(row)
    return records


def _get(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and _normalize_text(row[key]):
            return _normalize_text(row[key])
    return ""


def _extract_answer_key(raw: str) -> str:
    text = _normalize_text(raw).upper()
    if not text:
        return ""
    letters = re.findall(r"\b([A-F])\b", text)
    if letters:
        unique = sorted(set(letters))
        return ",".join(unique)
    if text in {"ALL OF THE ABOVE", "ALL"}:
        return "ALL"
    return text


def _build_question(case_context: str, question: str) -> str:
    case_context = _normalize_text(case_context)
    question = _normalize_text(question)
    if case_context:
        return f"Case context:\n{case_context}\n\nQuestion:\n{question}"
    return question


def convert_mcq(rows: List[Dict[str, str]], split: str) -> tuple[List[Dict[str, object]], ConversionStats, List[Dict[str, str]]]:
    option_cols = ["option_a", "option_b", "option_c", "option_d", "option_e", "option_f"]
    converted: List[Dict[str, object]] = []
    stats = ConversionStats(total_rows=len(rows))
    skipped: List[Dict[str, str]] = []

    for row in rows:
        options = [_normalize_text(row.get(col, "")) for col in option_cols]
        options = [x for x in options if x]
        answer_key = _extract_answer_key(_get(row, "correct_answer", "answer_key"))
        question = _build_question(
            _get(row, "case_context", "case"),
            _get(row, "question"),
        )

        if not question or len(options) < 2 or not answer_key:
            stats.skipped_rows += 1
            raw_id = _get(row, "id") or str(stats.kept_rows + 1)
            skipped.append(
                {
                    "id": raw_id,
                    "reason": "missing_required_fields",
                    "has_question": str(bool(question)),
                    "options_count": str(len(options)),
                    "has_answer_key": str(bool(answer_key)),
                }
            )
            continue

        raw_id = _get(row, "id") or str(stats.kept_rows + 1)
        source = _get(row, "source")
        difficulty = _get(row, "difficulty")
        explanation = _get(row, "explanation")

        converted.append(
            {
                "sample_id": f"mcq_{raw_id}",
                "track": "mcq",
                "split": split,
                "question": question,
                "options": options,
                "answer_key": answer_key,
                "domain": "tpn",
                "proficiency": difficulty.lower() if difficulty else None,
                "source_doc": source or None,
                "metadata": {
                    "difficulty": difficulty,
                    "case_context": _get(row, "case_context", "case"),
                    "explanation": explanation,
                    "source": source,
                    "raw_id": raw_id,
                },
            }
        )
        stats.kept_rows += 1

    return converted, stats, skipped


def convert_open_ended(rows: List[Dict[str, str]], split: str) -> tuple[List[Dict[str, object]], ConversionStats, List[Dict[str, str]]]:
    converted: List[Dict[str, object]] = []
    stats = ConversionStats(total_rows=len(rows))
    skipped: List[Dict[str, str]] = []

    for row in rows:
        question = _build_question(
            _get(row, "case_context", "case"),
            _get(row, "question"),
        )
        expected = _get(row, "expected_answer", "reference_answer", "answer")

        if not question or not expected:
            stats.skipped_rows += 1
            raw_id = _get(row, "id") or str(stats.kept_rows + 1)
            skipped.append(
                {
                    "id": raw_id,
                    "reason": "missing_required_fields",
                    "has_question": str(bool(question)),
                    "has_expected_answer": str(bool(expected)),
                    "question_preview": (question or "")[:140],
                }
            )
            continue

        raw_id = _get(row, "id") or str(stats.kept_rows + 1)
        source = _get(row, "source")
        difficulty = _get(row, "difficulty")

        converted.append(
            {
                "sample_id": f"open_{raw_id}",
                "track": "open_ended",
                "split": split,
                "question": question,
                "reference_answer": expected,
                "domain": "tpn",
                "proficiency": difficulty.lower() if difficulty else None,
                "source_doc": source or None,
                "metadata": {
                    "difficulty": difficulty,
                    "case_context": _get(row, "case_context", "case"),
                    "source": source,
                    "raw_id": raw_id,
                },
            }
        )
        stats.kept_rows += 1

    return converted, stats, skipped


def write_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert eval XLSX files to strict benchmark JSONL")
    parser.add_argument(
        "--mcq-xlsx",
        type=Path,
        default=Path("data/eval/MCQ_Evaluation_Set_Final.xlsx"),
        help="Path to MCQ evaluation workbook",
    )
    parser.add_argument(
        "--open-xlsx",
        type=Path,
        default=Path("data/eval/QandA_Evaluation_Set.xlsx"),
        help="Path to open-ended evaluation workbook",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/data/benchmark_2026-02-05"),
        help="Output directory for converted datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="holdout",
        choices=["train", "valid", "test", "holdout"],
        help="Dataset split label to assign to converted rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.mcq_xlsx.exists():
        raise FileNotFoundError(f"MCQ workbook not found: {args.mcq_xlsx}")
    if not args.open_xlsx.exists():
        raise FileNotFoundError(f"Open-ended workbook not found: {args.open_xlsx}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mcq_rows = read_xlsx_rows(args.mcq_xlsx)
    open_rows = read_xlsx_rows(args.open_xlsx)

    mcq_records, mcq_stats, mcq_skipped = convert_mcq(mcq_rows, split=args.split)
    open_records, open_stats, open_skipped = convert_open_ended(open_rows, split=args.split)

    mcq_path = args.out_dir / "mcq_holdout.jsonl"
    open_path = args.out_dir / "open_ended_holdout.jsonl"
    manifest_path = args.out_dir / "conversion_manifest.json"

    write_jsonl(mcq_records, mcq_path)
    write_jsonl(open_records, open_path)

    manifest = {
        "mcq_xlsx": str(args.mcq_xlsx),
        "open_xlsx": str(args.open_xlsx),
        "split": args.split,
        "sample_id_scheme": {
            "mcq": "mcq_<id>",
            "open_ended": "open_<id>",
        },
        "outputs": {
            "mcq": str(mcq_path),
            "open_ended": str(open_path),
        },
        "hashes": {
            "mcq_xlsx_sha256": _sha256_file(args.mcq_xlsx),
            "open_xlsx_sha256": _sha256_file(args.open_xlsx),
            "mcq_jsonl_sha256": _sha256_file(mcq_path),
            "open_jsonl_sha256": _sha256_file(open_path),
        },
        "stats": {
            "mcq": {
                "input_rows": mcq_stats.total_rows,
                "kept_rows": mcq_stats.kept_rows,
                "skipped_rows": mcq_stats.skipped_rows,
            },
            "open_ended": {
                "input_rows": open_stats.total_rows,
                "kept_rows": open_stats.kept_rows,
                "skipped_rows": open_stats.skipped_rows,
            },
        },
        "skipped": {
            "mcq": mcq_skipped,
            "open_ended": open_skipped,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Conversion complete:")
    print(f"  MCQ: {mcq_path} ({mcq_stats.kept_rows}/{mcq_stats.total_rows} rows)")
    print(f"  Open-ended: {open_path} ({open_stats.kept_rows}/{open_stats.total_rows} rows)")
    print(f"  Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
