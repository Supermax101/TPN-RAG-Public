import csv
import json
from pathlib import Path

from scripts.convert_calc_csv import convert_calc_rows, read_calc_csv, select_calc_50


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "S.No",
        "Provider",
        "Question",
        "Answer",
        "Reasoning (Chain of Thought)",
        "Complexity",
        "Source Document",
        "Verification Confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_convert_calc_csv_to_strict_records(tmp_path):
    csv_path = tmp_path / "calc.csv"
    _write_csv(
        csv_path,
        [
            {
                "S.No": "1",
                "Provider": "google",
                "Question": "Calculate GIR for a 5 kg infant receiving 12.5% dextrose at 10 mL/hr.",
                "Answer": "GIR is 4.175 mg/kg/min.",
                "Reasoning (Chain of Thought)": "x",
                "Complexity": "advanced",
                "Source Document": "5PN Components",
                "Verification Confidence": "0.98",
            },
            {
                "S.No": "2",
                "Provider": "anthropic",
                "Question": "What is the hourly infusion rate for 4.4 mL over 24 hours?",
                "Answer": "0.183 mL/hr (rounded 0.2 mL/hr).",
                "Reasoning (Chain of Thought)": "x",
                "Complexity": "basic",
                "Source Document": "2022 NICU New Fellow PN",
                "Verification Confidence": "0.95",
            },
        ],
    )

    rows = read_calc_csv(csv_path)
    records, stats = convert_calc_rows(rows, split="holdout")
    assert stats.total_rows == 2
    assert stats.kept_rows == 2
    assert stats.skipped_rows == 0
    assert records[0]["track"] == "open_ended"
    assert records[0]["split"] == "holdout"
    assert records[0]["sample_id"] == "calc_001"
    assert records[0]["source_doc"] == "5PN Components"
    assert "metadata" in records[0]


def test_select_calc_50_filters_and_ranks(tmp_path):
    # Only 2 eligible rows -> selection returns both (<= 50).
    csv_path = tmp_path / "calc.csv"
    _write_csv(
        csv_path,
        [
            {
                "S.No": "10",
                "Provider": "openai",
                "Question": "Calculate dose: 2 g/kg/day for 2 kg.",
                "Answer": "Dose is 4 g/day.",
                "Reasoning (Chain of Thought)": "x",
                "Complexity": "advanced",
                "Source Document": "DocA",
                "Verification Confidence": "0.93",
            },
            {
                "S.No": "11",
                "Provider": "openai",
                "Question": "Calculate GIR for 1 kg at 3 mL/hr D10.",
                "Answer": "GIR is 5 mg/kg/min.",
                "Reasoning (Chain of Thought)": "x",
                "Complexity": "basic",
                "Source Document": "DocB",
                "Verification Confidence": "0.99",
            },
            {
                "S.No": "12",
                "Provider": "openai",
                "Question": "Non-calc question.",
                "Answer": "No numbers here.",
                "Reasoning (Chain of Thought)": "x",
                "Complexity": "advanced",
                "Source Document": "DocC",
                "Verification Confidence": "0.99",
            },
        ],
    )
    rows = read_calc_csv(csv_path)
    records, _ = convert_calc_rows(rows, split="holdout")
    selected, manifest = select_calc_50(records)
    assert len(selected) == 2
    assert manifest["counts"]["eligible_records"] == 2
    assert manifest["counts"]["selected_records"] == 2
    # Ensure manifest is JSON-serializable.
    json.dumps(manifest)

