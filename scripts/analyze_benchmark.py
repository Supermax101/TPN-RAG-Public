#!/usr/bin/env python3
"""
Analyze benchmark run ledger and produce paper-grade statistical report JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.evaluation import build_analysis_report


def latest_records_file(root: Path) -> Path:
    files = sorted(root.glob("run_records_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No run_records_*.jsonl files in {root}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark run records")
    parser.add_argument(
        "--records",
        type=str,
        default="",
        help="Path to run_records_*.jsonl (defaults to latest in output dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results/benchmark",
        help="Directory containing benchmark outputs",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="analysis_report.json",
        help="Output analysis filename",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    records = Path(args.records) if args.records else latest_records_file(out_dir)
    output_path = out_dir / args.output_name
    build_analysis_report(records, output_path)
    print(f"Analysis saved: {output_path}")


if __name__ == "__main__":
    main()

