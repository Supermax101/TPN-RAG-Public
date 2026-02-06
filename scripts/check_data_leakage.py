#!/usr/bin/env python3
"""
Run leakage checks on benchmark dataset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.evaluation import check_data_leakage
from app.evaluation.data_leakage import load_records


def main():
    parser = argparse.ArgumentParser(description="Check train/holdout leakage")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    records = load_records(args.dataset)
    report = check_data_leakage(records)
    print(json.dumps(report, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()

