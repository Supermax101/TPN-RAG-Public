#!/usr/bin/env python3
"""
View benchmark results across all models and phases.

Usage:
  python scripts/view_results.py                          # scan default dirs
  python scripts/view_results.py --results-dir eval/results/phase1_sota
  python scripts/view_results.py --results-dir eval/results  # scan all phases
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_all_summaries(results_dir: Path) -> list[dict]:
    """Recursively find and load all summary_*.json files."""
    rows = []
    for summary_file in sorted(results_dir.rglob("summary_*.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            for row in data.get("rows", []):
                row["_source"] = str(summary_file.relative_to(results_dir))
                rows.append(row)
        except Exception as e:
            print(f"  Warning: Could not read {summary_file}: {e}", file=sys.stderr)
    return rows


def print_accuracy_table(rows: list[dict], title: str = "Benchmark Results"):
    """Print a formatted accuracy table."""
    if not rows:
        print(f"\n{title}: No results found.\n")
        return

    # Filter to MCQ rows with accuracy
    mcq_rows = [r for r in rows if "accuracy" in r]
    if not mcq_rows:
        print(f"\n{title}: No MCQ accuracy results found.\n")
        return

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(
        f"  {'Model':<20} {'Strategy':<10} {'RAG':<8} {'N':>5} "
        f"{'Accuracy':>10} {'Partial':>10} {'Latency(ms)':>12} {'Errors':>8}"
    )
    print(f"  {'-' * 85}")

    for row in sorted(mcq_rows, key=lambda r: (r["model_id"], r["rag_mode"], r["strategy"])):
        acc = row.get("accuracy", 0)
        partial = row.get("partial_rate", 0)
        latency = row.get("latency_ms_mean", 0)
        error_rate = row.get("error_rate", 0)
        print(
            f"  {row['model_id']:<20} {row['strategy']:<10} {row['rag_mode']:<8} {row['n']:>5} "
            f"{acc:>9.1%} {partial:>9.1%} {latency:>11.0f} {error_rate:>7.1%}"
        )

    print()


def print_rag_lift_table(rows: list[dict]):
    """Print RAG lift comparison: no_rag vs rag for each (model, strategy)."""
    mcq_rows = [r for r in rows if "accuracy" in r]
    if not mcq_rows:
        return

    # Group by (model_id, strategy)
    grouped = defaultdict(dict)
    for row in mcq_rows:
        key = (row["model_id"], row["strategy"])
        grouped[key][row["rag_mode"]] = row

    has_both = any("rag" in v and "no_rag" in v for v in grouped.values())
    if not has_both:
        return

    print(f"{'=' * 80}")
    print(f"  RAG Lift Analysis")
    print(f"{'=' * 80}")
    print(
        f"  {'Model':<20} {'Strategy':<10} {'No-RAG':>10} {'RAG':>10} {'Lift':>10} {'Lift%':>10}"
    )
    print(f"  {'-' * 75}")

    lifts = []
    for (model_id, strategy), modes in sorted(grouped.items()):
        if "rag" not in modes or "no_rag" not in modes:
            continue
        no_rag_acc = modes["no_rag"].get("accuracy", 0)
        rag_acc = modes["rag"].get("accuracy", 0)
        lift = rag_acc - no_rag_acc
        lift_pct = (lift / no_rag_acc * 100) if no_rag_acc > 0 else 0
        lifts.append(lift)
        marker = " ✓" if lift > 0 else " ✗" if lift < 0 else ""
        print(
            f"  {model_id:<20} {strategy:<10} {no_rag_acc:>9.1%} {rag_acc:>9.1%} "
            f"{lift:>+9.1%} {lift_pct:>+9.1f}%{marker}"
        )

    if lifts:
        avg_lift = sum(lifts) / len(lifts)
        print(f"  {'-' * 75}")
        print(f"  {'AVERAGE':<20} {'':10} {'':>10} {'':>10} {avg_lift:>+9.1%}")
    print()


def print_model_summary(rows: list[dict]):
    """Print per-model aggregate accuracy (across all strategies)."""
    mcq_rows = [r for r in rows if "accuracy" in r]
    if not mcq_rows:
        return

    # Group by (model_id, rag_mode)
    grouped = defaultdict(list)
    for row in mcq_rows:
        grouped[(row["model_id"], row["rag_mode"])].append(row)

    print(f"{'=' * 60}")
    print(f"  Model Summary (averaged across strategies)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<20} {'RAG':<8} {'Avg Accuracy':>14} {'Strategies':>12}")
    print(f"  {'-' * 55}")

    for (model_id, rag_mode), model_rows in sorted(grouped.items()):
        avg_acc = sum(r["accuracy"] for r in model_rows) / len(model_rows)
        print(
            f"  {model_id:<20} {rag_mode:<8} {avg_acc:>13.1%} {len(model_rows):>12}"
        )
    print()


def export_combined_csv(rows: list[dict], output_path: Path):
    """Export all results to a single combined CSV."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(col for row in rows for col in row.keys()))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Combined CSV exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="View benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval/results"),
        help="Root results directory to scan (default: eval/results)",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Export combined results to CSV",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        return 1

    # Find all phases
    phase_dirs = sorted(
        d for d in args.results_dir.iterdir()
        if d.is_dir() and any(d.rglob("summary_*.json"))
    )

    if not phase_dirs:
        # Maybe the results_dir itself has summaries
        rows = load_all_summaries(args.results_dir)
        if rows:
            print_accuracy_table(rows, title=f"Results: {args.results_dir.name}")
            print_rag_lift_table(rows)
            print_model_summary(rows)
        else:
            print(f"No summary files found in {args.results_dir}")
            return 1
    else:
        all_rows = []
        for phase_dir in phase_dirs:
            rows = load_all_summaries(phase_dir)
            all_rows.extend(rows)
            print_accuracy_table(rows, title=f"Phase: {phase_dir.name}")

        if all_rows:
            print_rag_lift_table(all_rows)
            print_model_summary(all_rows)

    if args.export_csv:
        all_rows = load_all_summaries(args.results_dir)
        export_combined_csv(all_rows, args.export_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
