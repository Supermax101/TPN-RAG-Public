#!/usr/bin/env python3
"""
DeepEval scoring for the calc benchmark (open-ended track).

This script is intentionally generation-model-agnostic: it consumes an existing
benchmark ledger (run_records_*.jsonl) and applies DeepEval judge metrics.

Typical usage:
  python scripts/deepeval_calc_eval.py \
    --dataset eval/data/calc_50_holdout.jsonl \
    --records eval/results/benchmark/run_records_YYYYMMDD_HHMMSS.jsonl \
    --snapshots eval/cache/retrieval_snapshots_calc_50.jsonl \
    --out-dir eval/results/deepeval/calc_50
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _latest_records_file(root: Path) -> Path:
    files = sorted(root.glob("run_records_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No run_records_*.jsonl files in {root}")
    return files[-1]


def _normalize_doc(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepEval scoring for calc benchmark open-ended runs")
    p.add_argument(
        "--dataset",
        type=str,
        default="eval/data/calc_50_holdout.jsonl",
        help="Strict open-ended dataset JSONL (calc_50_holdout.jsonl or calc_200_holdout.jsonl)",
    )
    p.add_argument(
        "--records",
        type=str,
        default="",
        help="Path to run_records_*.jsonl (defaults to latest in eval/results/benchmark)",
    )
    p.add_argument(
        "--records-dir",
        type=str,
        default="eval/results/benchmark",
        help="Directory to search for latest run_records_*.jsonl if --records not provided",
    )
    p.add_argument(
        "--snapshots",
        type=str,
        default="",
        help="Optional retrieval snapshots JSONL (written by tpnctl precompute-retrieval). Required for RAG-only metrics.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="eval/results/deepeval/calc",
        help="Output directory for DeepEval artifacts",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model name passed to DeepEval metrics (OpenAI-compatible).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge calls (DeepEval AsyncConfig).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build test cases and write nothing; do not call judge models.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = (PROJECT_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    records_path = (
        (PROJECT_ROOT / args.records).resolve()
        if args.records
        else _latest_records_file((PROJECT_ROOT / args.records_dir).resolve())
    )
    snapshots_path = None
    if args.snapshots:
        snapshots_path = (PROJECT_ROOT / args.snapshots).resolve() if not Path(args.snapshots).is_absolute() else Path(args.snapshots)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Records not found: {records_path}")
    if snapshots_path and not snapshots_path.exists():
        raise FileNotFoundError(f"Snapshots not found: {snapshots_path}")

    dataset_rows = _load_jsonl(dataset_path)
    dataset_by_id = {r["sample_id"]: r for r in dataset_rows if r.get("sample_id")}

    records_raw = _load_jsonl(records_path)
    open_records = [
        r
        for r in records_raw
        if str(r.get("track")) == "open_ended" and str(r.get("sample_id")) in dataset_by_id and not r.get("error")
    ]

    if not open_records:
        raise SystemExit(f"No open-ended records found for dataset {dataset_path.name} in {records_path.name}")

    snapshots_by_sample_id: Dict[str, dict] = {}
    if snapshots_path:
        for row in _load_jsonl(snapshots_path):
            sid = str(row.get("sample_id") or "")
            snap = row.get("snapshot")
            if sid and isinstance(snap, dict):
                snapshots_by_sample_id[sid] = snap

    if args.dry_run:
        rag = sum(1 for r in open_records if r.get("rag_enabled"))
        print(f"Dry run OK: dataset={len(dataset_rows)} open_records={len(open_records)} rag_records={rag} snapshots={len(snapshots_by_sample_id)}")
        return 0

    # Lazy imports to keep dry-run fast.
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.metrics.g_eval.g_eval import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_records_path = out_dir / f"deepeval_records_{stamp}.jsonl"
    out_summary_path = out_dir / f"deepeval_summary_{stamp}.json"
    out_summary_csv_path = out_dir / f"deepeval_summary_{stamp}.csv"

    # --- Metrics ---
    # 1) Primary correctness judge (reference-based).
    correctness = GEval(
        name="TPN_CalcCorrectness",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        criteria=(
            "Score the answer for clinical correctness and calculation accuracy compared to the expected output. "
            "The numeric values and units (e.g., mg/kg/min, g/kg/day, mEq/L, mL/hr) must match the expected output "
            "within normal clinical rounding tolerance. Penalize missing required quantities, wrong units, or "
            "mathematical errors. Prefer concise, minimally sufficient calculation steps."
        ),
        model=args.judge_model,
        threshold=0.8,
        async_mode=True,
        verbose_mode=False,
    )

    relevancy = AnswerRelevancyMetric(model=args.judge_model, threshold=0.6, async_mode=True)

    # RAG-only grounding and retrieval diagnostics.
    faithfulness = FaithfulnessMetric(model=args.judge_model, threshold=0.8, async_mode=True)
    c_precision = ContextualPrecisionMetric(model=args.judge_model, threshold=0.6, async_mode=True)
    c_recall = ContextualRecallMetric(model=args.judge_model, threshold=0.6, async_mode=True)
    c_relevancy = ContextualRelevancyMetric(model=args.judge_model, threshold=0.6, async_mode=True)

    async_cfg = AsyncConfig(run_async=True, max_concurrent=args.max_concurrent)
    display_cfg = DisplayConfig(show_indicator=True, print_results=False, verbose_mode=False)

    # --- Build test cases ---
    baseline_cases: List[LLMTestCase] = []
    rag_cases: List[LLMTestCase] = []
    meta_by_name: Dict[str, dict] = {}

    for r in open_records:
        sid = str(r["sample_id"])
        d = dataset_by_id[sid]
        tc = LLMTestCase(
            input=str(d.get("question") or ""),
            actual_output=str(r.get("response_text") or ""),
            expected_output=str(d.get("reference_answer") or ""),
            additional_metadata={
                "run_id": str(r.get("run_id") or ""),
                "sample_id": sid,
                "model_id": str(r.get("model_id") or ""),
                "strategy": str(r.get("prompt_strategy") or ""),
                "rag_enabled": bool(r.get("rag_enabled")),
                "gold_source_doc": str(d.get("source_doc") or ""),
            },
        )
        # DeepEval uses the test case name in some outputs; keep stable for joins.
        tc.name = f"{sid}:{r.get('run_id')}"
        meta_by_name[tc.name] = dict(tc.additional_metadata or {})

        if r.get("rag_enabled"):
            snap = snapshots_by_sample_id.get(sid)
            if snap:
                chunks = snap.get("chunks") or []
                retrieval_context = []
                for c in chunks:
                    src = str(c.get("source") or "unknown")
                    page = c.get("page")
                    content = str(c.get("content") or "")
                    header = f"{src}{f' (p.{page})' if page is not None else ''}"
                    retrieval_context.append(f"{header}\n{content}")
                tc.retrieval_context = retrieval_context
            rag_cases.append(tc)
        else:
            baseline_cases.append(tc)

    # Note: we score baseline metrics for ALL cases; RAG-only metrics only for rag_cases that have retrieval_context.
    all_cases = baseline_cases + rag_cases
    result_base = evaluate(
        test_cases=all_cases,
        metrics=[correctness, relevancy],
        async_config=async_cfg,
        display_config=display_cfg,
    )

    result_rag = None
    rag_scored = [tc for tc in rag_cases if tc.retrieval_context]
    if rag_scored:
        result_rag = evaluate(
            test_cases=rag_scored,
            metrics=[faithfulness, c_precision, c_recall, c_relevancy],
            async_config=async_cfg,
            display_config=display_cfg,
        )

    # --- Serialize results ---
    def _test_results(er) -> List[dict]:
        # EvaluationResult has been stable as {test_results:[...]} across DeepEval 2.x/3.x.
        raw = getattr(er, "model_dump", None)
        if callable(raw):
            data = er.model_dump()
        else:
            data = er.__dict__
        return list(data.get("test_results") or data.get("testResults") or [])

    base_rows = _test_results(result_base)
    rag_rows = _test_results(result_rag) if result_rag is not None else []

    by_name = defaultdict(dict)
    for row in base_rows:
        name = row.get("name") or row.get("testCase", {}).get("name")
        if name:
            by_name[name]["base"] = row
    for row in rag_rows:
        name = row.get("name") or row.get("testCase", {}).get("name")
        if name:
            by_name[name]["rag"] = row

    with out_records_path.open("w", encoding="utf-8") as f:
        for name, payload in by_name.items():
            meta = meta_by_name.get(name, {})
            f.write(
                json.dumps(
                    {
                        "name": name,
                        "meta": meta,
                        "base": payload.get("base"),
                        "rag": payload.get("rag"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _metrics_map(test_result: dict) -> Dict[str, dict]:
        metrics = test_result.get("metrics_data") or test_result.get("metricsData") or []
        out: Dict[str, dict] = {}
        for m in metrics or []:
            name = m.get("name")
            if not name:
                continue
            out[str(name)] = {
                "score": m.get("score"),
                "success": m.get("success"),
                "threshold": m.get("threshold"),
                "reason": m.get("reason"),
            }
        return out

    def _group_summary(rows: List[dict], kind: str) -> List[dict]:
        grouped: Dict[tuple, List[dict]] = defaultdict(list)
        for row in rows:
            name = row.get("name")
            meta = meta_by_name.get(name or "", {})
            if not meta:
                continue
            key = (
                meta.get("model_id", ""),
                meta.get("strategy", ""),
                "rag" if meta.get("rag_enabled") else "no_rag",
            )
            grouped[key].append(row)

        out_rows: List[dict] = []
        for (model_id, strategy, rag_mode), items in sorted(grouped.items()):
            metrics = [_metrics_map(x) for x in items]
            metric_names = sorted({n for mm in metrics for n in mm.keys()})
            row: Dict[str, Any] = {
                "kind": kind,
                "model_id": model_id,
                "strategy": strategy,
                "rag_mode": rag_mode,
                "n": len(items),
            }
            for mn in metric_names:
                scores = [m[mn].get("score") for m in metrics if mn in m and m[mn].get("score") is not None]
                succ = [m[mn].get("success") for m in metrics if mn in m and m[mn].get("success") is not None]
                if scores:
                    row[f"{mn}_mean"] = sum(float(s) for s in scores) / len(scores)
                if succ:
                    row[f"{mn}_pass_rate"] = sum(1 for s in succ if bool(s)) / len(succ)
            out_rows.append(row)
        return out_rows

    base_summary_rows = _group_summary(base_rows, kind="base")
    rag_summary_rows = _group_summary(rag_rows, kind="rag") if rag_rows else []

    # RAG lift for GEval correctness (if present in both modes).
    lift_rows: List[dict] = []
    base_index = {(r["model_id"], r["strategy"], r["rag_mode"]): r for r in base_summary_rows}
    for (model_id, strategy) in sorted({(r["model_id"], r["strategy"]) for r in base_summary_rows}):
        no_rag = base_index.get((model_id, strategy, "no_rag"))
        rag = base_index.get((model_id, strategy, "rag"))
        if not no_rag or not rag:
            continue
        # Prefer GEval metric name; fall back to any *_mean if missing.
        metric_key = "TPN_CalcCorrectness_mean"
        if metric_key in no_rag and metric_key in rag:
            lift_rows.append(
                {
                    "model_id": model_id,
                    "strategy": strategy,
                    "metric": "TPN_CalcCorrectness",
                    "no_rag_mean": no_rag[metric_key],
                    "rag_mean": rag[metric_key],
                    "delta": rag[metric_key] - no_rag[metric_key],
                }
            )

    summary = {
        "dataset": str(dataset_path),
        "records": str(records_path),
        "snapshots": str(snapshots_path) if snapshots_path else None,
        "counts": {
            "open_records": len(open_records),
            "scored_base": len(all_cases),
            "scored_rag": len(rag_scored),
        },
        "summary_rows_base": base_summary_rows,
        "summary_rows_rag": rag_summary_rows,
        "rag_lift": lift_rows,
        # Aggregate analysis is intentionally minimal here; the JSONL is the source of truth.
    }
    out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write a flat CSV for quick inspection.
    combined_rows = base_summary_rows + rag_summary_rows
    if combined_rows:
        fieldnames = list(dict.fromkeys(k for r in combined_rows for k in r.keys()))
        with out_summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in combined_rows:
                w.writerow(r)

    print("DeepEval complete:")
    print(f"  Records: {out_records_path}")
    print(f"  Summary: {out_summary_path}")
    print(f"  Summary CSV: {out_summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
