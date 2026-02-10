"""
QandA20 (open-ended) paper-run report generator.

Consumes a single run folder created by:
  tpnctl paper-open-qanda20

It merges:
- generation run_records_*.jsonl (per model/condition)
- DeepEval per-testcase records (per model/condition/judge)

Outputs (under <run_dir>/reports/open/qanda20/):
- per_sample.csv
- summary_by_model_condition.csv
- REPORT.md (boss-friendly narrative + rankings + examples)
- figures/*.png (best-effort) + figures/*.html (always)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_DISPLAY = {
    "gemini-3-flash": "Gemini 3 Flash",
    "gpt-5.2": "GPT-5.2",
    "gpt-5-mini": "GPT-5 Mini",
    "claude-sonnet": "Claude Sonnet 4.5",
    "grok-4.1-fast": "Grok 4.1 Fast",
    "kimi-k2.5": "Kimi K2.5",
    "phi-4": "Phi-4 (14B)",
    "gpt-oss-20b": "GPT-OSS (20B)",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "medgemma-27b": "MedGemma 27B",
    "gemma3-27b": "Gemma 3 27B",
}

MODEL_ORDER = [
    "gpt-5-mini",
    "gpt-5.2",
    "claude-sonnet",
    "gemini-3-flash",
    "grok-4.1-fast",
    "kimi-k2.5",
    "medgemma-27b",
    "gemma3-27b",
    "qwen3-30b-a3b",
    "phi-4",
    "gpt-oss-20b",
]

CONDITION_ORDER = ["no_rag", "rag_gated", "rag_always"]

PRIMARY_JUDGE_DIR = "openai__gpt-4.1-mini-2025-04-14"

# Reporting classification:
# - "Open" = local HF open-weights + Kimi (API-served but open weights).
# - "API" = proprietary closed models.
OPEN_MODEL_KEYS = {
    "phi-4",
    "gpt-oss-20b",
    "qwen3-30b-a3b",
    "medgemma-27b",
    "gemma3-27b",
    "kimi-k2.5",
}
API_MODEL_KEYS = {
    "gpt-5-mini",
    "gpt-5.2",
    "claude-sonnet",
    "gemini-3-flash",
    "grok-4.1-fast",
}


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _latest_file(dir_path: Path, glob_pat: str) -> Optional[Path]:
    files = sorted(dir_path.glob(glob_pat))
    return files[-1] if files else None


def _parse_judge_dirname(dirname: str) -> Tuple[str, str, str]:
    """
    Return (provider, model, judge_id) from a judge output directory name.

    New format (portable): "openai__gpt-4.1-mini-..." (":" -> "__")
    Legacy format (linux-only): "openai:gpt-4.1-mini-..."
    """
    name = (dirname or "").strip()
    if "__" in name:
        provider, model = name.split("__", 1)
        return provider, model, f"{provider}:{model}"
    if ":" in name:
        provider, model = name.split(":", 1)
        return provider, model, f"{provider}:{model}"
    # Fallback: treat as opaque
    return name, "", name


def _metrics_map(test_result: dict) -> Dict[str, dict]:
    metrics = test_result.get("metrics_data") or test_result.get("metricsData") or []
    out: Dict[str, dict] = {}
    for m in metrics or []:
        n = m.get("name")
        if not n:
            continue
        out[str(n)] = m
    return out


def _safe_write_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{stem}.html"
    fig.write_html(html_path)
    try:
        png_path = out_dir / f"{stem}.png"
        fig.write_image(png_path, scale=3)
    except Exception:
        # kaleido may not be installed; HTML is still generated
        pass


def _format_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return ""


def _format_float(x: Any, ndigits: int = 3) -> str:
    try:
        if x is None:
            return ""
        if pd.isna(x):
            return ""
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return ""


def _to_markdown_table(rows: List[dict], columns: List[str]) -> str:
    if not rows:
        return "_(no rows)_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = [str(r.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _select_rag_lift_examples(
    gen_df: pd.DataFrame,
    primary_provider: str,
    n: int = 4,
) -> List[dict]:
    """
    Pick illustrative examples where RAG (rag_gated) meaningfully improves
    correctness over no_rag, using the primary judge scores.
    """
    cor_col = f"{primary_provider}_correctness_score"
    rel_col = f"{primary_provider}_relevancy_score"

    needed_cols = {"sample_id", "model_id", "condition", "question", "reference_answer", "response_text", "format_ok", cor_col, rel_col, "grade_primary"}
    missing = [c for c in needed_cols if c not in gen_df.columns]
    if missing:
        return []

    df = gen_df.copy()
    df["model_id_str"] = df["model_id"].astype(str)
    df = df[df["condition"].isin(["no_rag", "rag_gated"])].copy()

    # Pivot so each (sample_id, model) has both conditions.
    pivot = (
        df.pivot_table(
            index=["sample_id", "model_id_str"],
            columns="condition",
            values=[cor_col, rel_col, "grade_primary", "format_ok", "response_text"],
            aggfunc="first",
        )
        .reset_index()
    )

    def _get(col: str, cond: str) -> str:
        return f"{col}|{cond}"

    # Flatten multiindex columns from pivot_table.
    pivot.columns = [
        f"{a}|{b}" if isinstance(a, str) and isinstance(b, str) else str(a)
        for (a, b) in getattr(pivot.columns, "to_list", lambda: list(pivot.columns))()
    ]

    cor_no = _get(cor_col, "no_rag")
    cor_rag = _get(cor_col, "rag_gated")
    grade_no = _get("grade_primary", "no_rag")
    grade_rag = _get("grade_primary", "rag_gated")
    fmt_no = _get("format_ok", "no_rag")
    fmt_rag = _get("format_ok", "rag_gated")

    if cor_no not in pivot.columns or cor_rag not in pivot.columns:
        return []

    pivot["delta_correctness"] = pivot[cor_rag] - pivot[cor_no]

    # Candidate: RAG PASS but no_rag FAIL (or much lower), and both formats ok.
    cand = pivot.copy()
    cand = cand[(cand[fmt_no].fillna(True).astype(bool)) & (cand[fmt_rag].fillna(True).astype(bool))]
    cand = cand[cand[grade_rag] == "PASS"]
    cand = cand[(cand[grade_no] == "FAIL") | (cand[cor_no] < 0.6)]
    cand = cand[cand["delta_correctness"] >= 0.20]
    cand = cand.sort_values("delta_correctness", ascending=False)

    # Avoid picking the same model repeatedly if possible.
    selected: List[dict] = []
    used_models: set[str] = set()
    for _, row in cand.iterrows():
        model_id = str(row.get("model_id_str") or "")
        if model_id in used_models and len(selected) < n:
            continue
        sid = str(row.get("sample_id") or "")

        # Recover question/reference from original df.
        base_row = df[(df["sample_id"] == sid) & (df["model_id"].astype(str) == model_id) & (df["condition"] == "no_rag")].head(1)
        rag_row = df[(df["sample_id"] == sid) & (df["model_id"].astype(str) == model_id) & (df["condition"] == "rag_gated")].head(1)
        if base_row.empty or rag_row.empty:
            continue

        selected.append(
            {
                "sample_id": sid,
                "model_id": model_id,
                "model_display": MODEL_DISPLAY.get(model_id, model_id),
                "question": str(base_row.iloc[0].get("question") or ""),
                "reference_answer": str(base_row.iloc[0].get("reference_answer") or ""),
                "no_rag_answer": str(base_row.iloc[0].get("response_text") or ""),
                "rag_gated_answer": str(rag_row.iloc[0].get("response_text") or ""),
                "no_rag_correctness": row.get(cor_no),
                "rag_gated_correctness": row.get(cor_rag),
                "delta_correctness": row.get("delta_correctness"),
            }
        )
        used_models.add(model_id)
        if len(selected) >= n:
            break

    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QandA20 v2 report from a paper run folder")
    p.add_argument("--run-dir", type=str, required=True, help="Run folder under eval/paper_runs/<run_set_id>")
    p.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl"),
        help="QandA20 dataset JSONL (default: repo canonical path)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    dataset_path = Path(args.dataset).resolve()

    open_root = run_dir / "open" / "qanda20"
    deepeval_root = run_dir / "deepeval" / "open" / "qanda20"
    report_root = run_dir / "reports" / "open" / "qanda20"
    figures_dir = report_root / "figures"
    report_root.mkdir(parents=True, exist_ok=True)

    # --- Dataset (ground truth) ---
    dataset_rows = _load_jsonl(dataset_path)
    ds_by_id = {str(r.get("sample_id")): r for r in dataset_rows if r.get("sample_id")}

    # --- Generation records ---
    gen_rows: List[dict] = []
    if open_root.exists():
        for model_dir in sorted([p for p in open_root.iterdir() if p.is_dir()]):
            model_key = model_dir.name
            for cond_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                condition = cond_dir.name
                records_path = _latest_file(cond_dir, "run_records_*.jsonl")
                if records_path is None:
                    continue
                for r in _load_jsonl(records_path):
                    if str(r.get("track")) != "open_ended":
                        continue
                    if r.get("error"):
                        continue
                    sid = str(r.get("sample_id") or "")
                    ds = ds_by_id.get(sid, {})
                    metrics = r.get("metrics") or {}
                    gen_rows.append(
                        {
                            "name": f"{sid}:{r.get('run_id')}",
                            "sample_id": sid,
                            "run_id": str(r.get("run_id") or ""),
                            "model_id": str(r.get("model_id") or model_key),
                            "provider": str(r.get("provider") or ""),
                            "model_name": str(r.get("model_name") or ""),
                            "condition": condition,
                            "prompt_strategy": str(getattr(r.get("prompt_strategy"), "value", r.get("prompt_strategy"))),
                            "question": str(ds.get("question") or r.get("question") or ""),
                            "reference_answer": str(ds.get("reference_answer") or ""),
                            "response_text": str(r.get("response_text") or ""),
                            "rag_enabled": bool(r.get("rag_enabled")),
                            "rag_context_used": bool(metrics.get("rag_context_used")),
                            "rag_gate_reason": metrics.get("rag_gate_reason"),
                            "rag_top_score": metrics.get("rag_top_score"),
                            "rag_context_chars": metrics.get("rag_context_chars"),
                            "rag_returned_chunks": metrics.get("rag_returned_chunks"),
                            # Deterministic metrics (subset; full set remains in run_records JSONL)
                            "final_key_f1": metrics.get("final_key_f1"),
                            "final_quantity_f1": metrics.get("final_quantity_f1"),
                            "final_unit_mismatch_count": metrics.get("final_unit_mismatch_count"),
                            # Format contract metrics
                            "format_ok": metrics.get("format_ok"),
                            "format_retry_used": metrics.get("format_retry_used"),
                            "format_violation_reason": metrics.get("format_violation_reason"),
                            "format_violation_reason_after_retry": metrics.get("format_violation_reason_after_retry"),
                            "format_retry_error": metrics.get("format_retry_error"),
                        }
                    )

    if not gen_rows:
        raise SystemExit(f"No generation records found under {open_root}")

    gen_df = pd.DataFrame(gen_rows)

    # --- DeepEval records (per judge) ---
    # judge_key -> name -> metric_name -> score/success
    deepeval_by_judge: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if deepeval_root.exists():
        for model_dir in sorted([p for p in deepeval_root.iterdir() if p.is_dir()]):
            for cond_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                for judge_dir in sorted([p for p in cond_dir.iterdir() if p.is_dir()]):
                    judge_key = judge_dir.name
                    records_path = _latest_file(judge_dir, "deepeval_records_*.jsonl")
                    if records_path is None:
                        continue
                    for row in _load_jsonl(records_path):
                        name = str(row.get("name") or "")
                        base = row.get("base") or {}
                        rag = row.get("rag") or {}

                        mm_base = _metrics_map(base)
                        mm_rag = _metrics_map(rag) if rag else {}

                        d: Dict[str, Any] = {}
                        for metric_name, payload in {**mm_base, **mm_rag}.items():
                            d[f"{metric_name}.score"] = payload.get("score")
                            d[f"{metric_name}.success"] = payload.get("success")
                            d[f"{metric_name}.threshold"] = payload.get("threshold")
                        deepeval_by_judge.setdefault(judge_key, {})[name] = d

    # Attach judge columns to gen_df
    for judge_key, by_name in deepeval_by_judge.items():
        provider, model, judge_id = _parse_judge_dirname(judge_key)
        prefix = provider  # keep columns short; provider identifies judge in this tri-judge setup

        def _get(name: str, k: str) -> Any:
            return by_name.get(name, {}).get(k)

        gen_df[f"{prefix}_judge_id"] = judge_id
        gen_df[f"{prefix}_correctness_score"] = gen_df["name"].apply(
            lambda n: _get(n, "TPN_OpenCorrectness.score")
        )
        gen_df[f"{prefix}_relevancy_score"] = gen_df["name"].apply(
            lambda n: _get(n, "Answer Relevancy.score") or _get(n, "AnswerRelevancyMetric.score")
        )
        gen_df[f"{prefix}_faithfulness_score"] = gen_df["name"].apply(
            lambda n: _get(n, "Faithfulness.score")
        )
        gen_df[f"{prefix}_ctx_precision_score"] = gen_df["name"].apply(
            lambda n: _get(n, "Contextual Precision.score")
            or _get(n, "ContextualPrecisionMetric.score")
        )
        gen_df[f"{prefix}_ctx_recall_score"] = gen_df["name"].apply(
            lambda n: _get(n, "Contextual Recall.score")
            or _get(n, "ContextualRecallMetric.score")
        )
        gen_df[f"{prefix}_ctx_relevancy_score"] = gen_df["name"].apply(
            lambda n: _get(n, "Contextual Relevancy.score")
            or _get(n, "ContextualRelevancyMetric.score")
        )

    # --- Derived PASS/PARTIAL/FAIL using primary judge (OpenAI) ---
    primary_provider = "openai"
    if f"{primary_provider}_correctness_score" not in gen_df.columns:
        # Fallback: choose any available provider that has correctness.
        for col in gen_df.columns:
            if col.endswith("_correctness_score"):
                primary_provider = col.split("_", 1)[0]
                break

    def _grade(row: pd.Series) -> str:
        if not bool(row.get("format_ok", True)):
            return "FAIL"
        c = row.get(f"{primary_provider}_correctness_score")
        r = row.get(f"{primary_provider}_relevancy_score")
        try:
            c = float(c) if c is not None else None
        except Exception:
            c = None
        try:
            r = float(r) if r is not None else None
        except Exception:
            r = None
        if c is None or r is None:
            return "FAIL"
        if c >= 0.8 and r >= 0.8:
            return "PASS"
        if c >= 0.6 and r >= 0.8:
            return "PARTIAL"
        return "FAIL"

    gen_df["primary_provider"] = primary_provider
    gen_df["grade_primary"] = gen_df.apply(_grade, axis=1)

    # Normalize order/display
    gen_df["model_display"] = gen_df["model_id"].map(MODEL_DISPLAY).fillna(gen_df["model_id"])
    gen_df["model_id"] = pd.Categorical(gen_df["model_id"], categories=MODEL_ORDER, ordered=True)
    gen_df["condition"] = pd.Categorical(gen_df["condition"], categories=CONDITION_ORDER, ordered=True)

    per_sample_path = report_root / "per_sample.csv"
    gen_df.sort_values(["model_id", "condition", "sample_id"]).to_csv(per_sample_path, index=False)

    # --- Summary table ---
    agg_cols = {}
    for provider in ["openai", "anthropic", "gemini"]:
        if f"{provider}_correctness_score" in gen_df.columns:
            agg_cols[f"{provider}_correctness_mean"] = (f"{provider}_correctness_score", "mean")
            agg_cols[f"{provider}_relevancy_mean"] = (f"{provider}_relevancy_score", "mean")
            agg_cols[f"{provider}_faithfulness_mean"] = (f"{provider}_faithfulness_score", "mean")
            agg_cols[f"{provider}_ctx_precision_mean"] = (f"{provider}_ctx_precision_score", "mean")
            agg_cols[f"{provider}_ctx_recall_mean"] = (f"{provider}_ctx_recall_score", "mean")
            agg_cols[f"{provider}_ctx_relevancy_mean"] = (f"{provider}_ctx_relevancy_score", "mean")

    summary = (
        gen_df.groupby(["model_id", "condition"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            format_ok_rate=("format_ok", lambda s: float(pd.Series(s).fillna(True).astype(bool).mean())),
            rag_context_used_rate=("rag_context_used", "mean"),
            pass_rate=("grade_primary", lambda s: float((pd.Series(s) == "PASS").mean())),
            partial_rate=("grade_primary", lambda s: float((pd.Series(s) == "PARTIAL").mean())),
            fail_rate=("grade_primary", lambda s: float((pd.Series(s) == "FAIL").mean())),
            **agg_cols,  # type: ignore[arg-type]
        )
        .reset_index()
    )
    summary["model_display"] = summary["model_id"].astype(str).map(MODEL_DISPLAY).fillna(summary["model_id"].astype(str))
    summary_path = report_root / "summary_by_model_condition.csv"
    summary.sort_values(["model_id", "condition"]).to_csv(summary_path, index=False)

    # --- Figures (primary judge) ---
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Mean correctness bars (primary provider)
    if f"{primary_provider}_correctness_mean" in summary.columns:
        fig = px.bar(
            summary,
            x="model_display",
            y=f"{primary_provider}_correctness_mean",
            color="condition",
            barmode="group",
            category_orders={
                "condition": CONDITION_ORDER,
                "model_display": [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
            },
            title=f"QandA20 GEval Correctness (mean) — Primary judge: {primary_provider}",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Correctness (0-1)", legend_title_text="")
        _safe_write_fig(fig, figures_dir, "01_correctness_mean_primary")

    # PASS/PARTIAL/FAIL stacked bars
    stacked = (
        summary[["model_display", "condition", "pass_rate", "partial_rate", "fail_rate"]]
        .melt(id_vars=["model_display", "condition"], var_name="grade", value_name="rate")
    )
    fig = px.bar(
        stacked,
        x="model_display",
        y="rate",
        color="grade",
        facet_col="condition",
        category_orders={
            "condition": CONDITION_ORDER,
            "model_display": [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
            "grade": ["pass_rate", "partial_rate", "fail_rate"],
        },
        title="QandA20 PASS/PARTIAL/FAIL rates (primary judge)",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Rate", legend_title_text="")
    _safe_write_fig(fig, figures_dir, "02_pass_partial_fail")

    # RAG lift (rag_gated - no_rag, rag_always - no_rag) on primary correctness
    if f"{primary_provider}_correctness_mean" in summary.columns:
        pivot = summary.pivot(index="model_display", columns="condition", values=f"{primary_provider}_correctness_mean")
        lift_rows = []
        for model_display in pivot.index:
            base = pivot.loc[model_display].get("no_rag")
            for cond in ["rag_gated", "rag_always"]:
                val = pivot.loc[model_display].get(cond)
                if pd.notna(base) and pd.notna(val):
                    lift_rows.append({"model_display": model_display, "condition": cond, "lift": float(val - base)})
        if lift_rows:
            lift_df = pd.DataFrame(lift_rows)
            fig = px.bar(
                lift_df,
                x="model_display",
                y="lift",
                color="condition",
                barmode="group",
                category_orders={
                    "condition": ["rag_gated", "rag_always"],
                    "model_display": [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
                },
                title=f"QandA20 RAG Lift vs no_rag — Primary judge: {primary_provider}",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Correctness lift", legend_title_text="")
            _safe_write_fig(fig, figures_dir, "03_rag_lift_primary")

    # Retrieval bottleneck diagnostic: ctx_recall vs correctness (rag only)
    ctx_col = f"{primary_provider}_ctx_recall_score"
    cor_col = f"{primary_provider}_correctness_score"
    if ctx_col in gen_df.columns and cor_col in gen_df.columns:
        rag_only = gen_df[gen_df["condition"].isin(["rag_gated", "rag_always"])].copy()
        rag_only = rag_only[pd.notna(rag_only[ctx_col]) & pd.notna(rag_only[cor_col])]
        if not rag_only.empty:
            fig = px.scatter(
                rag_only,
                x=ctx_col,
                y=cor_col,
                color="model_display",
                symbol="condition",
                title=f"QandA20: Contextual Recall vs Correctness (primary judge: {primary_provider})",
            )
            fig.update_layout(xaxis_title="Contextual Recall", yaxis_title="Correctness")
            _safe_write_fig(fig, figures_dir, "04_ctx_recall_vs_correctness")

    # --- Boss-friendly narrative report ---
    report_md_path = report_root / "REPORT.md"
    ranking_condition = "rag_gated"
    primary_score_mean_col = f"{primary_provider}_correctness_mean"
    primary_rel_mean_col = f"{primary_provider}_relevancy_mean"

    # Ranking rows (use primary judge, rag_gated condition).
    rank_df = summary.copy()
    rank_df["model_id_str"] = rank_df["model_id"].astype(str)
    rank_df = rank_df[rank_df["condition"] == ranking_condition].copy()
    if primary_score_mean_col in rank_df.columns:
        rank_df = rank_df[pd.notna(rank_df[primary_score_mean_col])].copy()
        rank_df = rank_df.sort_values(primary_score_mean_col, ascending=False)

    def _rank_table(df: pd.DataFrame, title: str) -> str:
        if df.empty or primary_score_mean_col not in df.columns:
            return f"### {title}\n_(not available yet; DeepEval scoring incomplete)_\n\n"
        rows = []
        for i, r in enumerate(df.itertuples(index=False), 1):
            mid = str(getattr(r, "model_id_str", "") or "")
            md = MODEL_DISPLAY.get(mid, mid)
            rows.append(
                {
                    "rank": i,
                    "model": md,
                    "correctness_mean": _format_float(getattr(r, primary_score_mean_col)),
                    "relevancy_mean": _format_float(getattr(r, primary_rel_mean_col, None)),
                    "pass_rate": _format_pct(getattr(r, "pass_rate")),
                }
            )
        return f"### {title}\n" + _to_markdown_table(rows, ["rank", "model", "correctness_mean", "relevancy_mean", "pass_rate"]) + "\n"

    open_rank = rank_df[rank_df["model_id_str"].isin(sorted(OPEN_MODEL_KEYS))].copy()
    api_rank = rank_df[rank_df["model_id_str"].isin(sorted(API_MODEL_KEYS))].copy()
    overall_rank = rank_df.copy()

    # RAG lift summary across all models (primary judge correctness mean).
    rag_lift_summary = ""
    if primary_score_mean_col in summary.columns:
        pivot = summary.pivot(index="model_display", columns="condition", values=primary_score_mean_col)
        if "no_rag" in pivot.columns and "rag_gated" in pivot.columns:
            deltas = (pivot["rag_gated"] - pivot["no_rag"]).dropna()
            if not deltas.empty:
                rag_lift_summary = f"**Average RAG lift (rag_gated - no_rag)**: `{deltas.mean():.3f}` (primary judge)."

    # Examples showing RAG improvement.
    examples = _select_rag_lift_examples(gen_df=gen_df, primary_provider=primary_provider, n=4)
    examples_md = ""
    if examples:
        blocks = []
        for ex in examples:
            blocks.append(
                "\n".join(
                    [
                        f"#### Example: {ex['sample_id']} ({ex['model_display']})",
                        "",
                        "**Question**",
                        "",
                        f"> {ex['question']}",
                        "",
                        "**Reference answer (ground truth)**",
                        "",
                        "```text",
                        ex["reference_answer"].strip(),
                        "```",
                        "",
                        f"**No RAG** (correctness={_format_float(ex['no_rag_correctness'])})",
                        "",
                        "```text",
                        ex["no_rag_answer"].strip(),
                        "```",
                        "",
                        f"**RAG (gated)** (correctness={_format_float(ex['rag_gated_correctness'])}, delta={_format_float(ex['delta_correctness'])})",
                        "",
                        "```text",
                        ex["rag_gated_answer"].strip(),
                        "```",
                        "",
                    ]
                )
            )
        examples_md = "### RAG vs No‑RAG Examples\n\n" + "\n".join(blocks) + "\n"
    else:
        examples_md = "### RAG vs No‑RAG Examples\n_(not available yet; requires DeepEval scores for both no_rag and rag_gated)_\n\n"

    # Chart references (png preferred, html always).
    figs = [
        ("01_correctness_mean_primary", "Correctness by model/condition (primary judge)"),
        ("02_pass_partial_fail", "PASS/PARTIAL/FAIL rates (primary judge)"),
        ("03_rag_lift_primary", "RAG lift vs no_rag (primary judge)"),
        ("04_ctx_recall_vs_correctness", "Retrieval bottleneck diagnostic (ctx recall vs correctness)"),
    ]
    fig_md_lines = ["### Key Figures\n"]
    for stem, label in figs:
        png = figures_dir / f"{stem}.png"
        html = figures_dir / f"{stem}.html"
        if png.exists():
            rel = f"figures/{stem}.png"
            fig_md_lines.append(f"- {label}: `{rel}`")
        elif html.exists():
            rel = f"figures/{stem}.html"
            fig_md_lines.append(f"- {label}: `{rel}`")
        else:
            fig_md_lines.append(f"- {label}: _(not generated yet)_")
    fig_md = "\n".join(fig_md_lines) + "\n\n"

    # Narrative: concise, boss-friendly.
    report_md = "\n".join(
        [
            "# QandA20 Open‑Ended Benchmark v2 (Paper‑Grade)",
            "",
            f"- Run folder: `{run_dir}`",
            "- Dataset: QandA20 holdout (N=20) open‑ended TPN clinical questions",
            "- Prompting: zero‑shot only (ZS), strict `Final answer:` output contract (no citations, no chain‑of‑thought)",
            "- Conditions per model: `no_rag`, `rag_gated`, `rag_always`",
            "- Determinism: RAG uses **precomputed retrieval snapshots** (no query‑embedding calls during benchmarks)",
            "",
            "## What We Score (and Why)",
            "",
            "**Primary outcome (open‑ended): GEval correctness** (`TPN_OpenCorrectness`, 0–1)",
            "- Implemented via DeepEval’s `GEval` metric with a clinical rubric (paraphrase‑tolerant).",
            "- Interprets: “Did the answer match the expected clinical content (including key numbers/units)?”",
            "",
            "**Guardrail: Answer relevancy** (0–1)",
            "- Prevents rewarding answers that are generally correct but do not address the question asked.",
            "",
            "**RAG‑only diagnostics (only when retrieval context is injected and used):**",
            "- **Faithfulness** (0–1): are the answer’s claims supported by retrieved context (hallucination check).",
            "- **Contextual Precision / Recall / Relevancy** (0–1): separates retrieval quality from generation quality.",
            "",
            "**Deterministic diagnostics (non‑judge):**",
            "- `format_ok`: output contract compliance (plus one automatic retry).",
            "- `final_key_f1`, `final_quantity_f1`, `final_unit_mismatch_count`: lightweight checks for calculation‑like items.",
            "",
            "## Judges (Tri‑Judge)",
            "",
            "- OpenAI: `gpt-4.1-mini-2025-04-14`",
            "- Anthropic: `claude-haiku-4-5-20251001`",
            "- Gemini: `gemini-2.5-flash-lite`",
            "",
            "## Pass / Fail Policy (paper-facing)",
            "",
            "- **PASS**: correctness ≥ 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`",
            "- **PARTIAL**: 0.60 ≤ correctness < 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`",
            "- **FAIL**: otherwise",
            "",
            "## Top Models (Primary Judge, rag_gated)",
            "",
            "_Note: Kimi is API-served but treated as an **Open** model in reporting._",
            "",
            _rank_table(open_rank, "Best Open Models (HF + Kimi)"),
            _rank_table(api_rank, "Best API Models (Closed)"),
            _rank_table(overall_rank, "Overall Ranking (All Models)"),
            "## RAG Lift (Primary Judge)",
            "",
            rag_lift_summary or "_(not available yet; DeepEval scoring incomplete)_",
            "",
            fig_md.strip(),
            "",
            examples_md.strip(),
            "",
            "## Where The Numbers Come From (files)",
            "",
            "- Generation per model/condition: `open/qanda20/<model>/<condition>/run_records_*.jsonl`",
            "- DeepEval per model/condition/judge: `deepeval/open/qanda20/<model>/<condition>/<judge>/deepeval_records_*.jsonl`",
            "- Canonical merged table: `per_sample.csv`",
            "- Summary table: `summary_by_model_condition.csv`",
            "",
        ]
    ).strip() + "\n"

    report_md_path.write_text(report_md, encoding="utf-8")

    print("Wrote:")
    print(f"  {per_sample_path}")
    print(f"  {summary_path}")
    print(f"  {report_md_path}")
    print(f"  {figures_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
