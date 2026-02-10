#!/usr/bin/env python3
"""
Nature-grade figures for QandA20 open-ended benchmark v2.

Reads the canonical merged CSVs produced by:
  data_viz/open_qanda20_v2_report.py

Outputs 6 high-resolution PNGs (plus one Plotly HTML) into:
  <run_dir>/reports/open/qanda20/figures_nature/

Uses:
  - matplotlib
  - seaborn
  - plotly (exported to PNG via kaleido)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px


CONDITION_ORDER = ["no_rag", "rag_gated", "rag_always"]
CONDITION_LABEL = {
    "no_rag": "No RAG",
    "rag_gated": "RAG (gated)",
    "rag_always": "RAG (always)",
}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Nature-grade QandA20 v2 figures (PNG).")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run folder under eval/paper_runs/<run_set_id>",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output folder (default: <run_dir>/reports/open/qanda20/figures_nature)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for bootstrapped confidence intervals.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples per group for CI.",
    )
    return p.parse_args()


def _bootstrap_ci(
    x: np.ndarray,
    *,
    seed: int,
    iters: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Return (mean, lo, hi) where lo/hi is a percentile bootstrap CI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = x.size
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        samp = rng.choice(x, size=n, replace=True)
        means[i] = float(np.mean(samp))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(np.mean(x)), lo, hi


def _model_type(model_id: str) -> str:
    return "Open" if str(model_id) in OPEN_MODEL_KEYS else "API"


def _ensure_out_dir(run_dir: Path, out_dir: Optional[str]) -> Path:
    if out_dir:
        p = Path(out_dir).expanduser().resolve()
    else:
        p = run_dir / "reports" / "open" / "qanda20" / "figures_nature"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_mpl_style() -> None:
    # Clean, paper-friendly defaults.
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def _load_inputs(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_root = run_dir / "reports" / "open" / "qanda20"
    per_sample = report_root / "per_sample.csv"
    summary = report_root / "summary_by_model_condition.csv"
    if not per_sample.exists():
        raise FileNotFoundError(f"Missing: {per_sample}")
    if not summary.exists():
        raise FileNotFoundError(f"Missing: {summary}")
    gen_df = pd.read_csv(per_sample)
    sum_df = pd.read_csv(summary)
    return gen_df, sum_df


def _model_order_by(sum_df: pd.DataFrame, *, condition: str, col: str) -> List[str]:
    d = sum_df[sum_df["condition"] == condition].copy()
    d = d[pd.notna(d[col])].copy()
    d = d.sort_values(col, ascending=False)
    return [str(x) for x in d["model_id"].tolist()]


def _label_models(sum_df: pd.DataFrame) -> Dict[str, str]:
    m = {}
    for r in sum_df.itertuples(index=False):
        mid = str(getattr(r, "model_id"))
        disp = str(getattr(r, "model_display", mid) or mid)
        m[mid] = disp
    return m


def fig01_correctness_grouped_bar(
    gen_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    iters: int,
) -> None:
    """
    Mean correctness by model + condition, with bootstrap 95% CI (OpenAI judge).
    """
    primary = "openai_correctness_score"
    df = gen_df.copy()
    df = df[pd.notna(df[primary])].copy()
    df["model_type"] = df["model_id"].astype(str).map(_model_type)

    model_labels = _label_models(sum_df)
    order = _model_order_by(sum_df, condition="rag_gated", col="openai_correctness_mean")

    rows = []
    for i, (mid, cond) in enumerate(df.groupby(["model_id", "condition"]).groups.keys()):
        x = df[(df["model_id"] == mid) & (df["condition"] == cond)][primary].to_numpy()
        mean, lo, hi = _bootstrap_ci(x, seed=seed + i, iters=iters)
        rows.append(
            {
                "model_id": str(mid),
                "model_display": model_labels.get(str(mid), str(mid)),
                "condition": str(cond),
                "mean": mean,
                "lo": lo,
                "hi": hi,
            }
        )
    stat = pd.DataFrame(rows)
    stat = stat[stat["condition"].isin(CONDITION_ORDER)].copy()
    stat["model_id"] = pd.Categorical(stat["model_id"], categories=order, ordered=True)
    stat = stat.sort_values(["model_id", "condition"])

    # Plot: horizontal grouped bars.
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    palette = {
        "no_rag": "#7a7a7a",
        "rag_gated": "#1f77b4",
        "rag_always": "#17a2a4",
    }
    bar_h = 0.23
    y = np.arange(len(order), dtype=float)
    offsets = {
        "no_rag": -bar_h,
        "rag_gated": 0.0,
        "rag_always": bar_h,
    }

    for cond in CONDITION_ORDER:
        sub = stat[stat["condition"] == cond].copy()
        sub = sub.set_index("model_id").reindex(order).reset_index()
        means = sub["mean"].to_numpy()
        lo = sub["lo"].to_numpy()
        hi = sub["hi"].to_numpy()
        xerr = np.vstack([means - lo, hi - means])
        ax.barh(
            y + offsets[cond],
            means,
            height=bar_h * 0.9,
            xerr=xerr,
            color=palette.get(cond),
            edgecolor="none",
            capsize=2,
            label=CONDITION_LABEL.get(cond, cond),
        )

    ax.set_yticks(y)
    ax.set_yticklabels([model_labels.get(m, m) for m in order])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("GEval correctness (mean, 0–1) — OpenAI judge")
    ax.set_ylabel("")
    ax.set_title("QandA20: Correctness by Model and RAG Condition")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_correctness_by_model_condition_openai.png")
    plt.close(fig)


def fig02_correctness_heatmap(
    sum_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Heatmap of mean correctness for model x condition (OpenAI judge).
    """
    df = sum_df.copy()
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    df = df[pd.notna(df["openai_correctness_mean"])].copy()

    order = _model_order_by(sum_df, condition="rag_gated", col="openai_correctness_mean")
    model_labels = _label_models(sum_df)
    df["model_id"] = pd.Categorical(df["model_id"].astype(str), categories=order, ordered=True)
    df["condition"] = pd.Categorical(df["condition"].astype(str), categories=CONDITION_ORDER, ordered=True)

    pivot = df.pivot(index="model_id", columns="condition", values="openai_correctness_mean").reindex(order)
    pivot.index = [model_labels.get(m, m) for m in pivot.index.astype(str)]
    pivot = pivot.rename(columns=CONDITION_LABEL)

    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Correctness (mean)"},
    )
    ax.set_title("QandA20: Mean Correctness (OpenAI judge)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_correctness_heatmap_openai.png")
    plt.close(fig)


def fig03_rag_lift(
    gen_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    iters: int,
) -> None:
    """
    RAG lift per model (delta correctness vs no_rag), with CI.
    """
    primary = "openai_correctness_score"
    df = gen_df.copy()
    df = df[df["condition"].isin(["no_rag", "rag_gated", "rag_always"])].copy()
    df = df[pd.notna(df[primary])].copy()

    # Pivot per (sample_id, model): compute deltas.
    piv = (
        df.pivot_table(
            index=["sample_id", "model_id", "model_display"],
            columns="condition",
            values=primary,
            aggfunc="first",
        )
        .reset_index()
    )

    # Flatten columns
    piv.columns = [c if isinstance(c, str) else str(c) for c in piv.columns]
    for col in ["no_rag", "rag_gated", "rag_always"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["delta_gated"] = piv["rag_gated"] - piv["no_rag"]
    piv["delta_always"] = piv["rag_always"] - piv["no_rag"]

    order = _model_order_by(sum_df, condition="rag_gated", col="openai_correctness_mean")
    model_labels = _label_models(sum_df)

    rows = []
    for i, mid in enumerate(order):
        sub = piv[piv["model_id"].astype(str) == mid]
        for j, (name, col) in enumerate([("RAG (gated)", "delta_gated"), ("RAG (always)", "delta_always")]):
            x = sub[col].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_ci(x, seed=seed + i * 10 + j, iters=iters)
            rows.append(
                {
                    "model_id": mid,
                    "model_display": model_labels.get(mid, mid),
                    "series": name,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                }
            )
    stat = pd.DataFrame(rows)
    stat["model_id"] = pd.Categorical(stat["model_id"], categories=order, ordered=True)
    stat = stat.sort_values(["model_id", "series"])

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(order), dtype=float)
    palette = {"RAG (gated)": "#1f77b4", "RAG (always)": "#17a2a4"}
    h = 0.18
    offsets = {"RAG (gated)": -h, "RAG (always)": h}
    for series in ["RAG (gated)", "RAG (always)"]:
        sub = stat[stat["series"] == series].set_index("model_id").reindex(order).reset_index()
        means = sub["mean"].to_numpy()
        lo = sub["lo"].to_numpy()
        hi = sub["hi"].to_numpy()
        xerr = np.vstack([means - lo, hi - means])
        ax.errorbar(
            means,
            y + offsets[series],
            xerr=xerr,
            fmt="o",
            ms=5,
            color=palette[series],
            ecolor=palette[series],
            capsize=2,
            label=series,
        )

    ax.axvline(0.0, color="#333333", lw=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([model_labels.get(m, m) for m in order])
    ax.set_xlabel("RAG lift in correctness (delta vs no RAG)")
    ax.set_ylabel("")
    ax.set_title("QandA20: RAG Lift by Model (OpenAI judge)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_rag_lift_by_model_openai.png")
    plt.close(fig)


def fig04_pass_partial_fail(
    gen_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    out_dir: Path,
    *,
    condition: str = "rag_gated",
) -> None:
    """
    PASS/PARTIAL/FAIL stacked bars for one condition (default: rag_gated).
    """
    df = gen_df.copy()
    df = df[df["condition"] == condition].copy()
    if "grade_primary" not in df.columns:
        raise ValueError("per_sample.csv missing grade_primary; regenerate the report first.")

    order = _model_order_by(sum_df, condition="rag_gated", col="openai_correctness_mean")
    model_labels = _label_models(sum_df)

    counts = (
        df.groupby(["model_id", "grade_primary"])["sample_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(order)
    )
    for k in ["PASS", "PARTIAL", "FAIL"]:
        if k not in counts.columns:
            counts[k] = 0
    counts = counts[["PASS", "PARTIAL", "FAIL"]]
    rates = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(order), dtype=float)
    left = np.zeros(len(order), dtype=float)
    colors = {"PASS": "#2ca02c", "PARTIAL": "#ffbf00", "FAIL": "#d62728"}
    for k in ["PASS", "PARTIAL", "FAIL"]:
        vals = rates[k].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=colors[k], edgecolor="none", label=k)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([model_labels.get(m, m) for m in order])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Rate")
    ax.set_ylabel("")
    ax.set_title(f"QandA20: PASS/PARTIAL/FAIL (Primary policy) — {CONDITION_LABEL.get(condition, condition)}")
    ax.legend(loc="lower right", frameon=True, ncols=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_pass_partial_fail_rag_gated.png")
    plt.close(fig)


def fig05_ctx_recall_vs_correctness(
    gen_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Retrieval diagnostic: contextual recall vs correctness (rag conditions only).
    """
    df = gen_df.copy()
    df = df[df["condition"].isin(["rag_gated", "rag_always"])].copy()
    df = df[df["rag_context_used"].fillna(False).astype(bool)].copy()
    df = df[pd.notna(df["openai_ctx_recall_score"]) & pd.notna(df["openai_correctness_score"])].copy()
    df["model_type"] = df["model_id"].astype(str).map(_model_type)
    df["condition_label"] = df["condition"].map(CONDITION_LABEL)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=df,
        x="openai_ctx_recall_score",
        y="openai_correctness_score",
        hue="condition_label",
        style="model_type",
        alpha=0.35,
        s=28,
        ax=ax,
        palette={"RAG (gated)": "#1f77b4", "RAG (always)": "#17a2a4"},
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Contextual recall (OpenAI judge)")
    ax.set_ylabel("Correctness (OpenAI judge)")
    ax.set_title("QandA20: Retrieval Bottleneck (Recall vs Correctness)")
    ax.legend(title="", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_ctx_recall_vs_correctness_scatter.png")
    plt.close(fig)


def fig06_judge_agreement_plotly(
    gen_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Judge agreement: OpenAI vs Anthropic correctness (Plotly), exported to PNG.
    """
    df = gen_df.copy()
    df = df[pd.notna(df["openai_correctness_score"]) & pd.notna(df["anthropic_correctness_score"])].copy()
    df["condition_label"] = df["condition"].map(CONDITION_LABEL)
    df["model_type"] = df["model_id"].astype(str).map(_model_type)

    x = df["openai_correctness_score"].to_numpy(dtype=float)
    y = df["anthropic_correctness_score"].to_numpy(dtype=float)
    if x.size:
        pear = float(np.corrcoef(x, y)[0, 1])
    else:
        pear = float("nan")

    fig = px.scatter(
        df,
        x="openai_correctness_score",
        y="anthropic_correctness_score",
        color="condition_label",
        symbol="model_type",
        opacity=0.5,
        title=f"Judge Agreement: Correctness (Pearson r={pear:.2f})",
        labels={
            "openai_correctness_score": "OpenAI correctness (0–1)",
            "anthropic_correctness_score": "Claude correctness (0–1)",
            "condition_label": "",
            "model_type": "",
        },
        width=850,
        height=650,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=1))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))

    # Save PNG + HTML (HTML is useful for inspection; PNG is paper-ready).
    fig.write_image(out_dir / "fig06_judge_agreement_openai_vs_claude.png", scale=3)
    fig.write_html(out_dir / "fig06_judge_agreement_openai_vs_claude.html")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = _ensure_out_dir(run_dir, args.out_dir)

    _set_mpl_style()
    gen_df, sum_df = _load_inputs(run_dir)

    fig01_correctness_grouped_bar(gen_df, sum_df, out_dir, seed=int(args.seed), iters=int(args.bootstrap))
    fig02_correctness_heatmap(sum_df, out_dir)
    fig03_rag_lift(gen_df, sum_df, out_dir, seed=int(args.seed), iters=int(args.bootstrap))
    fig04_pass_partial_fail(gen_df, sum_df, out_dir, condition="rag_gated")
    fig05_ctx_recall_vs_correctness(gen_df, out_dir)
    fig06_judge_agreement_plotly(gen_df, out_dir)

    print("Wrote figures to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

