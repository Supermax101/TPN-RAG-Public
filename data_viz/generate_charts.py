"""
TPN-RAG Benchmark Visualization Suite
======================================
Generates publication-quality charts from MCQ benchmark results.
Separate charts for API (SOTA) models and Open-Source (HuggingFace) models.

Usage:
    python data_viz/generate_charts.py

Output:
    data_viz/figures/*.png   — static images for papers
    data_viz/figures/*.html  — interactive Plotly charts
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Configuration ───────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical result files (124 MCQ holdout, best config per model)
# NOTE: Claude Sonnet 4.5 excluded — CoT responses truncated, parser cannot
#       extract answers (47/124 empty on CoT). See CLAUDE_SONNET_DATA_ISSUE.md
CANONICAL_RUNS = {
    # API Models
    "gemini-3-flash": RESULTS_DIR / "benchmark_api_gemini3flash_mcq124_v2" / "gemini-3-flash" / "accuracy_20260207_144137.csv",
    "gpt-5.2":        RESULTS_DIR / "benchmark_api_remaining_mcq124" / "gpt-5.2" / "accuracy_20260207_080046.csv",
    # "claude-sonnet":  RESULTS_DIR / "benchmark_api_remaining_mcq124" / "claude-sonnet" / "accuracy_20260207_080046.csv",  # EXCLUDED — parser bug, see CLAUDE_SONNET_DATA_ISSUE.md
    "grok-4.1-fast":  RESULTS_DIR / "benchmark_api_remaining_mcq124" / "grok-4.1-fast" / "accuracy_20260207_080046.csv",
    "kimi-k2.5":      RESULTS_DIR / "benchmark_api_remaining_mcq124" / "kimi-k2.5" / "accuracy_20260207_080046.csv",
    # HuggingFace Open Models
    "phi-4":          RESULTS_DIR / "benchmark_hf_phi4_mcq124_zs_few_cot_rap_gated_kb_books" / "phi-4" / "accuracy_20260207_020552.csv",
    "gpt-oss-20b":    RESULTS_DIR / "benchmark_hf_glm_gptoss_mcq124" / "gpt-oss-20b" / "accuracy_20260207_081516.csv",
    "qwen3-30b-a3b":  RESULTS_DIR / "benchmark_hf_qwen3_30b_mcq124" / "qwen3-30b-a3b" / "accuracy_20260207_130900.csv",
}

# gpt-5-mini split across two runs
GPT5MINI_RUNS = [
    RESULTS_DIR / "benchmark_gpt5mini_mcq124_zs_gated_kb_books" / "accuracy_20260206_232214.csv",
    RESULTS_DIR / "benchmark_gpt5mini_mcq124_few_cot_rap_gated_kb_books" / "accuracy_20260207_004109.csv",
]

# Model display names
MODEL_DISPLAY = {
    "gemini-3-flash": "Gemini 3 Flash",
    "gpt-5.2":        "GPT-5.2",
    # "claude-sonnet":  "Claude Sonnet 4.5",  # EXCLUDED
    "grok-4.1-fast":  "Grok 4.1 Fast",
    "kimi-k2.5":      "Kimi K2.5",
    "gpt-5-mini":     "GPT-5 Mini",
    "phi-4":          "Phi-4 (14B)",
    "gpt-oss-20b":    "GPT-OSS (20B)",
    "qwen3-30b-a3b":  "Qwen3-30B-A3B",
}

MODEL_TIER = {
    "gemini-3-flash": "API",
    "gpt-5.2":        "API",
    # "claude-sonnet":  "API",  # EXCLUDED
    "grok-4.1-fast":  "API",
    "gpt-5-mini":     "API",
    "kimi-k2.5":      "Open",
    "phi-4":          "Open",
    "gpt-oss-20b":    "Open",
    "qwen3-30b-a3b":  "Open",
}

# Model ordering within each tier (by expected quality)
API_MODEL_ORDER = ["gemini-3-flash", "grok-4.1-fast", "gpt-5.2", "gpt-5-mini"]  # claude-sonnet excluded
OPEN_MODEL_ORDER = ["kimi-k2.5", "qwen3-30b-a3b", "phi-4", "gpt-oss-20b"]
ALL_MODEL_ORDER = API_MODEL_ORDER + OPEN_MODEL_ORDER

STRATEGY_DISPLAY = {
    "ZS": "Zero-Shot",
    "FEW_SHOT": "Few-Shot",
    "COT": "Chain-of-Thought",
    "COT_SC": "CoT-SC",
    "RAP": "RAP",
}

CORE_STRATEGIES = ["ZS", "FEW_SHOT", "COT"]

# Colors — Nature Publishing Group (NPG) colorblind-friendly palette
COLOR_NO_RAG = "#8491B4"   # NPG steel-blue (muted baseline)
COLOR_RAG = "#3C5488"      # NPG dark-blue (RAG-enhanced)
COLOR_LIFT_POS = "#00A087" # NPG teal (positive)
COLOR_LIFT_NEG = "#E64B35" # NPG vermillion (negative)

# Per-model accent colors — NPG palette (colorblind-safe)
API_COLORS = {
    "gemini-3-flash": "#3C5488",  # NPG dark blue
    "gpt-5.2":        "#00A087",  # NPG teal
    # "claude-sonnet":  "#F39B7F",  # NPG salmon — EXCLUDED
    "grok-4.1-fast":  "#4DBBD5",  # NPG cyan
    "gpt-5-mini":     "#91D1C2",  # NPG mint
}
OPEN_COLORS = {
    "kimi-k2.5":      "#E64B35",  # NPG vermillion
    "phi-4":          "#8491B4",  # NPG steel blue
    "gpt-oss-20b":    "#7E6148",  # NPG brown
    "qwen3-30b-a3b":  "#B09C85",  # NPG sand
}
ALL_COLORS = {**API_COLORS, **OPEN_COLORS}

# Legend always at bottom, title always at top with enough room
LEGEND_BOTTOM = dict(
    orientation="h",
    yanchor="top",
    y=-0.15,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#e2e8f0",
    borderwidth=1,
)

LAYOUT_THEME = dict(
    font=dict(family="Inter, Arial, sans-serif", size=13),
    plot_bgcolor="#fafbfc",
    paper_bgcolor="#ffffff",
    margin=dict(l=60, r=30, t=100, b=80),
)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    """Load all canonical accuracy CSVs into a unified DataFrame."""
    frames = []

    for model_id, csv_path in CANONICAL_RUNS.items():
        if not csv_path.exists():
            print(f"  ⚠️  Missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["model_id"] = model_id
        frames.append(df)

    for csv_path in GPT5MINI_RUNS:
        if not csv_path.exists():
            print(f"  ⚠️  Missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["model_id"] = "gpt-5-mini"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["model_display"] = combined["model_id"].map(MODEL_DISPLAY)
    combined["model_tier"] = combined["model_id"].map(MODEL_TIER)
    combined["strategy_display"] = combined["strategy"].map(STRATEGY_DISPLAY)
    combined["accuracy_pct"] = combined["accuracy"] * 100
    return combined


# ─── Chart: Per-Model RAG vs No-RAG (3 strategies side by side) ─────────────

def _per_model_rag_chart(df_model: pd.DataFrame, model_id: str, model_display: str, accent_color: str):
    """
    Create a single model's RAG vs No-RAG chart with 3 grouped bar pairs
    (ZS, FEW_SHOT, COT). Each pair has No-RAG (grey) and +RAG (blue) bar.
    Returns a Figure.
    """
    core = df_model[df_model["strategy"].isin(CORE_STRATEGIES)].copy()

    strategies = [STRATEGY_DISPLAY[s] for s in CORE_STRATEGIES]

    no_rag_vals = []
    rag_vals = []
    lifts = []

    for s in CORE_STRATEGIES:
        sd = STRATEGY_DISPLAY[s]
        nr = core[(core["strategy"] == s) & (core["rag_mode"] == "no_rag")]
        r = core[(core["strategy"] == s) & (core["rag_mode"] == "rag")]
        nr_val = nr["accuracy_pct"].values[0] if len(nr) > 0 else 0
        r_val = r["accuracy_pct"].values[0] if len(r) > 0 else 0
        no_rag_vals.append(nr_val)
        rag_vals.append(r_val)
        lifts.append(r_val - nr_val)

    fig = go.Figure()

    # No RAG bars — accuracy value inside the bar
    fig.add_trace(go.Bar(
        name="No RAG",
        x=strategies,
        y=no_rag_vals,
        marker_color=COLOR_NO_RAG,
        text=[f"{v:.1f}%" for v in no_rag_vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=12, color="white"),
        width=0.3,
    ))

    # RAG bars — accuracy value inside the bar
    fig.add_trace(go.Bar(
        name="+ RAG",
        x=strategies,
        y=rag_vals,
        marker_color=accent_color,
        text=[f"{v:.1f}%" for v in rag_vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=12, color="white"),
        width=0.3,
    ))

    # Add lift annotations above the taller bar in each pair
    for i, (s, lift) in enumerate(zip(strategies, lifts)):
        color = COLOR_LIFT_POS if lift >= 0 else COLOR_LIFT_NEG
        sign = "+" if lift >= 0 else ""
        fig.add_annotation(
            x=s,
            y=max(no_rag_vals[i], rag_vals[i]) + 3,
            text=f"<b>{sign}{lift:.1f}pp</b>",
            showarrow=False,
            font=dict(size=14, color=color),
        )

    best_acc = max(max(rag_vals), max(no_rag_vals))
    best_strategy = strategies[rag_vals.index(max(rag_vals))]

    fig.update_layout(
        title=dict(
            text=f"<b>{model_display}</b><br>"
                 f"<sup>Best: {max(rag_vals):.1f}% ({best_strategy} + RAG) · "
                 f"124 MCQ holdout questions</sup>",
            x=0.5, xanchor="center",
        ),
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 115], dtick=10),
        xaxis_title="",
        barmode="group",
        bargap=0.25,
        height=420,
        width=550,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    return fig


def chart_per_model_api(df: pd.DataFrame):
    """Generate individual RAG vs No-RAG charts for each API model."""
    api_df = df[df["model_tier"] == "API"]

    for model_id in API_MODEL_ORDER:
        model_df = api_df[api_df["model_id"] == model_id]
        if model_df.empty:
            continue
        display = MODEL_DISPLAY[model_id]
        color = API_COLORS.get(model_id, "#3b82f6")
        fig = _per_model_rag_chart(model_df, model_id, display, color)

        safe_name = model_id.replace(".", "").replace("-", "_")
        fig.write_html(OUTPUT_DIR / f"api_{safe_name}_rag.html")
        fig.write_image(OUTPUT_DIR / f"api_{safe_name}_rag.png", scale=3)
        print(f"  ✅ API Model: {display}")


def chart_per_model_open(df: pd.DataFrame):
    """Generate individual RAG vs No-RAG charts for each Open model."""
    open_df = df[df["model_tier"] == "Open"]

    for model_id in OPEN_MODEL_ORDER:
        model_df = open_df[open_df["model_id"] == model_id]
        if model_df.empty:
            continue
        display = MODEL_DISPLAY[model_id]
        color = OPEN_COLORS.get(model_id, "#a855f7")
        fig = _per_model_rag_chart(model_df, model_id, display, color)

        safe_name = model_id.replace(".", "").replace("-", "_")
        fig.write_html(OUTPUT_DIR / f"open_{safe_name}_rag.html")
        fig.write_image(OUTPUT_DIR / f"open_{safe_name}_rag.png", scale=3)
        print(f"  ✅ Open Model: {display}")


# ─── Chart: API Models Combined Grid ────────────────────────────────────────

def chart_api_grid(df: pd.DataFrame):
    """2×3 subplot grid of all API models, each with RAG vs No-RAG × 3 strategies."""
    api_df = df[(df["model_tier"] == "API") & (df["strategy"].isin(CORE_STRATEGIES))]
    models = [m for m in API_MODEL_ORDER if m in api_df["model_id"].unique()]

    rows = 2
    cols = 3
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[MODEL_DISPLAY[m] for m in models],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    strategies = [STRATEGY_DISPLAY[s] for s in CORE_STRATEGIES]

    for idx, model_id in enumerate(models):
        r = (idx // cols) + 1
        c = (idx % cols) + 1
        m_df = api_df[api_df["model_id"] == model_id]
        color = API_COLORS.get(model_id, "#3b82f6")

        no_rag_vals = []
        rag_vals = []
        for s in CORE_STRATEGIES:
            nr = m_df[(m_df["strategy"] == s) & (m_df["rag_mode"] == "no_rag")]
            ra = m_df[(m_df["strategy"] == s) & (m_df["rag_mode"] == "rag")]
            no_rag_vals.append(nr["accuracy_pct"].values[0] if len(nr) > 0 else 0)
            rag_vals.append(ra["accuracy_pct"].values[0] if len(ra) > 0 else 0)

        show_legend = idx == 0

        fig.add_trace(go.Bar(
            name="No RAG", x=strategies, y=no_rag_vals,
            marker_color=COLOR_NO_RAG,
            text=[f"{v:.0f}" for v in no_rag_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(size=9, color="white"),
            showlegend=show_legend, legendgroup="no_rag",
            width=0.3,
        ), row=r, col=c)

        fig.add_trace(go.Bar(
            name="+ RAG", x=strategies, y=rag_vals,
            marker_color=color,
            text=[f"{v:.0f}" for v in rag_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(size=9, color="white"),
            showlegend=show_legend, legendgroup="rag",
            width=0.3,
        ), row=r, col=c)

        # Add RAG lift annotations
        for si, (strat, nr_v, r_v) in enumerate(zip(strategies, no_rag_vals, rag_vals)):
            lift = r_v - nr_v
            color_lift = COLOR_LIFT_POS if lift >= 0 else COLOR_LIFT_NEG
            sign = "+" if lift >= 0 else ""
            # Determine correct axis reference for this subplot
            axis_idx = (r - 1) * cols + c
            xref = "x" if axis_idx == 1 else f"x{axis_idx}"
            yref = "y" if axis_idx == 1 else f"y{axis_idx}"
            fig.add_annotation(
                x=strat, y=max(nr_v, r_v) + 5,
                xref=xref, yref=yref,
                text=f"<b>{sign}{lift:.1f}pp</b>",
                showarrow=False,
                font=dict(size=9, color=color_lift),
            )

        fig.update_yaxes(range=[0, 110], dtick=20, row=r, col=c)

    fig.update_layout(
        title=dict(
            text="<b>API Models: RAG vs No-RAG by Prompting Strategy</b><br>"
                 "<sup>124 RD Exam MCQs · TPN Clinical Decision Support Benchmark</sup>",
            x=0.5, xanchor="center",
        ),
        barmode="group",
        height=700,
        width=1100,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "api_all_models_grid.html")
    fig.write_image(OUTPUT_DIR / "api_all_models_grid.png", scale=3)
    print("  ✅ API Models Grid (2×3)")


def chart_open_grid(df: pd.DataFrame):
    """1×3 subplot grid of all Open models, each with RAG vs No-RAG × 3 strategies."""
    open_df = df[(df["model_tier"] == "Open") & (df["strategy"].isin(CORE_STRATEGIES))]
    models = [m for m in OPEN_MODEL_ORDER if m in open_df["model_id"].unique()]

    fig = make_subplots(
        rows=1, cols=len(models),
        subplot_titles=[MODEL_DISPLAY[m] for m in models],
        horizontal_spacing=0.08,
    )

    strategies = [STRATEGY_DISPLAY[s] for s in CORE_STRATEGIES]

    for idx, model_id in enumerate(models):
        c = idx + 1
        m_df = open_df[open_df["model_id"] == model_id]
        color = OPEN_COLORS.get(model_id, "#a855f7")

        no_rag_vals = []
        rag_vals = []
        for s in CORE_STRATEGIES:
            nr = m_df[(m_df["strategy"] == s) & (m_df["rag_mode"] == "no_rag")]
            ra = m_df[(m_df["strategy"] == s) & (m_df["rag_mode"] == "rag")]
            no_rag_vals.append(nr["accuracy_pct"].values[0] if len(nr) > 0 else 0)
            rag_vals.append(ra["accuracy_pct"].values[0] if len(ra) > 0 else 0)

        show_legend = idx == 0

        fig.add_trace(go.Bar(
            name="No RAG", x=strategies, y=no_rag_vals,
            marker_color=COLOR_NO_RAG,
            text=[f"{v:.0f}" for v in no_rag_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(size=9, color="white"),
            showlegend=show_legend, legendgroup="no_rag",
            width=0.3,
        ), row=1, col=c)

        fig.add_trace(go.Bar(
            name="+ RAG", x=strategies, y=rag_vals,
            marker_color=color,
            text=[f"{v:.0f}" for v in rag_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(size=9, color="white"),
            showlegend=show_legend, legendgroup="rag",
            width=0.3,
        ), row=1, col=c)

        # Add RAG lift annotations
        for si, (strat, nr_v, r_v) in enumerate(zip(strategies, no_rag_vals, rag_vals)):
            lift = r_v - nr_v
            color_lift = COLOR_LIFT_POS if lift >= 0 else COLOR_LIFT_NEG
            sign = "+" if lift >= 0 else ""
            xref = "x" if c == 1 else f"x{c}"
            yref = "y" if c == 1 else f"y{c}"
            fig.add_annotation(
                x=strat, y=max(nr_v, r_v) + 5,
                xref=xref, yref=yref,
                text=f"<b>{sign}{lift:.1f}pp</b>",
                showarrow=False,
                font=dict(size=9, color=color_lift),
            )

        fig.update_yaxes(range=[0, 110], dtick=20, row=1, col=c)

    fig.update_layout(
        title=dict(
            text="<b>Open-Source Models: RAG vs No-RAG by Prompting Strategy</b><br>"
                 "<sup>124 RD Exam MCQs · TPN Clinical Decision Support Benchmark</sup>",
            x=0.5, xanchor="center",
        ),
        barmode="group",
        height=450,
        width=1100,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "open_all_models_grid.html")
    fig.write_image(OUTPUT_DIR / "open_all_models_grid.png", scale=3)
    print("  ✅ Open Models Grid (1×3)")


# ─── Chart: Accuracy Heatmap (separate API / Open) ──────────────────────────

def _heatmap_for_tier(df: pd.DataFrame, tier: str, model_order: list, title_prefix: str, filename: str):
    """Generate accuracy heatmap for a single tier."""
    core = df[(df["model_tier"] == tier) & (df["strategy"].isin(CORE_STRATEGIES))].copy()
    core["condition"] = core["strategy_display"] + "<br>" + core["rag_mode"].map(
        {"no_rag": "No RAG", "rag": "+ RAG"}
    )

    col_order = []
    for s in CORE_STRATEGIES:
        sd = STRATEGY_DISPLAY[s]
        col_order.append(f"{sd}<br>No RAG")
        col_order.append(f"{sd}<br>+ RAG")

    # Use specified order, reversed for bottom-up display
    row_order = [MODEL_DISPLAY[m] for m in model_order if m in core["model_id"].unique()]
    row_order = row_order[::-1]  # best at top

    pivot = core.pivot_table(
        index="model_display", columns="condition",
        values="accuracy_pct", aggfunc="first"
    )
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=[[f"{v:.1f}%" if pd.notna(v) else "—" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale=[
            [0, "#fee2e2"],
            [0.25, "#fef3c7"],
            [0.45, "#d1fae5"],
            [0.65, "#6ee7b7"],
            [0.80, "#34d399"],
            [1, "#059669"],
        ],
        zmin=20,
        zmax=100,
        colorbar=dict(title="Accuracy %", ticksuffix="%"),
        hoverongaps=False,
    ))

    n_models = len(row_order)
    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}: Accuracy Heatmap</b><br>"
                 "<sup>Model × Prompting Strategy × RAG | 124 RD Exam MCQs</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Prompting Strategy",
        yaxis_title="",
        height=max(350, 80 * n_models + 120),
        width=900,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / f"{filename}.html")
    fig.write_image(OUTPUT_DIR / f"{filename}.png", scale=3)


def chart_heatmaps(df: pd.DataFrame):
    """Separate heatmaps for API and Open models."""
    _heatmap_for_tier(df, "API", API_MODEL_ORDER, "API (SOTA) Models", "heatmap_api")
    print("  ✅ Heatmap: API Models")
    _heatmap_for_tier(df, "Open", OPEN_MODEL_ORDER, "Open-Source Models", "heatmap_open")
    print("  ✅ Heatmap: Open Models")


# ─── Chart: RAG Lift (separate tiers) ───────────────────────────────────────

def _rag_lift_chart(df: pd.DataFrame, tier: str, model_order: list, title_prefix: str, filename: str):
    """RAG lift bar chart for a single tier."""
    core = df[(df["model_tier"] == tier) & (df["strategy"].isin(CORE_STRATEGIES))].copy()

    pivot = core.pivot_table(
        index=["model_display", "model_id", "strategy_display", "strategy"],
        columns="rag_mode", values="accuracy_pct", aggfunc="first"
    ).reset_index()

    if "no_rag" not in pivot.columns or "rag" not in pivot.columns:
        return

    pivot["lift"] = pivot["rag"] - pivot["no_rag"]

    # Sort by model order then strategy
    strat_order = {s: i for i, s in enumerate(CORE_STRATEGIES)}
    model_idx = {m: i for i, m in enumerate(model_order)}
    pivot["model_sort"] = pivot["model_id"].map(model_idx)
    pivot["strat_sort"] = pivot["strategy"].map(strat_order)
    pivot = pivot.sort_values(["model_sort", "strat_sort"], ascending=[False, True])

    pivot["label"] = pivot["model_display"] + " · " + pivot["strategy_display"]

    colors = [COLOR_LIFT_POS if v >= 0 else COLOR_LIFT_NEG for v in pivot["lift"]]

    fig = go.Figure(go.Bar(
        y=pivot["label"],
        x=pivot["lift"],
        orientation="h",
        marker_color=colors,
        text=[f"+{v:.1f}pp" if v >= 0 else f"{v:.1f}pp" for v in pivot["lift"]],
        textposition="outside",
        textfont=dict(size=12),
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#1e293b", line_width=1.5)

    # Add average lift annotation
    avg_lift = pivot["lift"].mean()
    fig.add_annotation(
        x=avg_lift, y=1.05, yref="paper",
        text=f"<b>Avg RAG lift: {avg_lift:+.1f}pp</b>",
        showarrow=False,
        font=dict(size=14, color=COLOR_LIFT_POS if avg_lift >= 0 else COLOR_LIFT_NEG),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}: RAG Lift (Accuracy Improvement)</b><br>"
                 "<sup>Percentage point change when RAG context is added</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="RAG Lift (percentage points)",
        yaxis_title="",
        height=max(400, len(pivot) * 32 + 120),
        width=900,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / f"{filename}.html")
    fig.write_image(OUTPUT_DIR / f"{filename}.png", scale=3)


def chart_rag_lifts(df: pd.DataFrame):
    """Separate RAG lift charts for API and Open models."""
    _rag_lift_chart(df, "API", API_MODEL_ORDER, "API (SOTA) Models", "lift_api")
    print("  ✅ RAG Lift: API Models")
    _rag_lift_chart(df, "Open", OPEN_MODEL_ORDER, "Open-Source Models", "lift_open")
    print("  ✅ RAG Lift: Open Models")


# ─── Chart: Leaderboard (separate tiers) ────────────────────────────────────

def _leaderboard_chart(df: pd.DataFrame, tier: str, model_order: list, tier_color: str, title_prefix: str, filename: str):
    """Best-accuracy leaderboard for a single tier."""
    tier_df = df[df["model_tier"] == tier]
    best = tier_df.loc[tier_df.groupby("model_id")["accuracy_pct"].idxmax()].copy()
    best["config_label"] = best["strategy_display"] + " " + best["rag_mode"].map(
        {"no_rag": "(No RAG)", "rag": "(+ RAG)"}
    )

    # Sort by accuracy ascending (for horizontal bar, bottom = lowest)
    best = best.sort_values("accuracy_pct", ascending=True)

    fig = go.Figure(go.Bar(
        y=best["model_display"],
        x=best["accuracy_pct"],
        orientation="h",
        marker_color=tier_color,
        text=[f"  {v:.1f}% — {c}" for v, c in zip(best["accuracy_pct"], best["config_label"])],
        textposition="inside",
        textfont=dict(color="white", size=13),
        insidetextanchor="start",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}: Leaderboard</b><br>"
                 "<sup>Best accuracy across all strategy × RAG configurations</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Accuracy (%)",
        xaxis=dict(range=[0, 100], dtick=10),
        yaxis_title="",
        height=max(300, len(best) * 55 + 120),
        width=800,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / f"{filename}.html")
    fig.write_image(OUTPUT_DIR / f"{filename}.png", scale=3)


def chart_leaderboards(df: pd.DataFrame):
    """Separate leaderboards for API and Open models."""
    _leaderboard_chart(df, "API", API_MODEL_ORDER, "#3b82f6", "API (SOTA) Models", "leaderboard_api")
    print("  ✅ Leaderboard: API Models")
    _leaderboard_chart(df, "Open", OPEN_MODEL_ORDER, "#a855f7", "Open-Source Models", "leaderboard_open")
    print("  ✅ Leaderboard: Open Models")


# ─── Chart: Strategy Comparison (separate tiers) ────────────────────────────

def _strategy_chart(df: pd.DataFrame, tier: str, title_prefix: str, filename: str):
    """Average accuracy by strategy, grouped No-RAG vs +RAG, for a single tier."""
    core = df[(df["model_tier"] == tier) & (df["strategy"].isin(CORE_STRATEGIES))].copy()
    avg = core.groupby(["strategy_display", "strategy", "rag_mode"])["accuracy_pct"].mean().reset_index()

    strategies = [STRATEGY_DISPLAY[s] for s in CORE_STRATEGIES]
    no_rag = avg[avg["rag_mode"] == "no_rag"].set_index("strategy_display")
    rag = avg[avg["rag_mode"] == "rag"].set_index("strategy_display")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="No RAG",
        x=strategies,
        y=[no_rag.loc[s, "accuracy_pct"] if s in no_rag.index else 0 for s in strategies],
        marker_color=COLOR_NO_RAG,
        text=[f"{no_rag.loc[s, 'accuracy_pct']:.1f}%" if s in no_rag.index else "" for s in strategies],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=13, color="white"),
        width=0.3,
    ))

    fig.add_trace(go.Bar(
        name="+ RAG",
        x=strategies,
        y=[rag.loc[s, "accuracy_pct"] if s in rag.index else 0 for s in strategies],
        marker_color=COLOR_RAG,
        text=[f"{rag.loc[s, 'accuracy_pct']:.1f}%" if s in rag.index else "" for s in strategies],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=13, color="white"),
        width=0.3,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}: Strategy Effectiveness</b><br>"
                 "<sup>Average accuracy across models, with and without RAG</sup>",
            x=0.5, xanchor="center",
        ),
        yaxis_title="Average Accuracy (%)",
        yaxis=dict(range=[0, 105], dtick=10),
        xaxis_title="Prompting Strategy",
        barmode="group",
        bargap=0.3,
        height=480,
        width=650,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / f"{filename}.html")
    fig.write_image(OUTPUT_DIR / f"{filename}.png", scale=3)


def chart_strategies(df: pd.DataFrame):
    """Separate strategy charts for API and Open models."""
    _strategy_chart(df, "API", "API (SOTA) Models", "strategy_api")
    print("  ✅ Strategy: API Models")
    _strategy_chart(df, "Open", "Open-Source Models", "strategy_open")
    print("  ✅ Strategy: Open Models")


# ─── Chart: Latency vs Accuracy (separate tiers) ────────────────────────────

def _latency_chart(df: pd.DataFrame, tier: str, colors: dict, title_prefix: str, filename: str):
    """Scatter: latency vs accuracy, one color per model."""
    core = df[(df["model_tier"] == tier) & (df["strategy"].isin(CORE_STRATEGIES))].copy()
    core["latency_s"] = core["latency_ms_mean"] / 1000

    fig = go.Figure()

    for model_id in sorted(core["model_id"].unique()):
        subset = core[core["model_id"] == model_id]
        fig.add_trace(go.Scatter(
            x=subset["latency_s"],
            y=subset["accuracy_pct"],
            mode="markers",
            name=MODEL_DISPLAY.get(model_id, model_id),
            marker=dict(
                size=16,
                color=colors.get(model_id, "#666"),
                line=dict(width=1.5, color="white"),
                opacity=0.9,
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Accuracy: %{y:.1f}%<br>"
                "Latency: %{x:.1f}s<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title_prefix}: Accuracy vs Latency</b><br>"
                 "<sup>Each point = one strategy × RAG configuration</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Mean Latency (seconds)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[20, 100], dtick=10),
        height=500,
        width=800,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / f"{filename}.html")
    fig.write_image(OUTPUT_DIR / f"{filename}.png", scale=3)


def chart_latency(df: pd.DataFrame):
    """Separate latency charts for API and Open models."""
    _latency_chart(df, "API", API_COLORS, "API (SOTA) Models", "latency_api")
    print("  ✅ Latency: API Models")
    _latency_chart(df, "Open", OPEN_COLORS, "Open-Source Models", "latency_open")
    print("  ✅ Latency: Open Models")


# ─── Chart: 10 Publication Charts (all models combined) ─────────────────────

def chart_01_heatmap(df: pd.DataFrame):
    """Chart 1: Combined accuracy heatmap — all 9 models."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()
    core["condition"] = core["strategy_display"] + "<br>" + core["rag_mode"].map(
        {"no_rag": "No RAG", "rag": "+ RAG"}
    )

    col_order = []
    for s in CORE_STRATEGIES:
        sd = STRATEGY_DISPLAY[s]
        col_order.append(f"{sd}<br>No RAG")
        col_order.append(f"{sd}<br>+ RAG")

    row_order = [MODEL_DISPLAY[m] for m in ALL_MODEL_ORDER if m in core["model_id"].unique()]
    row_order = row_order[::-1]  # best at top

    pivot = core.pivot_table(
        index="model_display", columns="condition",
        values="accuracy_pct", aggfunc="first"
    )
    pivot = pivot.reindex(index=row_order, columns=col_order)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=[[f"{v:.1f}%" if pd.notna(v) else "—" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[
            [0, "#fee2e2"], [0.30, "#fef3c7"], [0.50, "#d1fae5"],
            [0.70, "#6ee7b7"], [0.85, "#34d399"], [1, "#059669"],
        ],
        zmin=20, zmax=100,
        colorbar=dict(title="Accuracy %", ticksuffix="%"),
        hoverongaps=False,
    ))

    n_models = len(row_order)
    fig.update_layout(
        title=dict(
            text="<b>MCQ Accuracy: Model × Prompting Strategy × RAG</b><br>"
                 "<sup>124 RD Exam Questions · TPN Clinical Decision Support Benchmark</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Prompting Strategy",
        yaxis_title="",
        height=max(400, 60 * n_models + 140),
        width=950,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "01_accuracy_heatmap.html")
    fig.write_image(OUTPUT_DIR / "01_accuracy_heatmap.png", scale=3)
    print("  ✅ Chart 1: Accuracy Heatmap (all models)")


def chart_02_rag_impact(df: pd.DataFrame):
    """Chart 2: RAG vs No-RAG average accuracy per model (horizontal grouped bar)."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()
    avg = core.groupby(["model_id", "model_display", "model_tier", "rag_mode"])["accuracy_pct"].mean().reset_index()

    model_idx = {m: i for i, m in enumerate(ALL_MODEL_ORDER)}
    avg["sort"] = avg["model_id"].map(model_idx)
    avg = avg.sort_values("sort")

    no_rag = avg[avg["rag_mode"] == "no_rag"].copy()
    rag = avg[avg["rag_mode"] == "rag"].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=no_rag["model_display"], x=no_rag["accuracy_pct"],
        orientation="h", name="No RAG (Baseline)",
        marker_color=COLOR_NO_RAG,
        text=[f"{v:.1f}%" for v in no_rag["accuracy_pct"]],
        textposition="inside", textfont=dict(color="white", size=12),
    ))
    fig.add_trace(go.Bar(
        y=rag["model_display"], x=rag["accuracy_pct"],
        orientation="h", name="+ RAG (Retrieval-Augmented)",
        marker_color=COLOR_RAG,
        text=[f"{v:.1f}%" for v in rag["accuracy_pct"]],
        textposition="inside", textfont=dict(color="white", size=12),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Impact of RAG on Model Accuracy</b><br>"
                 "<sup>Average across Zero-Shot, Few-Shot, and Chain-of-Thought strategies</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Accuracy (%)", xaxis=dict(range=[0, 100], dtick=10),
        yaxis_title="",
        barmode="group", bargap=0.2, bargroupgap=0.1,
        height=500, width=900,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "02_rag_vs_no_rag.html")
    fig.write_image(OUTPUT_DIR / "02_rag_vs_no_rag.png", scale=3)
    print("  ✅ Chart 2: RAG vs No-RAG Impact (all models)")


def chart_03_rag_lift(df: pd.DataFrame):
    """Chart 3: RAG lift bar chart — all models combined, sorted by lift."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()
    pivot = core.pivot_table(
        index=["model_display", "model_id", "strategy_display", "strategy"],
        columns="rag_mode", values="accuracy_pct", aggfunc="first"
    ).reset_index()

    if "no_rag" not in pivot.columns or "rag" not in pivot.columns:
        return

    pivot["lift"] = pivot["rag"] - pivot["no_rag"]
    pivot = pivot.sort_values("lift", ascending=True)
    pivot["label"] = pivot["model_display"] + " · " + pivot["strategy_display"]
    colors = [COLOR_LIFT_POS if v >= 0 else COLOR_LIFT_NEG for v in pivot["lift"]]

    fig = go.Figure(go.Bar(
        y=pivot["label"], x=pivot["lift"],
        orientation="h", marker_color=colors,
        text=[f"+{v:.1f}pp" if v >= 0 else f"{v:.1f}pp" for v in pivot["lift"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#1e293b", line_width=1.5)

    avg_lift = pivot["lift"].mean()
    fig.add_annotation(
        x=avg_lift, y=1.05, yref="paper",
        text=f"<b>Avg RAG lift: {avg_lift:+.1f}pp</b>",
        showarrow=False,
        font=dict(size=14, color=COLOR_LIFT_POS if avg_lift >= 0 else COLOR_LIFT_NEG),
    )

    fig.update_layout(
        title=dict(
            text="<b>RAG Lift: Accuracy Improvement from Retrieval-Augmented Generation</b><br>"
                 "<sup>Percentage point (pp) change when RAG context is added</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="RAG Lift (percentage points)",
        yaxis_title="",
        height=max(500, len(pivot) * 28 + 140),
        width=950,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "03_rag_lift.html")
    fig.write_image(OUTPUT_DIR / "03_rag_lift.png", scale=3)
    print("  ✅ Chart 3: RAG Lift (all models)")


def chart_04_leaderboard(df: pd.DataFrame):
    """Chart 4: Best-accuracy leaderboard — all 9 models, color-coded by tier."""
    best = df.loc[df.groupby("model_id")["accuracy_pct"].idxmax()].copy()
    best["config_label"] = best["strategy_display"] + " " + best["rag_mode"].map(
        {"no_rag": "(No RAG)", "rag": "(+ RAG)"}
    )
    best = best.sort_values("accuracy_pct", ascending=True)

    tier_colors = {"API": "#3C5488", "Open": "#E64B35"}
    bar_colors = [tier_colors.get(t, "#666") for t in best["model_tier"]]

    fig = go.Figure(go.Bar(
        y=best["model_display"], x=best["accuracy_pct"],
        orientation="h", marker_color=bar_colors,
        text=[f"  {v:.1f}% — {c}" for v, c in zip(best["accuracy_pct"], best["config_label"])],
        textposition="inside", textfont=dict(color="white", size=13),
        insidetextanchor="start",
        showlegend=False,
    ))

    # Legend entries for tier colors
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#3C5488",
                         name="API Model", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#E64B35",
                         name="Open-Source Model", showlegend=True))

    fig.update_layout(
        title=dict(
            text="<b>Model Leaderboard: Best Accuracy Across All Configurations</b><br>"
                 "<sup>124 RD Exam MCQs · TPN Clinical Decision Support Benchmark</sup>",
            x=0.5, xanchor="center",
        ),
        xaxis_title="Accuracy (%)", xaxis=dict(range=[0, 100], dtick=10),
        yaxis_title="",
        height=max(350, len(best) * 50 + 130),
        width=900,
        showlegend=True,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "04_leaderboard.html")
    fig.write_image(OUTPUT_DIR / "04_leaderboard.png", scale=3)
    print("  ✅ Chart 4: Leaderboard (all models)")


def chart_05_latency(df: pd.DataFrame):
    """Chart 5: Accuracy vs Latency scatter — all models, color per model."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()
    core["latency_s"] = core["latency_ms_mean"] / 1000

    fig = go.Figure()
    for model_id in ALL_MODEL_ORDER:
        subset = core[core["model_id"] == model_id]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["latency_s"], y=subset["accuracy_pct"],
            mode="markers",
            name=MODEL_DISPLAY.get(model_id, model_id),
            marker=dict(
                size=14, color=ALL_COLORS.get(model_id, "#666"),
                line=dict(width=1, color="white"), opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Accuracy: %{y:.1f}%<br>"
                "Latency: %{x:.1f}s<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="<b>Accuracy vs Latency Trade-off</b><br>"
                 "<sup>Each point = one model × strategy × RAG combination</sup>",
            x=0.5, xanchor="center",
        ),
        yaxis_title="Accuracy (%)", yaxis=dict(range=[20, 100], dtick=10),
        xaxis_title="Mean Latency (seconds)", xaxis=dict(type="log"),
        height=600, width=950,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "05_latency_vs_accuracy.html")
    fig.write_image(OUTPUT_DIR / "05_latency_vs_accuracy.png", scale=3)
    print("  ✅ Chart 5: Latency vs Accuracy (all models)")


def chart_06_strategy_comparison(df: pd.DataFrame):
    """Chart 6: Strategy effectiveness — avg accuracy per strategy across all models."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()
    avg = core.groupby(["strategy_display", "rag_mode"])["accuracy_pct"].mean().reset_index()

    strat_order = [STRATEGY_DISPLAY[s] for s in CORE_STRATEGIES]
    no_rag = avg[avg["rag_mode"] == "no_rag"].copy()
    rag = avg[avg["rag_mode"] == "rag"].copy()
    no_rag["sort"] = no_rag["strategy_display"].map({s: i for i, s in enumerate(strat_order)})
    rag["sort"] = rag["strategy_display"].map({s: i for i, s in enumerate(strat_order)})
    no_rag = no_rag.sort_values("sort")
    rag = rag.sort_values("sort")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=no_rag["strategy_display"], y=no_rag["accuracy_pct"],
        name="No RAG", marker_color=COLOR_NO_RAG,
        text=[f"{v:.1f}%" for v in no_rag["accuracy_pct"]],
        textposition="outside", textfont=dict(size=13),
    ))
    fig.add_trace(go.Bar(
        x=rag["strategy_display"], y=rag["accuracy_pct"],
        name="+ RAG", marker_color=COLOR_RAG,
        text=[f"{v:.1f}%" for v in rag["accuracy_pct"]],
        textposition="outside", textfont=dict(size=13),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Prompting Strategy Effectiveness</b><br>"
                 "<sup>Average accuracy across all 9 models per strategy</sup>",
            x=0.5, xanchor="center",
        ),
        yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100], dtick=10),
        xaxis_title="Prompting Strategy",
        barmode="group", bargap=0.25,
        height=500, width=800,
        legend=LEGEND_BOTTOM,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "06_strategy_comparison.html")
    fig.write_image(OUTPUT_DIR / "06_strategy_comparison.png", scale=3)
    print("  ✅ Chart 6: Strategy Comparison")


def chart_07_tier_comparison(df: pd.DataFrame):
    """Chart 7: API vs Open-Source tier head-to-head comparison."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()

    # Aggregate per tier
    tier_avg = core.groupby(["model_tier", "rag_mode"])["accuracy_pct"].mean().reset_index()
    tier_best = core.groupby("model_tier")["accuracy_pct"].max().reset_index()
    tier_best.columns = ["model_tier", "best_accuracy"]
    tier_latency = core.groupby("model_tier")["latency_ms_mean"].mean().reset_index()
    tier_latency["latency_s"] = tier_latency["latency_ms_mean"] / 1000

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "<b>Avg Accuracy (No RAG vs RAG)</b>",
            "<b>Best Single Score</b>",
            "<b>Avg Latency</b>",
        ],
        horizontal_spacing=0.12,
    )

    tier_colors_map = {"API": "#3C5488", "Open": "#E64B35"}

    # Panel 1: Avg accuracy by tier × RAG
    for rag_mode, label, pattern in [("no_rag", "No RAG", ""), ("rag", "+ RAG", "/")]:
        subset = tier_avg[tier_avg["rag_mode"] == rag_mode]
        fig.add_trace(go.Bar(
            x=subset["model_tier"], y=subset["accuracy_pct"],
            name=label,
            marker_color=[tier_colors_map.get(t, "#666") for t in subset["model_tier"]],
            marker_pattern_shape=pattern,
            text=[f"{v:.1f}%" for v in subset["accuracy_pct"]],
            textposition="outside", textfont=dict(size=12),
            showlegend=(rag_mode == "no_rag"),
        ), row=1, col=1)

    # Panel 2: Best score per tier
    fig.add_trace(go.Bar(
        x=tier_best["model_tier"], y=tier_best["best_accuracy"],
        marker_color=[tier_colors_map.get(t, "#666") for t in tier_best["model_tier"]],
        text=[f"{v:.1f}%" for v in tier_best["best_accuracy"]],
        textposition="outside", textfont=dict(size=14, color="#1e293b"),
        showlegend=False,
    ), row=1, col=2)

    # Panel 3: Avg latency per tier
    fig.add_trace(go.Bar(
        x=tier_latency["model_tier"], y=tier_latency["latency_s"],
        marker_color=[tier_colors_map.get(t, "#666") for t in tier_latency["model_tier"]],
        text=[f"{v:.1f}s" for v in tier_latency["latency_s"]],
        textposition="outside", textfont=dict(size=14, color="#1e293b"),
        showlegend=False,
    ), row=1, col=3)

    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=2)
    fig.update_yaxes(title_text="Latency (s)", row=1, col=3)

    fig.update_layout(
        title=dict(
            text="<b>API vs Open-Source: Tier Comparison</b><br>"
                 "<sup>4 API models vs 4 open-source models on 124 RD exam MCQs</sup>",
            x=0.5, xanchor="center",
        ),
        barmode="group",
        height=500, width=1100,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "07_tier_comparison.html")
    fig.write_image(OUTPUT_DIR / "07_tier_comparison.png", scale=3)
    print("  ✅ Chart 7: API vs Open-Source Tier Comparison")


def chart_08_radar(df: pd.DataFrame):
    """Chart 8: Radar / spider chart — each model's accuracy profile across strategies."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()

    # Build categories: ZS-noRAG, ZS-RAG, FEW-noRAG, FEW-RAG, COT-noRAG, COT-RAG
    categories = []
    for s in CORE_STRATEGIES:
        sd = STRATEGY_DISPLAY[s]
        categories.append(f"{sd}\nNo RAG")
        categories.append(f"{sd}\n+ RAG")

    fig = go.Figure()

    for model_id in ALL_MODEL_ORDER:
        subset = core[core["model_id"] == model_id]
        if subset.empty:
            continue

        values = []
        for s in CORE_STRATEGIES:
            for rm in ["no_rag", "rag"]:
                row = subset[(subset["strategy"] == s) & (subset["rag_mode"] == rm)]
                values.append(row["accuracy_pct"].values[0] if len(row) > 0 else 0)

        # Close the polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            name=MODEL_DISPLAY.get(model_id, model_id),
            line=dict(color=ALL_COLORS.get(model_id, "#666"), width=2),
            fill="none",
            opacity=0.8,
        ))

    fig.update_layout(
        title=dict(
            text="<b>Model Accuracy Profiles</b><br>"
                 "<sup>Performance across all strategy × RAG combinations</sup>",
            x=0.5, xanchor="center",
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", dtick=20),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=650, width=750,
        legend=dict(
            orientation="v", x=1.05, y=0.5,
            font=dict(size=11),
        ),
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "08_radar_profiles.html")
    fig.write_image(OUTPUT_DIR / "08_radar_profiles.png", scale=3)
    print("  ✅ Chart 8: Radar Profiles (all models)")


def chart_09_accuracy_distribution(df: pd.DataFrame):
    """Chart 9: Box plot — accuracy distribution per model across all configs."""
    core = df[df["strategy"].isin(CORE_STRATEGIES)].copy()

    model_idx = {m: i for i, m in enumerate(ALL_MODEL_ORDER)}
    core["sort"] = core["model_id"].map(model_idx)
    core = core.sort_values("sort")

    fig = go.Figure()
    for model_id in ALL_MODEL_ORDER:
        subset = core[core["model_id"] == model_id]
        if subset.empty:
            continue
        tier = MODEL_TIER.get(model_id, "API")
        fig.add_trace(go.Box(
            y=subset["accuracy_pct"],
            name=MODEL_DISPLAY.get(model_id, model_id),
            marker_color=ALL_COLORS.get(model_id, "#666"),
            boxmean=True,
            line=dict(width=2),
            jitter=0.3,
            pointpos=-1.8,
            boxpoints="all",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Accuracy Distribution Per Model</b><br>"
                 "<sup>Spread across all strategy × RAG configurations (box = IQR, diamond = mean)</sup>",
            x=0.5, xanchor="center",
        ),
        yaxis_title="Accuracy (%)", yaxis=dict(range=[0, 100], dtick=10),
        xaxis_title="",
        height=550, width=1000,
        showlegend=False,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "09_accuracy_distribution.html")
    fig.write_image(OUTPUT_DIR / "09_accuracy_distribution.png", scale=3)
    print("  ✅ Chart 9: Accuracy Distribution (box plot)")


def chart_10_top_configs(df: pd.DataFrame):
    """Chart 10: Top configurations table — best config per model with full details."""
    best = df.loc[df.groupby("model_id")["accuracy_pct"].idxmax()].copy()
    model_idx = {m: i for i, m in enumerate(ALL_MODEL_ORDER)}
    best["sort"] = best["model_id"].map(model_idx)
    best = best.sort_values("accuracy_pct", ascending=False)

    # Build table data
    header_vals = [
        "<b>Rank</b>", "<b>Model</b>", "<b>Tier</b>", "<b>Strategy</b>",
        "<b>RAG</b>", "<b>Accuracy</b>",
    ]

    ranks = list(range(1, len(best) + 1))
    models = best["model_display"].tolist()
    tiers = best["model_tier"].tolist()
    strategies = best["strategy_display"].tolist()
    rag_labels = best["rag_mode"].map({"no_rag": "No", "rag": "Yes"}).tolist()
    accuracies = [f"{v:.1f}%" for v in best["accuracy_pct"]]

    # Color rows by tier
    row_colors = []
    for t in tiers:
        if t == "API":
            row_colors.append("#e0ecff")
        else:
            row_colors.append("#f0e6ff")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color="#1e293b",
            font=dict(color="white", size=13),
            align="center",
            height=35,
        ),
        cells=dict(
            values=[ranks, models, tiers, strategies, rag_labels, accuracies],
            fill_color=[row_colors] * 6,
            font=dict(size=12),
            align="center",
            height=30,
        ),
    )])

    fig.update_layout(
        title=dict(
            text="<b>Best Configuration Per Model</b><br>"
                 "<sup>Optimal strategy and RAG setting for each model on 124 RD exam MCQs</sup>",
            x=0.5, xanchor="center",
        ),
        height=max(350, len(best) * 35 + 180),
        width=1050,
        **LAYOUT_THEME,
    )

    fig.write_html(OUTPUT_DIR / "10_top_configurations.html")
    fig.write_image(OUTPUT_DIR / "10_top_configurations.png", scale=3)
    print("  ✅ Chart 10: Top Configurations Table")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TPN-RAG Benchmark Visualization Suite")
    print("=" * 60)

    print("\n📂 Loading results...")
    df = load_all_results()
    print(f"   Loaded {len(df)} rows across {df['model_id'].nunique()} models")
    api_models = [m for m in API_MODEL_ORDER if m in df["model_id"].unique()]
    open_models = [m for m in OPEN_MODEL_ORDER if m in df["model_id"].unique()]
    print(f"   API Models ({len(api_models)}):  {', '.join(api_models)}")
    print(f"   Open Models ({len(open_models)}): {', '.join(open_models)}")

    print("\n📊 Generating 10 Publication Charts (all models combined)...")
    chart_01_heatmap(df)
    chart_02_rag_impact(df)
    chart_03_rag_lift(df)
    chart_04_leaderboard(df)
    chart_05_latency(df)
    chart_06_strategy_comparison(df)
    chart_07_tier_comparison(df)
    chart_08_radar(df)
    chart_09_accuracy_distribution(df)
    chart_10_top_configs(df)

    print(f"\n{'=' * 60}")
    print(f"✅ All 10 charts saved to: {OUTPUT_DIR.resolve()}")
    print(f"   Static: *.png  |  Interactive: *.html")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

