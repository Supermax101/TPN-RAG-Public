#!/usr/bin/env python3
"""
TPN benchmark control-plane CLI.

Provides orchestration commands (status, benchmark, ingest, …) plus
interactive model tools (list-models, quick-test, ask).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table


console = Console()
app = typer.Typer(add_completion=False, help="Control-plane CLI for TPN RAG benchmark workflows")

ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], cwd: Optional[Path] = None, dry_run: bool = False) -> int:
    location = cwd or ROOT
    console.print(f"[cyan]$ {' '.join(cmd)}[/cyan]")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(location))
    return result.returncode


@app.command("status")
def status() -> None:
    """Show quick project status for benchmark readiness."""
    table = Table(title="TPN Benchmark Status")
    table.add_column("Check")
    table.add_column("Path / Value")
    table.add_column("State")

    checks = [
        ("Docs corpus", ROOT / "data/documents", True),
        ("MCQ holdout", ROOT / "eval/data/benchmark_2026-02-05/mcq_holdout.jsonl", False),
        ("Open-ended holdout", ROOT / "eval/data/benchmark_2026-02-05/open_ended_holdout.jsonl", False),
        ("BM25 index", ROOT / "data/bm25", True),
        ("Chroma index", ROOT / "data/chromadb", True),
        ("Benchmark runner", ROOT / "scripts/run_benchmark.py", False),
        ("Prompt preview", ROOT / "scripts/preview_prompts.py", False),
    ]

    for label, path, is_dir in checks:
        ok = path.is_dir() if is_dir else path.exists()
        table.add_row(label, str(path.relative_to(ROOT)), "OK" if ok else "MISSING")

    console.print(table)


@app.command("convert-eval")
def convert_eval(
    mcq_xlsx: Path = typer.Option(
        ROOT / "data/eval/MCQ_Evaluation_Set_Final.xlsx",
        help="MCQ workbook path",
    ),
    open_xlsx: Path = typer.Option(
        ROOT / "data/eval/QandA_Evaluation_Set.xlsx",
        help="Open-ended workbook path",
    ),
    out_dir: Path = typer.Option(
        ROOT / "eval/data/benchmark_2026-02-05",
        help="Output folder",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Convert Excel eval sheets to strict benchmark JSONL."""
    cmd = [
        sys.executable,
        "scripts/convert_eval_xlsx.py",
        "--mcq-xlsx",
        str(mcq_xlsx),
        "--open-xlsx",
        str(open_xlsx),
        "--out-dir",
        str(out_dir),
    ]
    code = _run(cmd, dry_run=dry_run)
    raise typer.Exit(code=code)


@app.command("check-leakage")
def check_leakage(
    dataset: Path = typer.Argument(..., help="Path to JSONL dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run split leakage checks for a dataset."""
    cmd = [sys.executable, "scripts/check_data_leakage.py", "--dataset", str(dataset)]
    code = _run(cmd, dry_run=dry_run)
    raise typer.Exit(code=code)


@app.command("ingest")
def ingest(
    docs_dir: Path = typer.Option(ROOT / "data/documents", help="Source markdown docs"),
    persist_dir: Path = typer.Option(ROOT / "data", help="Persist path for indexes"),
    no_vector_store: bool = typer.Option(False, "--no-vector-store", help="Build BM25 only"),
    chunk_size: int = typer.Option(1000, help="Chunk size"),
    chunk_overlap: int = typer.Option(200, help="Chunk overlap"),
    embedding_provider: str = typer.Option(
        "openai",
        help="Embedding provider: openai | huggingface",
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-large",
        help="Embedding model name",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """
    Build retrieval indexes from document corpus.

    Note: If vector-store creation is enabled, embedding API/model calls may occur.
    """
    cmd = [
        sys.executable,
        "scripts/ingest.py",
        "--docs-dir",
        str(docs_dir),
        "--persist-dir",
        str(persist_dir),
        "--chunk-size",
        str(chunk_size),
        "--chunk-overlap",
        str(chunk_overlap),
        "--embedding-provider",
        embedding_provider,
        "--embedding-model",
        embedding_model,
    ]
    if no_vector_store:
        cmd.append("--no-vector-store")
    code = _run(cmd, dry_run=dry_run)
    raise typer.Exit(code=code)


@app.command("preview-prompts")
def preview_prompts(
    strategy: str = typer.Option("all", help="all | ZS | FEW_SHOT | COT | COT_SC | RAP"),
    question: str = typer.Option(
        "A preterm infant on PN has serum potassium 2.9 mEq/L. How would you classify this?",
        help="Question for prompt rendering",
    ),
) -> None:
    """Render prompt templates without running any model."""
    cmd = [
        sys.executable,
        "scripts/preview_prompts.py",
        "--strategy",
        strategy,
        "--question",
        question,
    ]
    code = _run(cmd, dry_run=False)
    raise typer.Exit(code=code)


@app.command("benchmark")
def benchmark(
    mcq_dataset: Path = typer.Option(..., help="MCQ JSONL path"),
    open_dataset: Optional[Path] = typer.Option(None, help="Open-ended JSONL path"),
    persist_dir: Path = typer.Option(ROOT / "data", help="Persist dir containing indexes"),
    output_dir: Path = typer.Option(ROOT / "eval/results/benchmark", help="Benchmark output folder"),
    repeats: int = typer.Option(5, help="Repeat count"),
    candidate_k: int = typer.Option(60, help="Retrieval candidate pool size"),
    retrieval_iterations: int = typer.Option(2, help="Iterative retrieval passes"),
    max_decompositions: int = typer.Option(4, help="Max decomposition queries"),
    disable_iterative_retrieval: bool = typer.Option(
        False,
        "--disable-iterative-retrieval",
        help="Disable decomposition + iterative retrieval loop",
    ),
    models: str = typer.Option(
        "gpt-5.2,claude-sonnet,gemini-3-flash,grok-4.1-fast,kimi-k2.5",
        help="Comma-separated model keys",
    ),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG conditions"),
    include_baseline: bool = typer.Option(False, "--include-baseline", help="Include no-RAG baseline (off by default)"),
    max_concurrent: int = typer.Option(5, "--max-concurrent", help="Max concurrent API calls"),
    agentic_retrieval: bool = typer.Option(False, "--agentic-retrieval", help="Enable LLM relevance judging"),
    agentic_judge_provider: str = typer.Option("openai", "--agentic-judge-provider", help="Provider for agentic judge"),
    agentic_judge_model: str = typer.Option("gpt-4o-mini", "--agentic-judge-model", help="Model for agentic judge"),
    dynamic_few_shot: bool = typer.Option(False, "--dynamic-few-shot", help="Enable embedding-based few-shot selection"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Run full benchmark matrix."""
    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--mcq-dataset",
        str(mcq_dataset),
        "--persist-dir",
        str(persist_dir),
        "--output-dir",
        str(output_dir),
        "--repeats",
        str(repeats),
        "--candidate-k",
        str(candidate_k),
        "--retrieval-iterations",
        str(retrieval_iterations),
        "--max-decompositions",
        str(max_decompositions),
        "--models",
        models,
        "--max-concurrent",
        str(max_concurrent),
    ]
    if disable_iterative_retrieval:
        cmd.append("--disable-iterative-retrieval")
    if open_dataset:
        cmd.extend(["--open-dataset", str(open_dataset)])
    if no_rag:
        cmd.append("--no-rag")
    if include_baseline:
        cmd.append("--include-baseline")
    if agentic_retrieval:
        cmd.extend(["--agentic-retrieval", "--agentic-judge-provider", agentic_judge_provider, "--agentic-judge-model", agentic_judge_model])
    if dynamic_few_shot:
        cmd.append("--dynamic-few-shot")
    code = _run(cmd, dry_run=dry_run)
    raise typer.Exit(code=code)


@app.command("analyze")
def analyze(
    records: Optional[Path] = typer.Option(None, help="Path to run_records_*.jsonl"),
    output_dir: Path = typer.Option(ROOT / "eval/results/benchmark", help="Results dir"),
    output_name: str = typer.Option("analysis_report.json", help="Output filename"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command only"),
) -> None:
    """Generate benchmark analysis report."""
    cmd = [
        sys.executable,
        "scripts/analyze_benchmark.py",
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
    ]
    if records:
        cmd.extend(["--records", str(records)])
    code = _run(cmd, dry_run=dry_run)
    raise typer.Exit(code=code)


@app.command("show-latest")
def show_latest(
    output_dir: Path = typer.Option(ROOT / "eval/results/benchmark", help="Results dir"),
) -> None:
    """Print latest summary and analysis files if present."""
    summaries = sorted(output_dir.glob("summary_*.json"))
    analyses = sorted(output_dir.glob("*analysis*.json"))
    records = sorted(output_dir.glob("run_records_*.jsonl"))

    table = Table(title="Latest Benchmark Artifacts")
    table.add_column("Type")
    table.add_column("File")

    if summaries:
        table.add_row("Summary", summaries[-1].name)
    if analyses:
        table.add_row("Analysis", analyses[-1].name)
    if records:
        table.add_row("Run records", records[-1].name)
    if not summaries and not analyses and not records:
        table.add_row("None", "No artifacts found")
    console.print(table)

    if summaries:
        try:
            data = json.loads(summaries[-1].read_text(encoding="utf-8"))
            console.print(f"[dim]Rows in latest summary: {len(data.get('rows', []))}[/dim]")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Provider / model discovery
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS = {
    "openai": ("gpt-5.2", "OPENAI_API_KEY"),
    "anthropic": ("claude-sonnet-4-5-20250929", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini-3-flash-preview", "GEMINI_API_KEY"),
    "xai": ("grok-4-1-fast-reasoning", "XAI_API_KEY"),
    "kimi": ("kimi-k2.5", "KIMI_API_KEY"),
}


@app.command("list-models")
def list_models(
    check_health: bool = typer.Option(False, "--check-health", help="Ping each provider API"),
) -> None:
    """Show available providers, default models, and API key status."""
    table = Table(title="TPN Provider Matrix")
    table.add_column("Provider")
    table.add_column("Default Model")
    table.add_column("API Key")
    if check_health:
        table.add_column("Health")

    for provider, (model, env_var) in _PROVIDER_DEFAULTS.items():
        key_status = "SET" if os.environ.get(env_var) else "MISSING"
        row = [provider, model, key_status]

        if check_health:
            if key_status == "MISSING":
                row.append("SKIP")
            else:
                try:
                    from app.evaluation.provider_adapter import create_provider_adapter

                    adapter = create_provider_adapter(provider, model)
                    ok = asyncio.run(adapter.provider.check_health())
                    row.append("OK" if ok else "FAIL")
                except Exception as exc:
                    row.append(f"ERR: {exc!s:.30}")

        table.add_row(*row)

    console.print(table)


@app.command("quick-test")
def quick_test(
    provider: str = typer.Option("openai", help="Provider name"),
    model: Optional[str] = typer.Option(None, help="Model override"),
    with_rag: bool = typer.Option(False, "--with-rag", help="Enable RAG retrieval"),
    persist_dir: Path = typer.Option(ROOT / "data", help="Retrieval index dir"),
) -> None:
    """Run one question through a single provider to verify setup."""
    question = "A preterm infant on PN has serum potassium 2.9 mEq/L. How would you classify this?"

    from scripts.ask_question import ask_one

    result = asyncio.run(ask_one(
        question=question,
        provider=provider,
        model=model,
        strategy="ZS",
        with_rag=with_rag,
        persist_dir=str(persist_dir),
    ))

    console.print(f"\n[bold]Provider:[/bold] {result['provider']} / {result['model']}")
    console.print(f"[bold]Strategy:[/bold] {result['strategy']}")
    console.print(f"[bold]Latency:[/bold] {result['latency_ms']:.0f}ms")
    if result["retrieval"]:
        r = result["retrieval"]
        console.print(f"[bold]RAG:[/bold] {r['chunks']} chunks, {r['context_chars']} chars")
    console.print(f"\n[green]{'='*60}[/green]")
    console.print(result["answer"])


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Clinical TPN question"),
    provider: str = typer.Option("openai", help="Provider name"),
    model: Optional[str] = typer.Option(None, help="Model override"),
    strategy: str = typer.Option("ZS", help="Prompt strategy: ZS, FEW_SHOT, COT, COT_SC, RAP"),
    with_rag: bool = typer.Option(True, "--with-rag/--no-rag", help="Enable/disable RAG"),
    persist_dir: Path = typer.Option(ROOT / "data", help="Retrieval index dir"),
) -> None:
    """Ask a clinical TPN question with optional RAG and prompt strategy."""
    from scripts.ask_question import ask_one

    result = asyncio.run(ask_one(
        question=question,
        provider=provider,
        model=model,
        strategy=strategy,
        with_rag=with_rag,
        persist_dir=str(persist_dir),
    ))

    console.print(f"\n[bold]Provider:[/bold] {result['provider']} / {result['model']}")
    console.print(f"[bold]Strategy:[/bold] {result['strategy']}")
    console.print(f"[bold]Latency:[/bold] {result['latency_ms']:.0f}ms")
    if result["retrieval"]:
        r = result["retrieval"]
        console.print(f"[bold]RAG:[/bold] {r['chunks']} chunks from {r['sources']}")
    console.print(f"\n[green]{'='*60}[/green]")
    console.print(result["answer"])


if __name__ == "__main__":
    app()
