#!/usr/bin/env python3
"""
Run the publishable benchmark matrix.

This script uses the unified evaluation stack in app/evaluation/*
and writes:
- run_records_*.jsonl
- summary_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging so progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

from app.evaluation import (
    BenchmarkRunner,
    ExperimentConfig,
    ModelSpec,
    ModelTier,
    PromptStrategy,
    RetrieverAdapter,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _fingerprint_prompt_templates(template_dir: Path) -> str:
    """
    Fingerprint the full prompt template set used by the benchmark.

    We hash filenames + contents to detect any prompt drift between runs.
    """
    h = hashlib.sha256()
    files = sorted([p for p in template_dir.glob("*.txt") if p.is_file()])
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\n")
        h.update(p.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


DEFAULT_MODEL_MATRIX: Dict[str, ModelSpec] = {
    # --- Phase 1: SOTA API models ---
    "gpt-5.2": ModelSpec(model_id="gpt-5.2", provider="openai", model_name="gpt-5.2", tier=ModelTier.SOTA),
    "gpt-5-mini": ModelSpec(model_id="gpt-5-mini", provider="openai", model_name="gpt-5-mini", tier=ModelTier.SOTA),
    "claude-sonnet": ModelSpec(model_id="claude-sonnet", provider="anthropic", model_name="claude-sonnet-4-5-20250929", tier=ModelTier.SOTA),
    "gemini-3-flash": ModelSpec(model_id="gemini-3-flash", provider="gemini", model_name="gemini-3-flash-preview", tier=ModelTier.SOTA),
    "grok-4.1-fast": ModelSpec(model_id="grok-4.1-fast", provider="xai", model_name="grok-4-1-fast-reasoning", tier=ModelTier.SOTA),
    "kimi-k2.5": ModelSpec(model_id="kimi-k2.5", provider="kimi", model_name="kimi-k2.5", tier=ModelTier.SOTA),
    # --- Phase 2: Open-source HuggingFace models ---
    "gpt-oss-120b": ModelSpec(model_id="gpt-oss-120b", provider="huggingface", model_name="openai/gpt-oss-120b", tier=ModelTier.OPEN),
    "gpt-oss-20b": ModelSpec(model_id="gpt-oss-20b", provider="huggingface", model_name="openai/gpt-oss-20b", tier=ModelTier.OPEN),
    "qwen2.5-32b": ModelSpec(model_id="qwen2.5-32b", provider="huggingface", model_name="Qwen/Qwen2.5-32B-Instruct", tier=ModelTier.OPEN),
    "qwen3-30b-a3b": ModelSpec(model_id="qwen3-30b-a3b", provider="huggingface", model_name="Qwen/Qwen3-30B-A3B-Instruct-2507", tier=ModelTier.OPEN),
    "medgemma-27b": ModelSpec(model_id="medgemma-27b", provider="huggingface", model_name="google/medgemma-27b-text-it", tier=ModelTier.OPEN),
    "gemma3-27b": ModelSpec(model_id="gemma3-27b", provider="huggingface", model_name="google/gemma-3-27b-it", tier=ModelTier.OPEN),
    "phi-4": ModelSpec(model_id="phi-4", provider="huggingface", model_name="microsoft/phi-4", tier=ModelTier.OPEN),
    "glm-4.7-flash": ModelSpec(model_id="glm-4.7-flash", provider="huggingface", model_name="zai-org/GLM-4.7-Flash", tier=ModelTier.OPEN),
    "deepseek-r1-32b": ModelSpec(model_id="deepseek-r1-32b", provider="huggingface", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", tier=ModelTier.OPEN),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run publishable TPN benchmark")
    parser.add_argument("--mcq-dataset", type=str, help="Path to MCQ JSONL dataset")
    parser.add_argument("--open-dataset", type=str, help="Path to open-ended JSONL dataset")
    parser.add_argument("--persist-dir", type=str, default="./data", help="Persisted retrieval index root")
    parser.add_argument("--output-dir", type=str, default="eval/results/benchmark", help="Benchmark output directory")
    parser.add_argument("--repeats", type=int, default=5, help="Repeat runs per condition")
    parser.add_argument("--top-k", type=int, default=6, help="Top-k chunks in retrieval snapshot")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=40,
        help="Retrieval candidate pool size before context packing/rerank",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Maximum retrieved context length injected into prompts",
    )
    parser.add_argument(
        "--disable-rag-gating",
        action="store_true",
        help="Disable RAG context gating (always inject retrieved context when RAG is enabled)",
    )
    parser.add_argument(
        "--rag-min-top-score",
        type=float,
        default=0.62,
        help="Minimum reranker top score required to inject retrieved context (default: 0.62)",
    )
    parser.add_argument(
        "--rag-min-returned-chunks",
        type=int,
        default=2,
        help="Minimum number of retrieved chunks required to inject context (default: 2)",
    )
    parser.add_argument(
        "--rag-min-context-chars",
        type=int,
        default=200,
        help="Minimum context length required to inject retrieved context (default: 200)",
    )
    parser.add_argument(
        "--retrieval-iterations",
        type=int,
        default=2,
        help="Iterative retrieval passes per query (default: 2)",
    )
    parser.add_argument(
        "--max-decompositions",
        type=int,
        default=3,
        help="Max decomposition/expansion queries per iteration",
    )
    parser.add_argument(
        "--disable-iterative-retrieval",
        action="store_true",
        help="Disable decomposition + iterative retrieval loop",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_MATRIX.keys()),
        help="Comma-separated model keys. Use --list-models to inspect.",
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG conditions")
    parser.add_argument("--no-baseline", action="store_true", help="Disable no-RAG baseline runs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--list-models", action="store_true", help="List default model keys and exit")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls (default: 5)")
    parser.add_argument("--agentic-retrieval", action="store_true", help="Enable LLM relevance judging on retrieved chunks")
    parser.add_argument("--agentic-judge-provider", type=str, default="openai", help="Provider for agentic judge (default: openai)")
    parser.add_argument("--agentic-judge-model", type=str, default="gpt-4o-mini", help="Model for agentic judge (default: gpt-4o-mini)")
    parser.add_argument("--dynamic-few-shot", action="store_true", help="Enable embedding-based few-shot example selection")
    parser.add_argument("--retrieval-snapshots-in", type=str, default="", help="Path to precomputed retrieval snapshots JSONL")
    parser.add_argument(
        "--strategies",
        type=str,
        default="ZS,FEW_SHOT,COT",
        help="Comma-separated prompt strategies: ZS, FEW_SHOT, COT, COT_SC, RAP (default: ZS,FEW_SHOT,COT)",
    )
    return parser.parse_args()


def _parse_strategies_arg(value: str) -> List[PromptStrategy]:
    raw = (value or "").strip()
    if not raw or raw.lower() in {"default"}:
        return [PromptStrategy.ZS, PromptStrategy.FEW_SHOT, PromptStrategy.COT]
    if raw.lower() in {"all"}:
        return [PromptStrategy.ZS, PromptStrategy.FEW_SHOT, PromptStrategy.COT, PromptStrategy.COT_SC, PromptStrategy.RAP]

    mapping = {s.value: s for s in PromptStrategy}
    out: List[PromptStrategy] = []
    seen = set()

    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        norm = part.replace("-", "_").upper()
        norm = {
            "ZERO_SHOT": "ZS",
            "ZEROSHOT": "ZS",
            "FEWSHOT": "FEW_SHOT",
            "FEW": "FEW_SHOT",
            "CHAIN_OF_THOUGHT": "COT",
            "COTSC": "COT_SC",
        }.get(norm, norm)
        if norm not in mapping:
            raise SystemExit(f"Unknown strategy '{part}'. Use --strategies ZS,FEW_SHOT,COT (or 'all').")
        s = mapping[norm]
        if s.value not in seen:
            seen.add(s.value)
            out.append(s)

    if not out:
        raise SystemExit("No strategies selected. Provide a comma-separated list or 'default'.")
    return out


def main():
    args = parse_args()

    if args.list_models:
        print("Default model keys:")
        for key in DEFAULT_MODEL_MATRIX:
            spec = DEFAULT_MODEL_MATRIX[key]
            print(f"  - {key}: {spec.provider}/{spec.model_name} ({spec.tier.value})")
        return

    selected_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    missing = [k for k in selected_keys if k not in DEFAULT_MODEL_MATRIX]
    if missing:
        raise SystemExit(f"Unknown model keys: {missing}. Use --list-models.")

    # Apply seed from CLI
    random.seed(args.seed)

    models = [DEFAULT_MODEL_MATRIX[k] for k in selected_keys]
    prompt_strategies = _parse_strategies_arg(args.strategies)

    config = ExperimentConfig(
        name="tpn_publishable_benchmark",
        repeats=args.repeats,
        top_k=args.top_k,
        retrieval_candidate_k=args.candidate_k,
        iterative_retrieval=not args.disable_iterative_retrieval,
        retrieval_iterations=args.retrieval_iterations,
        max_query_decompositions=args.max_decompositions,
        max_context_chars=args.max_context_chars,
        rag_gating_enabled=not args.disable_rag_gating,
        rag_min_top_score=args.rag_min_top_score,
        rag_min_returned_chunks=args.rag_min_returned_chunks,
        rag_min_context_chars=args.rag_min_context_chars,
        include_no_rag=not args.no_baseline,
        include_rag=not args.no_rag,
        prompt_strategies=prompt_strategies,
        models=models,
        mcq_dataset_path=args.mcq_dataset,
        open_dataset_path=args.open_dataset,
        output_dir=args.output_dir,
        require_holdout_only=True,
        seed=args.seed,
        max_concurrent=args.max_concurrent,
        agentic_retrieval=args.agentic_retrieval,
        agentic_judge_provider=args.agentic_judge_provider,
        agentic_judge_model=args.agentic_judge_model,
        dynamic_few_shot=args.dynamic_few_shot,
    )

    retriever = None
    precomputed_snapshots = None
    meta = None
    if args.retrieval_snapshots_in:
        from app.evaluation.retrieval_snapshot_io import load_retrieval_snapshots

        precomputed_snapshots, meta = load_retrieval_snapshots(args.retrieval_snapshots_in)
        logging.info("Loaded %d precomputed retrieval snapshots", len(precomputed_snapshots))
        if meta:
            logging.info("Retrieval snapshot meta: %s", meta)
    elif config.include_rag:
        retriever = RetrieverAdapter(
            persist_dir=args.persist_dir,
            top_k=config.top_k,
            candidate_k=config.retrieval_candidate_k,
            max_context_chars=config.max_context_chars,
            iterative_retrieval=config.iterative_retrieval,
            retrieval_iterations=config.retrieval_iterations,
            max_query_decompositions=config.max_query_decompositions,
        )

    runner = BenchmarkRunner(config=config, retriever=retriever, precomputed_snapshots=precomputed_snapshots)
    result = asyncio.run(runner.run())
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Records (JSONL):  {result['records_path']}")
    print(f"  Summary (JSON):   {result['summary_path']}")
    print(f"  Accuracy (CSV):   {result['csv_path']}")
    if result.get("model_dirs"):
        print(f"\n  Per-model output directories:")
        for d in result["model_dirs"]:
            print(f"    {d}/")
    print("=" * 60)

    # Print accuracy table to stdout
    summary = result.get("summary", {})
    rows = summary.get("rows", [])
    if rows:
        print(f"\n{'Model':<20} {'Strategy':<10} {'Track':<10} {'RAG':<8} {'N':>4} {'Score':>10} {'Latency(ms)':>12}")
        print("-" * 84)
        for row in rows:
            track = row.get("track", "")
            if track == "mcq":
                score = row.get("accuracy", 0)
            else:
                score = (
                    row.get("final_key_f1_mean")
                    or row.get("key_f1_mean")
                    or row.get("quantity_f1_mean")
                    or row.get("f1_mean", 0)
                )
            print(
                f"{row['model_id']:<20} {row['strategy']:<10} {track:<10} {row['rag_mode']:<8} "
                f"{row['n']:>4} {score:>9.1%} {row['latency_ms_mean']:>11.0f}"
            )

    # ------------------------------------------------------------------
    # Write a run manifest for provenance (paper-grade reproducibility)
    # ------------------------------------------------------------------
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = Path(result["summary_path"])
        stamp = summary_path.stem.split("_", 1)[-1] if "_" in summary_path.stem else time.strftime("%Y%m%d_%H%M%S")
        manifest_path = output_dir / f"run_manifest_{stamp}.json"

        datasets = {}
        if args.mcq_dataset:
            p = Path(args.mcq_dataset)
            if not p.is_absolute():
                p = project_root / p
            datasets["mcq"] = {"path": str(p.resolve()), "sha256": _sha256_file(p)}
        if args.open_dataset:
            p = Path(args.open_dataset)
            if not p.is_absolute():
                p = project_root / p
            datasets["open_ended"] = {"path": str(p.resolve()), "sha256": _sha256_file(p)}

        ingestion_manifest = Path(args.persist_dir) / "ingestion_manifest.json"
        if not ingestion_manifest.is_absolute():
            ingestion_manifest = (project_root / ingestion_manifest).resolve()
        kb = {
            "persist_dir": str(Path(args.persist_dir).resolve()),
            "ingestion_manifest_path": str(ingestion_manifest.resolve()),
            "ingestion_manifest_sha256": _sha256_file(ingestion_manifest) if ingestion_manifest.exists() else "",
        }

        snapshots = {}
        if args.retrieval_snapshots_in:
            p = Path(args.retrieval_snapshots_in)
            if not p.is_absolute():
                p = project_root / p
            snapshots = {
                "path": str(p.resolve()),
                "sha256": _sha256_file(p),
                "meta": meta or {},
            }

        rag_policy = "disabled"
        if config.include_rag:
            rag_policy = "gated" if config.rag_gating_enabled else "always"

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git_head_sha(project_root),
            "experiment": config.model_dump(),
            "datasets": datasets,
            "kb": kb,
            "retrieval_snapshots": snapshots,
            "prompt_templates_sha256": _fingerprint_prompt_templates(project_root / "app/prompting/templates"),
            "rag_policy": rag_policy,
            "notes": {
                "baseline_included": bool(config.include_no_rag),
                "rag_included": bool(config.include_rag),
            },
        }

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n  Run manifest:     {manifest_path}")
    except Exception as e:
        print(f"\n[WARN] Failed to write run manifest: {e}")


if __name__ == "__main__":
    main()
