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
import random
import sys
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


DEFAULT_MODEL_MATRIX: Dict[str, ModelSpec] = {
    "gpt-4o": ModelSpec(model_id="gpt-4o", provider="openai", model_name="gpt-4o", tier=ModelTier.SOTA),
    "claude-sonnet": ModelSpec(model_id="claude-sonnet", provider="anthropic", model_name="claude-sonnet-4-20250514", tier=ModelTier.SOTA),
    "gemini-2.5-pro": ModelSpec(model_id="gemini-2.5-pro", provider="gemini", model_name="gemini-2.5-pro", tier=ModelTier.SOTA),
    "grok-4": ModelSpec(model_id="grok-4", provider="xai", model_name="grok-4", tier=ModelTier.SOTA),
    "kimi-k2": ModelSpec(model_id="kimi-k2", provider="kimi", model_name="kimi-k2", tier=ModelTier.SOTA),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run publishable TPN benchmark")
    parser.add_argument("--mcq-dataset", type=str, help="Path to MCQ JSONL dataset")
    parser.add_argument("--open-dataset", type=str, help="Path to open-ended JSONL dataset")
    parser.add_argument("--persist-dir", type=str, default="./data", help="Persisted retrieval index root")
    parser.add_argument("--output-dir", type=str, default="eval/results/benchmark", help="Benchmark output directory")
    parser.add_argument("--repeats", type=int, default=5, help="Repeat runs per condition")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k chunks in retrieval snapshot")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=60,
        help="Retrieval candidate pool size before context packing/rerank",
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
        default=4,
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
    parser.add_argument("--include-baseline", action="store_true", help="Include no-RAG baseline (off by default)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--list-models", action="store_true", help="List default model keys and exit")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls (default: 5)")
    parser.add_argument("--agentic-retrieval", action="store_true", help="Enable LLM relevance judging on retrieved chunks")
    parser.add_argument("--agentic-judge-provider", type=str, default="openai", help="Provider for agentic judge (default: openai)")
    parser.add_argument("--agentic-judge-model", type=str, default="gpt-4o-mini", help="Model for agentic judge (default: gpt-4o-mini)")
    parser.add_argument("--dynamic-few-shot", action="store_true", help="Enable embedding-based few-shot example selection")
    return parser.parse_args()


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
    prompt_strategies = [
        PromptStrategy.ZS,
        PromptStrategy.FEW_SHOT,
        PromptStrategy.COT,
        PromptStrategy.COT_SC,
        PromptStrategy.RAP,
    ]

    config = ExperimentConfig(
        name="tpn_publishable_benchmark",
        repeats=args.repeats,
        top_k=args.top_k,
        retrieval_candidate_k=args.candidate_k,
        iterative_retrieval=not args.disable_iterative_retrieval,
        retrieval_iterations=args.retrieval_iterations,
        max_query_decompositions=args.max_decompositions,
        include_no_rag=args.include_baseline,
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
    if config.include_rag:
        retriever = RetrieverAdapter(
            persist_dir=args.persist_dir,
            top_k=config.top_k,
            candidate_k=config.retrieval_candidate_k,
            max_context_chars=config.max_context_chars,
            iterative_retrieval=config.iterative_retrieval,
            retrieval_iterations=config.retrieval_iterations,
            max_query_decompositions=config.max_query_decompositions,
        )

    runner = BenchmarkRunner(config=config, retriever=retriever)
    result = asyncio.run(runner.run())
    print("Benchmark complete:")
    print(f"  Records: {result['records_path']}")
    print(f"  Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
