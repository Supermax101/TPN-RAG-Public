#!/usr/bin/env python3
"""
Model Comparison CLI - Compare multiple LLMs with and without RAG.

Requires Python 3.11+

Usage:
    # Compare HuggingFace models
    python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct -n 50

    # Include OpenAI baseline
    python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct openai:gpt-4o-mini -n 100

    # Use persisted retriever
    python scripts/compare_models.py --persist-dir ./data --models hf:Qwen/Qwen2.5-7B-Instruct -n 50

    # Skip baseline (RAG only)
    python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct --no-baseline -n 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default dataset path - use converted benchmark JSONL
DEFAULT_DATASET = "eval/data/benchmark_2026-02-05/mcq_holdout.jsonl"


def parse_model_spec(spec: str) -> tuple:
    """
    Parse model specification.

    Formats:
        - 'hf:Qwen/Qwen2.5-7B-Instruct' -> ('huggingface', 'Qwen/Qwen2.5-7B-Instruct')
        - 'huggingface:meta-llama/Llama-3.1-8B' -> ('huggingface', 'meta-llama/Llama-3.1-8B')
        - 'openai:gpt-4o' -> ('openai', 'gpt-4o')
        - 'anthropic:claude-sonnet-4-20250514' -> ('anthropic', 'claude-sonnet-4-20250514')
    """
    parts = spec.split(":", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid model spec: {spec}. Use format 'provider:model_name'")

    provider = parts[0]
    model_name = parts[1]

    # Normalize provider name
    if provider == "hf":
        provider = "huggingface"

    return provider, model_name


def load_retriever(persist_dir: str | None):
    """Load retriever from persisted data."""
    if not persist_dir:
        return None

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        logger.warning(f"Persist directory not found: {persist_dir}")
        return None

    try:
        from app.retrieval.hybrid import HybridRetriever, RRFConfig
        from rank_bm25 import BM25Okapi

        # Load BM25
        bm25_path = persist_path / "bm25"
        with open(bm25_path / "corpus.json") as f:
            corpus = json.load(f)
        with open(bm25_path / "metadata.json") as f:
            metadata = json.load(f)
        with open(bm25_path / "tokenized.json") as f:
            tokenized = json.load(f)

        bm25_index = BM25Okapi(tokenized)
        logger.info(f"Loaded BM25 with {len(corpus)} documents")

        # Load ChromaDB if available
        vector_collection = None
        chroma_path = persist_path / "chromadb"
        if not chroma_path.exists():
            chroma_path = persist_path / "chroma"
        if chroma_path.exists():
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=str(chroma_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                vector_collection = client.get_collection("tpn_documents")
                logger.info(f"Loaded ChromaDB with {vector_collection.count()} documents")
            except Exception as e:
                logger.warning(f"ChromaDB not loaded: {e}")

        # Create retriever
        config = RRFConfig(vector_k=20, bm25_k=20, final_k=10)
        retriever = HybridRetriever(
            vector_collection=vector_collection,
            bm25_index=bm25_index,
            bm25_corpus=corpus,
            bm25_metadata=metadata,
            config=config,
        )

        return retriever

    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        return None


class SimpleRetrieverAdapter:
    """Adapter to make HybridRetriever work with comparison framework."""

    def __init__(self, retriever):
        self._retriever = retriever

    def retrieve(self, query: str, top_k: int = 5) -> list:
        results = self._retriever.retrieve(query, top_k=top_k)
        return [
            {"content": r.content, "metadata": r.metadata}
            for r in results
        ]


def run_comparison(args):
    """Run the model comparison via the benchmark runner."""
    from app.evaluation.benchmark_types import (
        ExperimentConfig,
        ModelSpec,
        ModelTier,
        PromptStrategy,
    )
    from app.evaluation.benchmark_runner import run_benchmark

    # Parse model specs into ModelSpec objects
    model_specs = []
    for spec in args.models:
        try:
            provider, model_name = parse_model_spec(spec)
            tier = ModelTier.OPEN if provider == "huggingface" else ModelTier.SOTA
            model_specs.append(ModelSpec(
                model_id=f"{provider}/{model_name}",
                provider=provider,
                model_name=model_name,
                tier=tier,
            ))
            logger.info(f"Added model: {provider}/{model_name}")
        except ValueError as e:
            logger.error(str(e))
            return 1

    if not model_specs:
        logger.error("No valid models specified")
        return 1

    # Load retriever if persist-dir provided
    retriever = None
    if args.persist_dir:
        try:
            from app.evaluation.retriever_adapter import RetrieverAdapter
            retriever = RetrieverAdapter(
                persist_dir=args.persist_dir,
                top_k=args.top_k,
            )
            logger.info("Loaded retriever from %s", args.persist_dir)
        except Exception as e:
            logger.warning("Failed to load retriever: %s", e)

    has_retriever = retriever is not None
    if not has_retriever and args.no_baseline:
        logger.error("--no-baseline requires --persist-dir: cannot skip baseline without a retriever for RAG.")
        return 1
    if not has_retriever and not args.no_baseline:
        logger.warning("No retriever loaded (--persist-dir not set). Running no-RAG only.")

    # Sub-sample dataset if -n is smaller than full set
    dataset_path = args.dataset
    tmp_path = None
    try:
        with open(args.dataset) as f:
            all_lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    if args.sample_size and args.sample_size < len(all_lines):
        rng = random.Random(args.seed)
        sampled = rng.sample(all_lines, args.sample_size)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="compare_subset_",
        )
        tmp.writelines(sampled)
        tmp.close()
        tmp_path = tmp.name
        dataset_path = tmp_path
        logger.info(f"Sampled {args.sample_size} of {len(all_lines)} samples")

    config = ExperimentConfig(
        name="compare_models",
        seed=args.seed,
        repeats=1,
        top_k=args.top_k,
        include_no_rag=not args.no_baseline,
        include_rag=has_retriever,
        prompt_strategies=[PromptStrategy.ZS],
        models=model_specs,
        mcq_dataset_path=dataset_path,
        require_holdout_only=True,
        output_dir=args.output_dir,
    )

    try:
        logger.info(f"Starting comparison with {len(model_specs)} models...")
        result = asyncio.run(run_benchmark(config, retriever=retriever))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    summary = result.get("summary", {})
    for row in summary.get("rows", []):
        acc = row.get("accuracy", row.get("f1_mean", "N/A"))
        label = f"{row['model_id']} | {row['strategy']} | {row['rag_mode']}"
        if isinstance(acc, float):
            print(f"  {label}: {acc:.1%} (n={row['n']})")
        else:
            print(f"  {label}: {acc} (n={row['n']})")

    print(f"\nRecords saved to: {result.get('records_path', args.output_dir)}")
    print(f"Summary saved to: {result.get('summary_path', args.output_dir)}")


def list_models():
    """List available models with dynamic HuggingFace discovery."""
    from app.providers.huggingface import search_models as search_hf_models, list_trending_models

    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    print("\nFetching trending HuggingFace models...")
    models = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-sonnet-4-20250514"],
        "gemini": ["gemini-2.5-pro"],
        "xai": ["grok-4"],
        "kimi": ["kimi-k2"],
        "huggingface": list_trending_models(limit=10),
    }

    for provider, model_list in models.items():
        print(f"\n{provider.upper()}:")
        for m in model_list:
            print(f"  - {provider}:{m}")

    print("\n" + "-" * 60)
    print("SEARCH HUGGINGFACE MODELS")
    print("-" * 60)
    print("\nYou can use any model from HuggingFace Hub.")
    print("Search examples:")
    print("  - search_hf_models('Qwen instruct')     # Qwen models")
    print("  - search_hf_models('Llama instruct')    # Llama models")
    print("  - search_hf_models('Mistral')           # Mistral models")

    print("\n" + "-" * 60)
    print("USAGE")
    print("-" * 60)
    print("\nFormat: --models provider:model_name [provider:model_name ...]")
    print("\nExamples:")
    print("  # HuggingFace models")
    print("  --models hf:Qwen/Qwen2.5-7B-Instruct")
    print("  --models hf:meta-llama/Llama-3.1-8B-Instruct")
    print("")
    print("  # OpenAI models")
    print("  --models openai:gpt-4o-mini")
    print("")
    print("  # Compare multiple")
    print("  --models hf:Qwen/Qwen2.5-7B-Instruct openai:gpt-4o-mini")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple LLMs with and without RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare HuggingFace models
  python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct -n 50

  # Compare multiple HuggingFace models
  python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct hf:meta-llama/Llama-3.1-8B-Instruct -n 50

  # Include OpenAI baseline
  python scripts/compare_models.py --models hf:Qwen/Qwen2.5-7B-Instruct openai:gpt-4o-mini -n 100

  # List available models (fetches from HuggingFace Hub)
  python scripts/compare_models.py --list-models
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to compare (format: provider:model_name)"
    )
    parser.add_argument(
        "-n", "--sample-size",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        help="Directory with persisted ChromaDB and BM25 data"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline (no-RAG) evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comparison_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return 0

    if not args.models:
        parser.print_help()
        print("\nError: --models is required. Use --list-models to see available options.")
        return 1

    rc = run_comparison(args)
    return rc if rc else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
