#!/usr/bin/env python3
"""
Model Comparison CLI - Compare multiple LLMs with and without RAG.

Usage:
    # Compare Ollama models
    python scripts/compare_models.py --models ollama:qwen3:8b ollama:llama3.2:3b -n 50

    # Include OpenAI baseline
    python scripts/compare_models.py --models ollama:qwen3:8b openai:gpt-4o-mini -n 100

    # Use persisted retriever
    python scripts/compare_models.py --persist-dir ./data --models ollama:qwen3:8b -n 50

    # Skip baseline (RAG only)
    python scripts/compare_models.py --models ollama:qwen3:8b --no-baseline -n 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default dataset path
DEFAULT_DATASET = "/Users/chandra/Desktop/TPN2.OFinetuning/data/final/test.jsonl"


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
        chroma_path = persist_path / "chroma"
        if chroma_path.exists():
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=str(chroma_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                vector_collection = client.get_collection("tpn_rag")
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
    """Run the model comparison."""
    from app.evaluation.comparison import ModelComparison

    # Parse model specs
    models = []
    for spec in args.models:
        try:
            provider, model_name = parse_model_spec(spec)
            models.append((provider, model_name))
            logger.info(f"Added model: {provider}/{model_name}")
        except ValueError as e:
            logger.error(str(e))
            return

    if not models:
        logger.error("No valid models specified")
        return

    # Load retriever
    retriever = None
    if args.persist_dir:
        raw_retriever = load_retriever(args.persist_dir)
        if raw_retriever:
            retriever = SimpleRetrieverAdapter(raw_retriever)

    if not retriever and not args.no_baseline:
        logger.warning("No retriever loaded. Running baseline only (no RAG).")

    # Create comparison
    comparison = ModelComparison(
        dataset_path=args.dataset,
        retriever=retriever,
        top_k=args.top_k,
    )

    # Add models
    for provider, model_name in models:
        comparison.add_model(provider, model_name)

    # Run comparison
    logger.info(f"\nStarting comparison with {args.sample_size} samples...")
    logger.info(f"Models: {[f'{p}/{m}' for p, m in models]}")
    logger.info(f"Include baseline: {not args.no_baseline}")

    results = comparison.run(
        sample_size=args.sample_size,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        save_results=True,
        output_dir=args.output_dir,
    )

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(results.to_markdown())

    # Print RAG lift for each model
    print("\n" + "=" * 60)
    print("RAG LIFT ANALYSIS")
    print("=" * 60)

    model_names = set(m.model_name for m in results.models)
    for name in model_names:
        lift = results.get_rag_lift(name)
        if lift is not None:
            print(f"  {name}: {lift:+.1%} improvement with RAG")

    print(f"\nResults saved to: {args.output_dir}/")


def list_models():
    """List available models with dynamic HuggingFace discovery."""
    from app.models import list_available_models, search_hf_models

    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    print("\nFetching trending HuggingFace models...")
    models = list_available_models(fetch_hf=True, hf_limit=10)

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
        return

    if not args.models:
        parser.print_help()
        print("\nError: --models is required. Use --list-models to see available options.")
        return

    run_comparison(args)


if __name__ == "__main__":
    main()
