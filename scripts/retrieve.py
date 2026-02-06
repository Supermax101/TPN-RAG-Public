#!/usr/bin/env python3
"""
Retrieval CLI - Test and run the unified retrieval pipeline.

Usage:
    python scripts/retrieve.py --query "protein requirements for preterm infants"
    python scripts/retrieve.py --persist-dir ./data --query "lipid dosing"
    python scripts/retrieve.py --demo
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run demo with mock data to test all components."""
    print("=" * 60)
    print("UNIFIED RETRIEVAL PIPELINE DEMO")
    print("=" * 60)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: rank_bm25 not installed. Run: pip install rank-bm25")
        return

    # Import retrieval components directly (not through app module)
    from app.retrieval.hybrid import HybridRetriever, RRFConfig, RetrievalResult
    from app.retrieval.reranker import CrossEncoderReranker, RerankerConfig, RerankResult

    # Mock corpus
    corpus = [
        "Protein requirements for preterm infants are 3-4 g/kg/day according to ASPEN guidelines.",
        "Dextrose should be initiated at 6-8 mg/kg/min in neonates and advanced to 10-14 mg/kg/min.",
        "Lipid emulsions provide essential fatty acids. Start at 1 g/kg/day, advance to 3 g/kg/day.",
        "Monitor serum triglycerides when on lipid infusion. Levels should be below 400 mg/dL.",
        "Calcium and phosphorus must be balanced to prevent precipitation in TPN solutions.",
        "Electrolyte requirements vary based on gestational age and clinical condition.",
        "Trace elements including zinc, copper, manganese are essential for growth.",
        "Multivitamins should be added to TPN according to ASPEN recommendations.",
    ]

    metadata = [
        {"source": "ASPEN Handbook", "page": 10, "type": "text"},
        {"source": "NICU Guidelines", "page": 15, "type": "text"},
        {"source": "Lipid Manual", "page": 20, "type": "text"},
        {"source": "Monitoring Guide", "page": 5, "type": "text"},
        {"source": "Compatibility Guide", "page": 30, "type": "text"},
        {"source": "Electrolyte Guide", "page": 12, "type": "text"},
        {"source": "Trace Elements", "page": 8, "type": "text"},
        {"source": "Vitamin Guide", "page": 6, "type": "text"},
    ]

    # Create BM25 index
    tokenized = [doc.lower().split() for doc in corpus]
    bm25_index = BM25Okapi(tokenized)

    # Create hybrid retriever (BM25 only for demo)
    rrf_config = RRFConfig(bm25_k=10, final_k=5)
    hybrid = HybridRetriever(
        bm25_index=bm25_index,
        bm25_corpus=corpus,
        bm25_metadata=metadata,
        config=rrf_config,
    )

    # Create reranker
    reranker_config = RerankerConfig(top_k=3)
    reranker = CrossEncoderReranker(config=reranker_config)

    # Test queries
    queries = [
        "What is the protein requirement for preterm infants?",
        "How to monitor lipid infusion?",
        "What trace elements are needed in TPN?",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("=" * 50)

        # Step 1: Hybrid retrieval (BM25 only in demo)
        candidates = hybrid.retrieve_bm25_only(query, top_k=5)
        print(f"\n--- BM25 Retrieval ({len(candidates)} results) ---")
        for i, r in enumerate(candidates[:3]):
            print(f"  {i+1}. [{r.score:.3f}] {r.content[:60]}...")
            print(f"       Source: {r.metadata.get('source', 'Unknown')}")

        # Step 2: Reranking
        print("\n--- After Cross-Encoder Reranking ---")
        reranked = reranker.rerank(query, candidates, top_k=3)
        for r in reranked:
            change = r.original_rank - r.new_rank
            arrow = "↑" if change > 0 else ("↓" if change < 0 else "=")
            print(f"  {r.new_rank}. [{r.rerank_score:.3f}] {r.content[:60]}...")
            print(f"       Was rank {r.original_rank} {arrow}")


def run_retrieval(persist_dir: str, query: str, top_k: int = 5):
    """Run retrieval on actual persisted data."""
    from app.retrieval.hybrid import HybridRetriever, RRFConfig
    from app.retrieval.reranker import CrossEncoderReranker, RerankerConfig

    persist_path = Path(persist_dir)

    # Load BM25 data
    bm25_path = persist_path / "bm25"
    if not bm25_path.exists():
        print(f"ERROR: BM25 data not found at {bm25_path}")
        print("Run: python scripts/ingest.py first")
        return

    try:
        from rank_bm25 import BM25Okapi

        with open(bm25_path / "corpus.json") as f:
            corpus = json.load(f)
        with open(bm25_path / "metadata.json") as f:
            metadata = json.load(f)
        with open(bm25_path / "tokenized.json") as f:
            tokenized = json.load(f)

        bm25_index = BM25Okapi(tokenized)
        print(f"Loaded BM25 index with {len(corpus)} documents")

    except Exception as e:
        print(f"ERROR loading BM25: {e}")
        return

    # Load ChromaDB
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
            print(f"Loaded ChromaDB with {vector_collection.count()} documents")
        except Exception as e:
            print(f"WARNING: ChromaDB not loaded: {e}")

    # Create retriever
    rrf_config = RRFConfig(
        vector_k=20,
        bm25_k=20,
        vector_weight=0.5,
        bm25_weight=0.5,
        final_k=top_k * 2,
    )

    retriever = HybridRetriever(
        vector_collection=vector_collection,
        bm25_index=bm25_index,
        bm25_corpus=corpus,
        bm25_metadata=metadata,
        config=rrf_config,
    )

    # Retrieve
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)

    results = retriever.retrieve(query, top_k=top_k * 2)
    print(f"\n--- Hybrid Retrieval ({len(results)} results) ---")

    # Rerank
    reranker = CrossEncoderReranker(RerankerConfig(top_k=top_k))
    reranked = reranker.rerank(query, results, top_k=top_k)

    print(f"\n--- Top {top_k} Results (after reranking) ---")
    for r in reranked:
        print(f"\n  [{r.new_rank}] Score: {r.rerank_score:.3f}")
        print(f"  Source: {r.metadata.get('source', 'Unknown')}")
        print(f"  Content: {r.content[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Test the unified retrieval pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with mock data"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data",
        help="Directory with persisted ChromaDB and BM25 data"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to search for"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.query:
        run_retrieval(args.persist_dir, args.query, args.top_k)
    else:
        print("Usage:")
        print("  python scripts/retrieve.py --demo")
        print("  python scripts/retrieve.py --query 'your query here'")


if __name__ == "__main__":
    main()
