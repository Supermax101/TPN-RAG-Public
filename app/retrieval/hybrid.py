"""
Hybrid Retriever with Reciprocal Rank Fusion.

Combines vector similarity search with BM25 keyword search
using Reciprocal Rank Fusion (RRF) for optimal results.

RRF Formula: score = sum(1 / (k + rank_i)) for each retriever

Example:
    >>> retriever = HybridRetriever(chroma_collection, bm25_index)
    >>> results = retriever.retrieve("protein requirements", top_k=5)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion."""

    # RRF constant (higher = more weight to lower ranks)
    k: int = 60

    # Number of results from each retriever before fusion
    vector_k: int = 20
    bm25_k: int = 20

    # Weights for each retriever (should sum to 1.0)
    vector_weight: float = 0.5
    bm25_weight: float = 0.5

    # Final number of results after fusion
    final_k: int = 10


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""

    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # "vector", "bm25", or "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
        }


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both retrievers, leveraging the strengths of each:
    - Vector search: Semantic similarity, handles paraphrasing
    - BM25: Exact keyword matching, handles specific terms

    Example:
        >>> config = RRFConfig(vector_weight=0.6, bm25_weight=0.4)
        >>> retriever = HybridRetriever(collection, bm25, config)
        >>> results = retriever.retrieve("What is the protein dose for neonates?")
    """

    def __init__(
        self,
        vector_collection: Any = None,
        bm25_index: Any = None,
        bm25_corpus: List[str] = None,
        bm25_metadata: List[Dict] = None,
        config: Optional[RRFConfig] = None,
        embedding_function: Optional[Callable] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_collection: ChromaDB collection for vector search
            bm25_index: BM25Okapi index for keyword search
            bm25_corpus: List of document texts for BM25
            bm25_metadata: List of metadata dicts for BM25 results
            config: RRF configuration
            embedding_function: Function to embed queries (if not using collection's)
        """
        self.vector_collection = vector_collection
        self.bm25_index = bm25_index
        self.bm25_corpus = bm25_corpus or []
        self.bm25_metadata = bm25_metadata or []
        self.config = config or RRFConfig()
        self.embedding_function = embedding_function

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search with RRF fusion.

        Args:
            query: Search query string
            top_k: Number of results to return (overrides config)
            filter_dict: Optional metadata filter for vector search

        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        final_k = top_k or self.config.final_k

        # Get results from both retrievers
        vector_results = self._vector_search(query, filter_dict)
        bm25_results = self._bm25_search(query)

        # Fuse results using RRF
        fused = self._rrf_fusion(vector_results, bm25_results)

        # Return top-k
        return fused[:final_k]

    def _vector_search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
    ) -> List[tuple[str, Dict, float]]:
        """
        Perform vector similarity search.

        Returns:
            List of (content, metadata, score) tuples
        """
        if self.vector_collection is None:
            return []

        try:
            # Query ChromaDB
            results = self.vector_collection.query(
                query_texts=[query],
                n_results=self.config.vector_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            # Convert distances to similarity scores (ChromaDB returns distances)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Convert cosine distance to similarity: similarity = 1 - distance
            # For cosine distance in [0, 2], similarity in [-1, 1]
            # Normalize to [0, 1]: (1 - distance/2)
            scores = [max(0, 1 - d/2) for d in distances]

            return list(zip(documents, metadatas, scores))

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _bm25_search(self, query: str) -> List[tuple[str, Dict, float]]:
        """
        Perform BM25 keyword search.

        Returns:
            List of (content, metadata, score) tuples
        """
        if self.bm25_index is None or not self.bm25_corpus:
            return []

        try:
            from .tokenizer import clinical_tokenize

            # Tokenize query with clinical awareness
            query_tokens = clinical_tokenize(query)

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:self.config.bm25_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    content = self.bm25_corpus[idx]
                    metadata = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
                    # Normalize BM25 scores to [0, 1] range approximately
                    normalized_score = min(scores[idx] / 30.0, 1.0)  # Empirical normalization
                    results.append((content, metadata, normalized_score))

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: List[tuple[str, Dict, float]],
        bm25_results: List[tuple[str, Dict, float]],
    ) -> List[RetrievalResult]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF assigns scores based on rank position rather than raw scores,
        making it robust to different score distributions.

        Formula: RRF_score = sum(weight_i / (k + rank_i))
        """
        k = self.config.k

        # Track RRF scores by content hash (to handle duplicates)
        rrf_scores: Dict[str, float] = {}
        content_data: Dict[str, tuple[str, Dict]] = {}

        # Process vector results
        for rank, (content, metadata, score) in enumerate(vector_results):
            content_hash = hash(content[:500])  # Hash first 500 chars

            # RRF score contribution from vector
            rrf_contribution = self.config.vector_weight / (k + rank + 1)

            if content_hash in rrf_scores:
                rrf_scores[content_hash] += rrf_contribution
            else:
                rrf_scores[content_hash] = rrf_contribution
                content_data[content_hash] = (content, metadata)

        # Process BM25 results
        for rank, (content, metadata, score) in enumerate(bm25_results):
            content_hash = hash(content[:500])

            # RRF score contribution from BM25
            rrf_contribution = self.config.bm25_weight / (k + rank + 1)

            if content_hash in rrf_scores:
                rrf_scores[content_hash] += rrf_contribution
            else:
                rrf_scores[content_hash] = rrf_contribution
                content_data[content_hash] = (content, metadata)

        # Sort by RRF score and create results
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)

        results = []
        for content_hash in sorted_hashes:
            content, metadata = content_data[content_hash]
            results.append(RetrievalResult(
                content=content,
                metadata=metadata,
                score=rrf_scores[content_hash],
                source="hybrid",
            ))

        return results

    def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve using only vector search."""
        results = self._vector_search(query)[:top_k]
        return [
            RetrievalResult(content=c, metadata=m, score=s, source="vector")
            for c, m, s in results
        ]

    def retrieve_bm25_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve using only BM25 search."""
        results = self._bm25_search(query)[:top_k]
        return [
            RetrievalResult(content=c, metadata=m, score=s, source="bm25")
            for c, m, s in results
        ]


def demo_hybrid():
    """Demo function to test hybrid retrieval."""
    print("=" * 60)
    print("HYBRID RETRIEVER DEMO")
    print("=" * 60)

    # Create mock data
    corpus = [
        "Protein requirements for preterm infants are 3-4 g/kg/day.",
        "Dextrose should be initiated at 6-8 mg/kg/min in neonates.",
        "Lipid emulsions provide essential fatty acids for TPN.",
        "Monitor serum triglycerides when on lipid infusion.",
        "Calcium and phosphorus must be balanced to prevent precipitation.",
    ]

    metadata = [
        {"source": "ASPEN Handbook", "page": 10},
        {"source": "NICU Guidelines", "page": 15},
        {"source": "TPN Manual", "page": 20},
        {"source": "Lipid Guidelines", "page": 5},
        {"source": "Electrolyte Guide", "page": 30},
    ]

    # Create BM25 index
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        # Create retriever (no vector store for demo)
        config = RRFConfig(bm25_k=5, final_k=3)
        retriever = HybridRetriever(
            bm25_index=bm25,
            bm25_corpus=corpus,
            bm25_metadata=metadata,
            config=config,
        )

        # Test retrieval
        query = "What is the protein requirement for preterm infants?"
        print(f"\nQuery: {query}")
        print("\nBM25 Results:")

        results = retriever.retrieve_bm25_only(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r.score:.3f}] {r.content[:60]}...")
            print(f"     Source: {r.metadata.get('source', 'Unknown')}")

    except ImportError:
        print("rank_bm25 not installed. Run: pip install rank-bm25")


if __name__ == "__main__":
    demo_hybrid()
