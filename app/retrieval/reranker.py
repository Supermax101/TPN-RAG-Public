"""
Cross-Encoder Reranker for improving retrieval precision.

Cross-encoders process query-document pairs together, providing
more accurate relevance scores than bi-encoders (separate embeddings).

Recommended model: BAAI/bge-reranker-v2-m3
- State-of-the-art open-source reranker
- Multilingual support
- Good balance of speed and quality

Example:
    >>> reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    >>> reranked = reranker.rerank(query, candidates, top_k=5)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranking."""

    # Model to use for reranking
    model_name: str = "BAAI/bge-reranker-v2-m3"

    # Number of results to return after reranking
    top_k: int = 5

    # Minimum score threshold (0-1 after normalization)
    score_threshold: float = 0.0

    # Batch size for processing (for efficiency)
    batch_size: int = 32

    # Whether to normalize scores to [0, 1]
    normalize_scores: bool = True

    # Device for model inference
    device: Optional[str] = None  # None = auto-detect


@dataclass
class RerankResult:
    """A single reranked result."""

    content: str
    metadata: dict
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
        }


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Cross-encoders are more accurate than bi-encoders because they
    process the query and document together, allowing the model to
    capture fine-grained interactions.

    Trade-off:
    - Bi-encoder: Fast (separate embeddings), less accurate
    - Cross-encoder: Slow (joint processing), more accurate

    Best practice: Use bi-encoder for initial retrieval (top 50-100),
    then cross-encoder for reranking (top 5-10).

    Example:
        >>> reranker = CrossEncoderReranker(config)
        >>> results = reranker.rerank(
        ...     query="protein requirements for preterm",
        ...     candidates=initial_results,
        ...     top_k=5
        ... )
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the cross-encoder reranker.

        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()
        self._model = None
        self._initialized = False

    def _initialize(self) -> bool:
        """Lazy initialization of the reranker model."""
        if self._initialized:
            return True

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.config.model_name}")

            self._model = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
            )

            self._initialized = True
            logger.info("Reranker model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return False

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Rerank candidate documents using the cross-encoder.

        Args:
            query: Search query
            candidates: List of candidate documents (with .content and .metadata)
            top_k: Number of results to return (overrides config)

        Returns:
            List of RerankResult objects sorted by rerank score
        """
        top_k = top_k or self.config.top_k

        if not candidates:
            return []

        # Initialize model if needed
        if not self._initialize():
            # Fallback: return original candidates without reranking
            logger.warning("Reranker not available, returning original order")
            return self._fallback_rerank(candidates, top_k)

        # Extract content from candidates
        contents = []
        for c in candidates:
            if hasattr(c, 'content'):
                contents.append(c.content)
            elif isinstance(c, dict):
                contents.append(c.get('content', str(c)))
            else:
                contents.append(str(c))

        # Create query-document pairs
        pairs = [(query, content) for content in contents]

        # Get reranker scores
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._fallback_rerank(candidates, top_k)

        # Normalize scores if configured
        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        # Create results with both original and rerank scores
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            # Extract metadata
            if hasattr(candidate, 'metadata'):
                metadata = candidate.metadata
            elif isinstance(candidate, dict):
                metadata = candidate.get('metadata', {})
            else:
                metadata = {}

            # Get original score
            if hasattr(candidate, 'score'):
                original_score = candidate.score
            elif isinstance(candidate, dict):
                original_score = candidate.get('score', 0.0)
            else:
                original_score = 0.0

            results.append(RerankResult(
                content=contents[i],
                metadata=metadata,
                original_score=original_score,
                rerank_score=float(score),
                original_rank=i + 1,
                new_rank=0,  # Will be set after sorting
            ))

        # Sort by rerank score
        results.sort(key=lambda r: r.rerank_score, reverse=True)

        # Apply score threshold
        if self.config.score_threshold > 0:
            results = [r for r in results if r.rerank_score >= self.config.score_threshold]

        # Assign new ranks
        for i, result in enumerate(results):
            result.new_rank = i + 1

        return results[:top_k]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using sigmoid."""
        import math

        def sigmoid(x):
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        return [sigmoid(s) for s in scores]

    def _fallback_rerank(
        self,
        candidates: List[Any],
        top_k: int,
    ) -> List[RerankResult]:
        """Fallback when reranker model is not available."""
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            if hasattr(candidate, 'content'):
                content = candidate.content
                metadata = getattr(candidate, 'metadata', {})
                score = getattr(candidate, 'score', 0.0)
            elif isinstance(candidate, dict):
                content = candidate.get('content', str(candidate))
                metadata = candidate.get('metadata', {})
                score = candidate.get('score', 0.0)
            else:
                content = str(candidate)
                metadata = {}
                score = 0.0

            results.append(RerankResult(
                content=content,
                metadata=metadata,
                original_score=score,
                rerank_score=score,
                original_rank=i + 1,
                new_rank=i + 1,
            ))

        return results

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Useful for debugging or custom scoring.
        """
        if not self._initialize():
            return 0.0

        try:
            score = self._model.predict([(query, document)])[0]
            if self.config.normalize_scores:
                score = self._normalize_scores([score])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0


def demo_reranker():
    """Demo function to test the reranker."""
    print("=" * 60)
    print("CROSS-ENCODER RERANKER DEMO")
    print("=" * 60)

    # Create mock candidates
    @dataclass
    class MockCandidate:
        content: str
        metadata: dict
        score: float

    candidates = [
        MockCandidate(
            content="Lipid emulsions provide essential fatty acids for TPN patients.",
            metadata={"source": "Lipid Guide"},
            score=0.8,
        ),
        MockCandidate(
            content="Protein requirements for preterm infants are 3-4 g/kg/day according to ASPEN.",
            metadata={"source": "ASPEN Handbook"},
            score=0.7,
        ),
        MockCandidate(
            content="Dextrose should be initiated at 6-8 mg/kg/min in neonates.",
            metadata={"source": "NICU Guidelines"},
            score=0.75,
        ),
        MockCandidate(
            content="Monitor serum triglycerides when administering lipid infusions.",
            metadata={"source": "Monitoring Guide"},
            score=0.65,
        ),
    ]

    query = "What is the protein requirement for preterm infants?"

    print(f"\nQuery: {query}")
    print("\n--- Original Order (by vector score) ---")
    for i, c in enumerate(candidates):
        print(f"  {i+1}. [{c.score:.2f}] {c.content[:60]}...")

    # Try to use actual reranker
    config = RerankerConfig(top_k=3)
    reranker = CrossEncoderReranker(config)

    print("\n--- Attempting Reranking ---")
    results = reranker.rerank(query, candidates)

    print("\n--- After Reranking ---")
    for r in results:
        change = r.original_rank - r.new_rank
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "=")
        print(f"  {r.new_rank}. [{r.rerank_score:.3f}] {r.content[:60]}...")
        print(f"     Was rank {r.original_rank} {arrow}")


if __name__ == "__main__":
    demo_reranker()
