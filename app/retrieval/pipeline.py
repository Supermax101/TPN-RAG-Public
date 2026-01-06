"""
Unified Retrieval Pipeline for TPN RAG System.

Combines all retrieval techniques into a configurable pipeline:
1. Query Processing: HyDE and/or Multi-Query expansion
2. Retrieval: Hybrid search (Vector + BM25 with RRF)
3. Post-processing: Cross-encoder reranking

Example:
    >>> pipeline = RetrievalPipeline.from_config(config)
    >>> results = pipeline.retrieve("protein requirements for preterm infants")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Protocol, Union

from .hybrid import HybridRetriever, RRFConfig, RetrievalResult
from .hyde import HyDERetriever, HyDEConfig
from .multi_query import MultiQueryRetriever, MultiQueryConfig
from .reranker import CrossEncoderReranker, RerankerConfig, RerankResult

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Protocol for LLM implementations."""

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class RetrievalConfig:
    """
    Configuration for the unified retrieval pipeline.

    Controls which techniques are enabled and their parameters.
    """

    # Query expansion
    enable_hyde: bool = True
    enable_multi_query: bool = True

    # Retrieval
    enable_vector: bool = True
    enable_bm25: bool = True

    # Reranking
    enable_reranking: bool = True

    # Component configs
    rrf_config: RRFConfig = field(default_factory=RRFConfig)
    hyde_config: HyDEConfig = field(default_factory=HyDEConfig)
    multi_query_config: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)

    # Final output
    final_top_k: int = 5
    min_score_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_hyde": self.enable_hyde,
            "enable_multi_query": self.enable_multi_query,
            "enable_vector": self.enable_vector,
            "enable_bm25": self.enable_bm25,
            "enable_reranking": self.enable_reranking,
            "final_top_k": self.final_top_k,
            "min_score_threshold": self.min_score_threshold,
            "rrf": {
                "k": self.rrf_config.k,
                "vector_weight": self.rrf_config.vector_weight,
                "bm25_weight": self.rrf_config.bm25_weight,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfig":
        """Create config from dictionary."""
        rrf_data = data.get("rrf", {})
        rrf_config = RRFConfig(
            k=rrf_data.get("k", 60),
            vector_weight=rrf_data.get("vector_weight", 0.5),
            bm25_weight=rrf_data.get("bm25_weight", 0.5),
        )

        return cls(
            enable_hyde=data.get("enable_hyde", True),
            enable_multi_query=data.get("enable_multi_query", True),
            enable_vector=data.get("enable_vector", True),
            enable_bm25=data.get("enable_bm25", True),
            enable_reranking=data.get("enable_reranking", True),
            final_top_k=data.get("final_top_k", 5),
            min_score_threshold=data.get("min_score_threshold", 0.0),
            rrf_config=rrf_config,
        )


@dataclass
class PipelineResult:
    """Result from the retrieval pipeline."""

    query: str
    results: List[RerankResult]
    expanded_queries: List[str] = field(default_factory=list)
    hypothetical_doc: Optional[str] = None
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "expanded_queries": self.expanded_queries,
            "hypothetical_doc": self.hypothetical_doc,
            "retrieval_time_ms": self.retrieval_time_ms,
            "rerank_time_ms": self.rerank_time_ms,
        }


class RetrievalPipeline:
    """
    Unified retrieval pipeline combining all techniques.

    Pipeline flow:
    1. Query Expansion (optional)
       - HyDE: Generate hypothetical document
       - Multi-Query: Generate query variations

    2. Hybrid Retrieval
       - Vector search (semantic similarity)
       - BM25 search (keyword matching)
       - RRF fusion to combine results

    3. Reranking (optional)
       - Cross-encoder for fine-grained relevance

    Example:
        >>> # Full pipeline
        >>> config = RetrievalConfig(
        ...     enable_hyde=True,
        ...     enable_multi_query=True,
        ...     enable_reranking=True,
        ... )
        >>> pipeline = RetrievalPipeline(
        ...     config=config,
        ...     vector_collection=chroma_collection,
        ...     bm25_index=bm25,
        ...     llm=ollama_client,
        ... )
        >>> results = pipeline.retrieve("protein requirements")

        >>> # Retrieval-only (no query expansion or reranking)
        >>> simple_config = RetrievalConfig(
        ...     enable_hyde=False,
        ...     enable_multi_query=False,
        ...     enable_reranking=False,
        ... )
        >>> simple_pipeline = RetrievalPipeline(config=simple_config, ...)
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        vector_collection: Any = None,
        bm25_index: Any = None,
        bm25_corpus: List[str] = None,
        bm25_metadata: List[Dict] = None,
        llm: Optional[LLMProtocol] = None,
    ):
        """
        Initialize the retrieval pipeline.

        Args:
            config: Pipeline configuration
            vector_collection: ChromaDB collection for vector search
            bm25_index: BM25Okapi index for keyword search
            bm25_corpus: Document texts for BM25
            bm25_metadata: Metadata for BM25 documents
            llm: LLM for HyDE and Multi-Query (required if those are enabled)
        """
        self.config = config or RetrievalConfig()

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_collection=vector_collection,
            bm25_index=bm25_index,
            bm25_corpus=bm25_corpus or [],
            bm25_metadata=bm25_metadata or [],
            config=self.config.rrf_config,
        )

        # Initialize optional components
        self.hyde_retriever = None
        self.multi_query_retriever = None
        self.reranker = None

        if self.config.enable_hyde and llm:
            self.hyde_retriever = HyDERetriever(
                llm=llm,
                base_retriever=self.hybrid_retriever,
                config=self.config.hyde_config,
            )

        if self.config.enable_multi_query and llm:
            self.multi_query_retriever = MultiQueryRetriever(
                llm=llm,
                base_retriever=self.hybrid_retriever,
                config=self.config.multi_query_config,
            )

        if self.config.enable_reranking:
            self.reranker = CrossEncoderReranker(
                config=self.config.reranker_config,
            )

        self.llm = llm

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full retrieval pipeline.

        Args:
            query: Search query
            top_k: Number of results (overrides config)

        Returns:
            PipelineResult with results and metadata
        """
        import time

        top_k = top_k or self.config.final_top_k

        result = PipelineResult(query=query)
        start_time = time.time()

        # Step 1: Query Expansion
        expanded_queries = [query]
        hypothetical = None

        if self.config.enable_hyde and self.hyde_retriever:
            hypothetical = self.hyde_retriever.generate_hypothetical(query)
            result.hypothetical_doc = hypothetical
            expanded_queries.append(hypothetical)

        if self.config.enable_multi_query and self.multi_query_retriever:
            multi_queries = self.multi_query_retriever.generate_queries(query)
            expanded_queries.extend(multi_queries[1:])  # Skip original (already added)

        result.expanded_queries = expanded_queries

        # Step 2: Hybrid Retrieval
        # Use the most relevant query (original or with expansions)
        if len(expanded_queries) > 1:
            # Combine all queries for retrieval
            combined_query = " ".join(expanded_queries[:3])  # Limit to 3 to avoid noise
        else:
            combined_query = query

        # Increase initial retrieval for reranking
        initial_k = top_k * 4 if self.config.enable_reranking else top_k

        candidates = self.hybrid_retriever.retrieve(combined_query, top_k=initial_k)
        retrieval_end = time.time()
        result.retrieval_time_ms = (retrieval_end - start_time) * 1000

        # Step 3: Reranking
        if self.config.enable_reranking and self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            result.results = reranked
            result.rerank_time_ms = (time.time() - retrieval_end) * 1000
        else:
            # Convert to RerankResult format
            result.results = [
                RerankResult(
                    content=c.content,
                    metadata=c.metadata,
                    original_score=c.score,
                    rerank_score=c.score,
                    original_rank=i + 1,
                    new_rank=i + 1,
                )
                for i, c in enumerate(candidates[:top_k])
            ]

        # Apply score threshold
        if self.config.min_score_threshold > 0:
            result.results = [
                r for r in result.results
                if r.rerank_score >= self.config.min_score_threshold
            ]

        return result

    def retrieve_simple(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Simple retrieval returning just content and metadata.

        Convenient for use with LLM context building.
        """
        result = self.retrieve(query, top_k=top_k)
        return [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.rerank_score,
            }
            for r in result.results
        ]

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_chars: int = 4000,
    ) -> str:
        """
        Build context string for LLM from retrieved documents.

        Args:
            query: Search query
            top_k: Number of documents to include
            max_chars: Maximum context length

        Returns:
            Formatted context string
        """
        docs = self.retrieve_simple(query, top_k=top_k)

        context_parts = []
        total_chars = 0

        for doc in docs:
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]

            # Check if we have room
            entry = f"[Source: {source}]\n{content}"
            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(context_parts)

    @classmethod
    def from_persisted(
        cls,
        persist_dir: str | Path,
        llm: Optional[LLMProtocol] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> "RetrievalPipeline":
        """
        Create pipeline from persisted ChromaDB and BM25 index.

        Args:
            persist_dir: Directory containing chroma/ and bm25/ subdirectories
            llm: LLM for query expansion (optional)
            config: Pipeline configuration (optional)

        Returns:
            Configured RetrievalPipeline
        """
        persist_dir = Path(persist_dir)
        config = config or RetrievalConfig()

        # Load ChromaDB
        vector_collection = None
        chroma_path = persist_dir / "chroma"
        if chroma_path.exists():
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=str(chroma_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                vector_collection = client.get_collection("tpn_documents")
                logger.info(f"Loaded ChromaDB collection with {vector_collection.count()} documents")
            except Exception as e:
                logger.warning(f"Failed to load ChromaDB: {e}")

        # Load BM25
        bm25_index = None
        bm25_corpus = []
        bm25_metadata = []
        bm25_path = persist_dir / "bm25"

        if bm25_path.exists():
            try:
                from rank_bm25 import BM25Okapi

                with open(bm25_path / "corpus.json") as f:
                    bm25_corpus = json.load(f)

                with open(bm25_path / "metadata.json") as f:
                    bm25_metadata = json.load(f)

                with open(bm25_path / "tokenized.json") as f:
                    tokenized = json.load(f)

                bm25_index = BM25Okapi(tokenized)
                logger.info(f"Loaded BM25 index with {len(bm25_corpus)} documents")

            except Exception as e:
                logger.warning(f"Failed to load BM25: {e}")

        return cls(
            config=config,
            vector_collection=vector_collection,
            bm25_index=bm25_index,
            bm25_corpus=bm25_corpus,
            bm25_metadata=bm25_metadata,
            llm=llm,
        )


def demo_pipeline():
    """Demo function to test the retrieval pipeline."""
    print("=" * 60)
    print("RETRIEVAL PIPELINE DEMO")
    print("=" * 60)

    # Create mock corpus
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

    try:
        from rank_bm25 import BM25Okapi

        # Create BM25 index
        tokenized = [doc.lower().split() for doc in corpus]
        bm25_index = BM25Okapi(tokenized)

        # Create simple config (no query expansion or reranking for demo)
        config = RetrievalConfig(
            enable_hyde=False,
            enable_multi_query=False,
            enable_vector=False,  # No vector store in demo
            enable_bm25=True,
            enable_reranking=False,
            final_top_k=3,
        )

        # Create pipeline
        pipeline = RetrievalPipeline(
            config=config,
            bm25_index=bm25_index,
            bm25_corpus=corpus,
            bm25_metadata=metadata,
        )

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

            result = pipeline.retrieve(query)
            print(f"Retrieval time: {result.retrieval_time_ms:.1f}ms")
            print(f"\nTop {len(result.results)} results:")

            for r in result.results:
                print(f"\n  [{r.rerank_score:.3f}] {r.content[:70]}...")
                print(f"         Source: {r.metadata.get('source', 'Unknown')}")

    except ImportError:
        print("rank_bm25 not installed. Run: pip install rank-bm25")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_pipeline()
