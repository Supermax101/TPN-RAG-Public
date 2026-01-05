"""
Ingestion Pipeline for TPN RAG system.

Orchestrates the full document processing workflow:
1. Load markdown files from DPT2 output
2. Clean OCR artifacts
3. Chunk with clinical-aware boundaries
4. Generate embeddings with Ollama
5. Store in ChromaDB vector store
6. Build BM25 keyword index

Example usage:
    >>> pipeline = IngestionPipeline(
    ...     docs_dir="/path/to/dpt2/output",
    ...     persist_dir="./chroma_db"
    ... )
    >>> stats = pipeline.run()
    >>> print(f"Ingested {stats.total_chunks} chunks")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

from .cleaner import DocumentCleaner, CleaningStats
from .chunker import SemanticChunker, Chunk, ChunkingStats

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from the full ingestion pipeline."""

    files_processed: int = 0
    files_failed: int = 0
    total_chunks: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    total_original_chars: int = 0
    total_cleaned_chars: int = 0
    total_anchors_removed: int = 0
    total_figures_removed: int = 0
    total_procedures_preserved: int = 0
    total_tables_preserved: int = 0
    avg_chunk_size: float = 0.0
    vector_store_created: bool = False
    bm25_index_created: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def cleaning_reduction_percent(self) -> float:
        """Percentage reduction from cleaning."""
        if self.total_original_chars == 0:
            return 0.0
        return (1 - self.total_cleaned_chars / self.total_original_chars) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_chunks": self.total_chunks,
            "text_chunks": self.text_chunks,
            "table_chunks": self.table_chunks,
            "total_original_chars": self.total_original_chars,
            "total_cleaned_chars": self.total_cleaned_chars,
            "cleaning_reduction_percent": round(self.cleaning_reduction_percent, 1),
            "total_anchors_removed": self.total_anchors_removed,
            "total_figures_removed": self.total_figures_removed,
            "total_procedures_preserved": self.total_procedures_preserved,
            "total_tables_preserved": self.total_tables_preserved,
            "avg_chunk_size": round(self.avg_chunk_size, 0),
            "vector_store_created": self.vector_store_created,
            "bm25_index_created": self.bm25_index_created,
            "errors": self.errors,
        }


class IngestionPipeline:
    """
    Orchestrates document ingestion from raw DPT2 output to searchable index.

    This pipeline:
    1. Discovers all markdown files in the source directory
    2. Cleans each file to remove OCR artifacts
    3. Chunks into semantic units with clinical awareness
    4. Optionally generates embeddings and stores in ChromaDB
    5. Optionally builds BM25 keyword index

    Example:
        >>> pipeline = IngestionPipeline(
        ...     docs_dir="/path/to/dpt2",
        ...     chunk_size=1000,
        ...     chunk_overlap=200
        ... )
        >>> stats = pipeline.run()
    """

    def __init__(
        self,
        docs_dir: str | Path,
        persist_dir: Optional[str | Path] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        embedding_model: str = "qwen3-embedding:0.6b",
        collection_name: str = "tpn_rag",
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            docs_dir: Path to directory containing DPT2 markdown files
            persist_dir: Path to persist ChromaDB and BM25 index.
                        If None, uses in-memory storage.
            chunk_size: Target chunk size in characters (default 1000)
            chunk_overlap: Overlap between chunks (default 200)
            min_chunk_size: Minimum chunk size (default 100)
            embedding_model: Ollama embedding model name
            collection_name: ChromaDB collection name
        """
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Initialize cleaner and chunker
        self.cleaner = DocumentCleaner(preserve_table_ids=False)
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

        # Will be initialized when needed
        self._chroma_client = None
        self._collection = None
        self._embedding_function = None
        self._bm25_corpus = []
        self._bm25_metadata = []

    def discover_files(self) -> List[Path]:
        """
        Discover all markdown files in the source directory.

        Returns:
            List of Path objects for each markdown file
        """
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.docs_dir}")

        md_files = list(self.docs_dir.glob("*.md"))
        logger.info(f"Discovered {len(md_files)} markdown files in {self.docs_dir}")
        return sorted(md_files)

    def process_file(self, file_path: Path) -> tuple[List[Chunk], CleaningStats]:
        """
        Process a single file through cleaning and chunking.

        Args:
            file_path: Path to the markdown file

        Returns:
            Tuple of (chunks, cleaning_stats)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Clean the document
        cleaned_text, clean_stats = self.cleaner.clean(raw_text, source_name=file_path.name)

        # Chunk the cleaned text
        chunks = self.chunker.chunk(
            cleaned_text,
            source=file_path.name,
            additional_metadata={"file_path": str(file_path)},
        )

        return chunks, clean_stats

    def run(
        self,
        create_vector_store: bool = True,
        create_bm25_index: bool = True,
        save_stats: bool = True,
    ) -> IngestionStats:
        """
        Run the full ingestion pipeline.

        Args:
            create_vector_store: Whether to create ChromaDB vector store
            create_bm25_index: Whether to create BM25 keyword index
            save_stats: Whether to save stats to JSON file

        Returns:
            IngestionStats with pipeline metrics
        """
        stats = IngestionStats()

        # Discover files
        try:
            md_files = self.discover_files()
        except FileNotFoundError as e:
            stats.errors.append(str(e))
            return stats

        if not md_files:
            stats.errors.append("No markdown files found")
            return stats

        # Process each file
        all_chunks: List[Chunk] = []
        chunk_sizes: List[int] = []

        for file_path in md_files:
            try:
                chunks, clean_stats = self.process_file(file_path)

                # Accumulate stats
                stats.files_processed += 1
                stats.total_original_chars += clean_stats.original_length
                stats.total_cleaned_chars += clean_stats.cleaned_length
                stats.total_anchors_removed += clean_stats.anchors_removed
                stats.total_figures_removed += clean_stats.figures_removed
                stats.total_procedures_preserved += clean_stats.procedures_preserved
                stats.total_tables_preserved += clean_stats.tables_preserved

                # Count chunk types
                for chunk in chunks:
                    if chunk.is_table:
                        stats.table_chunks += 1
                    else:
                        stats.text_chunks += 1
                    chunk_sizes.append(chunk.length)

                all_chunks.extend(chunks)

                logger.debug(
                    f"Processed {file_path.name}: {len(chunks)} chunks, "
                    f"{clean_stats.reduction_percent:.1f}% reduction"
                )

            except Exception as e:
                stats.files_failed += 1
                stats.errors.append(f"Failed to process {file_path.name}: {e}")
                logger.error(f"Error processing {file_path.name}: {e}")

        stats.total_chunks = len(all_chunks)
        if chunk_sizes:
            stats.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)

        logger.info(
            f"Processed {stats.files_processed} files, "
            f"created {stats.total_chunks} chunks"
        )

        # Create vector store
        if create_vector_store and all_chunks:
            try:
                self._create_vector_store(all_chunks)
                stats.vector_store_created = True
                logger.info("Vector store created successfully")
            except Exception as e:
                stats.errors.append(f"Vector store creation failed: {e}")
                logger.error(f"Vector store creation failed: {e}")

        # Create BM25 index
        if create_bm25_index and all_chunks:
            try:
                self._create_bm25_index(all_chunks)
                stats.bm25_index_created = True
                logger.info("BM25 index created successfully")
            except Exception as e:
                stats.errors.append(f"BM25 index creation failed: {e}")
                logger.error(f"BM25 index creation failed: {e}")

        # Save stats
        if save_stats and self.persist_dir:
            self._save_stats(stats)

        return stats

    def _create_vector_store(self, chunks: List[Chunk]) -> None:
        """
        Create ChromaDB vector store from chunks.

        Uses Ollama embedding model for generating embeddings.
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        # Create or connect to ChromaDB
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            chroma_path = self.persist_dir / "chroma"
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._chroma_client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )

        # Create embedding function for Ollama
        self._embedding_function = self._create_ollama_embedding_function()

        # Delete existing collection if it exists
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection didn't exist

        # Create new collection
        self._collection = self._chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk.content)
            metadatas.append({
                **chunk.metadata,
                "is_table": chunk.is_table,
                "chunk_length": chunk.length,
            })
            ids.append(f"chunk_{i}")

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self._collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx],
            )
            logger.debug(f"Added chunks {i} to {end_idx} to vector store")

        logger.info(f"Created vector store with {len(documents)} chunks")

    def _create_ollama_embedding_function(self):
        """Create Ollama embedding function for ChromaDB."""
        try:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        except ImportError:
            # Fallback for older chromadb versions
            from chromadb.utils import embedding_functions

            class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
                def __init__(self, model_name: str, url: str = "http://localhost:11434"):
                    self.model_name = model_name
                    self.url = url

                def __call__(self, input: List[str]) -> List[List[float]]:
                    import requests
                    embeddings = []
                    for text in input:
                        response = requests.post(
                            f"{self.url}/api/embeddings",
                            json={"model": self.model_name, "prompt": text},
                        )
                        response.raise_for_status()
                        embeddings.append(response.json()["embedding"])
                    return embeddings

        return OllamaEmbeddingFunction(
            model_name=self.embedding_model,
            url="http://localhost:11434",
        )

    def _create_bm25_index(self, chunks: List[Chunk]) -> None:
        """
        Create BM25 keyword index from chunks.

        Stores the corpus and metadata for BM25 retrieval.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = []
        for chunk in chunks:
            # Simple tokenization - can be improved with better NLP
            tokens = chunk.content.lower().split()
            tokenized_corpus.append(tokens)
            self._bm25_corpus.append(chunk.content)
            self._bm25_metadata.append(chunk.metadata)

        # Create BM25 index
        self._bm25 = BM25Okapi(tokenized_corpus)

        # Save to disk if persist_dir is set
        if self.persist_dir:
            bm25_path = self.persist_dir / "bm25"
            bm25_path.mkdir(parents=True, exist_ok=True)

            # Save corpus and metadata
            with open(bm25_path / "corpus.json", "w") as f:
                json.dump(self._bm25_corpus, f)

            with open(bm25_path / "metadata.json", "w") as f:
                json.dump(self._bm25_metadata, f)

            # Save tokenized corpus for BM25 reconstruction
            with open(bm25_path / "tokenized.json", "w") as f:
                json.dump(tokenized_corpus, f)

            logger.info(f"BM25 index saved to {bm25_path}")

    def _save_stats(self, stats: IngestionStats) -> None:
        """Save ingestion stats to JSON file."""
        if not self.persist_dir:
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.persist_dir / "ingestion_stats.json"

        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Stats saved to {stats_path}")

    def get_collection(self):
        """Get the ChromaDB collection if created."""
        return self._collection

    def search_bm25(self, query: str, top_k: int = 5) -> List[tuple[str, dict, float]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (document, metadata, score) tuples
        """
        if not hasattr(self, "_bm25") or self._bm25 is None:
            raise ValueError("BM25 index not created. Run pipeline first.")

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((
                    self._bm25_corpus[idx],
                    self._bm25_metadata[idx],
                    float(scores[idx]),
                ))

        return results


def demo_pipeline():
    """Demo function to test the ingestion pipeline."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Use test directory or command line arg
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/chandra/Desktop/DPT2 Output"

    print("=" * 60)
    print("INGESTION PIPELINE DEMO")
    print("=" * 60)

    # Create pipeline (no vector store or BM25 for demo)
    pipeline = IngestionPipeline(
        docs_dir=docs_dir,
        persist_dir=None,  # In-memory for demo
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Discover files
    files = pipeline.discover_files()
    print(f"\nFound {len(files)} markdown files")

    # Process just first 3 files for demo
    print("\nProcessing first 3 files...")
    for file_path in files[:3]:
        chunks, clean_stats = pipeline.process_file(file_path)
        chunk_stats = pipeline.chunker.get_stats(chunks)
        print(f"\n  {file_path.name}:")
        print(f"    Cleaned: {clean_stats.original_length} -> {clean_stats.cleaned_length} chars ({clean_stats.reduction_percent:.1f}% reduction)")
        print(f"    Chunks: {chunk_stats.total_chunks} (text: {chunk_stats.text_chunks}, tables: {chunk_stats.table_chunks})")

    print("\n" + "=" * 60)
    print("Pipeline demo complete!")
    print("Run with vector store: pipeline.run(create_vector_store=True)")
    print("=" * 60)


if __name__ == "__main__":
    demo_pipeline()
