#!/usr/bin/env python3
"""
Document Ingestion Script for TPN RAG System.

Requires Python 3.11+

Processes OCR markdown files through the ingestion pipeline:
1. Cleans OCR artifacts
2. Chunks with clinical-aware boundaries
3. Creates ChromaDB vector store with configurable embeddings
   (OpenAI text-embedding-3-large recommended for benchmark accuracy)
4. Creates BM25 keyword index

Usage:
    python scripts/ingest.py [--docs-dir PATH] [--persist-dir PATH] [options]

Examples:
    # Basic ingestion with defaults
    python scripts/ingest.py --docs-dir data/documents --persist-dir ./data

    # OpenAI embedding (recommended for benchmark runs)
    python scripts/ingest.py --embedding-provider openai --embedding-model text-embedding-3-large

    # HuggingFace embedding
    python scripts/ingest.py --embedding-provider huggingface --embedding-model Qwen/Qwen3-Embedding-8B

    # Skip vector store (just BM25)
    python scripts/ingest.py --no-vector-store
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

# Add project root to path so `app.*` imports work when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file so os.getenv() picks up API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLEANER
# =============================================================================

@dataclass
class CleaningStats:
    """Statistics from document cleaning process."""
    original_length: int = 0
    cleaned_length: int = 0
    anchors_removed: int = 0
    figures_removed: int = 0
    caption_errors_removed: int = 0
    tables_preserved: int = 0
    embedded_tables_extracted: int = 0
    procedures_preserved: int = 0

    @property
    def reduction_percent(self) -> float:
        if self.original_length == 0:
            return 0.0
        return (1 - self.cleaned_length / self.original_length) * 100


class DocumentCleaner:
    """Clean DPT2 OCR artifacts from markdown documents."""

    PATTERNS = {
        "anchor": re.compile(
            r"<a\s+id=['\"][0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}['\"]\s*>\s*</a>",
            re.IGNORECASE
        ),
        "figure_block": re.compile(r"<::(.*?)::>", re.DOTALL),
        "caption_error": re.compile(r"^\s*CAPTION\s+ERROR\s*$", re.MULTILINE | re.IGNORECASE),
        "excess_newlines": re.compile(r"\n{3,}"),
        "table_ids": re.compile(r'\s+id="[^"]*"'),
    }

    TABLE_INDICATORS = ["|", "---", "<table", "<tr>"]

    VALUABLE_CONTENT_INDICATORS = [
        "step", "1.", "2.", "hours", "timeline", "procedure",
        "algorithm", "formula", "calculate", "monitor",
        "mg/kg", "g/kg", "ml/kg", "kcal",
    ]

    REMOVABLE_PATTERNS = [
        re.compile(r"^logo:", re.IGNORECASE),
        re.compile(r"^.*?logo\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"naspghan\s*foundation", re.IGNORECASE),
        re.compile(r"^.*?illustration\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    def __init__(self, preserve_table_ids: bool = False):
        self.preserve_table_ids = preserve_table_ids

    def clean(self, text: str, source_name: Optional[str] = None) -> tuple[str, CleaningStats]:
        stats = CleaningStats(original_length=len(text))

        text, stats.anchors_removed = self._remove_pattern(text, "anchor")
        text, stats.figures_removed, stats.embedded_tables_extracted, stats.procedures_preserved = self._process_figures(text)
        stats.tables_preserved = len(re.findall(r"<table", text, re.IGNORECASE))
        text, stats.caption_errors_removed = self._remove_pattern(text, "caption_error")

        if not self.preserve_table_ids:
            text = self.PATTERNS["table_ids"].sub("", text)

        text = self._normalize(text)
        stats.cleaned_length = len(text)
        return text, stats

    def _remove_pattern(self, text: str, name: str) -> tuple[str, int]:
        matches = self.PATTERNS[name].findall(text)
        return self.PATTERNS[name].sub("", text), len(matches)

    def _process_figures(self, text: str) -> tuple[str, int, int, int]:
        figures_removed, tables_extracted, procedures_preserved = 0, 0, 0

        def replace(match):
            nonlocal figures_removed, tables_extracted, procedures_preserved
            content = match.group(1)

            if len(content.strip()) < 50 or any(p.search(content) for p in self.REMOVABLE_PATTERNS):
                figures_removed += 1
                return ""

            if any(ind in content for ind in self.TABLE_INDICATORS):
                if "|" in content and "---" in content:
                    lines = [l.strip() for l in content.split("\n") if "|" in l.strip()]
                    if lines:
                        tables_extracted += 1
                        return "\n" + "\n".join(lines) + "\n"

            if any(ind in content.lower() for ind in self.VALUABLE_CONTENT_INDICATORS):
                procedures_preserved += 1
                return "\n" + content.strip() + "\n"

            figures_removed += 1
            return ""

        return self.PATTERNS["figure_block"].sub(replace, text), figures_removed, tables_extracted, procedures_preserved

    def _normalize(self, text: str) -> str:
        text = self.PATTERNS["excess_newlines"].sub("\n\n", text)
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()


# =============================================================================
# CHUNKER
# =============================================================================

@dataclass
class Chunk:
    """A document chunk with content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    is_table: bool = False

    @property
    def length(self) -> int:
        return len(self.content)


@dataclass
class ChunkingStats:
    """Statistics from the chunking process."""
    total_chunks: int = 0
    table_chunks: int = 0
    text_chunks: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0


class SemanticChunker:
    """Semantic chunker with clinical-aware boundaries."""

    SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n# ", "\n\n", "\n", ". ", "; ", ", ", " "]

    TABLE_PATTERNS = {
        "html": re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE),
        "md": re.compile(r"(?:^|\n)(\|[^\n]+\|)\n(\|[-:| ]+\|)\n((?:\|[^\n]+\|\n?)+)", re.MULTILINE),
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    _PLACEHOLDER_RE = re.compile(r"\[TABLE\]")

    def chunk(self, text: str, source: Optional[str] = None, additional_metadata: Optional[dict] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []

        text_without_tables, tables = self._extract_tables(text)
        text_chunks = self._recursive_split(text_without_tables)

        all_items: List[Tuple[int, Chunk]] = []
        search_start = 0

        # Text chunks: split on placeholders, find each part with advancing cursor
        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue
            parts = self._PLACEHOLDER_RE.split(chunk_text)
            first_part = True
            for part in parts:
                # Strip lines containing partial placeholder tokens leaked via overlap
                cleaned = "\n".join(
                    ln for ln in part.split("\n") if "TABLE" not in ln or not ln.strip().startswith("[")
                ).strip()
                if not cleaned:
                    continue
                part = cleaned
                prefix = part[:80]
                if first_part:
                    find_from = max(0, search_start - self.chunk_overlap)
                    first_part = False
                else:
                    find_from = search_start
                pos = text_without_tables.find(prefix, find_from)
                if pos == -1:
                    pos = len(text_without_tables)
                else:
                    search_start = pos + len(prefix)
                all_items.append((pos, Chunk(
                    content=part,
                    metadata={"type": "text"},
                    is_table=False,
                )))

        # Table chunks: locate by placeholder positions in text_without_tables
        ph_search = 0
        for _, table_content in tables:
            ph_pos = text_without_tables.find("[TABLE]", ph_search)
            if ph_pos == -1:
                ph_pos = len(text_without_tables)
            else:
                ph_search = ph_pos + len("[TABLE]")
            all_items.append((ph_pos, Chunk(
                content=table_content.strip(),
                metadata={"type": "table"},
                is_table=True,
            )))

        all_items.sort(key=lambda x: x[0])
        all_chunks = [chunk for _, chunk in all_items]

        base_meta = additional_metadata or {}
        if source:
            base_meta["source"] = source

        for i, chunk in enumerate(all_chunks):
            chunk.metadata.update(base_meta)
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        return all_chunks

    def _extract_tables(self, text: str) -> tuple[str, List[tuple[int, str]]]:
        tables = [(m.start(), m.group(0)) for m in self.TABLE_PATTERNS["html"].finditer(text)]
        tables += [(m.start(), m.group(0)) for m in self.TABLE_PATTERNS["md"].finditer(text)]
        tables.sort(key=lambda x: x[0])

        text_without = text
        offset = 0
        for pos, table in tables:
            adj_pos = pos - offset
            placeholder = "\n[TABLE]\n"
            text_without = text_without[:adj_pos] + placeholder + text_without[adj_pos + len(table):]
            offset += len(table) - len(placeholder)

        return text_without, tables

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        if separators is None:
            separators = self.SEPARATORS

        if not text or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for i, sep in enumerate(separators):
            if sep in text:
                splits = text.split(sep)
                chunks = []
                for j, split in enumerate(splits):
                    if j > 0 and sep.strip():
                        split = sep.lstrip() + split
                    if split.strip():
                        chunks.append(split)

                final = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size and i < len(separators) - 1:
                        final.extend(self._recursive_split(chunk, separators[i + 1:]))
                    else:
                        final.append(chunk)

                return self._merge_with_overlap(final)

        return self._split_by_size(text)

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current) < self.min_chunk_size or len(current) + len(next_chunk) <= self.chunk_size:
                current = current + " " + next_chunk.lstrip()
            else:
                merged.append(current.strip())
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    overlap = current[-self.chunk_overlap:]
                    space = overlap.find(" ")
                    if space > 0:
                        overlap = overlap[space + 1:]
                    current = overlap + " " + next_chunk.lstrip()
                else:
                    current = next_chunk

        if current.strip():
            merged.append(current.strip())

        return merged

    def _split_by_size(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        return [c for c in chunks if c]

    def get_stats(self, chunks: List[Chunk]) -> ChunkingStats:
        if not chunks:
            return ChunkingStats()
        sizes = [c.length for c in chunks]
        return ChunkingStats(
            total_chunks=len(chunks),
            table_chunks=sum(1 for c in chunks if c.is_table),
            text_chunks=sum(1 for c in chunks if not c.is_table),
            avg_chunk_size=sum(sizes) / len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
        )


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

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
        if self.total_original_chars == 0:
            return 0.0
        return (1 - self.total_cleaned_chars / self.total_original_chars) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_chunks": self.total_chunks,
            "text_chunks": self.text_chunks,
            "table_chunks": self.table_chunks,
            "total_original_chars": self.total_original_chars,
            "total_cleaned_chars": self.total_cleaned_chars,
            "cleaning_reduction_percent": round(self.cleaning_reduction_percent, 1),
            "avg_chunk_size": round(self.avg_chunk_size, 0),
            "vector_store_created": self.vector_store_created,
            "bm25_index_created": self.bm25_index_created,
            "errors": self.errors,
        }


class IngestionPipeline:
    """Orchestrates document ingestion from raw DPT2 output to searchable index."""

    def __init__(
        self,
        docs_dir: str | Path,
        kb_manifest: Optional[str | Path] = None,
        persist_dir: Optional[str | Path] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        collection_name: str = "tpn_documents",
    ):
        self.docs_dir = Path(docs_dir)
        self.kb_manifest_path = Path(kb_manifest) if kb_manifest else None
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.cleaner = DocumentCleaner()
        self.chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self._chroma_client = None
        self._collection = None
        self._bm25 = None
        self._bm25_corpus = []
        self._bm25_metadata = []

    def _kb_manifest_sha256(self) -> str:
        if not self.kb_manifest_path:
            return ""
        try:
            raw = self.kb_manifest_path.read_bytes()
        except Exception:
            return ""
        return hashlib.sha256(raw).hexdigest()

    def _load_kb_manifest(self) -> Dict[str, Any]:
        if not self.kb_manifest_path:
            return {}
        try:
            return json.loads(self.kb_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def discover_files(self) -> List[Path]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.docs_dir}")
        if self.kb_manifest_path:
            manifest = self._load_kb_manifest()
            included = manifest.get("included_files") or []
            md_files: List[Path] = []
            for rel in included:
                p = (self.docs_dir / str(rel)).resolve()
                if p.exists() and p.is_file() and p.suffix.lower() == ".md":
                    md_files.append(p)
            logger.info(
                "Discovered %d markdown files (KB manifest: %s)",
                len(md_files),
                str(self.kb_manifest_path),
            )
            return sorted(md_files)

        md_files = list(self.docs_dir.glob("*.md"))
        logger.info(f"Discovered {len(md_files)} markdown files")
        return sorted(md_files)

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def process_file(self, file_path: Path) -> tuple[List[Chunk], CleaningStats, str]:
        raw_text = file_path.read_text(encoding="utf-8")
        file_sha256 = self._sha256_text(raw_text)
        cleaned_text, clean_stats = self.cleaner.clean(raw_text, source_name=file_path.name)
        chunks = self.chunker.chunk(cleaned_text, source=file_path.name, additional_metadata={"file_path": str(file_path)})
        return chunks, clean_stats, file_sha256

    def _manifest_path(self) -> Optional[Path]:
        if not self.persist_dir:
            return None
        return self.persist_dir / "ingestion_manifest.json"

    def _load_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path()
        if not path:
            return
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(
        self,
        create_vector_store: bool = True,
        create_bm25_index: bool = True,
        save_stats: bool = True,
        rebuild_vector_store: bool = False,
    ) -> IngestionStats:
        stats = IngestionStats()

        try:
            md_files = self.discover_files()
        except FileNotFoundError as e:
            stats.errors.append(str(e))
            return stats

        if not md_files:
            stats.errors.append("No markdown files found")
            return stats

        all_chunks: List[Chunk] = []
        chunk_sizes: List[int] = []
        per_file_chunks: Dict[str, List[Chunk]] = {}
        per_file_hash: Dict[str, str] = {}
        per_file_chunk_count: Dict[str, int] = {}

        manifest: Dict[str, Any] = {}
        force_rebuild = bool(rebuild_vector_store)
        changed_files: set[str] = set()
        kb_manifest_sha256 = self._kb_manifest_sha256()

        if create_vector_store and self.persist_dir:
            manifest = self._load_manifest()
            # If manifest exists but config changed, force rebuild for correctness.
            if manifest:
                expected = {
                    "collection_name": self.collection_name,
                    "chunk_size": self.chunker.chunk_size,
                    "chunk_overlap": self.chunker.chunk_overlap,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": self.embedding_model,
                    "kb_manifest_sha256": kb_manifest_sha256,
                }
                for k, v in expected.items():
                    if manifest.get(k) != v:
                        logger.info("Ingestion config changed (%s). Forcing vector store rebuild.", k)
                        force_rebuild = True
                        break

        for file_path in md_files:
            try:
                chunks, clean_stats, file_sha256 = self.process_file(file_path)
                per_file_hash[file_path.name] = file_sha256
                per_file_chunk_count[file_path.name] = len(chunks)

                stats.files_processed += 1
                stats.total_original_chars += clean_stats.original_length
                stats.total_cleaned_chars += clean_stats.cleaned_length
                stats.total_anchors_removed += clean_stats.anchors_removed
                stats.total_figures_removed += clean_stats.figures_removed
                stats.total_procedures_preserved += clean_stats.procedures_preserved
                stats.total_tables_preserved += clean_stats.tables_preserved

                for chunk in chunks:
                    if chunk.is_table:
                        stats.table_chunks += 1
                    else:
                        stats.text_chunks += 1
                    chunk_sizes.append(chunk.length)

                all_chunks.extend(chunks)

                # Track per-file chunks for incremental vector updates.
                per_file_chunks[file_path.name] = chunks
                if create_vector_store and self.persist_dir:
                    if force_rebuild:
                        changed_files.add(file_path.name)
                    else:
                        prev = (manifest.get("files") or {}).get(file_path.name) if manifest else None
                        if not prev or prev.get("sha256") != file_sha256:
                            changed_files.add(file_path.name)

            except Exception as e:
                stats.files_failed += 1
                stats.errors.append(f"Failed to process {file_path.name}: {e}")
                logger.error(f"Error processing {file_path.name}: {e}")

        stats.total_chunks = len(all_chunks)
        if chunk_sizes:
            stats.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)

        logger.info(f"Processed {stats.files_processed} files, created {stats.total_chunks} chunks")

        if create_vector_store and all_chunks:
            try:
                self._create_vector_store(
                    all_chunks,
                    per_file_chunks=per_file_chunks,
                    per_file_hash=per_file_hash,
                    per_file_chunk_count=per_file_chunk_count,
                    manifest=manifest,
                    changed_files=changed_files,
                    rebuild=force_rebuild,
                )
                stats.vector_store_created = True
                logger.info("Vector store created successfully")
            except Exception as e:
                stats.errors.append(f"Vector store creation failed: {e}")
                logger.error(f"Vector store creation failed: {e}")

        if create_bm25_index and all_chunks:
            try:
                self._create_bm25_index(all_chunks)
                stats.bm25_index_created = True
                logger.info("BM25 index created successfully")
            except Exception as e:
                stats.errors.append(f"BM25 index creation failed: {e}")
                logger.error(f"BM25 index creation failed: {e}")

        if save_stats and self.persist_dir:
            self._save_stats(stats)

        return stats

    def _create_vector_store(
        self,
        chunks: List[Chunk],
        per_file_chunks: Dict[str, List[Chunk]],
        per_file_hash: Dict[str, str],
        per_file_chunk_count: Dict[str, int],
        manifest: Dict[str, Any],
        changed_files: set[str],
        rebuild: bool,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            chroma_path = self.persist_dir / "chromadb"
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

        embedding_fn = self._create_embedding_function()

        # Resolve existing collection (for incremental update) unless forced rebuild.
        existing_collection = None
        if not rebuild:
            try:
                existing_collection = self._chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_fn,
                )
            except Exception:
                existing_collection = None

        # Migration safety: if we're switching to manifest-based incremental
        # ingestion but a non-empty collection already exists and no manifest is
        # present, incremental IDs will not match legacy IDs and would create
        # duplicates. Force a one-time rebuild in that case.
        if not rebuild and existing_collection is not None and not (manifest.get("files") if manifest else None):
            try:
                if int(existing_collection.count() or 0) > 0:
                    logger.info(
                        "No ingestion_manifest.json found but existing Chroma collection is non-empty; "
                        "forcing a one-time rebuild to avoid duplicate chunks."
                    )
                    rebuild = True
                    existing_collection = None
            except Exception:
                pass

        if rebuild or existing_collection is None:
            try:
                self._chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            # Full rebuild: treat all files as changed.
            changed_files = set(per_file_chunks.keys())
            manifest = {}
        else:
            self._collection = existing_collection

        # Incremental strategy:
        # - delete chunks for changed/removed files
        # - add chunks for changed/new files
        manifest_files = dict((manifest.get("files") or {}) if manifest else {})
        current_files = set(per_file_chunks.keys())

        removed_files = set(manifest_files.keys()) - current_files
        for fname in sorted(removed_files):
            prior = manifest_files.get(fname) or {}
            prior_count = int(prior.get("chunk_count") or 0)
            if prior_count > 0:
                ids = [f"{fname}::chunk_{i}" for i in range(prior_count)]
                try:
                    self._collection.delete(ids=ids)
                except Exception:
                    pass
            manifest_files.pop(fname, None)

        for fname in sorted(changed_files):
            new_chunks = per_file_chunks.get(fname) or []

            # Delete prior chunks for this file (if any).
            prior = manifest_files.get(fname) or {}
            prior_count = int(prior.get("chunk_count") or 0)
            if prior_count > 0:
                ids = [f"{fname}::chunk_{i}" for i in range(prior_count)]
                try:
                    self._collection.delete(ids=ids)
                except Exception:
                    pass

            documents = [c.content for c in new_chunks]
            metadatas = [{**c.metadata, "is_table": c.is_table, "chunk_length": c.length} for c in new_chunks]
            ids = [
                f"{fname}::chunk_{int(c.metadata.get('chunk_index', i))}"
                for i, c in enumerate(new_chunks)
            ]

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self._collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx],
                )
            manifest_files[fname] = {
                "sha256": per_file_hash.get(fname, ""),
                "chunk_count": int(per_file_chunk_count.get(fname, len(new_chunks))),
            }

        # Persist manifest so future ingests can skip unchanged docs.
        new_manifest = {
            "version": 1,
            "collection_name": self.collection_name,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "kb_manifest_sha256": kb_manifest_sha256,
            "files": manifest_files,
        }
        self._save_manifest(new_manifest)

        logger.info(
            "Vector store update complete: %d changed, %d removed, %d total files",
            len(changed_files),
            len(removed_files),
            len(new_manifest["files"]),
        )

    def _create_embedding_function(self):
        """Create embedding function for ChromaDB."""
        from chromadb.utils import embedding_functions

        if self.embedding_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for --embedding-provider openai."
                )
            logger.info("Using OpenAI embeddings: %s", self.embedding_model)
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.embedding_model,
            )

        if self.embedding_provider in {"huggingface", "hf"}:
            from sentence_transformers import SentenceTransformer
            import torch

            class HuggingFaceEmbedding(embedding_functions.EmbeddingFunction):
                def __init__(self, model_name: str):
                    self.model_name = model_name
                    self._model = None

                def _load_model(self):
                    if self._model is None:
                        logger.info("Loading embedding model: %s", self.model_name)
                        self._model = SentenceTransformer(
                            self.model_name,
                            trust_remote_code=True,
                            model_kwargs={"torch_dtype": torch.bfloat16},
                        )
                    return self._model

                def __call__(self, input: List[str]) -> List[List[float]]:
                    model = self._load_model()
                    # Prefer document/passage encoding when available (SentenceTransformers >= 2.3).
                    # Some models do not define a "document" prompt name; encode_document() handles that
                    # gracefully by falling back to any available passage/corpus prompt or no prompt.
                    if hasattr(model, "encode_document"):
                        embeddings = model.encode_document(input, show_progress_bar=False)
                    else:
                        embeddings = model.encode(input, show_progress_bar=False)
                    return embeddings.tolist()

            return HuggingFaceEmbedding(model_name=self.embedding_model)

        raise ValueError(
            f"Unknown embedding provider '{self.embedding_provider}'. Use 'openai' or 'huggingface'."
        )

    def _create_bm25_index(self, chunks: List[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        from app.retrieval.tokenizer import clinical_tokenize

        tokenized_corpus = []
        for chunk in chunks:
            tokens = clinical_tokenize(chunk.content)
            tokenized_corpus.append(tokens)
            self._bm25_corpus.append(chunk.content)
            self._bm25_metadata.append(chunk.metadata)

        self._bm25 = BM25Okapi(tokenized_corpus)

        if self.persist_dir:
            bm25_path = self.persist_dir / "bm25"
            bm25_path.mkdir(parents=True, exist_ok=True)

            with open(bm25_path / "corpus.json", "w") as f:
                json.dump(self._bm25_corpus, f)
            with open(bm25_path / "metadata.json", "w") as f:
                json.dump(self._bm25_metadata, f)
            with open(bm25_path / "tokenized.json", "w") as f:
                json.dump(tokenized_corpus, f)

            logger.info(f"BM25 index saved to {bm25_path}")

    def _save_stats(self, stats: IngestionStats) -> None:
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.persist_dir / "ingestion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        logger.info(f"Stats saved to {stats_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest DPT2 documents into TPN RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--docs-dir",
        default="data/documents",
        help="Path to markdown files (default: %(default)s)",
    )
    parser.add_argument(
        "--kb-manifest",
        default="",
        help="Optional KB manifest JSON (data/kb_manifests/kb_clean.json, kb_max.json). When set, only listed .md files are ingested.",
    )
    parser.add_argument(
        "--persist-dir",
        default="./data",
        help="Path to persist vector store and index (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target chunk size in characters (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "huggingface", "hf"],
        help="Embedding provider (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model name (default: %(default)s)",
    )
    parser.add_argument(
        "--collection-name",
        default="tpn_documents",
        help="ChromaDB collection name (default: %(default)s)",
    )
    parser.add_argument(
        "--no-vector-store",
        action="store_true",
        help="Skip vector store creation",
    )
    parser.add_argument(
        "--rebuild-vector-store",
        action="store_true",
        help="Force a full rebuild of the vector store (re-embeds all docs).",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Skip BM25 index creation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("TPN RAG Document Ingestion")
    print("=" * 60)
    print(f"\nSource: {args.docs_dir}")
    if args.kb_manifest:
        print(f"KB manifest: {args.kb_manifest}")
    print(f"Persist: {args.persist_dir}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    print(f"Embedding provider: {args.embedding_provider}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vector store: {'disabled' if args.no_vector_store else 'enabled'}")
    print(f"Vector rebuild: {'yes' if args.rebuild_vector_store else 'no (incremental)'}")
    print(f"BM25 index: {'disabled' if args.no_bm25 else 'enabled'}")
    print()

    pipeline = IngestionPipeline(
        docs_dir=args.docs_dir,
        kb_manifest=(args.kb_manifest or None),
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
    )

    stats = pipeline.run(
        create_vector_store=not args.no_vector_store,
        create_bm25_index=not args.no_bm25,
        save_stats=True,
        rebuild_vector_store=bool(args.rebuild_vector_store),
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles processed: {stats.files_processed}")
    print(f"Files failed: {stats.files_failed}")
    print(f"\nTotal chunks: {stats.total_chunks:,}")
    print(f"  Text chunks: {stats.text_chunks:,}")
    print(f"  Table chunks: {stats.table_chunks:,}")
    print(f"\nAvg chunk size: {stats.avg_chunk_size:.0f} chars")
    print(f"\nCleaning stats:")
    print(f"  Original: {stats.total_original_chars:,} chars ({stats.total_original_chars/1024:.0f} KB)")
    print(f"  Cleaned: {stats.total_cleaned_chars:,} chars ({stats.total_cleaned_chars/1024:.0f} KB)")
    print(f"  Reduction: {stats.cleaning_reduction_percent:.1f}%")
    print(f"\nVector store: {'created' if stats.vector_store_created else 'not created'}")
    print(f"BM25 index: {'created' if stats.bm25_index_created else 'not created'}")

    if stats.errors:
        print(f"\nErrors ({len(stats.errors)}):")
        for error in stats.errors[:5]:
            print(f"  - {error}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")

    if stats.files_failed > 0 or stats.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
