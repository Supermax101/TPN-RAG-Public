"""
Semantic boundary chunker using embedding similarity drops.

Splits documents at topic boundaries detected by cosine similarity
drops between consecutive sentence embeddings, producing chunks that
are more topically coherent than character-based recursive splitting.

Opt-in via ``chunker_type="semantic"`` in config or IngestionPipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from .chunker import Chunk

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving clinical abbreviations."""
    # Split on sentence-ending punctuation, but not abbreviations or decimals
    pattern = r"(?<!\d)(?<!p)(?<!e\.g)(?<!i\.e)(?<!vs)(?<!Dr)(?<!Mr)(?<!Mrs)[.!?]+\s+"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if s.strip()]


class SemanticBoundaryChunker:
    """
    Splits at topic boundaries detected by embedding similarity drops.

    Algorithm:
    1. Split text into sentences
    2. Embed all sentences
    3. Compute cosine similarity between consecutive sentence embeddings
    4. Split where similarity < threshold (topic boundary)
    5. Merge small chunks up to max_chunk_size

    Deterministic: no randomness, same input always produces same output.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
    ):
        self._model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading semantic chunker model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def chunk(
        self,
        text: str,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: The full document text.
            source: Source filename for metadata.
            metadata: Additional metadata dict to merge into each chunk.

        Returns:
            List of Chunk objects with content and metadata.
        """
        if not text or not text.strip():
            return []

        self._ensure_model()
        assert self._model is not None

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Single sentence → single chunk
        if len(sentences) == 1:
            return [self._make_chunk(sentences[0], 0, source, metadata)]

        # Embed all sentences
        embeddings = self._model.encode(sentences, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Compute cosine similarity between consecutive sentences
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        similarities = np.sum(normed[:-1] * normed[1:], axis=1)

        # Find split points where similarity drops below threshold
        split_indices = [0]  # Start of first group
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)

        # Group sentences into raw chunks
        raw_groups: List[List[str]] = []
        for i, start in enumerate(split_indices):
            end = split_indices[i + 1] if i + 1 < len(split_indices) else len(sentences)
            raw_groups.append(sentences[start:end])

        # Merge small groups with neighbors
        merged_groups = self._merge_small_groups(raw_groups)

        # Split oversized groups
        final_groups = []
        for group in merged_groups:
            joined = " ".join(group)
            if len(joined) > self.max_chunk_size:
                final_groups.extend(self._split_oversized(group))
            else:
                final_groups.append(group)

        # Create Chunk objects
        chunks = []
        for idx, group in enumerate(final_groups):
            content = " ".join(group)
            if content.strip():
                chunks.append(self._make_chunk(content, idx, source, metadata))

        return chunks

    def _merge_small_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """Merge groups that are below min_chunk_size with their neighbor."""
        if not groups:
            return groups

        merged: List[List[str]] = [groups[0]]
        for group in groups[1:]:
            current_text = " ".join(merged[-1])
            new_text = " ".join(group)

            if len(current_text) < self.min_chunk_size:
                merged[-1].extend(group)
            elif len(new_text) < self.min_chunk_size:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    def _split_oversized(self, sentences: List[str]) -> List[List[str]]:
        """Split a group of sentences that exceeds max_chunk_size."""
        result: List[List[str]] = []
        current: List[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) + 1 > self.max_chunk_size and current:
                result.append(current)
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent) + 1

        if current:
            result.append(current)

        return result

    def _make_chunk(
        self,
        content: str,
        index: int,
        source: Optional[str],
        extra_metadata: Optional[Dict],
    ) -> Chunk:
        meta: Dict = {"chunk_index": index}
        if source:
            meta["source"] = source
        if extra_metadata:
            meta.update(extra_metadata)
        return Chunk(content=content, metadata=meta)
