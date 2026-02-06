"""
Semantic Chunker for clinical documents.

This module provides intelligent chunking with clinical-aware boundaries,
preserving tables and structured content while splitting at natural breakpoints.

Key features:
- Clinical-aware separators (section headers, paragraphs, sentences)
- Table preservation (keeps HTML and markdown tables as single chunks)
- Configurable chunk size and overlap
- Metadata extraction for each chunk
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class Chunk:
    """
    A document chunk with content and metadata.

    Attributes:
        content: The text content of the chunk
        metadata: Associated metadata (source, section, chunk_index, etc.)
        is_table: Whether this chunk contains a table
    """

    content: str
    metadata: dict = field(default_factory=dict)
    is_table: bool = False

    @property
    def length(self) -> int:
        """Character length of the chunk."""
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
    """
    Semantic chunker with clinical-aware boundaries.

    This chunker:
    1. Detects and preserves tables as single chunks
    2. Splits remaining text at natural boundaries
    3. Respects markdown structure (headers, lists)
    4. Adds overlap between chunks for context continuity

    Example usage:
        >>> chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
        >>> chunks = chunker.chunk(cleaned_text, source="5PN_Components.md")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.metadata['chunk_index']}: {chunk.length} chars")
    """

    # Clinical-aware separators in order of preference
    # Split at larger boundaries first, then smaller ones
    SEPARATORS = [
        "\n## ",        # Major section headers
        "\n### ",       # Subsection headers
        "\n#### ",      # Sub-subsection headers
        "\n# ",         # Top-level headers (less common in body)
        "\n\n",         # Paragraph breaks
        "\n",           # Line breaks
        ". ",           # Sentence boundaries
        "; ",           # Clinical list separators
        ", ",           # Comma-separated items
        " ",            # Words (last resort)
    ]

    # Patterns to identify table content
    TABLE_PATTERNS = {
        "html_table": re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE),
        "markdown_table": re.compile(
            r"(?:^|\n)(\|[^\n]+\|)\n(\|[-:| ]+\|)\n((?:\|[^\n]+\|\n?)+)",
            re.MULTILINE
        ),
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size for each chunk in characters (default 1000)
            chunk_overlap: Overlap between chunks for context continuity (default 200)
            min_chunk_size: Minimum chunk size to avoid tiny fragments (default 100)
            separators: Custom list of separators (default uses CLINICAL_SEPARATORS)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or self.SEPARATORS

    def chunk(
        self,
        text: str,
        source: Optional[str] = None,
        additional_metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk text with clinical-aware boundaries.

        Args:
            text: The text to chunk (should be pre-cleaned)
            source: Source filename for metadata
            additional_metadata: Extra metadata to add to all chunks

        Returns:
            List of Chunk objects with content and metadata
        """
        if not text or not text.strip():
            return []

        # Step 1: Extract tables first (they should be single chunks)
        text_without_tables, tables = self._extract_tables(text)

        # Step 2: Chunk the remaining text
        text_chunks = self._recursive_split(text_without_tables)

        # Step 3: Combine tables and text chunks in document order
        all_chunks = self._merge_chunks_with_tables(text_without_tables, text_chunks, tables)

        # Step 4: Add metadata to all chunks
        base_metadata = additional_metadata or {}
        if source:
            base_metadata["source"] = source

        for i, chunk in enumerate(all_chunks):
            chunk.metadata.update(base_metadata)
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        return all_chunks

    def _extract_tables(self, text: str) -> tuple[str, List[tuple[int, str]]]:
        """
        Extract tables from text, returning text without tables and table info.

        Returns:
            Tuple of (text_without_tables, list of (position, table_content))
        """
        tables = []

        # Find HTML tables
        for match in self.TABLE_PATTERNS["html_table"].finditer(text):
            tables.append((match.start(), match.group(0)))

        # Find markdown tables
        for match in self.TABLE_PATTERNS["markdown_table"].finditer(text):
            tables.append((match.start(), match.group(0)))

        # Sort by position
        tables.sort(key=lambda x: x[0])

        # Remove tables from text (replace with placeholder)
        text_without_tables = text
        offset = 0
        for pos, table in tables:
            adjusted_pos = pos - offset
            table_len = len(table)
            placeholder = f"\n[TABLE_PLACEHOLDER_{len(tables)}]\n"
            text_without_tables = (
                text_without_tables[:adjusted_pos] +
                placeholder +
                text_without_tables[adjusted_pos + table_len:]
            )
            offset += table_len - len(placeholder)

        return text_without_tables, tables

    def _recursive_split(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Recursively split text using separators in order of preference.

        This implements a recursive character text splitter that:
        1. Tries to split at the largest separator first
        2. If chunks are too big, recursively splits with smaller separators
        3. Merges small chunks with overlap
        """
        if separators is None:
            separators = self.separators

        if not text or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator in text:
                splits = text.split(separator)

                # Re-add separator to the beginning of each split (except first)
                chunks = []
                for j, split in enumerate(splits):
                    if j > 0 and separator.strip():
                        split = separator.lstrip() + split
                    if split.strip():
                        chunks.append(split)

                # Check if any chunk is too big
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size and i < len(separators) - 1:
                        # Recursively split with remaining separators
                        sub_chunks = self._recursive_split(chunk, separators[i + 1:])
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)

                # Merge with overlap
                return self._merge_with_overlap(final_chunks)

        # No separator found, split by character count
        return self._split_by_size(text)

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small and add overlap between chunks.
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current) < self.min_chunk_size:
                # Chunk too small, merge with next
                current = current + " " + next_chunk.lstrip()
            elif len(current) + len(next_chunk) <= self.chunk_size:
                # Can fit both in one chunk
                current = current + " " + next_chunk.lstrip()
            else:
                # Add current chunk and start new one with overlap
                merged.append(current.strip())

                # Add overlap from end of current to start of next
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    overlap = current[-self.chunk_overlap:]
                    # Try to start overlap at a word boundary
                    space_idx = overlap.find(" ")
                    if space_idx > 0:
                        overlap = overlap[space_idx + 1:]
                    current = overlap + " " + next_chunk.lstrip()
                else:
                    current = next_chunk

        if current.strip():
            merged.append(current.strip())

        return merged

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by character count when no separators are found."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to end at a word boundary
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return [c for c in chunks if c]

    # Regex matching [TABLE_PLACEHOLDER_N] tokens injected by _extract_tables
    _PLACEHOLDER_RE = re.compile(r"\[TABLE_PLACEHOLDER_\d+\]")

    def _merge_chunks_with_tables(
        self,
        text_without_tables: str,
        text_chunks: List[str],
        tables: List[tuple[int, str]],
    ) -> List[Chunk]:
        """
        Merge text chunks and table chunks in document order.

        Searches in text_without_tables (where the chunks originated) using
        an advancing cursor so repeated text gets the correct position.
        Tables are located by their placeholder positions in the same text.
        """
        all_items: List[Tuple[int, Chunk]] = []
        search_start = 0

        # Text chunks: split on placeholders, find each part with advancing cursor
        for chunk_text in text_chunks:
            if not chunk_text.strip():
                continue
            parts = self._PLACEHOLDER_RE.split(chunk_text)
            first_part = True
            for part in parts:
                # Strip lines containing partial placeholder tokens leaked via
                # chunk overlap slicing through [TABLE_PLACEHOLDER_N] markers.
                cleaned = "\n".join(
                    ln for ln in part.split("\n") if "TABLE_PLACEHOLDER" not in ln
                ).strip()
                if not cleaned:
                    continue
                part = cleaned
                prefix = part[:80]
                if first_part:
                    # At chunk boundary, back up by overlap for adjacent-chunk overlap
                    find_from = max(0, search_start - self.chunk_overlap)
                    first_part = False
                else:
                    # Within same chunk (parts separated by placeholders), advance only
                    find_from = search_start
                pos = text_without_tables.find(prefix, find_from)
                if pos == -1:
                    pos = len(text_without_tables)  # fallback: put at end
                else:
                    search_start = pos + len(prefix)
                all_items.append((pos, Chunk(
                    content=part,
                    metadata={"type": "text"},
                    is_table=False,
                )))

        # Table chunks: locate by placeholder positions in text_without_tables
        placeholder_tag = f"[TABLE_PLACEHOLDER_{len(tables)}]"
        ph_search = 0
        for _, table_content in tables:
            ph_pos = text_without_tables.find(placeholder_tag, ph_search)
            if ph_pos == -1:
                ph_pos = len(text_without_tables)
            else:
                ph_search = ph_pos + len(placeholder_tag)
            all_items.append((ph_pos, Chunk(
                content=table_content.strip(),
                metadata={"type": "table"},
                is_table=True,
            )))

        all_items.sort(key=lambda x: x[0])
        return [chunk for _, chunk in all_items]

    def get_stats(self, chunks: List[Chunk]) -> ChunkingStats:
        """Calculate statistics for a list of chunks."""
        if not chunks:
            return ChunkingStats()

        sizes = [c.length for c in chunks]
        table_chunks = sum(1 for c in chunks if c.is_table)

        return ChunkingStats(
            total_chunks=len(chunks),
            table_chunks=table_chunks,
            text_chunks=len(chunks) - table_chunks,
            avg_chunk_size=sum(sizes) / len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
        )


def demo_chunker():
    """Demo function to test the chunker on sample content."""
    sample = """# Components of PN

Typical Nutritional Components of PN include water and various macronutrients.

## Macronutrients

There are three major macronutrients in parenteral nutrition:
* Carbohydrates (Dextrose)
* Fat (Intravenous Lipid Emulsion - ILE)
* Protein (Amino Acids - AA)

Macronutrients may be combined in PN solution as follows:
* 2-in-1: Dextrose + AA (lipids administered separately)
* 3-in-1: Dextrose + AA + Lipids

### Protein Requirements

Protein requirements vary by patient population. For preterm infants, the recommended intake is 3-4 g/kg/day. For term infants, the range is 2.5-3 g/kg/day.

<table>
<tr><td>Patient Type</td><td>Protein (g/kg/day)</td><td>Dextrose (mg/kg/min)</td></tr>
<tr><td>Preterm Infant</td><td>3-4</td><td>10-14</td></tr>
<tr><td>Term Infant</td><td>2.5-3</td><td>6-8</td></tr>
<tr><td>Child (1-10 yr)</td><td>1.5-2.5</td><td>8-10</td></tr>
</table>

## Micronutrients

Micronutrients include electrolytes, minerals, vitamins, and trace elements. These must be carefully balanced to prevent deficiencies and toxicities.

### Electrolytes

Key electrolytes include sodium, potassium, chloride, and bicarbonate. Requirements vary based on clinical condition, losses, and renal function.

Reference: Corkins MR, et al. The A.S.P.E.N. Pediatric Nutrition Support Core Curriculum, 2nd Ed. ASPEN; 2015.
"""

    chunker = SemanticChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk(sample, source="demo.md")
    stats = chunker.get_stats(chunks)

    print("=" * 60)
    print("SEMANTIC CHUNKER DEMO")
    print("=" * 60)
    print(f"\nTotal chunks: {stats.total_chunks}")
    print(f"Table chunks: {stats.table_chunks}")
    print(f"Text chunks: {stats.text_chunks}")
    print(f"Avg chunk size: {stats.avg_chunk_size:.0f} chars")
    print(f"Min/Max: {stats.min_chunk_size} / {stats.max_chunk_size}")

    print("\n" + "=" * 60)
    print("CHUNKS:")
    print("=" * 60)

    for chunk in chunks:
        chunk_type = "TABLE" if chunk.is_table else "TEXT"
        print(f"\n[{chunk_type}] Chunk {chunk.metadata.get('chunk_index', '?')} ({chunk.length} chars):")
        print("-" * 40)
        preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(preview)


if __name__ == "__main__":
    demo_chunker()
