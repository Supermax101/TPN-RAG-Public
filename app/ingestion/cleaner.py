"""
Document Cleaner for DPT2 OCR output.

This module removes OCR artifacts from DPT2-processed markdown files
while preserving clinically valuable content like tables, headers, and references.

Artifacts removed:
- UUID anchor tags: <a id='UUID'></a>
- Figure/logo descriptions: <::...::>
- CAPTION ERROR markers
- Excessive whitespace

Content preserved:
- HTML tables with clinical data
- Section headers (# ## ###)
- Bullet lists
- References and citations
- Clinical text and values
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


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
        """Calculate percentage reduction in document size."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.cleaned_length / self.original_length) * 100


class DocumentCleaner:
    """
    Clean DPT2 OCR artifacts from markdown documents.

    This cleaner is specifically designed for DPT2 output which contains:
    - UUID anchors for element tracking
    - Figure/logo descriptions in <::...::> format
    - CAPTION ERROR placeholders
    - Tables embedded in figure descriptions

    Example usage:
        >>> cleaner = DocumentCleaner()
        >>> cleaned_text, stats = cleaner.clean(raw_markdown)
        >>> print(f"Reduced by {stats.reduction_percent:.1f}%")
    """

    # Regex patterns for DPT2 artifacts
    PATTERNS = {
        # UUID anchor tags: <a id='225e3420-d6a6-4765-8cca-f34797da05d6'></a>
        "anchor": re.compile(
            r"<a\s+id=['\"][0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}['\"]\s*>\s*</a>",
            re.IGNORECASE
        ),

        # Figure/logo descriptions: <::...::>
        # But we need to check if it contains a table first
        "figure_block": re.compile(
            r"<::(.*?)::>",
            re.DOTALL
        ),

        # CAPTION ERROR standalone markers
        "caption_error": re.compile(
            r"^\s*CAPTION\s+ERROR\s*$",
            re.MULTILINE | re.IGNORECASE
        ),

        # Multiple consecutive blank lines
        "excess_newlines": re.compile(r"\n{3,}"),

        # HTML table id attributes (keep table, remove ids)
        "table_ids": re.compile(r'\s+id="[^"]*"'),
    }

    # Patterns that indicate a figure block contains valuable table data
    TABLE_INDICATORS = [
        "|",           # Markdown table separator
        "---",         # Table header separator
        "<table",      # HTML table
        "<tr>",        # HTML table row
    ]

    # Patterns that indicate a figure block contains valuable procedural/structured content
    VALUABLE_CONTENT_INDICATORS = [
        "step",        # Procedural steps
        "1.",          # Numbered steps
        "2.",          # Numbered steps
        "hours",       # Time-based protocols
        "timeline",    # Timelines
        "procedure",   # Procedural content
        "algorithm",   # Clinical algorithms
        "formula",     # Formulas
        "calculate",   # Calculations
        "monitor",     # Monitoring protocols
        "mg/kg",       # Dosing info
        "g/kg",        # Dosing info
        "ml/kg",       # Dosing info
        "kcal",        # Caloric info
    ]

    # Patterns that indicate a figure is just a logo or simple image description (remove)
    REMOVABLE_PATTERNS = [
        re.compile(r"^logo:", re.IGNORECASE),
        re.compile(r"^.*?logo\s*:?\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"naspghan\s*foundation", re.IGNORECASE),
        re.compile(r"^.*?illustration\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^:\s*figure\s*:?:?\s*>?\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    def __init__(self, preserve_table_ids: bool = False):
        """
        Initialize the document cleaner.

        Args:
            preserve_table_ids: If True, keep HTML id attributes on table elements.
                               Default False removes them for cleaner output.
        """
        self.preserve_table_ids = preserve_table_ids

    def clean(self, text: str, source_name: Optional[str] = None) -> tuple[str, CleaningStats]:
        """
        Clean DPT2 artifacts from markdown text.

        Args:
            text: Raw markdown text from DPT2 output
            source_name: Optional source filename for logging

        Returns:
            Tuple of (cleaned_text, cleaning_stats)

        Example:
            >>> cleaner = DocumentCleaner()
            >>> text = "<a id='abc-123'></a>\\n\\nSome content"
            >>> cleaned, stats = cleaner.clean(text)
            >>> print(cleaned)
            "Some content"
        """
        stats = CleaningStats(original_length=len(text))

        # Step 1: Remove UUID anchors
        text, anchor_count = self._remove_anchors(text)
        stats.anchors_removed = anchor_count

        # Step 2: Process figure blocks (extract tables, remove descriptions, preserve procedures)
        text, figure_count, table_count, procedure_count = self._process_figure_blocks(text)
        stats.figures_removed = figure_count
        stats.embedded_tables_extracted = table_count
        stats.procedures_preserved = procedure_count

        # Step 3: Count preserved HTML tables
        stats.tables_preserved = len(re.findall(r"<table", text, re.IGNORECASE))

        # Step 4: Remove CAPTION ERROR markers
        text, caption_count = self._remove_caption_errors(text)
        stats.caption_errors_removed = caption_count

        # Step 5: Clean up table IDs if requested
        if not self.preserve_table_ids:
            text = self.PATTERNS["table_ids"].sub("", text)

        # Step 6: Normalize whitespace
        text = self._normalize_whitespace(text)

        stats.cleaned_length = len(text)

        return text, stats

    def _remove_anchors(self, text: str) -> tuple[str, int]:
        """Remove UUID anchor tags and count removals."""
        matches = self.PATTERNS["anchor"].findall(text)
        cleaned = self.PATTERNS["anchor"].sub("", text)
        return cleaned, len(matches)

    def _process_figure_blocks(self, text: str) -> tuple[str, int, int, int]:
        """
        Process <::...::> figure blocks.

        - If block contains table data, extract and convert to markdown
        - If block contains valuable procedural/structured content, preserve it
        - Otherwise, remove the block entirely

        Returns:
            Tuple of (processed_text, figures_removed, tables_extracted, procedures_preserved)
        """
        figures_removed = 0
        tables_extracted = 0
        procedures_preserved = 0

        def replace_figure_block(match):
            nonlocal figures_removed, tables_extracted, procedures_preserved
            content = match.group(1)
            content_lower = content.lower()

            # First check if it's clearly a removable pattern (logos, simple illustrations)
            if self._is_removable_figure(content):
                figures_removed += 1
                return ""

            # Check if this contains table data
            if self._contains_table_data(content):
                table = self._extract_table_from_figure(content)
                if table:
                    tables_extracted += 1
                    return "\n" + table + "\n"

            # Check if this contains valuable procedural/structured content
            if self._contains_valuable_content(content):
                # Clean and preserve the content
                cleaned_content = self._clean_figure_content(content)
                if cleaned_content:
                    procedures_preserved += 1
                    return "\n" + cleaned_content + "\n"

            # Default: remove the figure block
            figures_removed += 1
            return ""

        cleaned = self.PATTERNS["figure_block"].sub(replace_figure_block, text)
        return cleaned, figures_removed, tables_extracted, procedures_preserved

    def _is_removable_figure(self, content: str) -> bool:
        """Check if figure is clearly removable (logo, simple illustration)."""
        # Short content that's just a figure marker
        if len(content.strip()) < 50:
            return True

        # Check against removable patterns
        for pattern in self.REMOVABLE_PATTERNS:
            if pattern.search(content):
                return True

        return False

    def _contains_valuable_content(self, content: str) -> bool:
        """Check if figure block contains valuable procedural/structured content."""
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in self.VALUABLE_CONTENT_INDICATORS)

    def _clean_figure_content(self, content: str) -> Optional[str]:
        """
        Clean figure content to extract valuable text.

        Removes image descriptions but keeps structured information.
        """
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip lines that are just figure markers
            if line.lower() in [": figure", ": figure::", ": illustration", "figure::", "illustration::"]:
                continue
            # Skip lines that start with [Image:
            if line.lower().startswith("[image:"):
                continue
            # Skip lines that are just describing an image
            if line.lower().startswith("a ") and any(x in line.lower() for x in ["image", "photo", "picture", "diagram shows", "illustration shows"]):
                continue
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        # Only return if there's substantial content left
        return result if len(result) > 50 else None

    def _contains_table_data(self, content: str) -> bool:
        """Check if figure block content contains table data."""
        return any(indicator in content for indicator in self.TABLE_INDICATORS)

    def _extract_table_from_figure(self, content: str) -> Optional[str]:
        """
        Extract table content from a figure block.

        Handles both markdown tables and attempts to parse
        structured text into markdown table format.
        """
        # Check for markdown table format (pipes and dashes)
        if "|" in content and "---" in content:
            # Extract just the table portion
            lines = content.split("\n")
            table_lines = []
            in_table = False

            for line in lines:
                stripped = line.strip()
                if "|" in stripped:
                    in_table = True
                    table_lines.append(stripped)
                elif in_table and stripped and not stripped.startswith(":"):
                    # End of table if we hit non-table content
                    if "---" not in stripped and "|" not in stripped:
                        break
                    table_lines.append(stripped)

            if table_lines:
                return "\n".join(table_lines)

        # Check for HTML table
        if "<table" in content.lower():
            # Return the HTML table as-is
            table_match = re.search(r"<table.*?</table>", content, re.DOTALL | re.IGNORECASE)
            if table_match:
                return table_match.group(0)

        return None

    def _remove_caption_errors(self, text: str) -> tuple[str, int]:
        """Remove CAPTION ERROR markers."""
        matches = self.PATTERNS["caption_error"].findall(text)
        cleaned = self.PATTERNS["caption_error"].sub("", text)
        return cleaned, len(matches)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize excessive whitespace while preserving structure."""
        # Replace 3+ newlines with 2
        text = self.PATTERNS["excess_newlines"].sub("\n\n", text)

        # Strip trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Strip leading/trailing whitespace from document
        return text.strip()

    def clean_file(self, input_path: Path, output_path: Optional[Path] = None) -> CleaningStats:
        """
        Clean a markdown file and optionally save to output path.

        Args:
            input_path: Path to input markdown file
            output_path: Optional path for cleaned output. If None, modifies in place.

        Returns:
            CleaningStats from the cleaning process
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text, stats = self.clean(raw_text, source_name=input_path.name)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

        return stats


def demo_cleaner():
    """Demo function to test the cleaner on sample content."""
    sample = """<a id='225e3420-d6a6-4765-8cca-f34797da05d6'></a>

Components of PN

<a id='182251f5-0f7a-4fe7-a020-58d3e1f42a12'></a>

<::NASPGHAN FOUNDATION
: figure::>

<a id='7f218b6b-dfd5-41d9-9fa9-1c04c8f88617'></a>

CAPTION ERROR

<a id='c5903626-47a4-4335-8b4f-db79cd2644b4'></a>

Typical Nutritional Components of PN

<a id='9e82c334-48a8-4843-83e3-8c07bc221127'></a>

Water

Macronutrients
* Protein
* Carbohydrate
* Fat

<a id='fc44e57b-a056-4e69-87c4-cacb0d5c4736'></a>

# Micronutrients
* Electrolytes & Minerals
* Vitamins
* Trace elements

<table id="3-1">
<tr><td id="3-2">Protein (g/kg/day)</td><td id="3-3">3 – 4</td></tr>
<tr><td id="3-4">Dextrose (mg/kg/min)</td><td id="3-5">10 – 14</td></tr>
</table>

<::Children (1-10 yr) | Initiation | Goals
---|---|---
Protein (g/kg/day) | 1.5-2.5 | 1.5-2.5
Dextrose (mg/kg/min) | 3-6 | 8-10
: table::>

<::logo: NASPGHAN
The logo features dark blue text on gradient background.::>

Corkins MR, et al. The A.S.P.E.N. Pediatric Nutrition Support Core Curriculum, 2nd Ed. ASPEN; 2015.
"""

    cleaner = DocumentCleaner()
    cleaned, stats = cleaner.clean(sample)

    print("=" * 60)
    print("DOCUMENT CLEANER DEMO")
    print("=" * 60)
    print(f"\nOriginal length: {stats.original_length}")
    print(f"Cleaned length: {stats.cleaned_length}")
    print(f"Reduction: {stats.reduction_percent:.1f}%")
    print(f"\nAnchors removed: {stats.anchors_removed}")
    print(f"Figures removed: {stats.figures_removed}")
    print(f"Procedures preserved: {stats.procedures_preserved}")
    print(f"Caption errors removed: {stats.caption_errors_removed}")
    print(f"Tables preserved: {stats.tables_preserved}")
    print(f"Embedded tables extracted: {stats.embedded_tables_extracted}")
    print("\n" + "=" * 60)
    print("CLEANED OUTPUT:")
    print("=" * 60)
    print(cleaned)


if __name__ == "__main__":
    demo_cleaner()
