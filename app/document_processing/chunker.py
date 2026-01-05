"""
Semantic Chunking Pipeline for Clinical Documents.

This module implements production-grade document chunking specifically
designed for TPN (Total Parenteral Nutrition) clinical guidelines.

Key Features:
- RecursiveCharacterTextSplitter with clinical-aware separators
- Preserves medical terminology and dosing information
- Adds rich metadata for retrieval enhancement
- Handles tables and structured clinical data
"""

import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..logger import logger
from .models import ChunkMetadata, ProcessedDocument


# Clinical-specific separators that preserve semantic boundaries
CLINICAL_SEPARATORS = [
    "\n## ",           # Markdown H2 (major sections)
    "\n### ",          # Markdown H3 (subsections)
    "\n#### ",         # Markdown H4 (sub-subsections)
    "\n\n\n",          # Triple newline (major breaks)
    "\n\n",            # Double newline (paragraphs)
    "\n",              # Single newline
    ". ",              # Sentence boundary (period + space)
    "; ",              # Clinical listing separator
    ", ",              # Enumeration
    " ",               # Word boundary (last resort)
]

# Patterns to identify important clinical content
DOSING_PATTERNS = [
    r'\d+\.?\d*\s*(mg|g|mcg|mL|L|mEq|mmol|kcal|cal)/kg',
    r'\d+\.?\d*\s*%\s*(dextrose|amino acid|lipid)',
    r'\d+\.?\d*\s*(mg|g|mcg|mL|L|mEq|mmol)/(?:kg/)?(?:day|hr|hour|min)',
]

CALCULATION_PATTERNS = [
    r'=\s*\d+',
    r'×|÷|±',
    r'formula|equation|calculate|calculation',
]


class ClinicalTextSplitter(RecursiveCharacterTextSplitter):
    """
    Text splitter optimized for clinical TPN documents.
    
    Extends RecursiveCharacterTextSplitter with:
    - Clinical-aware separators
    - Medical terminology preservation
    - Dosing information detection
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CLINICAL_SEPARATORS,
            keep_separator=True,
            is_separator_regex=False,
            length_function=len,
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text while preserving clinical integrity."""
        # Pre-process: protect dosing patterns from splitting
        protected_text = self._protect_clinical_terms(text)
        
        # Split using parent method
        chunks = super().split_text(protected_text)
        
        # Post-process: restore protected terms
        chunks = [self._restore_clinical_terms(c) for c in chunks]
        
        # Filter out tiny/useless chunks
        chunks = [c for c in chunks if len(c.strip()) >= 50]
        
        return chunks
    
    def _protect_clinical_terms(self, text: str) -> str:
        """Protect clinical terms from being split."""
        # Protect mg/kg/day type patterns from splitting on /
        text = re.sub(r'(\d+\.?\d*)\s*/\s*(kg|day|hr|min)', r'\1⌀\2', text)
        return text
    
    def _restore_clinical_terms(self, text: str) -> str:
        """Restore protected clinical terms."""
        return text.replace('⌀', '/')


class SemanticChunker:
    """
    Production-grade semantic chunking for TPN clinical documents.
    
    This class orchestrates the complete chunking pipeline:
    1. Load raw documents (JSON or Markdown)
    2. Extract and merge content
    3. Apply semantic splitting
    4. Enrich with metadata
    5. Return LangChain Document objects
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.splitter = ClinicalTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"SemanticChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def process_json_document(self, json_path: Path) -> List[Document]:
        """
        Process a pre-parsed JSON document from the documents directory.
        
        This handles the existing JSON format from the document parsing service,
        but RE-CHUNKS the content properly instead of using arbitrary chunks.
        """
        start_time = time.time()
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {json_path}: {e}")
            return []
        
        metadata = data.get('metadata', {})
        source_filename = metadata.get('filename', json_path.stem)
        page_count = metadata.get('page_count', 0)
        
        # Strategy 1: If there's a 'markdown' field at root, use it
        if 'markdown' in data and data['markdown']:
            full_content = data['markdown']
        else:
            # Strategy 2: Merge chunks from parsing service, but we'll re-chunk
            chunks = data.get('chunks', [])
            content_parts = []
            
            for chunk in chunks:
                chunk_type = chunk.get('type', 'text')
                chunk_markdown = chunk.get('markdown', '')
                
                # Skip non-text elements like logos, figures without captions
                if chunk_type in ['logo', 'page_break', 'header', 'footer']:
                    continue
                if 'CAPTION ERROR' in chunk_markdown:
                    continue
                if len(chunk_markdown.strip()) < 20:
                    continue
                
                # Clean the content
                cleaned = self._clean_content(chunk_markdown)
                if cleaned:
                    content_parts.append(cleaned)
            
            full_content = '\n\n'.join(content_parts)
        
        if not full_content.strip():
            logger.warning(f"No content extracted from {json_path}")
            return []
        
        # NOW properly chunk the merged content
        text_chunks = self.splitter.split_text(full_content)
        
        # Create LangChain Document objects with rich metadata
        documents = []
        for i, chunk_text in enumerate(text_chunks):
            # Detect chunk characteristics
            has_dosing = self._contains_dosing(chunk_text)
            has_calculation = self._contains_calculation(chunk_text)
            section_header = self._extract_section_header(chunk_text)
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source_filename,
                    "doc_id": source_filename.replace('.pdf', '').replace('.md', ''),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "page_count": page_count,
                    "section_header": section_header,
                    "contains_dosing": has_dosing,
                    "contains_calculation": has_calculation,
                    "chunk_size": len(chunk_text),
                }
            )
            documents.append(doc)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Processed {source_filename}: {len(documents)} chunks in {processing_time:.0f}ms")
        
        return documents
    
    def process_markdown_file(self, md_path: Path) -> List[Document]:
        """Process a raw markdown file directly."""
        start_time = time.time()
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {md_path}: {e}")
            return []
        
        source_filename = md_path.name
        
        # Clean and chunk
        cleaned_content = self._clean_content(content)
        text_chunks = self.splitter.split_text(cleaned_content)
        
        # Create documents
        documents = []
        for i, chunk_text in enumerate(text_chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source_filename,
                    "doc_id": md_path.stem,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "section_header": self._extract_section_header(chunk_text),
                    "contains_dosing": self._contains_dosing(chunk_text),
                    "contains_calculation": self._contains_calculation(chunk_text),
                    "chunk_size": len(chunk_text),
                }
            )
            documents.append(doc)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Processed {source_filename}: {len(documents)} chunks in {processing_time:.0f}ms")
        
        return documents
    
    def process_all_documents(self, documents_dir: Path) -> List[Document]:
        """Process all documents in a directory."""
        all_documents = []
        
        # Process JSON files (from parsing service)
        json_files = list(documents_dir.glob("*_response.json"))
        logger.info(f"Found {len(json_files)} JSON documents")
        
        for json_file in json_files:
            docs = self.process_json_document(json_file)
            all_documents.extend(docs)
        
        # Also process raw markdown files if no corresponding JSON
        md_files = list(documents_dir.glob("*.md"))
        for md_file in md_files:
            # Skip if there's already a JSON version
            json_version = documents_dir / f"{md_file.stem}_response.json"
            if json_version.exists():
                continue
            
            docs = self.process_markdown_file(md_file)
            all_documents.extend(docs)
        
        logger.info(f"Total: {len(all_documents)} chunks from {len(json_files)} documents")
        return all_documents
    
    def _clean_content(self, text: str) -> str:
        """Clean markdown content for better chunking."""
        # Remove HTML anchor tags
        text = re.sub(r"<a id='[^']+'>\\s*</a>\\s*", '', text)
        # Remove figure placeholders
        text = re.sub(r'<::.*?::>', '', text, flags=re.DOTALL)
        # Normalize excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Remove excessive whitespace
        text = re.sub(r' {3,}', ' ', text)
        return text.strip()
    
    def _extract_section_header(self, text: str) -> Optional[str]:
        """Extract section header from chunk text."""
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Markdown header
                return line.lstrip('#').strip()[:100]
            if line.startswith('**') and line.endswith('**'):
                # Bold header
                return line.strip('*').strip()[:100]
        return None
    
    def _contains_dosing(self, text: str) -> bool:
        """Check if chunk contains dosing information."""
        text_lower = text.lower()
        for pattern in DOSING_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _contains_calculation(self, text: str) -> bool:
        """Check if chunk contains calculations or formulas."""
        text_lower = text.lower()
        for pattern in CALCULATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False


def rechunk_documents(
    documents_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Convenience function to rechunk all documents in a directory.
    
    Usage:
        from app.document_processing import rechunk_documents
        docs = rechunk_documents(Path("data/documents"))
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.process_all_documents(documents_dir)
