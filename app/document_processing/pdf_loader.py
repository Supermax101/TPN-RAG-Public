"""
PDF Loader for TPN Knowledge Base.

Supports loading PDF books and clinical documents using:
1. PyMuPDF4LLM (preferred - outputs Markdown with structure)
2. PyMuPDF (fallback - page-by-page extraction)
3. Unstructured (alternative - structure-aware)

Best Practices (from LangChain docs):
- Use chunk_size 512-1000 characters
- Use chunk_overlap 10-20% (100-200 characters)
- Preserve document structure (headers, tables)
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..logger import logger


# Clinical separators for medical documents
CLINICAL_SEPARATORS = [
    "\n## ",      # Markdown H2
    "\n### ",     # Markdown H3
    "\n#### ",    # Markdown H4
    "\n\n\n",     # Major break
    "\n\n",       # Paragraph
    "\n",         # Line
    ". ",         # Sentence
    "; ",         # Clinical list
    ", ",         # Enumeration
    " ",          # Word
]


class PDFLoader:
    """
    Production PDF loader for clinical documents.
    
    Usage:
        ```python
        loader = PDFLoader()
        documents = await loader.load_and_chunk("path/to/book.pdf")
        ```
    """
    
    def __init__(self):
        self._has_pymupdf4llm = self._check_pymupdf4llm()
        self._has_pymupdf = self._check_pymupdf()
    
    def _check_pymupdf4llm(self) -> bool:
        """Check if pymupdf4llm is available."""
        try:
            import pymupdf4llm
            return True
        except ImportError:
            return False
    
    def _check_pymupdf(self) -> bool:
        """Check if PyMuPDF is available."""
        try:
            import fitz  # PyMuPDF
            return True
        except ImportError:
            return False
    
    async def load_and_chunk(
        self,
        pdf_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        Load a PDF and chunk it for RAG.
        
        This method:
        1. Extracts text (preferring Markdown format)
        2. Cleans and normalizes content
        3. Chunks with clinical-aware separators
        4. Adds rich metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        # Try extraction methods in order of preference
        if self._has_pymupdf4llm:
            pages = self._extract_with_pymupdf4llm(pdf_path)
        elif self._has_pymupdf:
            pages = self._extract_with_pymupdf(pdf_path)
        else:
            pages = await self._extract_with_langchain(pdf_path)
        
        if not pages:
            logger.warning(f"No content extracted from {pdf_path.name}")
            return []
        
        # Chunk the content
        documents = self._chunk_content(
            pages=pages,
            source_file=pdf_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"Extracted {len(documents)} chunks from {pdf_path.name}")
        return documents
    
    def _extract_with_pymupdf4llm(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract with PyMuPDF4LLM (outputs Markdown)."""
        import pymupdf4llm
        
        logger.info("Using PyMuPDF4LLM for Markdown extraction")
        
        try:
            # Extract as Markdown (preserves structure)
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            
            # Split by pages if possible
            pages = []
            if "\n\n---\n\n" in md_text:
                # Some PDFs have page breaks
                page_texts = md_text.split("\n\n---\n\n")
                for i, text in enumerate(page_texts, 1):
                    pages.append({
                        "page_number": i,
                        "content": text.strip(),
                    })
            else:
                # Single document
                pages.append({
                    "page_number": 0,
                    "content": md_text.strip(),
                })
            
            return pages
            
        except Exception as e:
            logger.error(f"PyMuPDF4LLM extraction failed: {e}")
            return []
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract with PyMuPDF (page by page)."""
        import fitz
        
        logger.info("Using PyMuPDF for page-by-page extraction")
        
        pages = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                
                if text.strip():
                    pages.append({
                        "page_number": page_num,
                        "content": text.strip(),
                    })
            
            doc.close()
            return pages
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return []
    
    async def _extract_with_langchain(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract using LangChain's PDF loaders."""
        logger.info("Using LangChain PDF loader (fallback)")
        
        try:
            # Try different loaders
            try:
                from langchain_community.document_loaders import PyMuPDFLoader
                loader = PyMuPDFLoader(str(pdf_path))
            except ImportError:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(pdf_path))
            
            docs = loader.load()
            
            pages = []
            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1
                pages.append({
                    "page_number": page_num,
                    "content": doc.page_content.strip(),
                })
            
            return pages
            
        except Exception as e:
            logger.error(f"LangChain PDF loader failed: {e}")
            return []
    
    def _chunk_content(
        self,
        pages: List[Dict[str, Any]],
        source_file: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """Chunk extracted content with clinical-aware splitting."""
        
        # Create splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=CLINICAL_SEPARATORS,
            keep_separator=True,
            length_function=len,
        )
        
        all_documents = []
        chunk_index = 0
        
        for page in pages:
            page_num = page.get("page_number", 0)
            content = page.get("content", "")
            
            if not content or len(content.strip()) < 50:
                continue
            
            # Clean content
            content = self._clean_content(content)
            
            # Split
            chunks = splitter.split_text(content)
            
            for chunk_text in chunks:
                if len(chunk_text.strip()) < 50:
                    continue
                
                # Extract section header if present
                section_header = self._extract_section_header(chunk_text)
                
                # Detect content type
                has_dosing = self._contains_dosing(chunk_text)
                has_table = self._contains_table(chunk_text)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source_file,
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "section_header": section_header,
                        "contains_dosing": has_dosing,
                        "contains_table": has_table,
                        "chunk_size": len(chunk_text),
                        "doc_type": "pdf",
                    }
                )
                all_documents.append(doc)
                chunk_index += 1
        
        return all_documents
    
    def _clean_content(self, text: str) -> str:
        """Clean and normalize PDF text."""
        # Fix common PDF extraction issues
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Restore paragraph breaks
        text = re.sub(r'\. ([A-Z])', r'.\n\n\1', text)
        
        # Fix bullet points
        text = re.sub(r'•\s*', '\n• ', text)
        text = re.sub(r'‣\s*', '\n• ', text)
        
        # Fix numbered lists
        text = re.sub(r'(\d+)\.\s+', r'\n\1. ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+\s*\|\s*Chapter', '\nChapter', text)
        
        return text.strip()
    
    def _extract_section_header(self, text: str) -> Optional[str]:
        """Extract section header from chunk."""
        lines = text.split('\n')[:3]
        
        for line in lines:
            line = line.strip()
            
            # Markdown header
            if line.startswith('#'):
                return line.lstrip('#').strip()[:100]
            
            # Bold header
            if line.startswith('**') and line.endswith('**'):
                return line.strip('*').strip()[:100]
            
            # ALL CAPS header
            if line.isupper() and 10 < len(line) < 80:
                return line.title()[:100]
        
        return None
    
    def _contains_dosing(self, text: str) -> bool:
        """Check if chunk contains dosing information."""
        patterns = [
            r'\d+\.?\d*\s*(mg|g|mcg|mL|L|mEq|mmol|kcal)/kg',
            r'\d+\.?\d*\s*%\s*(dextrose|amino acid|lipid)',
            r'\d+\.?\d*\s*(mg|g|mcg|mL|L)/(?:kg/)?(?:day|hr)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _contains_table(self, text: str) -> bool:
        """Check if chunk appears to contain table data."""
        # Markdown tables
        if '|' in text and text.count('|') > 4:
            return True
        
        # Tab-separated data
        if '\t' in text and text.count('\t') > 2:
            return True
        
        return False


async def load_pdf_book(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Convenience function to load a PDF book.
    
    Usage:
        ```python
        from app.document_processing.pdf_loader import load_pdf_book
        
        docs = await load_pdf_book(Path("books/tpn_handbook.pdf"))
        ```
    """
    loader = PDFLoader()
    return await loader.load_and_chunk(pdf_path, chunk_size, chunk_overlap)
