"""
Optimized Data Loader for TPN Knowledge Base.

This loader is specifically optimized for the pre-parsed documents in data/documents.
These documents are already high-quality parsed content with:
- Anchor IDs for sections
- Pre-chunked segments in JSON
- Tables in HTML format
- Clinical dosing data

Strategy:
1. For JSON files: Use the pre-chunked segments (already optimally chunked)
2. For MD files: Use clinical-aware semantic chunking
3. Clean up parsing artifacts (CAPTION ERROR, logos, anchors)
4. Preserve table structures
5. Enrich metadata with section headers and dosing detection
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logger import logger


# =============================================================================
# CLINICAL SEPARATORS - Optimized for TPN content
# =============================================================================

CLINICAL_SEPARATORS = [
    # Major section breaks
    "\n# ",       # H1
    "\n## ",      # H2
    "\n### ",     # H3
    "\n#### ",    # H4
    
    # Clinical content boundaries
    "\n---\n",    # Horizontal rule
    "\n\n\n",     # Major paragraph break
    "\n\n",       # Paragraph
    
    # Table boundaries
    "</table>",   # End of table
    "<table",     # Start of table
    
    # Sentence/clause
    ". ",
    ";\n",
    "\n* ",       # Bullet points
    "\n- ",       # Dashes
    
    # Fallback
    "\n",
    " ",
]


# =============================================================================
# CONTENT CLEANING
# =============================================================================

def clean_content(text: str) -> str:
    """Clean parsing artifacts from content."""
    
    # Remove anchor tags
    text = re.sub(r'<a id=[\'"][^"\']+[\'"]></a>\s*', '', text)
    
    # Remove CAPTION ERROR
    text = re.sub(r'CAPTION ERROR\s*', '', text)
    
    # Remove logo descriptions
    text = re.sub(r'\s*<::logo:[^>]+:>\s*', '', text)
    text = re.sub(r'\s*<::[^:>]*NASPGHAN[^:>]*::\s*figure::\s*>\s*', '', text)
    text = re.sub(r'NASPGHAN\s*FOUNDATION\s*', '', text)
    
    # Remove figure markers that are just logos
    text = re.sub(r'<::NASPGHAN FOUNDATION[^>]*::\s*figure::\s*>', '', text)
    
    # Clean up table/figure markers but keep content
    text = re.sub(r'<::\s*table\s*::\s*>', '', text)
    text = re.sub(r'::\s*table::', '', text)
    text = re.sub(r'::\s*figure::', '', text)
    text = re.sub(r'<::', '', text)
    text = re.sub(r'::>', '', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Remove lines that are just whitespace
    lines = [line for line in text.split('\n') if line.strip() or not line]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_section_header(text: str) -> Optional[str]:
    """Extract section header from text."""
    lines = text.split('\n')[:5]
    
    for line in lines:
        line = line.strip()
        
        # Markdown header
        if line.startswith('#'):
            return line.lstrip('#').strip()[:100]
        
        # Title-case line (likely a header)
        if 10 < len(line) < 100 and not line.startswith('*') and not line.startswith('-'):
            if line[0].isupper():
                return line[:100]
    
    return None


def contains_clinical_data(text: str) -> Dict[str, bool]:
    """Detect clinical data types in text."""
    return {
        'has_dosing': bool(re.search(
            r'\d+\.?\d*\s*(mg|g|mcg|mL|mEq|mmol|kcal)/kg|'
            r'\d+\.?\d*\s*%\s*(dextrose|amino|lipid)|'
            r'\d+\.?\d*\s*g/kg/day',
            text, re.IGNORECASE
        )),
        'has_table': bool(re.search(r'<table|</table>|\|.*\|.*\|', text)),
        'has_calculation': bool(re.search(
            r'GIR|glucose infusion rate|'
            r'mOsm|osmolality|'
            r'ratio|formula',
            text, re.IGNORECASE
        )),
        'has_guidelines': bool(re.search(
            r'ASPEN|ESPGHAN|NASPGHAN|guideline|recommendation',
            text, re.IGNORECASE
        )),
    }


# =============================================================================
# JSON LOADER - Uses pre-chunked content
# =============================================================================

def load_json_document(json_path: Path) -> List[Document]:
    """
    Load a pre-chunked JSON document.
    
    These JSON files contain already-chunked segments from PDF parsing.
    We use these directly as they are already optimally segmented.
    """
    documents = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return []
    
    source_name = json_path.stem.replace('_response', '')
    
    # Check if this is a chunked document
    if 'chunks' in data and isinstance(data['chunks'], list):
        chunks = data['chunks']
    else:
        # Fallback: treat markdown as single doc
        chunks = [{'markdown': data.get('markdown', ''), 'type': 'text'}]
    
    chunk_index = 0
    for chunk in chunks:
        markdown = chunk.get('markdown', '')
        chunk_type = chunk.get('type', 'text')
        
        # Skip logos, figures without content
        if chunk_type in ['logo', 'figure'] and 'NASPGHAN' in markdown:
            continue
        if len(markdown.strip()) < 50:
            continue
        
        # Clean content
        cleaned = clean_content(markdown)
        if len(cleaned.strip()) < 50:
            continue
        
        # Extract metadata
        section = extract_section_header(cleaned)
        clinical = contains_clinical_data(cleaned)
        
        # Create document
        doc = Document(
            page_content=cleaned,
            metadata={
                'source': source_name,
                'chunk_type': chunk_type,
                'chunk_index': chunk_index,
                'section_header': section,
                'has_dosing': clinical['has_dosing'],
                'has_table': clinical['has_table'],
                'has_calculation': clinical['has_calculation'],
                'has_guidelines': clinical['has_guidelines'],
                'page': chunk.get('grounding', {}).get('page', 0) if 'grounding' in chunk else 0,
            }
        )
        documents.append(doc)
        chunk_index += 1
    
    logger.info(f"Loaded {len(documents)} chunks from {json_path.name}")
    return documents


# =============================================================================
# MARKDOWN LOADER - With semantic chunking
# =============================================================================

def load_markdown_document(
    md_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Load a Markdown document with semantic chunking.
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to load {md_path}: {e}")
        return []
    
    source_name = md_path.stem
    
    # Clean content
    cleaned = clean_content(content)
    
    if len(cleaned.strip()) < 100:
        return []
    
    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CLINICAL_SEPARATORS,
        keep_separator=True,
        length_function=len,
    )
    
    # Split
    chunks = splitter.split_text(cleaned)
    
    documents = []
    for i, chunk_text in enumerate(chunks):
        if len(chunk_text.strip()) < 50:
            continue
        
        section = extract_section_header(chunk_text)
        clinical = contains_clinical_data(chunk_text)
        
        doc = Document(
            page_content=chunk_text,
            metadata={
                'source': source_name,
                'chunk_type': 'text',
                'chunk_index': i,
                'section_header': section,
                'has_dosing': clinical['has_dosing'],
                'has_table': clinical['has_table'],
                'has_calculation': clinical['has_calculation'],
                'has_guidelines': clinical['has_guidelines'],
            }
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} chunks from {md_path.name}")
    return documents


# =============================================================================
# MAIN LOADER
# =============================================================================

class TPNDataLoader:
    """
    Production data loader for TPN knowledge base.
    
    Optimized for the pre-parsed documents in data/documents.
    """
    
    def __init__(
        self,
        data_dir: Path = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prefer_json: bool = True,
    ):
        from app.config import settings
        self.data_dir = data_dir or settings.documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prefer_json = prefer_json
    
    def load_all(self) -> List[Document]:
        """Load all documents from the data directory."""
        documents = []
        
        json_files = list(self.data_dir.glob('*_response.json'))
        md_files = list(self.data_dir.glob('*.md'))
        
        logger.info(f"Found {len(json_files)} JSON files, {len(md_files)} MD files")
        
        # Track which sources we've loaded (prefer JSON)
        loaded_sources = set()
        
        if self.prefer_json:
            # Load JSON first (pre-chunked)
            for json_file in json_files:
                docs = load_json_document(json_file)
                documents.extend(docs)
                source = json_file.stem.replace('_response', '')
                loaded_sources.add(source)
            
            # Load remaining MD files
            for md_file in md_files:
                source = md_file.stem
                if source not in loaded_sources:
                    docs = load_markdown_document(
                        md_file, self.chunk_size, self.chunk_overlap
                    )
                    documents.extend(docs)
        else:
            # Load all files
            for json_file in json_files:
                documents.extend(load_json_document(json_file))
            for md_file in md_files:
                documents.extend(load_markdown_document(
                    md_file, self.chunk_size, self.chunk_overlap
                ))
        
        logger.info(f"Loaded {len(documents)} total chunks")
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the data."""
        json_files = list(self.data_dir.glob('*_response.json'))
        md_files = list(self.data_dir.glob('*.md'))
        
        return {
            'json_files': len(json_files),
            'md_files': len(md_files),
            'data_dir': str(self.data_dir),
            'prefer_json': self.prefer_json,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_tpn_data() -> List[Document]:
    """Load all TPN documents."""
    loader = TPNDataLoader()
    return loader.load_all()


def rebuild_vectorstore(force: bool = False) -> int:
    """
    Rebuild the vector store from TPN data.
    
    Returns the number of documents indexed.
    """
    from app.config import settings
    
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
    
    logger.info("Loading TPN documents...")
    loader = TPNDataLoader()
    documents = loader.load_all()
    
    if not documents:
        logger.error("No documents loaded!")
        return 0
    
    logger.info(f"Loaded {len(documents)} chunks")
    
    # Initialize embeddings
    embed_model = settings.ollama_embed_model or "nomic-embed-text"
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url=settings.ollama_base_url
    )
    
    # Clear existing
    persist_dir = str(settings.chromadb_dir)
    
    if force:
        import shutil
        if settings.chromadb_dir.exists():
            shutil.rmtree(settings.chromadb_dir)
            logger.info("Cleared existing vector store")
    
    # Create vector store
    settings.chromadb_dir.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=persist_dir,
    )
    
    count = vectorstore._collection.count()
    logger.info(f"Vector store rebuilt with {count} chunks")
    
    return count
