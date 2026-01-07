"""
Simple TPN Data Loader - Uses Pre-Chunked Segments Directly.

The JSON files in data/documents already contain optimally chunked segments
from PDF parsing. We use these DIRECTLY without re-chunking.

This is the optimal approach because:
1. Chunks are already semantically segmented
2. Context is preserved (tables stay together, sections stay together)
3. We get page-level grounding for citations
4. No risk of breaking clinical tables or dosing info

Usage:
    from app.simple_loader import load_all_documents, rebuild_index
    
    # Load all pre-chunked documents
    documents = load_all_documents()
    
    # Rebuild the vector store
    count = rebuild_index()
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from app.config import settings
from app.logger import logger


# =============================================================================
# CONTENT CLEANING - Minimal, just remove noise
# =============================================================================

def clean_chunk(text: str) -> str:
    """Clean obvious noise from chunk content."""
    
    # Remove anchor tags (keep content clean)
    text = re.sub(r'<a id=[\'"][^"\']+[\'"]></a>\s*', '', text)
    
    # Remove CAPTION ERROR artifacts
    text = re.sub(r'\s*CAPTION ERROR\s*', '', text)
    
    # Remove standalone logo text
    text = re.sub(r'^\s*NASPGHAN\s*FOUNDATION\s*$', '', text, flags=re.MULTILINE)
    
    # Remove logo figure markers
    text = re.sub(r'<::NASPGHAN[^>]*::\s*\w+::\s*>', '', text)
    text = re.sub(r'<::logo:[^>]+:>', '', text)
    
    # Clean up empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def is_useful_chunk(chunk: dict, cleaned_text: str) -> bool:
    """Check if chunk contains useful content."""
    
    chunk_type = chunk.get('type', 'text')
    
    # Skip pure logos
    if chunk_type == 'logo':
        return False
    
    # Skip very short content
    if len(cleaned_text) < 30:
        return False
    
    # Skip chunks that are just organization names
    if cleaned_text.strip() in ['NASPGHAN', 'FOUNDATION', 'NASPGHAN FOUNDATION']:
        return False
    
    return True


def extract_metadata(chunk: dict, cleaned_text: str, source: str) -> Dict[str, Any]:
    """Extract useful metadata from chunk."""
    
    metadata = {
        'source': source,
        'chunk_id': chunk.get('id', ''),
        'chunk_type': chunk.get('type', 'text'),
    }
    
    # Page from grounding
    grounding = chunk.get('grounding', {})
    if grounding and 'page' in grounding:
        metadata['page'] = grounding['page']
    
    # Extract section header (first line if it looks like a header)
    lines = cleaned_text.split('\n')[:3]
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            metadata['section'] = line.lstrip('#').strip()[:100]
            break
        elif 10 < len(line) < 80 and line[0].isupper() and ':' in line[:50]:
            metadata['section'] = line[:100]
            break
    
    # Detect clinical content types
    metadata['has_dosing'] = bool(re.search(
        r'\d+\.?\d*\s*(mg|g|mcg|mL|mEq|mmol|kcal)/kg|'
        r'\d+\.?\d*\s*g/kg/day|'
        r'\d+\.?\d*\s*mg/kg/min',
        cleaned_text, re.IGNORECASE
    ))
    
    metadata['has_table'] = bool(re.search(r'<table|</table>|\|.*\|.*\|', cleaned_text))
    
    return metadata


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_json_file(json_path: Path) -> List[Document]:
    """Load pre-chunked segments from a JSON file."""
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {json_path.name}: {e}")
        return []
    
    source = json_path.stem.replace('_response', '')
    chunks = data.get('chunks', [])
    
    if not chunks:
        # Fallback: use the full markdown as a single chunk
        md = data.get('markdown', '')
        if md and len(md) > 100:
            cleaned = clean_chunk(md)
            return [Document(page_content=cleaned, metadata={'source': source, 'chunk_type': 'full'})]
        return []
    
    documents = []
    for chunk in chunks:
        markdown = chunk.get('markdown', '')
        cleaned = clean_chunk(markdown)
        
        if not is_useful_chunk(chunk, cleaned):
            continue
        
        metadata = extract_metadata(chunk, cleaned, source)
        
        doc = Document(page_content=cleaned, metadata=metadata)
        documents.append(doc)
    
    return documents


def load_all_documents() -> List[Document]:
    """
    Load all documents from data/documents.
    
    Uses the pre-chunked JSON files directly.
    """
    documents_dir = settings.documents_dir
    
    if not documents_dir.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        return []
    
    json_files = list(documents_dir.glob('*_response.json'))
    logger.info(f"Found {len(json_files)} JSON files to load")
    
    all_documents = []
    for json_file in json_files:
        docs = load_json_file(json_file)
        all_documents.extend(docs)
    
    logger.info(f"Loaded {len(all_documents)} chunks from {len(json_files)} files")
    return all_documents


def rebuild_index(force: bool = False) -> int:
    """
    Rebuild the vector store from pre-chunked documents.

    Returns the number of documents indexed.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Load documents
    documents = load_all_documents()

    if not documents:
        logger.error("No documents loaded!")
        return 0

    # Clear if force
    if force and settings.chromadb_dir.exists():
        import shutil
        shutil.rmtree(settings.chromadb_dir)
        logger.info("Cleared existing vector store")

    # Ensure directory exists
    settings.chromadb_dir.mkdir(parents=True, exist_ok=True)

    # Create HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"trust_remote_code": True}
    )

    # Build vector store
    logger.info(f"Creating vector store with {len(documents)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chromadb_dir),
    )

    count = vectorstore._collection.count()
    logger.info(f"Vector store ready with {count} chunks")

    return count


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rebuild':
        print("Rebuilding index...")
        count = rebuild_index(force=True)
        print(f"Done! Indexed {count} chunks.")
    else:
        print("Loading documents (dry run)...")
        docs = load_all_documents()
        print(f"Loaded {len(docs)} chunks")
        
        # Show sample
        if docs:
            sample = docs[0]
            print(f"\nSample chunk from '{sample.metadata.get('source')}':")
            print(f"  Type: {sample.metadata.get('chunk_type')}")
            print(f"  Has dosing: {sample.metadata.get('has_dosing')}")
            print(f"  Content preview: {sample.page_content[:200]}...")
