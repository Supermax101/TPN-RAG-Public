"""
Document loader service.
Reads pre-chunked JSON documents and loads them into the vector store.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..data_models import DocumentChunk
from ..config import settings
from ..logger import logger


class DocumentLoader:
    """Loads documents from JSON files into the RAG system."""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.documents_dir = settings.documents_dir
        
        if not self.documents_dir.exists():
            raise ValueError(f"Documents directory not found: {self.documents_dir}")
    
    async def load_all_documents(self) -> Dict[str, Any]:
        """Loads all JSON documents from the documents directory."""
        logger.info("Loading documents into vector store...")
        
        json_files = list(self.documents_dir.glob("*_response.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {self.documents_dir}")
            return {"loaded": 0, "failed": 0, "total_chunks": 0}
        
        logger.info(f"Found {len(json_files)} documents")
        
        loaded = 0
        failed = 0
        total_chunks = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                logger.info(f"Processing {i}/{len(json_files)}: {json_file.stem}")
                chunks = await self._load_document_chunks(json_file)
                
                if chunks:
                    doc_name = json_file.stem.replace("_response", "")
                    await self.rag_service.add_document_chunks(chunks, doc_name)
                    loaded += 1
                    total_chunks += len(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks from {doc_name}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to load {json_file.name}: {e}")
        
        result = {"loaded": loaded, "failed": failed, "total_chunks": total_chunks}
        logger.info(f"Loading complete: {result}")
        return result
    
    async def _load_document_chunks(self, json_file: Path) -> List[DocumentChunk]:
        """Parses a JSON file and returns a list of DocumentChunks."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}")
            return []
        
        metadata = data.get('metadata', {})
        source_filename = metadata.get('filename', json_file.stem)
        page_count = metadata.get('page_count', 0)
        chunks = data.get('chunks', [])
        
        if not chunks:
            return []
        
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('id', f"{source_filename}_chunk_{i}")
            chunk_type = chunk.get('type', 'unknown')
            chunk_markdown = chunk.get('markdown', '')
            grounding = chunk.get('grounding', {})
            
            cleaned_content = self._clean_content(chunk_markdown)
            
            # Filter noise
            if "CAPTION ERROR" in cleaned_content:
                continue
            if len(cleaned_content) < 20 and not cleaned_content.startswith('#'):
                continue
            
            page = grounding.get('page', 0)
            bounding_box = grounding.get('box', {})
            section_name = self._extract_section(cleaned_content)
            
            # Contextualize content for better embedding/retrieval
            # Prepending Document Name and Section helps retrieval finding the right scope
            contextualized_content = f"Source: {source_filename}\nSection: {section_name}\n\n{cleaned_content}"
            
            chunk_metadata = {
                "source_file": source_filename,
                "chunk_type": chunk_type,
                "page": page,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "bounding_box": json.dumps(bounding_box) if bounding_box else "",
                "page_count": page_count,
            }
            
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=source_filename,
                content=contextualized_content,
                chunk_type=chunk_type,
                section=section_name,
                metadata=chunk_metadata
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def _clean_content(self, markdown: str) -> str:
        """Removes HTML artifacts and normalizes whitespace."""
        content = re.sub(r"<a id='[^']+'>\s*</a>\s*", '', markdown)
        content = re.sub(r'<::.*?::>', '', content, flags=re.DOTALL) # Remove figures descriptions
        content = re.sub(r'\n\n\n+', '\n\n', content)
        return content.strip()
    
    def _extract_section(self, content: str) -> str:
        """Extracts section heading from content if present."""
        lines = content.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return line.replace('#', '').strip()
            if line.startswith('**') and line.endswith('**'):
                return line.replace('**', '').strip()
        
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith('<'):
                return line[:100]
        
        return "Content"
    
    async def load_single_document(self, filename: str) -> Optional[Dict[str, Any]]:
        """Loads a single document by filename."""
        json_file = self.documents_dir / f"{filename}_response.json"
        if not json_file.exists():
            json_file = self.documents_dir / f"{filename}.json"
        if not json_file.exists():
            return None
        
        try:
            chunks = await self._load_document_chunks(json_file)
            if chunks:
                doc_name = filename.replace("_response", "")
                await self.rag_service.add_document_chunks(chunks, doc_name)
                return {"document": doc_name, "chunks_loaded": len(chunks), "status": "success"}
            return {"document": filename, "chunks_loaded": 0, "status": "no_content"}
        except Exception as e:
            return {"document": filename, "chunks_loaded": 0, "status": "failed", "error": str(e)}
    
    def get_available_documents(self) -> List[str]:
        """Returns list of available document filenames."""
        json_files = list(self.documents_dir.glob("*_response.json"))
        return [f.stem.replace("_response", "") for f in json_files]
