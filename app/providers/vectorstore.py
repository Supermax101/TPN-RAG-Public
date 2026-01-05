"""
Vector store implementations.
Handles storage and retrieval of document embeddings.
"""
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from .base import VectorStore
from ..models import DocumentChunk
from ..config import settings


class ChromaVectorStore(VectorStore):
    """Persistent vector store using ChromaDB for similarity search."""
    
    # Distance metric for similarity search
    # Options: "cosine", "l2", "ip" (inner product)
    # Cosine is preferred for text embeddings as it measures angle between vectors
    DISTANCE_METRIC = "cosine"
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            import chromadb.telemetry
            chromadb.telemetry.telemetry = None
            
            self.client = chromadb.PersistentClient(
                path=str(settings.chromadb_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
            )
            
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except (NotFoundError, ValueError):
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.DISTANCE_METRIC}
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")
    
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_name: str
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.chunk_id or str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk.content)
            embeddings_list.append(embedding)
            
            metadata = {
                "doc_id": chunk.doc_id,
                "document_name": doc_name,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section or "",
                **chunk.metadata
            }
            if chunk.page_num is not None:
                metadata["page_num"] = chunk.page_num
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add chunks: {e}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filters if filters else None
            )
            
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score based on metric
                    distance = results["distances"][0][i]
                    score = self._distance_to_score(distance)
                    
                    result = {
                        "chunk_id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "score": score,
                        "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                        "document_name": results["metadatas"][0][i].get("document_name", "Unknown"),
                        "chunk_type": results["metadatas"][0][i].get("chunk_type", "text"),
                        "section": results["metadatas"][0][i].get("section", ""),
                        "page_num": results["metadatas"][0][i].get("page_num"),
                        "metadata": results["metadatas"][0][i]
                    }
                    search_results.append(result)
            
            return search_results
        except Exception as e:
            raise RuntimeError(f"Failed to search: {e}")
    
    async def delete_document(self, doc_id: str) -> None:
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as e:
            raise RuntimeError(f"Failed to delete document: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            if count == 0:
                return {"total_chunks": 0, "total_documents": 0, "collection_name": self.collection_name}
            
            sample = self.collection.get(limit=min(5000, count))
            unique_docs = set()
            if sample["metadatas"]:
                for m in sample["metadatas"]:
                    unique_docs.add(m.get("doc_id") or m.get("document_name", ""))
            
            return {
                "total_chunks": count,
                "total_documents": len(unique_docs),
                "collection_name": self.collection_name
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get stats: {e}")
    
    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.DISTANCE_METRIC}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to reset: {e}")
    
    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score [0, 1] based on the distance metric.
        
        For cosine distance: range is [0, 2], where 0 = identical, 2 = opposite
            score = 1 - (distance / 2)
        
        For L2 (Euclidean) distance: range is [0, ∞)
            score = 1 / (1 + distance)
        
        For inner product: range is (-∞, ∞), higher = more similar for normalized vectors
            score = (1 + distance) / 2  (assumes normalized vectors, distance in [-1, 1])
        """
        if self.DISTANCE_METRIC == "cosine":
            # Cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, 0 = opposite
            return max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        
        elif self.DISTANCE_METRIC == "l2":
            # L2 distance: 0 = identical, ∞ = very different
            # Convert using inverse: 1 = identical, approaches 0 as distance increases
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
        
        elif self.DISTANCE_METRIC == "ip":
            # Inner product for normalized vectors: 1 = identical, -1 = opposite
            # ChromaDB returns negative inner product as distance
            # So distance = -ip, and ip = -distance
            # Convert: score = (ip + 1) / 2 = (-distance + 1) / 2
            return max(0.0, min(1.0, (1.0 - distance) / 2.0))
        
        else:
            # Fallback to L2 formula
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
