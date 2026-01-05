"""
API route handlers.
"""
from typing import Dict
from fastapi import APIRouter, HTTPException, Depends

from ..models import SearchQuery, RAGQuery, SearchResponse, RAGResponse
from ..services.rag import RAGService
from .schemas import SearchRequest, RAGRequest, StatsResponse
from .dependencies import get_rag_service

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> SearchResponse:
    try:
        query = SearchQuery(
            query=request.query,
            limit=request.limit,
            filters=request.filters
        )
        return await rag_service.search(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/ask", response_model=RAGResponse)
async def ask_question(
    request: RAGRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> RAGResponse:
    try:
        query = RAGQuery(
            question=request.question,
            search_limit=request.search_limit,
            model=request.model,
            temperature=request.temperature
        )
        return await rag_service.ask(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_collection_stats(
    rag_service: RAGService = Depends(get_rag_service)
) -> StatsResponse:
    try:
        stats = await rag_service.get_collection_stats()
        return StatsResponse(
            total_chunks=stats.get("total_chunks", 0),
            total_documents=stats.get("total_documents", 0),
            collection_name=stats.get("collection_name", "unknown"),
            embedding_model=rag_service.embedding_provider.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, str]:
    try:
        await rag_service.remove_document(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
