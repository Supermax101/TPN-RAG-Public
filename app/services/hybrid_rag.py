"""
Hybrid RAG Service with advanced retrieval features.
Combines vector search with BM25, cross-encoder reranking, and optional graph search.
"""

from typing import List, Dict, Any, Optional
from ..models import SearchQuery, SearchResult, RAGQuery, RAGResponse, SearchResponse
from .rag import RAGService
from .advanced_rag import AdvancedRAG, AdvancedRAGConfig

ADVANCED_RAG_AVAILABLE = True
try:
    from .advanced_rag import AdvancedRAG, AdvancedRAGConfig
except ImportError:
    ADVANCED_RAG_AVAILABLE = False
    AdvancedRAG = None
    AdvancedRAGConfig = None


class HybridRAGService(RAGService):
    """Enhanced RAG with advanced retrieval techniques.
    
    Features:
    - BM25 + Vector hybrid search
    - Cross-encoder reranking
    - Multi-query generation
    - HyDE (Hypothetical Document Embeddings)
    - RRF fusion for combining results
    """
    
    def __init__(
        self,
        embedding_provider,
        vector_store,
        llm_provider,
        enable_advanced: bool = True,
        advanced_config: Optional[Any] = None
    ):
        super().__init__(embedding_provider, vector_store, llm_provider)
        
        self.advanced_rag = None
        if ADVANCED_RAG_AVAILABLE and enable_advanced:
            try:
                config = advanced_config or AdvancedRAGConfig()
                self.advanced_rag = AdvancedRAG(
                    llm_provider=llm_provider,
                    embedding_provider=embedding_provider,
                    config=config
                )
            except Exception as e:
                print(f"Failed to initialize advanced RAG: {e}")
                self.advanced_rag = None
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Hybrid search with BM25, vector, and cross-encoder reranking."""
        
        print(f"Hybrid RAG Search: {query.query[:60]}...")
        
        target_limit = query.limit or 10
        
        # Step 1: Multi-query generation
        queries_to_search = [query.query]
        if self.advanced_rag and self.advanced_rag.config.enable_multi_query:
            try:
                queries_to_search = await self.advanced_rag.multi_query_generation(query.query)
            except Exception as e:
                print(f"Multi-query generation failed: {e}")
        
        # Step 2: HyDE - add hypothetical answer as query
        if self.advanced_rag and self.advanced_rag.config.enable_hyde:
            try:
                hyde_query = await self.advanced_rag.generate_hyde_hypothesis(query.query)
                if hyde_query:
                    queries_to_search.append(hyde_query)
            except Exception as e:
                print(f"HyDE generation failed: {e}")
        
        # Step 3: Search with all query variants
        all_ranked_lists = []
        
        for i, q in enumerate(queries_to_search, 1):
            print(f"  Query variant {i}/{len(queries_to_search)}: {q[:60]}...")
            
            # Vector search
            sub_query = SearchQuery(
                query=q,
                limit=50,
                filters=query.filters
            )
            vector_results = await super().search(sub_query)
            
            # BM25 search on same chunks
            bm25_results = []
            if self.advanced_rag and self.advanced_rag.config.enable_bm25_hybrid:
                try:
                    bm25_results = await self.advanced_rag.bm25_search(
                        query=q,
                        all_chunks=vector_results.results,
                        top_k=target_limit * 2
                    )
                except Exception as e:
                    print(f"BM25 search failed: {e}")
            
            # Combine results
            combined = list(vector_results.results[:target_limit * 2]) + bm25_results
            all_ranked_lists.append(combined)
        
        # Step 4: RRF fusion
        if self.advanced_rag and self.advanced_rag.config.enable_rrf and len(all_ranked_lists) > 1:
            try:
                fused_results = await self.advanced_rag.reciprocal_rank_fusion(all_ranked_lists)
                unique_results = fused_results[:target_limit * 2]
            except Exception as e:
                print(f"RRF fusion failed: {e}")
                unique_results = all_ranked_lists[0][:target_limit * 2] if all_ranked_lists else []
        else:
            unique_results = all_ranked_lists[0][:target_limit * 2] if all_ranked_lists else []
        
        # Step 5: Cross-encoder reranking
        if self.advanced_rag and self.advanced_rag.config.enable_cross_encoder:
            try:
                unique_results = await self.advanced_rag.rerank_with_cross_encoder(
                    query=query.query,
                    documents=unique_results,
                    top_k=5
                )
            except Exception as e:
                print(f"Cross-encoder reranking failed: {e}")
                unique_results = unique_results[:5]
        else:
            unique_results = unique_results[:5]
        
        return SearchResponse(
            query=query,
            results=unique_results,
            total_results=sum(len(lst) for lst in all_ranked_lists),
            search_time_ms=0,
            model_used="hybrid_bm25_vector"
        )
    
    async def ask(self, rag_query: RAGQuery) -> RAGResponse:
        """Answer with hybrid retrieval."""
        return await super().ask(rag_query)
