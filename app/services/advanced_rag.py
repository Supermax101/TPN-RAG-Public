"""
Advanced RAG Improvements (2025 Best Practices)
Implements cutting-edge retrieval patterns for maximum accuracy.

Key Features:
1. Cross-Encoder Reranking (BGE reranker) - improves relevance ordering
2. Parent Document Retrieval - retrieves surrounding context
3. HyDE (Hypothetical Document Embeddings) - better query matching
4. Multi-Query Generation - query expansion with synonyms
5. BM25 Hybrid Search - keyword + semantic combination
6. Reciprocal Rank Fusion - combining multiple result lists
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None


class AdvancedRAGConfig(BaseModel):
    """Configuration for advanced RAG features."""
    
    enable_bm25_hybrid: bool = Field(default=True, description="Enable BM25 + Vector hybrid retrieval")
    bm25_weight: float = Field(default=0.5, description="Weight for BM25")
    vector_weight: float = Field(default=0.5, description="Weight for vector search")
    
    enable_multi_query: bool = Field(default=False, description="Generate multiple query variants")
    num_query_variants: int = Field(default=2, description="Number of query variants")
    
    enable_hyde: bool = Field(default=False, description="Generate hypothetical answer for retrieval")
    hyde_max_words: int = Field(default=50, description="Max words for hypothetical answer")
    
    enable_cross_encoder: bool = Field(default=True, description="Enable cross-encoder reranking")
    cross_encoder_model: str = Field(default="BAAI/bge-reranker-base", description="Cross-encoder model")
    
    enable_parent_retrieval: bool = Field(default=True, description="Retrieve parent context")
    parent_context_size: int = Field(default=2000, description="Parent chunk size in characters")
    
    enable_rrf: bool = Field(default=True, description="Enable RRF for fusing results")
    rrf_k: int = Field(default=60, description="RRF constant")


from ..logger import logger

class AdvancedRAG:
    """Implements advanced RAG techniques for improved retrieval accuracy."""
    
    def __init__(
        self,
        llm_provider,
        embedding_provider,
        config: Optional[AdvancedRAGConfig] = None
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.config = config or AdvancedRAGConfig()
        
        self.cross_encoder = None
        if self.config.enable_cross_encoder and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(self.config.cross_encoder_model)
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                self.cross_encoder = None
        
        self._log_config()
    
    def _log_config(self):
        logger.info("Advanced RAG Configuration:")
        logger.info(f"  - Multi-Query: {'enabled' if self.config.enable_multi_query else 'disabled'}")
        logger.info(f"  - BM25 Hybrid: {'enabled' if self.config.enable_bm25_hybrid else 'disabled'}")
        logger.info(f"  - HyDE: {'enabled' if self.config.enable_hyde else 'disabled'}")
        logger.info(f"  - Cross-Encoder: {'enabled' if self.cross_encoder else 'disabled'}")
        logger.info(f"  - RRF Fusion: {'enabled' if self.config.enable_rrf else 'disabled'}")
    
    async def rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Rerank documents using cross-encoder model."""
        if not self.cross_encoder or not documents:
            return documents
        
        top_k = top_k or len(documents)
        
        try:
            pairs = [[query, doc.content if hasattr(doc, 'content') else doc.chunk.content] 
                     for doc in documents]
            scores = self.cross_encoder.predict(pairs)
            scored_docs = [(score, doc) for score, doc in zip(scores, documents)]
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            reranked = [doc for score, doc in scored_docs[:top_k]]
            
            top_scores = [float(score) for score, _ in scored_docs[:min(3, len(scored_docs))]]
            logger.debug(f"Cross-encoder reranked {len(documents)} -> {len(reranked)} docs")
            logger.debug(f"  Top scores: {[round(s, 3) for s in top_scores]}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents[:top_k]
    
    async def bm25_search(self, query: str, all_chunks: List[Any], top_k: int = 10) -> List[Any]:
        """BM25 keyword search to complement vector search."""
        if not BM25_AVAILABLE or not self.config.enable_bm25_hybrid:
            return []
        
        try:
            corpus = []
            for chunk in all_chunks:
                if hasattr(chunk, 'content'):
                    corpus.append(chunk.content)
                elif hasattr(chunk, 'chunk') and hasattr(chunk.chunk, 'content'):
                    corpus.append(chunk.chunk.content)
                else:
                    corpus.append(str(chunk))
            
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            results = [all_chunks[idx] for idx in top_indices if idx < len(all_chunks)]
            
            logger.debug(f"BM25 search: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def multi_query_generation(self, query: str) -> List[str]:
        """Generate multiple query variants for better retrieval coverage."""
        if not self.config.enable_multi_query:
            return [query]
        
        try:
            prompt = f"""Generate {self.config.num_query_variants} alternative phrasings of this question.

Original: {query}

Requirements:
1. Use synonyms and alternative terms
2. Rephrase structure while keeping meaning
3. Keep same intent

Respond with ONLY the alternatives, one per line, numbered."""

            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            variants = []
            for line in response.strip().split('\n'):
                line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if line and line != query:
                    variants.append(line)
            
            all_queries = [query] + variants[:self.config.num_query_variants]
            
            logger.debug(f"Multi-Query: Generated {len(all_queries)} variants")
            return all_queries
            
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}")
            return [query]
    
    async def generate_hyde_hypothesis(self, question: str) -> Optional[str]:
        """Generate hypothetical answer for HyDE retrieval."""
        if not self.config.enable_hyde:
            return None
        
        try:
            prompt = f"""Write a concise, factual answer to this question.

Question: {question}

Requirements:
- Maximum 2-3 sentences (~{self.config.hyde_max_words} words)
- Be specific and factual
- Just state the facts directly

Answer:"""

            hypothesis = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=150
            )
            
            hypothesis = hypothesis.strip()
            words = hypothesis.split()
            if len(words) > self.config.hyde_max_words * 1.5:
                hypothesis = ' '.join(words[:self.config.hyde_max_words]) + "..."
            
            logger.debug(f"HyDE hypothesis generated ({len(hypothesis.split())} words)")
            return hypothesis
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return None
    
    async def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Any]],
        k: Optional[int] = None
    ) -> List[Any]:
        """Combine multiple ranked lists using Reciprocal Rank Fusion."""
        if not self.config.enable_rrf or len(ranked_lists) <= 1:
            return ranked_lists[0] if ranked_lists else []
        
        k = k or self.config.rrf_k
        
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, 1):
                doc_key = id(doc)
                score = 1.0 / (k + rank)
                
                if doc_key in rrf_scores:
                    rrf_scores[doc_key][0] += score
                else:
                    rrf_scores[doc_key] = [score, doc]
        
        fused = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        fused_docs = [doc for score, doc in fused]
        
        logger.debug(f"RRF fused {len(ranked_lists)} lists -> {len(fused_docs)} unique docs")
        return fused_docs
    
    async def retrieve_parent_context(
        self,
        document: Any,
        all_documents: List[Any]
    ) -> Optional[str]:
        """Retrieve parent document context for a matched chunk."""
        if not self.config.enable_parent_retrieval:
            return None
        
        try:
            doc_id = document.chunk.doc_id if hasattr(document, 'chunk') else getattr(document, 'doc_id', None)
            chunk_content = document.content if hasattr(document, 'content') else document.chunk.content
            
            if not doc_id:
                return None
            
            same_doc_chunks = [
                d for d in all_documents 
                if (hasattr(d, 'chunk') and d.chunk.doc_id == doc_id) or getattr(d, 'doc_id', None) == doc_id
            ]
            
            if len(same_doc_chunks) <= 1:
                return None
            
            parent_context_parts = [chunk_content]
            
            for other_chunk in same_doc_chunks[:3]:
                other_content = other_chunk.content if hasattr(other_chunk, 'content') else other_chunk.chunk.content
                if other_content != chunk_content:
                    parent_context_parts.append(other_content)
            
            parent_context = "\n\n".join(parent_context_parts[:3])
            
            if len(parent_context) > self.config.parent_context_size:
                parent_context = parent_context[:self.config.parent_context_size] + "..."
            
            return parent_context
            
        except Exception as e:
            logger.error(f"Parent context retrieval failed: {e}")
            return None
