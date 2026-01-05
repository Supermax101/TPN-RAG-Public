"""
Core RAG service.
Handles document search, context building, and answer generation.
"""
import time
from typing import List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..models import (
    SearchQuery, SearchResponse, SearchResult, DocumentChunk,
    RAGQuery, RAGResponse
)
from ..providers.base import EmbeddingProvider, VectorStore, LLMProvider
from ..logger import logger


class RAGService:
    """Retrieval-Augmented Generation service using vector search and LLM generation."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.prompt_template = self._build_prompt_template()
        self.output_parser = StrOutputParser()
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Performs vector similarity search."""
        start_time = time.time()
        
        try:
            query_embedding = await self.embedding_provider.embed_query(query.query)
            
            raw_results = await self.vector_store.search_similar(
                query_embedding,
                limit=query.limit,
                filters=query.filters
            )
            
            search_results = []
            for result in raw_results:
                chunk = DocumentChunk(
                    chunk_id=result["chunk_id"],
                    doc_id=result["doc_id"],
                    content=result["content"],
                    chunk_type=result.get("chunk_type", "text"),
                    page_num=result.get("page_num"),
                    section=result.get("section"),
                    metadata=result.get("metadata", {})
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result["score"],
                    document_name=result.get("document_name", "Unknown")
                )
                search_results.append(search_result)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            return SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                search_time_ms=search_time_ms,
                model_used=self.embedding_provider.model_name
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def ask(self, rag_query: RAGQuery) -> RAGResponse:
        """Answers a question using retrieved context and LLM generation."""
        start_time = time.time()
        logger.info(f"Processing Query: {rag_query.question}")
        
        search_query = SearchQuery(
            query=rag_query.question,
            limit=rag_query.search_limit
        )
        
        search_start = time.time()
        search_response = await self.search(search_query)
        search_time_ms = (time.time() - search_start) * 1000
        
        if not search_response.results:
            logger.warning("No relevant documents found for query.")
            total_time_ms = (time.time() - start_time) * 1000
            return RAGResponse(
                question=rag_query.question,
                answer="No relevant documents found. Please rephrase your question.",
                sources=[],
                search_time_ms=search_time_ms,
                generation_time_ms=0,
                total_time_ms=total_time_ms,
                model_used="no-model"
            )
        
        context = self._build_context(search_response.results)
        
        formatted_messages = self.prompt_template.format_messages(
            context=context,
            question=rag_query.question
        )
        prompt_str = self._format_messages(formatted_messages)
        
        generation_start = time.time()
        try:
            answer = await self.llm_provider.generate(
                prompt=prompt_str,
                model=rag_query.model,
                temperature=rag_query.temperature,
                max_tokens=600
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "I'm sorry, I encountered an error generating the response."
            
        generation_time_ms = (time.time() - generation_start) * 1000
        total_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Answer generated in {total_time_ms:.0f}ms (Search: {search_time_ms:.0f}ms)")
        
        return RAGResponse(
            question=rag_query.question,
            answer=answer.strip(),
            sources=search_response.results,
            search_time_ms=search_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            model_used=rag_query.model or "default"
        )
    
    def _format_messages(self, messages: List[Any]) -> str:
        """Converts LangChain messages to string format."""
        formatted_parts = []
        
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "").upper()
            if role == "SYSTEM":
                formatted_parts.append(f"SYSTEM:\n{msg.content}\n")
            elif role == "HUMAN":
                formatted_parts.append(f"USER:\n{msg.content}\n")
            elif role == "AI":
                formatted_parts.append(f"ASSISTANT:\n{msg.content}\n")
            else:
                formatted_parts.append(f"{msg.content}\n")
        
        formatted_parts.append("\nASSISTANT:")
        return "\n".join(formatted_parts)
    
    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Builds the prompt template for RAG responses."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant. Answer questions based on the provided context.
            
Guidelines:
- Base your answer only on the provided context
- Cite specific sources when possible
- If information is insufficient, state what's missing
- Be precise and helpful"""),
            ("human", """Context:
{context}

Question: {question}

Provide a clear, evidence-based answer using the context above.""")
        ])
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """Builds context string from search results with source attribution."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            doc_name = result.document_name[:60]
            section = result.chunk.section or "General"
            page = f", Page {result.chunk.page_num}" if result.chunk.page_num else ""
            
            context_parts.append(
                f"[Source {i}: {doc_name}{page}]\n"
                f"Section: {section}\n"
                f"{result.content}"
            )
        
        return "\n\n".join(context_parts)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Returns statistics about the document collection."""
        return await self.vector_store.get_stats()
    
    async def add_document_chunks(
        self,
        chunks: List[DocumentChunk],
        doc_name: str
    ) -> None:
        """Adds document chunks to the vector store in batches."""
        if not chunks:
            return
        
        batch_size = 50
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            logger.debug(f"Generating embeddings for batch {i//batch_size + 1}")
            try:
                batch_embeddings = await self.embedding_provider.embed_texts(batch_texts)
                await self.vector_store.add_chunks(batch_chunks, batch_embeddings, doc_name)
            except Exception as e:
                logger.error(f"Failed to process batch {i}: {e}")
                raise
        
        logger.info(f"Successfully processed all {len(chunks)} chunks")
    
    async def remove_document(self, doc_id: str) -> None:
        """Removes a document from the vector store."""
        await self.vector_store.delete_document(doc_id)
