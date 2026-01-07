"""
MCQ Chain - LangChain 1.x Production RAG for MCQ Questions

Implements a production-grade LCEL chain for answering MCQ questions
with retrieval-augmented generation.

Key Features:
- Optimized prompt structure (context FIRST)
- Structured output with Pydantic
- Multi-answer question support
- Error handling and fallback parsing
"""

from typing import Optional, List, Literal, Union, Dict, Any
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..parsers.mcq_parser import MCQAnswer, MCQMultiAnswer, parse_mcq_response, normalize_answer
from ..config import settings
from ..logger import logger
from .retrieval_chain import RetrievalChain, RetrievalConfig
from .tpn_prompts import TPN_SINGLE_ANSWER_PROMPT, TPN_MULTI_ANSWER_PROMPT


# =============================================================================
# Use TPN-Specific Prompts (imported from tpn_prompts.py)
# These prompts feature:
# - Context FIRST (before question)
# - Explicit grounding instructions
# - TPN/ASPEN domain terminology
# =============================================================================

SINGLE_ANSWER_PROMPT = TPN_SINGLE_ANSWER_PROMPT
MULTI_ANSWER_PROMPT = TPN_MULTI_ANSWER_PROMPT


# =============================================================================
# MCQ CHAIN
# =============================================================================

class MCQChainConfig(BaseModel):
    """Configuration for MCQ answering chain."""
    
    # LLM settings
    model: str = Field(default="qwen2.5:7b", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    max_tokens: int = Field(default=1000, description="Max response tokens")
    
    # Retrieval settings
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    enable_bm25: bool = Field(default=True, description="Enable BM25 hybrid search")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    
    # Output settings
    use_structured_output: bool = Field(default=True, description="Use Pydantic structured output")


class MCQChain:
    """
    Production MCQ answering chain using LangChain 1.x patterns.
    
    Example:
        ```python
        from app.chains import MCQChain
        
        chain = MCQChain()
        await chain.initialize()
        
        result = await chain.answer(
            question="What is the protein requirement for preterm infants?",
            options="A. 1g/kg/day | B. 2g/kg/day | C. 3-4g/kg/day | D. 5g/kg/day",
            answer_type="single"
        )
        
        print(result["answer"])  # "C"
        print(result["thinking"])  # Clinical reasoning
        ```
    """
    
    def __init__(
        self,
        config: Optional[MCQChainConfig] = None,
        retrieval_chain: Optional[RetrievalChain] = None,
    ):
        self.config = config or MCQChainConfig()
        self.retrieval_chain = retrieval_chain
        self.llm = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the chain components."""

        # Initialize LLM using HuggingFace
        from langchain_huggingface import HuggingFaceEndpoint

        model_name = self.config.model if "/" in self.config.model else settings.hf_llm_model
        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=self.config.temperature if self.config.temperature > 0 else 0.01,
            max_new_tokens=self.config.max_tokens,
        )
        
        # Initialize retrieval chain if not provided
        if self.retrieval_chain is None:
            retrieval_config = RetrievalConfig(
                k=self.config.retrieval_k * 2,
                final_k=self.config.retrieval_k,
                enable_bm25=self.config.enable_bm25,
                enable_reranking=self.config.enable_reranking,
            )
            self.retrieval_chain = RetrievalChain(config=retrieval_config)
        
        await self.retrieval_chain.initialize()
        
        self._initialized = True
        logger.info(f"MCQChain initialized with model: {self.config.model}")
    
    async def answer(
        self,
        question: str,
        options: str,
        answer_type: Literal["single", "multi"] = "single",
        case_context: str = "",
    ) -> Dict[str, Any]:
        """
        Answer an MCQ question using RAG.
        
        Args:
            question: The question text
            options: Answer options (e.g., "A. Choice | B. Choice")
            answer_type: "single" or "multi"
            case_context: Optional clinical case context
        
        Returns:
            Dict with keys: answer, thinking, confidence, sources, retrieval_scores
        """
        if not self._initialized:
            await self.initialize()
        
        # Step 1: Retrieve relevant context
        search_query = f"{case_context} {question}".strip() if case_context else question
        retrieved_docs = await self.retrieval_chain.retrieve(search_query)
        
        # Format context
        context = self._format_context(retrieved_docs)
        retrieval_scores = [
            doc.metadata.get("score", 0.5) for doc in retrieved_docs
        ]
        
        # Step 2: Select prompt
        prompt = MULTI_ANSWER_PROMPT if answer_type == "multi" else SINGLE_ANSWER_PROMPT
        
        # Step 3: Generate answer
        try:
            if self.config.use_structured_output:
                result = await self._generate_structured(
                    prompt, question, options, context, case_context, answer_type
                )
            else:
                result = await self._generate_unstructured(
                    prompt, question, options, context, case_context
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = {
                "answer": "ERROR",
                "thinking": f"Generation error: {str(e)}",
                "confidence": "low",
            }
        
        # Add retrieval info
        result["sources"] = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section_header", ""),
            }
            for doc in retrieved_docs[:3]
        ]
        result["retrieval_scores"] = retrieval_scores
        
        return result
    
    async def _generate_structured(
        self,
        prompt: ChatPromptTemplate,
        question: str,
        options: str,
        context: str,
        case_context: str,
        answer_type: str,
    ) -> Dict[str, Any]:
        """Generate answer with structured output."""
        
        # Get the appropriate model
        model_class = MCQMultiAnswer if answer_type == "multi" else MCQAnswer
        
        try:
            # Try structured output
            structured_llm = self.llm.with_structured_output(model_class)
            
            chain = prompt | structured_llm
            
            result = await chain.ainvoke({
                "question": question,
                "options": options,
                "context": context,
                "case_context": case_context or "",
            })
            
            if isinstance(result, MCQMultiAnswer):
                return {
                    "answer": result.answer_string,
                    "thinking": result.thinking,
                    "confidence": result.confidence,
                }
            else:
                return {
                    "answer": result.answer,
                    "thinking": result.thinking,
                    "confidence": result.confidence,
                }
        
        except Exception as e:
            logger.warning(f"Structured output failed, falling back: {e}")
            return await self._generate_unstructured(
                prompt, question, options, context, case_context
            )
    
    async def _generate_unstructured(
        self,
        prompt: ChatPromptTemplate,
        question: str,
        options: str,
        context: str,
        case_context: str,
    ) -> Dict[str, Any]:
        """Generate answer with string output and manual parsing."""
        
        chain = prompt | self.llm | StrOutputParser()
        
        raw_response = await chain.ainvoke({
            "question": question,
            "options": options,
            "context": context,
            "case_context": case_context or "",
        })
        
        # Parse the response
        answer, thinking, confidence = parse_mcq_response(raw_response)
        
        return {
            "answer": answer,
            "thinking": thinking,
            "confidence": confidence,
            "raw_response": raw_response[:500],
        }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant clinical reference found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")[:50]
            section = doc.metadata.get("section_header", "")
            
            header = f"[Source {i}: {source}]"
            if section:
                header += f"\nSection: {section}"
            
            context_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            "initialized": self._initialized,
            "config": self.config.model_dump(),
            "retrieval": self.retrieval_chain.get_stats() if self.retrieval_chain else None,
        }


def create_mcq_chain(
    model: str = "qwen2.5:7b",
    retrieval_k: int = 5,
    enable_reranking: bool = True,
) -> MCQChain:
    """
    Factory function to create a configured MCQ chain.
    
    Example:
        ```python
        chain = create_mcq_chain(model="mistral:7b", retrieval_k=5)
        await chain.initialize()
        
        result = await chain.answer(
            question="What is the protein requirement?",
            options="A. 1g | B. 2g | C. 3g | D. 4g"
        )
        ```
    """
    config = MCQChainConfig(
        model=model,
        retrieval_k=retrieval_k,
        enable_reranking=enable_reranking,
    )
    return MCQChain(config=config)
