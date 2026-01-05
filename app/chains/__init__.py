"""
Chains Module.
LangChain 1.x LCEL chains for RAG and MCQ answering.
"""

from .mcq_chain import MCQChain, create_mcq_chain
from .retrieval_chain import RetrievalChain, create_retrieval_chain
from .agentic_rag import AgenticMCQRAG, create_agentic_mcq_rag

__all__ = [
    # Standard chains
    "MCQChain",
    "create_mcq_chain",
    "RetrievalChain",
    "create_retrieval_chain",
    # LangGraph agentic pattern
    "AgenticMCQRAG",
    "create_agentic_mcq_rag",
]
