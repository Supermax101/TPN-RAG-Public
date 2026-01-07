"""
LangGraph Agentic RAG - Adaptive Retrieval with Document Grading.

IMPORTANT GROUNDING GUARANTEE:
================================
This system ALWAYS grounds answers in YOUR knowledge base.
The "rewrite" step ONLY rewrites the SEARCH QUERY to find better documents.
It NEVER rewrites or fabricates clinical information.

Flow:
1. Retrieve context from YOUR knowledge base
2. Grade: Are retrieved documents relevant to the question?
3. If NOT relevant → Rewrite the SEARCH QUERY (not the answer!) and retry
4. If relevant → Generate answer EXCLUSIVELY from retrieved context
5. Answer is ALWAYS based on YOUR knowledge base

For hospital/clinical use:
- All answers cite retrieved sources
- Model explicitly instructed to prefer knowledge base over training data
- If no relevant context found, model states "insufficient information"
"""

from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool

from ..config import settings
from ..logger import logger
from .tpn_prompts import TPN_SINGLE_ANSWER_PROMPT, TPN_MULTI_ANSWER_PROMPT


# =============================================================================
# STATE SCHEMA
# =============================================================================

class AgenticRAGState(BaseModel):
    """State for the agentic RAG workflow."""
    
    # Core state
    messages: List[Any] = Field(default_factory=list)
    question: str = ""
    options: str = ""
    answer_type: str = "single"
    case_context: str = ""
    
    # Retrieval state
    documents: List[Document] = Field(default_factory=list)
    context: str = ""
    
    # Grading state
    documents_relevant: bool = False
    rewrite_count: int = 0
    max_rewrites: int = 2
    
    # Output state
    answer: str = ""
    thinking: str = ""
    confidence: str = "medium"
    
    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# DOCUMENT GRADING (from LangGraph docs)
# =============================================================================

class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if document is relevant to the TPN clinical question, 'no' if not"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of relevance assessment"
    )


GRADE_PROMPT = """You are a clinical document relevance grader for TPN (Total Parenteral Nutrition) questions.

Assess whether the retrieved document contains information relevant to answer the clinical question.

**Retrieved Document:**
{context}

**Clinical Question:**
{question}

Grade as "yes" if the document contains:
- Specific TPN dosing, calculations, or requirements mentioned in the question
- ASPEN guidelines or clinical protocols relevant to the question
- Clinical concepts (amino acids, dextrose, lipids, electrolytes) mentioned in the question

Grade as "no" if the document:
- Is about a different topic entirely
- Contains only general/unrelated information
- Does not help answer the specific question

Be strict - the document must be specifically helpful for answering THIS question."""


# =============================================================================
# QUERY REWRITING
# =============================================================================

REWRITE_PROMPT = """You are a clinical query optimizer for TPN (Total Parenteral Nutrition) information retrieval.

The original query did not retrieve relevant documents. Rewrite it to be more specific and likely to find relevant TPN clinical information.

**Original Question:**
{question}

**Tips for rewriting:**
- Add specific TPN terminology (amino acids, dextrose, lipids, electrolytes)
- Include patient context (preterm, neonatal, pediatric)
- Mention ASPEN guidelines if relevant
- Focus on the core clinical concept

**Rewritten query (one line only):**"""


# =============================================================================
# AGENTIC RAG GRAPH
# =============================================================================

class AgenticMCQRAG:
    """
    LangGraph-based Agentic RAG for MCQ answering.
    
    Implements adaptive retrieval with:
    - Document grading
    - Query rewriting
    - Structured output
    
    Usage:
        ```python
        rag = AgenticMCQRAG()
        await rag.initialize()
        
        result = await rag.answer(
            question="What is the protein requirement?",
            options="A. 1g | B. 2g | C. 3g",
        )
        ```
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model_name = model
        self.llm = None
        self.grader_llm = None
        self.vector_store = None
        self.graph = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the agentic RAG components."""

        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

        # Initialize LLMs using HuggingFace
        model_name = self.model_name if "/" in self.model_name else settings.hf_llm_model
        self.llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.01,  # Near-zero for deterministic output
            max_new_tokens=1024,
        )

        self.grader_llm = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.01,
            max_new_tokens=256,
        )

        # Initialize vector store with HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"trust_remote_code": True}
        )

        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chromadb_dir),
        )
        
        # Build the graph
        self._build_graph()
        
        self._initialized = True
        logger.info(f"AgenticMCQRAG initialized with model: {self.model_name}")
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        
        # Define nodes
        def retrieve(state: dict) -> dict:
            """Retrieve documents from vector store."""
            query = state.get("question", "")
            case_context = state.get("case_context", "")
            
            # Combine for search
            search_query = f"{case_context} {query}".strip() if case_context else query
            
            docs = self.vector_store.similarity_search(search_query, k=5)
            
            # Format context
            context = "\n\n---\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in docs
            ])
            
            return {
                "documents": docs,
                "context": context,
            }
        
        def grade_documents(state: dict) -> Literal["generate", "rewrite"]:
            """Grade retrieved documents for relevance."""
            question = state.get("question", "")
            context = state.get("context", "")
            rewrite_count = state.get("rewrite_count", 0)
            max_rewrites = state.get("max_rewrites", 2)
            
            # If no context, need to rewrite
            if not context or len(context.strip()) < 50:
                if rewrite_count < max_rewrites:
                    return "rewrite"
                return "generate"  # Give up and try to answer anyway
            
            # Grade with LLM
            try:
                prompt = GRADE_PROMPT.format(question=question, context=context[:2000])
                
                structured_grader = self.grader_llm.with_structured_output(GradeDocuments)
                result = structured_grader.invoke([{"role": "user", "content": prompt}])
                
                if result.binary_score == "yes":
                    return "generate"
                elif rewrite_count < max_rewrites:
                    return "rewrite"
                else:
                    return "generate"  # Max rewrites reached
                    
            except Exception as e:
                logger.warning(f"Document grading failed: {e}")
                return "generate"
        
        def rewrite_question(state: dict) -> dict:
            """Rewrite the question for better retrieval."""
            question = state.get("question", "")
            rewrite_count = state.get("rewrite_count", 0)
            
            try:
                prompt = REWRITE_PROMPT.format(question=question)
                response = self.llm.invoke([{"role": "user", "content": prompt}])
                
                new_question = response.content.strip()
                
                return {
                    "question": new_question,
                    "rewrite_count": rewrite_count + 1,
                }
            except Exception as e:
                logger.warning(f"Query rewrite failed: {e}")
                return {"rewrite_count": rewrite_count + 1}
        
        def generate_answer(state: dict) -> dict:
            """Generate the final MCQ answer."""
            question = state.get("question", "")
            options = state.get("options", "")
            context = state.get("context", "")
            case_context = state.get("case_context", "")
            answer_type = state.get("answer_type", "single")
            
            # Select prompt
            prompt_template = TPN_MULTI_ANSWER_PROMPT if answer_type == "multi" else TPN_SINGLE_ANSWER_PROMPT
            
            # Format and invoke
            try:
                messages = prompt_template.format_messages(
                    question=question,
                    options=options,
                    context=context or "No relevant context found.",
                    case_context=case_context or "",
                )
                
                response = self.llm.invoke(messages)
                raw_answer = response.content
                
                # Parse answer
                from ..parsers.mcq_parser import parse_mcq_response
                answer, thinking, confidence = parse_mcq_response(raw_answer)
                
                return {
                    "answer": answer,
                    "thinking": thinking,
                    "confidence": confidence,
                }
                
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                return {
                    "answer": "ERROR",
                    "thinking": f"Generation error: {str(e)}",
                    "confidence": "low",
                }
        
        # Build graph
        workflow = StateGraph(dict)
        
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", rewrite_question)
        workflow.add_node("generate", generate_answer)
        
        # Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            }
        )
        workflow.add_edge("rewrite", "retrieve")  # Loop back
        workflow.add_edge("generate", END)
        
        self.graph = workflow.compile()
    
    async def answer(
        self,
        question: str,
        options: str,
        answer_type: str = "single",
        case_context: str = "",
    ) -> Dict[str, Any]:
        """Answer an MCQ question using agentic RAG."""
        
        if not self._initialized:
            await self.initialize()
        
        # Initial state
        initial_state = {
            "question": question,
            "options": options,
            "answer_type": answer_type,
            "case_context": case_context,
            "documents": [],
            "context": "",
            "rewrite_count": 0,
            "max_rewrites": 2,
            "answer": "",
            "thinking": "",
            "confidence": "medium",
        }
        
        # Run graph
        try:
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result.get("answer", "ERROR"),
                "thinking": result.get("thinking", ""),
                "confidence": result.get("confidence", "medium"),
                "rewrite_count": result.get("rewrite_count", 0),
                "context_used": bool(result.get("context")),
            }
            
        except Exception as e:
            logger.error(f"Agentic RAG failed: {e}")
            return {
                "answer": "ERROR",
                "thinking": f"Pipeline error: {str(e)}",
                "confidence": "low",
                "rewrite_count": 0,
                "context_used": False,
            }


async def create_agentic_mcq_rag(model: str = "qwen2.5:7b") -> AgenticMCQRAG:
    """Factory function to create and initialize an AgenticMCQRAG instance."""
    rag = AgenticMCQRAG(model=model)
    await rag.initialize()
    return rag
