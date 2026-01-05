"""
Prompt templates for RAG question answering.
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel
from ..models import SearchResult


class QuestionType(str, Enum):
    """Types of questions."""
    BOARD_STYLE = "board_style"
    CLINICAL_REASONING = "clinical_reasoning"
    DOSAGE_CALCULATION = "dosage_calculation"
    REFERENCE_VALUES = "reference_values"
    GENERAL = "general"


class PromptTemplate(BaseModel):
    """Template for prompts."""
    name: str
    description: str
    template: str
    question_type: QuestionType
    required_sources: int = 3
    max_sources: int = 10


class PromptEngine:
    """Prompt engineering for question answering."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[QuestionType, PromptTemplate]:
        """Initialize prompt templates."""
        
        templates = {}
        
        templates[QuestionType.BOARD_STYLE] = PromptTemplate(
            name="TPN Board-Style Question",
            description="Template for TPN/parenteral nutrition board-style MCQ questions",
            question_type=QuestionType.BOARD_STYLE,
            template="""You are a TPN (Total Parenteral Nutrition) Clinical Specialist answering board-style examination questions.

TPN CLINICAL QUESTION: {question}

REFERENCE KNOWLEDGE BASE:
{context}

APPROACH:
- Use the provided ASPEN guidelines and TPN protocols as your PRIMARY reference
- Supplement with your clinical training knowledge where the sources don't cover specifics
- For calculations, prefer formulas and ranges from the provided documents
- Cite sources when directly using information from the knowledge base

CLINICAL DECISION FRAMEWORK:
- Apply ASPEN evidence-based TPN guidelines as the primary authority
- Consider patient-specific factors (age, weight, gestational age, clinical status)
- Reference TPN component dosing: Amino acids (g/kg/day), Dextrose (mg/kg/min), Lipids (g/kg/day)
- Account for monitoring protocols and safety considerations

BOARD-STYLE ANSWER FORMAT:

1) **Answer:** [Letter choice with brief rationale]

2) **Clinical Analysis:**
   • Key clinical considerations
   • Relevant calculations or values (show work if applicable)
   • Why other options are incorrect

3) **Evidence Base:** [Source citations from knowledge base + clinical principles]

TPN CLINICAL ANSWER:""",
            required_sources=3,
            max_sources=6
        )
        
        templates[QuestionType.CLINICAL_REASONING] = PromptTemplate(
            name="Clinical Reasoning",
            description="Template for complex clinical reasoning scenarios",
            question_type=QuestionType.CLINICAL_REASONING,
            template="""You are providing clinical reasoning for a complex scenario using evidence-based sources.

CLINICAL SCENARIO: {question}

REFERENCE SOURCES:
{context}

Provide a structured clinical reasoning response:
1. Key clinical findings
2. Differential considerations
3. Recommended approach
4. Evidence supporting recommendation

CLINICAL REASONING:""",
            required_sources=3,
            max_sources=8
        )
        
        templates[QuestionType.DOSAGE_CALCULATION] = PromptTemplate(
            name="Dosage Calculation",
            description="Template for dosing calculations",
            question_type=QuestionType.DOSAGE_CALCULATION,
            template="""You are calculating dosages based on clinical guidelines.

CALCULATION REQUEST: {question}

REFERENCE GUIDELINES:
{context}

Provide:
1. Calculation method from guidelines
2. Step-by-step calculation
3. Final dosage with units
4. Safety range verification

CALCULATION:""",
            required_sources=2,
            max_sources=5
        )
        
        templates[QuestionType.REFERENCE_VALUES] = PromptTemplate(
            name="Reference Values",
            description="Template for reference value lookups",
            question_type=QuestionType.REFERENCE_VALUES,
            template="""You are looking up reference values from clinical guidelines.

QUERY: {question}

REFERENCE SOURCES:
{context}

Provide the reference values with:
1. Normal ranges
2. Age/weight specific adjustments if applicable
3. Source citation

REFERENCE VALUES:""",
            required_sources=2,
            max_sources=4
        )
        
        templates[QuestionType.GENERAL] = PromptTemplate(
            name="General Question",
            description="General question answering template",
            question_type=QuestionType.GENERAL,
            template="""Based on the following context, answer the question accurately.

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so clearly
- Be precise and helpful

ANSWER:""",
            required_sources=2,
            max_sources=6
        )
        
        return templates
    
    def get_template(self, question_type: QuestionType = QuestionType.GENERAL) -> PromptTemplate:
        """Get template for question type."""
        return self.templates.get(question_type, self.templates[QuestionType.GENERAL])
    
    def detect_question_type(self, question: str) -> QuestionType:
        """Detect the type of question based on keywords."""
        question_lower = question.lower()
        
        calculation_keywords = ["calculate", "compute", "dose", "dosing", "mg/kg", "g/kg", "ml/kg"]
        if any(kw in question_lower for kw in calculation_keywords):
            return QuestionType.DOSAGE_CALCULATION
        
        reference_keywords = ["normal range", "reference", "lab value", "level"]
        if any(kw in question_lower for kw in reference_keywords):
            return QuestionType.REFERENCE_VALUES
        
        board_keywords = ["which of the following", "best describes", "most likely", "least likely"]
        if any(kw in question_lower for kw in board_keywords):
            return QuestionType.BOARD_STYLE
        
        reasoning_keywords = ["why", "explain", "mechanism", "pathophysiology"]
        if any(kw in question_lower for kw in reasoning_keywords):
            return QuestionType.CLINICAL_REASONING
        
        return QuestionType.GENERAL
    
    def format_prompt(
        self,
        question: str,
        context: str,
        question_type: Optional[QuestionType] = None
    ) -> str:
        """Format a prompt using the appropriate template."""
        if question_type is None:
            question_type = self.detect_question_type(question)
        
        template = self.get_template(question_type)
        
        return template.template.format(
            question=question,
            context=context
        )
    
    def build_context_from_results(
        self,
        results: List[SearchResult],
        max_sources: int = 6
    ) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(results[:max_sources], 1):
            doc_name = result.document_name[:60]
            section = result.chunk.section or "General"
            page = f", Page {result.chunk.page_num}" if result.chunk.page_num else ""
            
            context_parts.append(
                f"[Source {i}: {doc_name}{page}]\n"
                f"Section: {section}\n"
                f"{result.content}"
            )
        
        return "\n\n".join(context_parts)
