"""
TPN-Specific Production Prompts.

These prompts are optimized for:
1. Context-FIRST placement (reduces hallucination)
2. Explicit grounding instructions (prioritize knowledge base)
3. TPN/ASPEN domain-specific terminology
4. Structured output format
"""

from langchain_core.prompts import ChatPromptTemplate

from ..prompting import TPN_SYSTEM_PROMPT

# Template addendum appended after the canonical system prompt
_SINGLE_ANSWER_ADDENDUM = """

## TPN CLINICAL KNOWLEDGE BASE (PRIMARY SOURCE)
{context}

---

## RESPONSE FORMAT (follow EXACTLY)
Provide a brief clinical reasoning (2-3 sentences), then state your answer.

Thinking: [Your clinical reasoning based on the context]
Answer: [ONLY the single letter A-F — no option text, no explanation, nothing else]"""

_MULTI_ANSWER_ADDENDUM = """

## TPN CLINICAL KNOWLEDGE BASE (PRIMARY SOURCE)
{context}

---

## RESPONSE FORMAT (follow EXACTLY)
Provide clinical reasoning for each selected answer, then list ALL correct answers.

Thinking: [Reasoning for each correct answer]
Answer: [ONLY the letter(s), e.g. A, C, D — no option text, no explanation, nothing else]"""


# =============================================================================
# SINGLE-ANSWER MCQ PROMPT
# =============================================================================

TPN_SINGLE_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TPN_SYSTEM_PROMPT + _SINGLE_ANSWER_ADDENDUM),
    
    ("human", """{case_context}

**QUESTION:** {question}

**OPTIONS:**
{options}

Based on the TPN clinical knowledge base provided, give your Thinking and Answer:""")
])


# =============================================================================
# MULTI-ANSWER MCQ PROMPT  
# =============================================================================

TPN_MULTI_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TPN_SYSTEM_PROMPT + _MULTI_ANSWER_ADDENDUM),
    
    ("human", """{case_context}

**QUESTION (Select ALL that apply):** {question}

**OPTIONS:**
{options}

Based on the TPN clinical knowledge base, identify ALL correct answers. Give your Thinking and Answer:""")
])


# =============================================================================
# RETRIEVAL QUERY ENHANCEMENT PROMPT (for HyDE)
# =============================================================================

TPN_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a TPN (Total Parenteral Nutrition) clinical documentation expert.

Given a clinical question, write a SHORT passage (2-3 sentences) that would appear in an ASPEN guideline or TPN textbook and answer the question.

Focus on:
- Neonatal/pediatric parenteral nutrition
- ASPEN recommendations and dosing
- Amino acids, dextrose, lipid emulsions
- Electrolyte and mineral requirements
- PN-related complications (cholestasis, etc.)

Write as if you're excerpting from a clinical guideline. Be specific and factual."""),
    
    ("human", """Question: {question}

Write a hypothetical clinical guideline passage that answers this:""")
])


# =============================================================================
# QUERY EXPANSION PROMPT (for Multi-Query)
# =============================================================================

TPN_MULTIQUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a TPN clinical information specialist. 

Given a clinical question, generate 3 alternative search queries that could help find relevant information in a TPN knowledge base.

Focus on:
- Different terminology (e.g., "protein" vs "amino acids")
- Related clinical concepts
- Specific vs general queries

Output exactly 3 queries, one per line."""),
    
    ("human", """Original question: {question}

Generate 3 alternative search queries:""")
])


# =============================================================================
# ANSWER VALIDATION/GROUNDING CHECK
# =============================================================================

TPN_GROUNDING_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical accuracy reviewer.

Evaluate whether the provided answer is GROUNDED in the given context.

Score from 1-5:
- 5: Fully grounded - answer directly supported by context
- 4: Mostly grounded - answer supported with minor inference
- 3: Partially grounded - some support but requires external knowledge
- 2: Weakly grounded - minimal context support
- 1: Not grounded - contradicts or ignores context

Be strict. Medical accuracy requires strong grounding."""),
    
    ("human", """Context:
{context}

Question: {question}
Answer given: {answer}
Reasoning: {thinking}

Grounding score (1-5) and brief explanation:""")
])


# =============================================================================
# EXPORT ALL PROMPTS
# =============================================================================

__all__ = [
    "TPN_SINGLE_ANSWER_PROMPT",
    "TPN_MULTI_ANSWER_PROMPT",
    "TPN_HYDE_PROMPT",
    "TPN_MULTIQUERY_PROMPT",
    "TPN_GROUNDING_CHECK_PROMPT",
]
