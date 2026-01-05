"""
MCQ Answer Parsing and Structured Output Models.

Provides Pydantic models for structured MCQ responses and robust parsing
logic that handles various LLM output formats.
"""

import re
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class MCQAnswer(BaseModel):
    """
    Structured output for single-answer MCQ questions.
    
    Use with LangChain's .with_structured_output() for reliable parsing:
    
        llm.with_structured_output(MCQAnswer)
    """
    
    thinking: str = Field(
        description="Step-by-step clinical reasoning explaining the answer choice"
    )
    answer: str = Field(
        description="Single letter answer: A, B, C, D, E, or F"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the answer"
    )
    source_used: bool = Field(
        default=True,
        description="Whether the provided context was used to answer"
    )
    
    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Ensure answer is a valid single letter A-F."""
        v = v.strip().upper()
        if v not in ['A', 'B', 'C', 'D', 'E', 'F']:
            # Try to extract a letter
            match = re.search(r'\b([A-F])\b', v)
            if match:
                return match.group(1)
            raise ValueError(f"Answer must be A, B, C, D, E, or F. Got: {v}")
        return v


class MCQMultiAnswer(BaseModel):
    """
    Structured output for multi-answer MCQ questions.
    
    Use when the question asks to "select all that apply".
    """
    
    thinking: str = Field(
        description="Step-by-step clinical reasoning for each selected answer"
    )
    answers: List[str] = Field(
        description="List of correct answer letters, e.g., ['A', 'B', 'D']"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the answers"
    )
    source_used: bool = Field(
        default=True,
        description="Whether the provided context was used to answer"
    )
    
    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v: List[str]) -> List[str]:
        """Ensure all answers are valid letters A-F."""
        valid = []
        for ans in v:
            ans = ans.strip().upper()
            if ans in ['A', 'B', 'C', 'D', 'E', 'F']:
                valid.append(ans)
        if not valid:
            raise ValueError("At least one valid answer (A-F) required")
        return sorted(set(valid))  # Dedupe and sort
    
    @property
    def answer_string(self) -> str:
        """Return comma-separated answer string for comparison."""
        return ",".join(sorted(self.answers))


def parse_mcq_response(
    raw_response: str,
    is_multi_answer: bool = False
) -> tuple[str, str, str]:
    """
    Robust fallback parser for MCQ responses when structured output fails.
    
    Returns:
        Tuple of (answer, thinking, confidence)
    
    This handles various response formats:
    - "Answer: A" / "The answer is A"
    - "A" (just the letter)
    - "A, B, C" (multi-answer)
    - JSON responses
    - Chain-of-thought with "Answer:" at end
    """
    
    # Clean the response
    clean_response = raw_response.strip()
    
    # Remove thinking tags if present
    clean_response = re.sub(
        r'<think>.*?</think>', '', clean_response, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    thinking = ""
    answer = ""
    confidence = "medium"
    
    # Strategy 1: Look for explicit "Answer:" pattern (most reliable)
    answer_match = re.search(
        r'(?:answer|choice|selection)(?:\s+is)?[:\s]+([A-F](?:\s*,\s*[A-F])*)',
        clean_response,
        re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).replace(" ", "").upper()
        # Everything before is thinking
        thinking = clean_response[:answer_match.start()].strip()
    
    # Strategy 2: Look for "Thinking:" then "Answer:" format
    if not answer:
        thinking_match = re.search(
            r'thinking[:\s]+(.+?)(?:answer[:\s]+([A-F](?:\s*,\s*[A-F])*))',
            clean_response,
            re.IGNORECASE | re.DOTALL
        )
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            answer = thinking_match.group(2).replace(" ", "").upper()
    
    # Strategy 3: Response starts with letter(s)
    if not answer:
        start_match = re.match(
            r'^([A-F](?:\s*,\s*[A-F])*)\b',
            clean_response.upper()
        )
        if start_match:
            answer = start_match.group(1).replace(" ", "")
    
    # Strategy 4: Find letter at start of a line
    if not answer:
        line_match = re.search(
            r'(?:^|\n)\s*([A-F])\s*[.:\)]\s',
            clean_response,
            re.IGNORECASE
        )
        if line_match:
            answer = line_match.group(1).upper()
    
    # Strategy 5: Last resort - find any A-F letter
    # BUT be careful not to pick up letters from explanations
    if not answer:
        # Look for isolated letters (word boundaries)
        letters = re.findall(r'\b([A-F])\b', clean_response.upper())
        if letters:
            if is_multi_answer:
                # For multi-answer, take unique letters
                answer = ",".join(sorted(set(letters)))
            else:
                # For single answer, prefer letters at the end of response
                # (more likely to be the final answer after reasoning)
                answer = letters[-1]
    
    # If still no answer, return parse error
    if not answer:
        answer = "PARSE_ERROR"
    
    # Extract confidence if mentioned
    if re.search(r'\b(high|highly)\s+confiden', clean_response, re.IGNORECASE):
        confidence = "high"
    elif re.search(r'\b(low|uncertain|unsure)', clean_response, re.IGNORECASE):
        confidence = "low"
    
    # Use the full response as thinking if we didn't extract it
    if not thinking and answer != "PARSE_ERROR":
        thinking = clean_response
    
    return answer, thinking, confidence


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Handles:
    - "A" -> "A"
    - "A,B,C" -> "A,B,C" (sorted, deduped)
    - "A, B, C" -> "A,B,C"
    - "a,b" -> "A,B"
    """
    answer = answer.strip().upper()
    
    # Handle special cases
    if answer in ["ALL", "ALL OF THE ABOVE"]:
        return "ALL"
    if answer in ["NONE", "NONE OF THE ABOVE"]:
        return "NONE"
    
    # Extract letters
    letters = re.findall(r'\b([A-F])\b', answer)
    if letters:
        return ",".join(sorted(set(letters)))
    
    return answer


def answers_match(predicted: str, expected: str) -> tuple[bool, bool]:
    """
    Compare predicted and expected answers.
    
    Returns:
        Tuple of (exact_match, partial_match)
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    exact_match = pred_norm == exp_norm
    
    # Partial match for multi-answer
    partial_match = False
    if not exact_match and "," in exp_norm:
        pred_set = set(pred_norm.split(","))
        exp_set = set(exp_norm.split(","))
        if pred_set & exp_set:  # Any overlap
            partial_match = True
    
    return exact_match, partial_match
