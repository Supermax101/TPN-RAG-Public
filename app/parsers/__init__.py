"""
Parsers Module.
Structured output parsers for MCQ and clinical question answering.
"""

from .mcq_parser import MCQAnswer, MCQMultiAnswer, parse_mcq_response

__all__ = [
    "MCQAnswer",
    "MCQMultiAnswer", 
    "parse_mcq_response",
]
