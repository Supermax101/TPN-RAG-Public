"""
Tests for the refactored evaluation pipeline.

Covers:
1. All 5 templates contain {context_block} placeholder
2. Context appears in rendered prompt when provided, absent when not
3. Parser doesn't match "ASPEN" -> "A" or "Based on Fluid balance" -> "F"
4. clinical_tokenize handles dosing expressions correctly
5. Few-shot examples are MCQ format (contain "Answer: <letter>")
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Template placeholder tests
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "app" / "prompting" / "templates"

TEMPLATE_FILES = [
    "zero_shot.txt",
    "few_shot.txt",
    "cot.txt",
    "cot_sc.txt",
    "rap.txt",
]


@pytest.mark.parametrize("template_file", TEMPLATE_FILES)
def test_template_contains_context_block(template_file):
    """All 5 prompt templates must include {context_block}."""
    path = TEMPLATE_DIR / template_file
    assert path.exists(), f"Template not found: {path}"
    content = path.read_text(encoding="utf-8")
    assert "{context_block}" in content, (
        f"{template_file} is missing {{context_block}} placeholder. "
        "RAG context will be silently dropped."
    )


@pytest.mark.parametrize("template_file", TEMPLATE_FILES)
def test_template_contains_question(template_file):
    """All templates must include {question} placeholder."""
    path = TEMPLATE_DIR / template_file
    content = path.read_text(encoding="utf-8")
    assert "{question}" in content


@pytest.mark.parametrize("template_file", TEMPLATE_FILES)
def test_template_contains_options_block(template_file):
    """All templates must include {options_block} placeholder."""
    path = TEMPLATE_DIR / template_file
    content = path.read_text(encoding="utf-8")
    assert "{options_block}" in content


# ---------------------------------------------------------------------------
# Renderer context injection tests
# ---------------------------------------------------------------------------

from app.prompting.renderer import render_prompt


def test_context_appears_in_rendered_prompt_when_provided():
    """When context is provided, it should appear in the rendered prompt."""
    context = "Protein requirements for preterm infants are 3-4 g/kg/day."
    prompt = render_prompt(
        strategy="ZS",
        question="What is the protein requirement?",
        options=["A) 1-2 g/kg/day", "B) 3-4 g/kg/day"],
        context=context,
    )
    assert "3-4 g/kg/day" in prompt
    assert "CLINICAL KNOWLEDGE BASE" in prompt


def test_context_absent_when_not_provided():
    """When no context is provided, the clinical knowledge base header should not appear."""
    prompt = render_prompt(
        strategy="ZS",
        question="What is the protein requirement?",
        options=["A) 1-2 g/kg/day", "B) 3-4 g/kg/day"],
        context=None,
    )
    assert "CLINICAL KNOWLEDGE BASE" not in prompt


def test_context_absent_when_empty():
    """Empty context string should behave like None."""
    prompt = render_prompt(
        strategy="ZS",
        question="What is TPN?",
        context="   ",
    )
    assert "CLINICAL KNOWLEDGE BASE" not in prompt


# ---------------------------------------------------------------------------
# Parser false-positive tests
# ---------------------------------------------------------------------------

from app.parsers.mcq_parser import parse_mcq_response


def test_parser_does_not_match_aspen_as_A():
    """Parser should NOT extract 'A' from the word 'ASPEN'."""
    response = "Based on ASPEN guidelines, the recommendation is to start with dextrose."
    answer, _, _ = parse_mcq_response(response)
    assert answer != "A", f"Parser incorrectly matched 'A' from 'ASPEN': got {answer}"


def test_parser_does_not_match_fluid_as_F():
    """Parser should NOT extract 'F' from 'Fluid'."""
    response = "Fluid balance is important in TPN management. Based on the evidence."
    answer, _, _ = parse_mcq_response(response)
    assert answer != "F", f"Parser incorrectly matched 'F' from 'Fluid': got {answer}"


def test_parser_does_not_match_based_as_B():
    """Parser should NOT extract 'B' from 'Based'."""
    response = "Based on the provided context, electrolyte monitoring is recommended."
    answer, _, _ = parse_mcq_response(response)
    assert answer != "B", f"Parser incorrectly matched 'B' from 'Based': got {answer}"


def test_parser_extracts_explicit_answer():
    """Parser correctly extracts explicit answer format."""
    response = "Reasoning: Preterm infants need 3-4 g/kg/day.\nAnswer: C"
    answer, _, _ = parse_mcq_response(response)
    assert answer == "C"


def test_parser_extracts_answer_at_start():
    """Parser correctly extracts single letter at start."""
    response = "B\nThe correct option is B because..."
    answer, _, _ = parse_mcq_response(response)
    assert answer == "B"


def test_parser_handles_therefore_cue():
    """Parser should find answer after 'therefore' cue."""
    response = "After analyzing the ASPEN guidelines, therefore D is the correct answer."
    answer, _, _ = parse_mcq_response(response)
    assert answer == "D"


# ---------------------------------------------------------------------------
# Clinical tokenizer tests
# ---------------------------------------------------------------------------

from app.retrieval.tokenizer import clinical_tokenize


def test_tokenize_dosing_expression():
    """Dosing expressions should be preserved as single tokens."""
    tokens = clinical_tokenize("3-4 g/kg/day protein for preterm")
    # Should contain a token with g_per_kg
    dosing_tokens = [t for t in tokens if "per" in t]
    assert len(dosing_tokens) >= 1, f"No dosing token found in: {tokens}"


def test_tokenize_removes_stopwords():
    """Common stopwords should be removed."""
    tokens = clinical_tokenize("the protein is for the infant")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "for" not in tokens
    assert "protein" in tokens
    assert "infant" in tokens


def test_tokenize_preserves_numeric_ranges():
    """Numeric ranges like 3-4 should be preserved."""
    tokens = clinical_tokenize("dose range of 3-4 units")
    assert any("3-4" in t for t in tokens), f"Numeric range not preserved in: {tokens}"


def test_tokenize_empty_input():
    """Empty input should return empty list."""
    assert clinical_tokenize("") == []
    assert clinical_tokenize("   ") == []


# ---------------------------------------------------------------------------
# Few-shot example format tests
# ---------------------------------------------------------------------------

from app.prompting.renderer import DEFAULT_FEW_SHOT_EXAMPLES


def test_few_shot_examples_are_mcq_format():
    """Few-shot examples should demonstrate MCQ-format answers."""
    for i, example in enumerate(DEFAULT_FEW_SHOT_EXAMPLES):
        answer_text = example.get("answer", "")
        assert "Answer:" in answer_text, (
            f"Few-shot example {i} missing 'Answer:' — got: {answer_text[:100]}"
        )
        # Should contain a letter answer
        import re
        assert re.search(r"Answer:\s*[A-F]", answer_text), (
            f"Few-shot example {i} missing letter answer after 'Answer:'"
        )
