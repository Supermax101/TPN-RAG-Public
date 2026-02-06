"""Smoke tests for AnswerMetrics (open-ended evaluation)."""

import pytest
from app.evaluation.metrics import AnswerMetrics, EvalResult


@pytest.fixture
def metrics():
    return AnswerMetrics()


def test_exact_match(metrics):
    result = metrics.evaluate_single(
        question="What is the protein requirement?",
        generated="Protein requirement is 3-4 g/kg/day",
        ground_truth="Protein requirement is 3-4 g/kg/day",
    )
    assert result.exact_match == 1.0
    assert result.f1_score == 1.0


def test_partial_overlap(metrics):
    result = metrics.evaluate_single(
        question="What is the protein requirement for preterm infants?",
        generated="Preterm infants require protein at 3-4 g/kg/day for growth.",
        ground_truth="Protein requirement for preterm infants is 3-4 g/kg/day.",
    )
    assert result.f1_score > 0.5
    assert result.exact_match == 0.0


def test_clinical_key_phrase_overlap(metrics):
    result = metrics.evaluate_single(
        question="Dosing?",
        generated="The dose is 3 g/kg/day for preterm infants.",
        ground_truth="Recommended dose: 3 g/kg/day in preterm neonates.",
    )
    assert result.key_phrase_overlap > 0.0


def test_no_overlap(metrics):
    result = metrics.evaluate_single(
        question="What is TPN?",
        generated="Totally unrelated answer about weather.",
        ground_truth="Total Parenteral Nutrition is an IV feeding method.",
    )
    assert result.f1_score < 0.3


def test_citation_match_by_source(metrics):
    result = metrics.evaluate_single(
        question="Source?",
        generated="According to ASPEN Guidelines 2020, protein is 3-4 g/kg/day.",
        ground_truth="Protein is 3-4 g/kg/day.",
        ground_truth_source="ASPEN_Guidelines_2020.md",
    )
    assert result.citation_match is True


def test_citation_match_by_page(metrics):
    result = metrics.evaluate_single(
        question="Source?",
        generated="See p.44 for details on protein dosing.",
        ground_truth="Protein dosing details.",
        ground_truth_source="SomeDoc.md",
        ground_truth_page=44,
    )
    assert result.citation_match is True


def test_citation_no_match(metrics):
    result = metrics.evaluate_single(
        question="Source?",
        generated="No source cited.",
        ground_truth="Protein dosing.",
        ground_truth_source="ASPEN_Guidelines.md",
        ground_truth_page=10,
    )
    assert result.citation_match is False


def test_empty_ground_truth(metrics):
    result = metrics.evaluate_single(
        question="Q?",
        generated="",
        ground_truth="",
    )
    assert result.f1_score == 1.0
    assert result.exact_match == 1.0


def test_eval_result_dataclass():
    r = EvalResult(f1_score=0.85, exact_match=0.0, key_phrase_overlap=0.5, citation_match=True)
    assert r.f1_score == 0.85
    assert r.citation_match is True
