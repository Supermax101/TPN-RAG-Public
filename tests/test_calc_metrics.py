import pytest

from app.evaluation.calc_metrics import (
    evaluate_calc_metrics,
    evaluate_doc_citations,
    extract_quantities,
)


def test_extract_quantities_range():
    qs = extract_quantities("Recommended fluids: 100–120 mL/kg/day.")
    assert len(qs) == 2
    values = sorted([q.raw_value for q in qs])
    assert values == [100.0, 120.0]
    assert all(q.family == "vol_ml" for q in qs)
    assert all(q.per_units == ("kg", "day") for q in qs)


def test_calc_metrics_mass_unit_conversion():
    r = evaluate_calc_metrics(
        expected_answer="Protein dose is 1 g/day.",
        output_answer="Protein dose is 1000 mg/day.",
    )
    assert r.quantity_recall == 1.0
    assert r.quantity_precision == 1.0
    assert r.quantity_f1 == 1.0
    assert r.matched_quantity_count == 1


def test_calc_metrics_unit_mismatch():
    r = evaluate_calc_metrics(
        expected_answer="Infuse lipid at 0.2 mL/hr.",
        output_answer="Infuse lipid at 0.2 mL/day.",
    )
    assert r.quantity_recall == 0.0
    assert r.unit_mismatch_count >= 1


def test_doc_citations_match_gold_and_context():
    out = "Final answer: ... Citations: [TPN Considerations]"
    c = evaluate_doc_citations(
        output_answer=out,
        gold_source_doc="TPN Considerations",
        retrieved_sources=["TPN Considerations", "Other Doc"],
    )
    assert c.citation_present is True
    assert c.cites_gold_source_doc is True
    assert c.cited_doc_in_retrieved_context is True
