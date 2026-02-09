from app.evaluation.format_metrics import validate_open_final_answer


def test_validate_open_final_answer_accepts_clean_final_answer():
    r = validate_open_final_answer("Final answer: 5.0 mg/kg/min")
    assert r.ok
    assert r.reason == ""


def test_validate_open_final_answer_rejects_work_section():
    r = validate_open_final_answer("Final answer: 5.0 mg/kg/min\nWork: 10*...")
    assert not r.ok
    assert "contains_banned_section_header" in r.reason


def test_validate_open_final_answer_rejects_missing_prefix():
    r = validate_open_final_answer("Answer: 5.0 mg/kg/min")
    assert not r.ok
    assert "missing_final_answer_prefix" in r.reason


def test_validate_open_final_answer_rejects_bracket_citation():
    r = validate_open_final_answer("Final answer: Use cyclic PN. [TPN Considerations]")
    assert not r.ok
    assert "contains_bracket_citation" in r.reason

