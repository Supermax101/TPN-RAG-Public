from app.prompting import get_open_ended_system_prompt, render_open_prompt
from app.evaluation.benchmark_types import PromptStrategy


def test_open_prompt_does_not_contain_mcq_output_rules():
    prompt = render_open_prompt(
        strategy=PromptStrategy.ZS,
        question="Calculate GIR for a 5 kg infant receiving D10 at 10 mL/hr.",
        context="",
    )
    assert "ONLY the letter" not in prompt
    assert "Answer: <ONLY the letter" not in prompt
    assert "Final answer:" in prompt
    assert "STRICT output rules" in prompt


def test_open_rag_system_prompt_forbids_citations():
    sp = get_open_ended_system_prompt(use_rag=True)
    assert "citation_rules" not in sp
    assert "Do NOT cite" in sp
