"""
Canonical prompt rendering for all benchmark strategies.

This module centralizes prompt templates so strategy changes happen in one place
and are reused by scripts/tests/runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": (
            "A 28-week preterm infant weighing 1.2 kg is started on TPN. "
            "What is the recommended initial amino acid dose?\n"
            "A. 0.5 g/kg/day\n"
            "B. 1.5-2.0 g/kg/day\n"
            "C. 3.0-4.0 g/kg/day\n"
            "D. 5.0 g/kg/day"
        ),
        "answer": (
            "Reasoning: ASPEN guidelines recommend initiating amino acids at "
            "1.5-2.0 g/kg/day in very preterm infants and advancing to "
            "3.0-4.0 g/kg/day. The initial starting dose is 1.5-2.0 g/kg/day.\n"
            "Answer: B"
        ),
    },
    {
        "question": (
            "Which of the following is the MOST appropriate initial glucose "
            "infusion rate (GIR) for a term neonate receiving TPN?\n"
            "A. 2-3 mg/kg/min\n"
            "B. 4-6 mg/kg/min\n"
            "C. 8-10 mg/kg/min\n"
            "D. 12-14 mg/kg/min"
        ),
        "answer": (
            "Reasoning: For term neonates, the standard starting GIR is "
            "4-6 mg/kg/min, then advanced based on glucose tolerance monitoring. "
            "Preterm infants may start at 6-8 mg/kg/min.\n"
            "Answer: B"
        ),
    },
]


def _normalize_strategy(strategy: object) -> str:
    if hasattr(strategy, "value"):
        value = getattr(strategy, "value")
        if isinstance(value, str):
            return value
    return str(strategy)


def _format_options(options: Optional[Sequence[str]]) -> str:
    if not options:
        return ""
    lines: List[str] = []
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        lines.append(f"{label}. {option}")
    return "\n".join(lines)


def _format_context(context: Optional[str]) -> str:
    """Wrap context with a clear header when present, return empty string when not."""
    text = (context or "").strip()
    if not text:
        return ""
    return f"## TPN CLINICAL KNOWLEDGE BASE\n{text}"


def _format_few_shots(examples: Optional[Iterable[Dict[str, str]]]) -> str:
    shots = list(examples or DEFAULT_FEW_SHOT_EXAMPLES)
    if not shots:
        return ""
    lines: List[str] = []
    for ex in shots:
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a:
            continue
        lines.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(lines)


@dataclass
class PromptRenderer:
    """
    Template-backed prompt renderer.

    Template placeholders:
    - {question}
    - {options_block}
    - {context_block}
    - {few_shots_block}
    """

    template_dir: Path = Path(__file__).parent / "templates"

    TEMPLATE_FILES: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.TEMPLATE_FILES is None:
            self.TEMPLATE_FILES = {
                "ZS": "zero_shot.txt",
                "FEW_SHOT": "few_shot.txt",
                "COT": "cot.txt",
                "COT_SC": "cot_sc.txt",
                "RAP": "rap.txt",
            }
        self._cache: Dict[str, str] = {}

    def _load_template(self, strategy_value: str) -> str:
        if strategy_value not in self.TEMPLATE_FILES:
            raise ValueError(f"Unsupported prompt strategy: {strategy_value}")

        filename = self.TEMPLATE_FILES[strategy_value]
        if filename not in self._cache:
            path = self.template_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Prompt template not found: {path}")
            self._cache[filename] = path.read_text(encoding="utf-8")
        return self._cache[filename]

    def render(
        self,
        strategy: object,
        question: str,
        options: Optional[Sequence[str]] = None,
        context: Optional[str] = None,
        few_shot_examples: Optional[Iterable[Dict[str, str]]] = None,
    ) -> str:
        strategy_value = _normalize_strategy(strategy)
        template = self._load_template(strategy_value)

        options_block = _format_options(options)
        context_block = _format_context(context)
        few_shots_block = _format_few_shots(few_shot_examples)

        prompt = template.format(
            question=question.strip(),
            options_block=options_block,
            context_block=context_block,
            few_shots_block=few_shots_block,
        ).strip()
        return prompt


_DEFAULT_RENDERER = PromptRenderer()


def render_prompt(
    strategy: object,
    question: str,
    options: Optional[Sequence[str]] = None,
    context: Optional[str] = None,
    few_shot_examples: Optional[Iterable[Dict[str, str]]] = None,
) -> str:
    """
    Convenience function to render a prompt using shared default templates.
    """
    return _DEFAULT_RENDERER.render(
        strategy=strategy,
        question=question,
        options=options,
        context=context,
        few_shot_examples=few_shot_examples,
    )
