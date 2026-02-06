#!/usr/bin/env python3
"""
Render prompt strategy variants without running any model.

Useful for prompt QA before launching benchmark runs on remote GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.evaluation.benchmark_types import PromptStrategy
from app.prompting import PromptRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview benchmark prompt strategies")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="One of: all, ZS, FEW_SHOT, COT, COT_SC, RAP",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="A preterm infant on PN has serum potassium 2.9 mEq/L. How would you classify this?",
        help="Question text",
    )
    parser.add_argument(
        "--options",
        type=str,
        default="Normal,Mild hypokalemia,Moderate hypokalemia,Severe hypokalemia",
        help="Comma-separated options",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=(
            "Normal neonatal serum potassium is 4-6.5 mEq/L. "
            "Mild hypokalemia: 3.5-3.9. Moderate: 2.5-3.5. Severe: <2.5."
        ),
        help="Retrieved context for RAP preview",
    )
    return parser.parse_args()


def resolve_strategies(raw: str) -> List[PromptStrategy]:
    if raw.lower() == "all":
        return list(PromptStrategy)
    try:
        return [PromptStrategy[raw.upper()]]
    except KeyError as exc:
        allowed = ", ".join(["all"] + [s.name for s in PromptStrategy])
        raise SystemExit(f"Invalid strategy '{raw}'. Allowed: {allowed}") from exc


def main() -> int:
    args = parse_args()
    renderer = PromptRenderer()
    options: List[str] = []
    for raw in [x.strip() for x in args.options.split(",") if x.strip()]:
        # Accept either "A. text" or plain "text".
        cleaned = raw
        if len(raw) >= 3 and raw[0].upper() in "ABCDEF" and raw[1] in [".", ")", ":"] and raw[2] == " ":
            cleaned = raw[3:].strip()
        options.append(cleaned)

    strategies = resolve_strategies(args.strategy)
    for strategy in strategies:
        prompt = renderer.render(
            strategy=strategy,
            question=args.question,
            options=options,
            context=args.context,
        )
        print("=" * 80)
        print(f"STRATEGY: {strategy.value}")
        print("=" * 80)
        print(prompt)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
