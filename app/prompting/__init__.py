"""
Prompt strategy package for benchmark and RAG evaluation flows.
"""

from .renderer import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    PromptRenderer,
    render_prompt,
)

__all__ = [
    "DEFAULT_FEW_SHOT_EXAMPLES",
    "PromptRenderer",
    "render_prompt",
]

