"""
Prompt strategy package for benchmark and RAG evaluation flows.
"""

from .renderer import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    PromptRenderer,
    render_prompt,
)
from .system_prompt import TPN_SYSTEM_PROMPT

__all__ = [
    "DEFAULT_FEW_SHOT_EXAMPLES",
    "PromptRenderer",
    "TPN_SYSTEM_PROMPT",
    "render_prompt",
]

