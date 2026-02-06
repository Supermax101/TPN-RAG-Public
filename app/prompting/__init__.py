"""
Prompt strategy package for benchmark and RAG evaluation flows.
"""

from .renderer import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    PromptRenderer,
    render_prompt,
)
from .system_prompt import (
    TPN_BASE_SYSTEM_PROMPT,
    TPN_RAG_SYSTEM_PROMPT,
    TPN_SYSTEM_PROMPT,
    get_system_prompt,
)

__all__ = [
    "DEFAULT_FEW_SHOT_EXAMPLES",
    "PromptRenderer",
    "TPN_BASE_SYSTEM_PROMPT",
    "TPN_RAG_SYSTEM_PROMPT",
    "TPN_SYSTEM_PROMPT",
    "get_system_prompt",
    "render_prompt",
]
