"""
Compatibility shim for prompt rendering.

Canonical implementation now lives in app/prompting/.
"""

from __future__ import annotations

from ..prompting import DEFAULT_FEW_SHOT_EXAMPLES, PromptRenderer, render_prompt

__all__ = [
    "DEFAULT_FEW_SHOT_EXAMPLES",
    "PromptRenderer",
    "render_prompt",
]
