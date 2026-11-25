"""LangGraph ``langgraph.constants`` equivalents."""

from __future__ import annotations

# LangGraph constants (identical values so conditionals and tags transfer unchanged).
CONF = "configurable"
CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"
TAG_HIDDEN = "langsmith:hidden"
TAG_NOSTREAM = "nostream"
TASKS = "__pregel_tasks"

from ._compat import NATIVE
from .warnings import LangGraphDeprecatedSinceV10

START = NATIVE.START
END = NATIVE.END

__all__ = [
    "CONF",
    "CONFIG_KEY_CHECKPOINTER",
    "END",
    "LangGraphDeprecatedSinceV10",
    "START",
    "TAG_HIDDEN",
    "TAG_NOSTREAM",
    "TASKS",
]
