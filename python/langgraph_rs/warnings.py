"""LangGraph ``langgraph.warnings`` equivalents.

These warning classes are emitted by LangGraph when deprecated APIs are used; they are defined
here so ``warnings.filterwarnings(..., category=LangGraphDeprecatedSinceV10)`` keeps working.
"""

from __future__ import annotations

__all__ = [
    "LangGraphDeprecatedSinceV05",
    "LangGraphDeprecatedSinceV10",
    "LangGraphDeprecatedSinceV11",
    "LangGraphDeprecationWarning",
]


class LangGraphDeprecationWarning(DeprecationWarning):
    """Base class for LangGraph deprecation warnings."""


class LangGraphDeprecatedSinceV05(LangGraphDeprecationWarning):
    """Deprecated since LangGraph 0.5."""


class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning):
    """Deprecated since LangGraph 1.0."""


class LangGraphDeprecatedSinceV11(LangGraphDeprecationWarning):
    """Deprecated since LangGraph 1.1."""
