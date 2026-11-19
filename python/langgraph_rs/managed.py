"""LangGraph ``langgraph.managed`` equivalents.

``IsLastStep`` and ``RemainingSteps`` are declared as state annotations in LangGraph::

    class State(TypedDict):
        remaining_steps: RemainingSteps
        is_last_step: IsLastStep

The Rust extension injects both values before each superstep and never persists them, so the
classes below only carry the annotation. ``is_last_step`` / ``remaining_steps`` remain available
as helper functions for imperative reads.
"""

from __future__ import annotations

from ._compat import NATIVE, module_getattr, re_export


class IsLastStep:
    """Annotation marking a channel as LangGraph's managed ``is_last_step`` value."""


class RemainingSteps:
    """Annotation marking a channel as LangGraph's managed ``remaining_steps`` value."""


__all__ = ["IsLastStep", "RemainingSteps", "is_last_step", "remaining_steps"]

_MISSING = re_export(globals(), ["is_last_step", "remaining_steps"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)
