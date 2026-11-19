"""LangGraph ``langgraph.managed`` equivalents.

LangGraph exposes ``IsLastStep`` and ``RemainingSteps`` as annotation classes. The Rust
extension provides the equivalent channel helpers under ``is_last_step`` and
``remaining_steps``; the LangGraph spellings are aliased here so imports keep working.
"""

from __future__ import annotations

from ._compat import NATIVE, module_getattr, re_export

__all__ = ["IsLastStep", "RemainingSteps", "is_last_step", "remaining_steps"]

_MISSING = re_export(globals(), __all__)
IsLastStep = NATIVE.is_last_step
RemainingSteps = NATIVE.remaining_steps
__getattr__ = module_getattr(__name__, globals(), _MISSING)
