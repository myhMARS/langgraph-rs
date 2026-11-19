"""LangGraph ``langgraph.runtime`` equivalents."""

from __future__ import annotations

from ._compat import NATIVE, module_getattr, re_export

__all__ = ["current_context", "current_runtime", "get_runtime"]

_MISSING = re_export(globals(), __all__)
get_runtime = NATIVE.current_runtime
__getattr__ = module_getattr(__name__, globals(), _MISSING)
