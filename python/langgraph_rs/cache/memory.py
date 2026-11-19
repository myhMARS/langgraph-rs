"""``langgraph.cache.memory`` equivalents."""

from __future__ import annotations

from .._compat import module_getattr, re_export

__all__ = ["InMemoryCache"]

_MISSING = re_export(globals(), __all__)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
