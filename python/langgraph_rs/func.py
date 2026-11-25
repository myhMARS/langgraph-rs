"""LangGraph ``langgraph.func`` equivalents."""

from __future__ import annotations

from ._compat import module_getattr, re_export
from .channels import EphemeralValue, LastValue
from .warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10

__all__ = [
    "BaseCache",
    "BaseCheckpointSaver",
    "CACHE_NS_WRITES",
    "EphemeralValue",
    "LangGraphDeprecatedSinceV05",
    "LangGraphDeprecatedSinceV10",
    "LastValue",
    "MISSING",
    "PREVIOUS",
    "BaseStore",
    "CachePolicy",
    "END",
    "RetryPolicy",
    "START",
    "StreamMode",
    "TimeoutPolicy",
    "entrypoint",
    "task",
]

CACHE_NS_WRITES = "__pregel_ns_writes"
PREVIOUS = "__previous__"
MISSING = object()

_MISSING = re_export(globals(), __all__)
__getattr__ = module_getattr(__name__, globals(), _MISSING)