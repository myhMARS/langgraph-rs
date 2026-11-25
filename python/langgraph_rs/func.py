"""LangGraph ``langgraph.func`` equivalents."""

from __future__ import annotations

from typing import Any

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
    "coerce_timeout_policy",
    "identifier",
    "is_async_callable",
    "sync_timeout_unsupported",
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

async def is_async_callable(func):

    import inspect as _inspect

    return _inspect.iscoroutinefunction(func)


def identifier(obj):

    return getattr(obj, "__name__", None) or type(obj).__name__


def coerce_timeout_policy(policy):

    import numbers

    if policy is None or hasattr(policy, "run_timeout") or hasattr(policy, "idle_timeout"):
        return policy
    if isinstance(policy, numbers.Real):
        return TimeoutPolicy(run_timeout=float(policy))
    raise TypeError(f"cannot coerce {type(policy).__name__} into a TimeoutPolicy")


def sync_timeout_unsupported():

    raise RuntimeError(
        "timeouts are only supported for async graphs; use ainvoke/astream or remove the timeout"
    )

_MISSING = re_export(globals(), __all__)
__getattr__ = module_getattr(__name__, globals(), _MISSING)