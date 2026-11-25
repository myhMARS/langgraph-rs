"""LangGraph ``langgraph.func`` equivalents.

The functional API (`entrypoint` / `task`) is implemented by the extension; the other names are
re-exported for import compatibility. LangGraph's own internal machinery (`Pregel`, `PregelNode`,
`ChannelWrite`, …) raises when used, because the extension runs its own engine.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Generic, TypeVar, cast, get_args, get_origin, overload

from ._compat import fill_unsupported, module_getattr, re_export
from .cache.base import BaseCache
from .channels import EphemeralValue, LastValue
from .checkpoint.base import BaseCheckpointSaver
from .store.base import BaseStore
from .types import DeprecatedKwargs, StreamMode
from .warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10

try:  # Python 3.11+ ships `Unpack` in typing.
    from typing import Unpack  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - 3.10 fallback
    from typing_extensions import Unpack  # type: ignore[no-redef]

__all__ = [
    "Any",
    "Awaitable",
    "BaseCache",
    "BaseCheckpointSaver",
    "BaseStore",
    "CACHE_NS_WRITES",
    "CachePolicy",
    "Callable",
    "ChannelWrite",
    "ChannelWriteEntry",
    "ContextT",
    "DeprecatedKwargs",
    "END",
    "EphemeralValue",
    "Generic",
    "LangGraphDeprecatedSinceV05",
    "LangGraphDeprecatedSinceV10",
    "LastValue",
    "MISSING",
    "P",
    "PREVIOUS",
    "Pregel",
    "PregelNode",
    "R",
    "RetryPolicy",
    "S",
    "START",
    "Sequence",
    "StreamMode",
    "SyncAsyncFuture",
    "T",
    "TimeoutPolicy",
    "TypeVar",
    "Unpack",
    "cast",
    "coerce_timeout_policy",
    "dataclass",
    "entrypoint",
    "functools",
    "get_args",
    "get_origin",
    "get_runnable_for_entrypoint",
    "identifier",
    "inspect",
    "is_async_callable",
    "overload",
    "sync_timeout_unsupported",
    "task",
    "timedelta",
    "warnings",
]

CACHE_NS_WRITES = "__pregel_ns_writes"
PREVIOUS = "__previous__"
MISSING = object()

# Generic parameters LangGraph re-exports from `pregel._call` / `langgraph.typing`.
ContextT = TypeVar("ContextT")
P = TypeVar("P")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")


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


class _EntrypointRunnable:
    """Minimal ``Runnable``-like wrapper around an entrypoint callable."""

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def invoke(self, input, config=None, **kwargs):
        if config is not None:
            return self.func(input, config=config)
        return self.func(input)

    async def ainvoke(self, input, config=None, **kwargs):
        result = self.invoke(input, config=config)
        if inspect.isawaitable(result):
            return await result
        return result


def get_runnable_for_entrypoint(func):
    """Wrap an entrypoint callable so it exposes `invoke` / `ainvoke`."""

    return _EntrypointRunnable(func)


# Real adapters are re-exported by the extension; names it does not implement become
# placeholders that raise with an explanation when used.
_MISSING = re_export(
    globals(),
    [
        "CachePolicy",
        "END",
        "RetryPolicy",
        "START",
        "TimeoutPolicy",
        "entrypoint",
        "task",
    ],
)
fill_unsupported(
    globals(),
    [name for name in ("Pregel", "PregelNode", "ChannelWrite", "ChannelWriteEntry", "SyncAsyncFuture") if name not in globals()],
    hint="The extension runs its own engine; use the compiled graph returned by StateGraph.compile().",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)