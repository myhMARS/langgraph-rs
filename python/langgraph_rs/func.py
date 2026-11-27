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
from . import langgraph_rs as _ext  # the compiled extension module

_NATIVE_ENTRYPOINT = _ext.entrypoint
_NATIVE_TASK = _ext.task

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


class _Resolved:
    """A finished task result, shaped like LangGraph's future."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value

    async def aresult(self) -> Any:
        return self._value

    def __await__(self) -> Any:
        async def _wait() -> Any:
            return self._value

        return _wait().__await__()


class _TaskProxy:
    """LangGraph hands back a future for every task call; the engine may resolve eagerly."""

    __slots__ = ("_handle",)

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outcome = self._handle(*args, **kwargs)
        if hasattr(outcome, "result") or hasattr(outcome, "aresult"):
            return outcome
        return _Resolved(outcome)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._handle, item)


def task(
    __func_or_none__: Any = None,
    *,
    name: str | None = None,
    retry_policy: Any = None,
    cache_policy: Any = None,
    timeout: Any = None,
    **kwargs: Any,
) -> Any:
    """LangGraph's ``task`` decorator (usable bare or with options)."""
    del retry_policy, cache_policy, timeout, kwargs  # the engine applies these at the call site

    def decorate(function: Any) -> Any:
        return _TaskProxy(_NATIVE_TASK(function, name))

    if __func_or_none__ is None:
        return decorate
    if callable(__func_or_none__):
        return decorate(__func_or_none__)
    raise TypeError("task() expects a callable or keyword options")


def _resolve_value(value: Any) -> Any:
    """Resolve task futures inside a returned value, as LangGraph does for entrypoint results."""
    if isinstance(value, _Resolved):
        return value.result()
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_value(item) for item in value)
    result = getattr(value, "result", None)
    if callable(result) and type(value).__name__ in ("SyncAsyncFuture", "Future", "_TaskFuture"):
        return _resolve_value(result())
    return value


_ENTRYPOINT_VALUE = "__entrypoint_value__"


class _EntrypointValue:
    """Adapts LangGraph's value-in/value-out entrypoint onto a graph with one value channel.

    The engine runs graphs over mapping states; LangGraph's functional API passes the input value
    straight to the decorated function and returns its result, so this adapter translates at the
    boundary (input -> channel, channel -> result).
    """

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    # --- translation helpers -------------------------------------------------
    def _to_state(self, value: Any) -> Any:
        from .types import Command

        if isinstance(value, Command):
            # A resumed run keeps its command untouched.
            return value
        return {_ENTRYPOINT_VALUE: value}

    def _from_state(self, state: Any) -> Any:
        if not isinstance(state, dict):
            return state
        if "__interrupt__" in state:
            return state
        return state.get(_ENTRYPOINT_VALUE)

    def _from_chunk(self, chunk: Any) -> Any:
        return self._from_state(chunk)

    # --- the Pregel-like surface LangGraph's entrypoint exposes --------------
    def invoke(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        result = self._compiled.invoke(self._to_state(value), config, **kwargs)
        return self._from_state(result)

    async def ainvoke(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        result = await self._compiled.ainvoke(self._to_state(value), config, **kwargs)
        return self._from_state(result)

    def stream(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        for chunk in self._compiled.stream(self._to_state(value), config, **kwargs):
            yield self._from_chunk(chunk)

    async def astream(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        async for chunk in self._compiled.astream(self._to_state(value), config, **kwargs):
            yield self._from_chunk(chunk)

    def resume(self, thread_id: str, value: Any = None, **kwargs: Any) -> Any:
        """Convenience wrapper for `invoke(Command(resume=value), config)`."""
        result = self._compiled.resume(thread_id, value, **kwargs)
        return self._from_state(result)

    def get_state(self, config: Any) -> Any:
        snapshot = self._compiled.get_state(config)
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict) and _ENTRYPOINT_VALUE in values:
            snapshot.values = values[_ENTRYPOINT_VALUE]
        return snapshot

    def get_state_history(self, config: Any, **kwargs: Any) -> Any:
        return self._compiled.get_state_history(config, **kwargs)

    def update_state(self, config: Any, values: Any, **kwargs: Any) -> Any:
        return self._compiled.update_state(config, self._to_state(values), **kwargs)

    def with_config(self, config: Any = None, **kwargs: Any) -> Any:
        return self._compiled.with_config(config, **kwargs)

    def __or__(self, other: Any) -> Any:
        return self._compiled.__or__(other)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._compiled, item)


def entrypoint(
    *args: Any,
    checkpointer: Any = None,
    store: Any = None,
    cache: Any = None,
    context_schema: Any = None,
    cache_policy: Any = None,
    retry_policy: Any = None,
    timeout: Any = None,
    **kwargs: Any,
) -> Any:
    """LangGraph's ``entrypoint`` decorator (bare, or with compiler options)."""
    del cache, cache_policy, retry_policy, timeout, kwargs
    function = None
    if args:
        candidate = args[0]
        if callable(candidate) and not hasattr(candidate, "get_tuple"):
            function = candidate
        else:
            checkpointer = candidate if checkpointer is None else checkpointer

    def decorate(function: Any) -> Any:
        from .graph.state import StateGraph

        graph = StateGraph({_ENTRYPOINT_VALUE: "last_value"})

        def node(state: Any) -> Any:
            value = state.get(_ENTRYPOINT_VALUE) if isinstance(state, dict) else state
            return {_ENTRYPOINT_VALUE: _resolve_value(function(value))}

        graph.add_node("entrypoint", node)
        graph.set_entry_point("entrypoint")
        graph.set_finish_point("entrypoint")
        compiled = graph.compile(
            checkpointer=checkpointer, store=store, context_schema=context_schema
        )
        return _EntrypointValue(compiled)

    if function is not None:
        # Used as `@entrypoint` / `entrypoint(fn, ...)`.
        return decorate(function)
    return decorate
