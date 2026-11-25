"""LangGraph ``langgraph.runtime`` equivalents.

`Runtime` mirrors LangGraph's dataclass (`context`, `store`, `stream_writer`, `heartbeat`,
`previous`, `execution_info`, `server_info`, `control`), and `get_runtime()` builds it from the
extension's current-run context. Outside a run it returns `DEFAULT_RUNTIME`, like LangGraph.

`current_context()` / `current_runtime()` remain as the extension's lower-level accessors
(`current_runtime` returns the raw mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar

from ._compat import NATIVE, module_getattr, re_export
from .store.base import BaseStore

__all__ = [
    "CONF",
    "CONFIG_KEY_RUNTIME",
    "BaseStore",
    "BaseUser",
    "ContextT",
    "DEFAULT_RUNTIME",
    "ExecutionInfo",
    "Runtime",
    "RunControl",
    "ServerInfo",
    "StreamWriter",
    "current_context",
    "current_runtime",
    "get_runtime",
]

CONF = "configurable"
CONFIG_KEY_RUNTIME = "__pregel_runtime"

ContextT = TypeVar("ContextT")


class BaseUser(Protocol):
    """Protocol for the authenticated user LangGraph exposes on `ServerInfo`."""

    identity: str
    display_name: str | None


@dataclass
class ExecutionInfo:
    """Where the current node invocation sits in the run."""

    checkpoint_id: str | None = None
    checkpoint_ns: str = ""
    task_id: str | None = None
    thread_id: str | None = None
    run_id: str | None = None
    node_attempt: int = 1
    node_first_attempt_time: float | None = None


@dataclass
class ServerInfo:
    """Deployment information (`assistant_id`, `graph_id`, `user`)."""

    assistant_id: str = ""
    graph_id: str | None = None
    user: BaseUser | None = None


def _no_op_stream_writer(*args: Any, **kwargs: Any) -> None:
    """Stream writer used outside a run (LangGraph's default)."""


def _no_op_heartbeat(*args: Any, **kwargs: Any) -> None:
    """Heartbeat used outside a run (LangGraph's default)."""


@dataclass
class Runtime(Generic[ContextT]):
    """Per-run information available from node code."""

    context: ContextT | None = None
    store: Any = None
    stream_writer: Callable[..., None] = _no_op_stream_writer
    heartbeat: Callable[..., None] = _no_op_heartbeat
    previous: Any = None
    execution_info: ExecutionInfo | None = None
    server_info: ServerInfo | None = None
    control: Any = None
    # Not part of LangGraph's dataclass: the extension also knows the namespace.
    namespace: str = ""

    def __post_init__(self) -> None:
        _ = field


DEFAULT_RUNTIME = Runtime()


def _native_stream_writer() -> Callable[..., None]:
    factory = getattr(NATIVE, "get_stream_writer", None)
    if factory is None:
        return _no_op_stream_writer
    try:
        writer = factory()
    except Exception:  # pragma: no cover - outside a node the factory raises
        return _no_op_stream_writer
    return writer if callable(writer) else _no_op_stream_writer


def get_runtime(context: Any = None) -> Runtime:
    """Runtime for the current run, or `DEFAULT_RUNTIME` outside a graph/node."""
    raw = NATIVE.current_runtime()
    if raw is None:
        return DEFAULT_RUNTIME
    context_value = context if context is not None else (raw or {}).get("context")
    thread_id = (raw or {}).get("thread_id")
    namespace = (raw or {}).get("namespace") or ""
    store = None
    if (raw or {}).get("has_store"):
        try:
            store = NATIVE.get_store()
        except Exception:  # pragma: no cover - store unavailable mid-teardown
            store = None
    return Runtime(
        context=context_value,
        store=store,
        stream_writer=_native_stream_writer(),
        execution_info=ExecutionInfo(thread_id=thread_id, checkpoint_ns=namespace),
        namespace=namespace,
    )


# LangGraph's runtime module also re-exports these two helpers.
get_config = NATIVE.get_config

_MISSING = re_export(globals(), ["RunControl", "StreamWriter", "current_context", "current_runtime"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)