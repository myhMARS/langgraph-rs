"""LangGraph ``langgraph.types`` equivalents."""

from __future__ import annotations

from collections.abc import Callable as AbcCallable
from typing import Any, Literal, NamedTuple, Protocol, TypedDict

from ._compat import fill_unsupported, module_getattr, re_export
from .warnings import LangGraphDeprecatedSinceV10, LangGraphDeprecatedSinceV11
from .types_extra import (
    PregelTask as _PregelTask,
    AnyMessage,
    BaseCheckpointSaver,
    CacheKey,
    CheckpointMetadata,
    CheckpointPayload,
    CheckpointStreamPart,
    CheckpointTask,
    Checkpointer,
    CustomStreamPart,
    DebugPayload,
    DebugStreamPart,
    Durability,
    GraphOutput,
    MessagesStreamPart,
    N,
    NS_SEP,
    KeyFuncT,
    OutputT,
    ResponseT,
    StateT,
    Overwrite,
    PregelExecutableTask,
    StateUpdate,
    StreamPart,
    TaskPayload,
    TaskResultPayload,
    TasksStreamPart,
    ToolOutputMixin,
    TracePolicy,
    UpdatesStreamPart,
    ValuesStreamPart,
    default_cache_key,
    default_retry_on,
    ensure_valid_checkpointer,
    get_cached_annotated_keys,
    get_update_as_tuples,
    omit_payload,
    xxh3_128_hexdigest,
)

__all__ = [
    "All",
    "AnyMessage",
    "BaseCheckpointSaver",
    "CacheKey",
    "CachePolicy",
    "CheckpointMetadata",
    "CheckpointPayload",
    "CheckpointStreamPart",
    "CheckpointTask",
    "Checkpointer",
    "Command",
    "CustomStreamPart",
    "DebugPayload",
    "DebugStreamPart",
    "Durability",
    "GraphInterrupt",
    "GraphOutput",
    "Interrupt",
    "LangGraphDeprecatedSinceV10",
    "LangGraphDeprecatedSinceV11",
    "MessagesStreamPart",
    "KeyFuncT",
    "N",
    "NS_SEP",
    "OutputT",
    "ResponseT",
    "StateT",
    "Overwrite",
    "PregelExecutableTask",
    "PregelTask",
    "RetryPolicy",
    "RunControl",
    "Send",
    "StateSnapshot",
    "StateUpdate",
    "StreamMode",
    "StreamPart",
    "StreamWriter",
    "Task",
    "TaskPayload",
    "TaskResultPayload",
    "TasksStreamPart",
    "TimeoutPolicy",
    "ToolOutputMixin",
    "TracePolicy",
    "UpdatesStreamPart",
    "ValuesStreamPart",
    "default_cache_key",
    "default_retry_on",
    "ensure_valid_checkpointer",
    "get_cached_annotated_keys",
    "get_update_as_tuples",
    "interrupt",
    "omit_payload",
    "xxh3_128_hexdigest",
]

# Runtime values (the extension's classes) are re-exported, but the *type* surface matches
# LangGraph: `StreamMode` / `All` are `Literal` aliases and `StreamWriter` is a callable type.
_MISSING = re_export(globals(), ["CachePolicy", "Command", "GraphInterrupt", "Interrupt", "PregelTask", "RetryPolicy", "RunControl", "Send", "StateSnapshot", "Task", "TimeoutPolicy", "interrupt"])

# ``PregelTask`` is a plain named tuple in LangGraph, so the Python definition is used.
PregelTask = _PregelTask


class StateSnapshot(NamedTuple):
    """LangGraph's snapshot of a thread: a named tuple with derived conveniences.

    The eight fields mirror LangGraph exactly (so ``isinstance(snapshot, tuple)`` and tuple
    unpacking work); the extension additionally answers mapping-style lookups and exposes the
    derived `thread_id` / `step` / `interrupted` helpers older code used.
    """

    values: Any
    next: Any
    config: Any
    metadata: Any
    created_at: Any
    parent_config: Any
    tasks: Any
    interrupts: Any

    @property
    def thread_id(self) -> Any:
        return (self.config or {}).get("configurable", {}).get("thread_id")

    @property
    def checkpoint_ns(self) -> Any:
        return (self.config or {}).get("configurable", {}).get("checkpoint_ns")

    @property
    def checkpoint_id(self) -> Any:
        return (self.config or {}).get("configurable", {}).get("checkpoint_id")

    @property
    def parent_checkpoint_id(self) -> Any:
        return (self.parent_config or {}).get("configurable", {}).get("checkpoint_id")

    @property
    def step(self) -> Any:
        return (self.metadata or {}).get("step")

    @property
    def interrupted(self) -> bool:
        return bool(self.interrupts)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError as error:
                raise KeyError(key) from error
        return tuple.__getitem__(self, key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Mapping-style lookup kept for callers that treated the snapshot as a dict."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> tuple[str, ...]:
        return self._fields

    def __contains__(self, item: Any) -> bool:
        # Tuple membership is by value; strings are treated as field names.
        if isinstance(item, str):
            return item in self._fields
        return tuple.__contains__(self, item)

    def items(self) -> Any:
        return ((key, self[key]) for key in self._fields)

    @classmethod
    def from_native(cls, raw: Any) -> "StateSnapshot":
        """Build a snapshot from the engine's JSON-shaped result."""
        if isinstance(raw, cls) or raw is None:
            return raw
        data = dict(raw)
        return cls(
            values=data.get("values"),
            next=data.get("next"),
            config=data.get("config"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            parent_config=data.get("parent_config"),
            tasks=data.get("tasks"),
            interrupts=data.get("interrupts"),
        )
fill_unsupported(globals(), _MISSING)

StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
"""How the stream method should emit outputs (LangGraph's `StreamMode`)."""

All = Literal["*"]
"""Send every node's output (`Command(goto=All)`)."""

StreamWriter = AbcCallable[[Any], None]
"""Writes a value to the `custom` stream from inside a node."""

MISSING: Any = object()
"""Sentinel for "argument not provided" (LangGraph's `MISSING`)."""


class DeprecatedKwargs(TypedDict, total=False):
    """Deprecated keyword arguments accepted for compatibility."""


class SendProtocol(Protocol):
    """Minimal shape of a `Send` (node name + argument)."""

    node: str
    arg: Any
fill_unsupported(globals(), _MISSING)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
# LangGraph also exposes these langchain-core / pydantic names from its `types` module.
try:  # pragma: no cover - langchain-core is optional
    from langchain_core.runnables import Runnable, RunnableConfig  # noqa: F401

    __all__.extend(["Runnable", "RunnableConfig"])
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - pydantic is optional
    from pydantic import TypeAdapter, ValidationError  # noqa: F401

    __all__.extend(["TypeAdapter", "ValidationError"])
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - Python 3.12+
    from typing import TypeAliasType  # noqa: F401
except ImportError:  # pragma: no cover
    try:
        from typing_extensions import TypeAliasType  # noqa: F401
    except ImportError:
        pass
else:
    if "TypeAliasType" not in __all__:
        __all__.append("TypeAliasType")