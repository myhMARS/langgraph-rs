"""LangGraph ``langgraph.types`` equivalents."""

from __future__ import annotations

from collections.abc import Callable as AbcCallable
from typing import Any, Literal, Protocol, TypedDict

from ._compat import fill_unsupported, module_getattr, re_export
from .warnings import LangGraphDeprecatedSinceV10, LangGraphDeprecatedSinceV11
from .types_extra import (
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