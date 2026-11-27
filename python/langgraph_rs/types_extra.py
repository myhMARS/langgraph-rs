"""LangGraph ``langgraph.types`` equivalents.

Besides the runtime values (`Command`, `Send`, `Interrupt`, ...) LangGraph exports a set of typing
shapes here -- the seven stream-part types, checkpoint/task payloads, and small helpers. They are
reproduced with the same names and keys so type annotations in migrating code keep type-checking.

`xxh3_128_hexdigest` delegates to the `xxhash` package (the same algorithm LangGraph uses); when it
is absent the function raises instead of returning a different digest.
"""

from __future__ import annotations

import pickle
from typing import Annotated, Any, Callable, Literal, NamedTuple, Sequence, TypeVar, TypedDict, Union

from ._compat import module_getattr, re_export

__all__ = [
    "AnyMessage",
    "BaseCheckpointSaver",
    "CacheKey",
    "CheckpointMetadata",
    "CheckpointPayload",
    "CheckpointStreamPart",
    "CheckpointTask",
    "Checkpointer",
    "CustomStreamPart",
    "DebugPayload",
    "DebugStreamPart",
    "Durability",
    "GraphOutput",
    "MessagesStreamPart",
    "KeyFuncT",
    "N",
    "NS_SEP",
    "OutputT",
    "ResponseT",
    "StateT",
    "Overwrite",
    "PregelExecutableTask",
    "StateUpdate",
    "StreamPart",
    "TaskPayload",
    "TaskResultPayload",
    "TasksStreamPart",
    "ToolOutputMixin",
    "TracePolicy",
    "UpdatesStreamPart",
    "ValuesStreamPart",
    "default_cache_key",
    "default_retry_on",
    "ensure_valid_checkpointer",
    "get_cached_annotated_keys",
    "get_update_as_tuples",
    "omit_payload",
    "xxh3_128_hexdigest",
]

NS_SEP = "|"
Durability = Literal["sync", "async", "exit"]

try:  # LangChain's message union, when available
    from langchain_core.messages import AnyMessage as _AnyMessage
except Exception:  # pragma: no cover - langchain_core is optional
    _AnyMessage = Any

AnyMessage = _AnyMessage

StateT = TypeVar("StateT")
KeyFuncT = TypeVar("KeyFuncT")
N = TypeVar("N")
OutputT = TypeVar("OutputT")
ResponseT = TypeVar("ResponseT")


class ValuesStreamPart(TypedDict):
    """`stream_mode="values"` chunk."""

    type: Literal["values"]
    ns: tuple[str, ...]
    data: Any
    interrupts: list[Any]


class UpdatesStreamPart(TypedDict):
    """`stream_mode="updates"` chunk."""

    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]


class MessagesStreamPart(TypedDict):
    """`stream_mode="messages"` chunk."""

    type: Literal["messages"]
    ns: tuple[str, ...]
    data: tuple[Any, dict[str, Any]]


class CustomStreamPart(TypedDict):
    """`stream_mode="custom"` chunk."""

    type: Literal["custom"]
    ns: tuple[str, ...]
    data: Any


class CheckpointStreamPart(TypedDict):
    """`stream_mode="checkpoints"` chunk."""

    type: Literal["checkpoints"]
    ns: tuple[str, ...]
    data: dict[str, Any]


class TasksStreamPart(TypedDict):
    """`stream_mode="tasks"` chunk."""

    type: Literal["tasks"]
    ns: tuple[str, ...]
    data: Any


class DebugStreamPart(TypedDict):
    """`stream_mode="debug"` chunk."""

    type: Literal["debug"]
    ns: tuple[str, ...]
    data: "DebugPayload"


StreamPart = Union[
    ValuesStreamPart,
    UpdatesStreamPart,
    MessagesStreamPart,
    CustomStreamPart,
    CheckpointStreamPart,
    TasksStreamPart,
    DebugStreamPart,
]


class CheckpointTask(TypedDict):
    """One task of a checkpoint snapshot."""

    id: str
    name: str
    error: str | None
    result: Any
    interrupts: list[Any]
    state: Any


class CheckpointPayload(TypedDict):
    """`get_state()` payload."""

    config: dict[str, Any]
    metadata: dict[str, Any]
    values: Any
    next: tuple[str, ...]
    parent_config: dict[str, Any] | None
    tasks: tuple[CheckpointTask, ...]


class TaskPayload(TypedDict):
    """Pending task of a run."""

    id: str
    name: str
    input: Any
    triggers: tuple[str, ...]
    metadata: dict[str, Any] | None


class TaskResultPayload(TypedDict):
    """Result of an executed task."""

    id: str
    name: str
    error: str | None
    interrupts: list[Any]
    result: Any


class DebugPayload(TypedDict):
    """`stream_mode="debug"` payload."""

    step: int
    timestamp: str
    type: str
    payload: dict[str, Any]


class DeltaChannelHistory(TypedDict):
    """Per-channel write history (`get_delta_channel_history` results)."""

    writes: list[tuple[str, str, Any]]
    seed: int


class CheckpointMetadata(TypedDict, total=False):
    """Metadata stored alongside a checkpoint."""

    source: Literal["input", "loop", "update", "fork"]
    step: int
    parents: dict[str, str]


class StateUpdate(NamedTuple):
    """A state update produced by a task (LangGraph's `StateUpdate`)."""

    values: dict[str, Any] | None
    as_node: str | None = None
    task_id: str | None = None


class Overwrite:
    """Sentinel that replaces a channel value instead of reducing into it."""

    __slots__ = ("value", "type")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.type = "overwrite"

    def __repr__(self) -> str:
        return f"Overwrite(value={self.value!r})"


class GraphOutput:
    """Return value of `invoke` when interrupts occurred."""

    __slots__ = ("value", "interrupts")

    def __init__(self, value: Any = None, interrupts: Sequence[Any] = ()) -> None:
        self.value = value
        self.interrupts = tuple(interrupts)

    def __repr__(self) -> str:
        return f"GraphOutput(value={self.value!r}, interrupts={self.interrupts!r})"


class TracePolicy:
    """Controls what tracing captures for a node."""

    __slots__ = ("process_inputs", "process_outputs")

    def __init__(
        self,
        process_inputs: Callable[[Any], Any] | None = None,
        process_outputs: Callable[[Any], Any] | None = None,
    ) -> None:
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs


class CacheKey(NamedTuple):
    """Cache entry key (namespace + key + optional TTL)."""

    ns: tuple[str, ...]
    key: str
    ttl: int | None

class PregelExecutableTask:
    """Internal executable task (LangGraph's dataclass shape)."""

    __slots__ = (
        "name",
        "input",
        "proc",
        "writes",
        "config",
        "triggers",
        "retry_policy",
        "cache_key",
        "id",
        "path",
        "writers",
        "subgraphs",
        "timeout",
    )

    def __init__(
        self,
        name: str = "",
        input: Any = None,
        proc: Any = None,
        writes: Any = None,
        config: Any = None,
        triggers: Any = None,
        retry_policy: Any = None,
        cache_key: Any = None,
        id: str = "",
        path: Any = None,
        writers: Any = (),
        subgraphs: Any = (),
        timeout: Any = None,
    ) -> None:
        self.name = name
        self.input = input
        self.proc = proc
        self.writes = writes if writes is not None else []
        self.config = config if config is not None else {}
        self.triggers = tuple(triggers or ())
        self.retry_policy = tuple(retry_policy or ())
        self.cache_key = cache_key
        self.id = id
        self.path = tuple(path or ())
        self.writers = tuple(writers or ())
        self.subgraphs = tuple(subgraphs or ())
        self.timeout = timeout

class ToolOutputMixin:
    """Mixin marking a message as tool output."""


Checkpointer = Any

# LangGraph exports the checkpointer base class from `types` too; reuse the real one.
from .checkpoint.base import BaseCheckpointSaver  # noqa: E402  (circular-safe tail import)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(value, key=repr))
    return value


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Pickle-based cache key, like LangGraph's default."""
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)


def xxh3_128_hexdigest(data: bytes) -> str:
    """XXH3-128 hex digest (same algorithm LangGraph uses, via the `xxhash` package)."""
    try:
        import xxhash
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "xxh3_128_hexdigest needs the `xxhash` package: pip install xxhash"
        ) from error
    return xxhash.xxh3_128_hexdigest(data)


def omit_payload(_value: Any) -> dict[str, Any]:
    """`TracePolicy` helper that records an empty payload, dropping the value entirely.

    Use as `process_inputs` and/or `process_outputs` on a `TracePolicy` to keep a node's span and
    its timing while omitting its inputs/outputs from the trace.
    """
    return {}


def default_retry_on(exception: BaseException) -> bool:
    """LangGraph's default retry predicate: retry transient and network errors."""
    if isinstance(exception, (ConnectionError, TimeoutError, ValueError, OSError)):
        return True
    import httpx

    return isinstance(exception, httpx.HTTPError)


def ensure_valid_checkpointer(checkpointer: Any) -> Any:
    """Raise when a checkpointer object does not follow the saver protocol."""
    if checkpointer is None or checkpointer is False:
        return checkpointer
    required = ("get_tuple", "put", "put_writes", "list")
    missing = [name for name in required if not hasattr(checkpointer, name)]
    if missing:
        raise TypeError(
            f"checkpointer {type(checkpointer).__name__} is missing {missing}; a saver must "
            "implement get_tuple/put/put_writes/list"
        )
    return checkpointer


def get_cached_annotated_keys(obj: Any) -> list[str]:
    """Keys of a class/schema annotated with `Annotated[...]` metadata."""
    import typing

    annotations = getattr(obj, "__annotations__", {}) or {}
    try:
        resolved = typing.get_type_hints(obj, include_extras=True)
    except Exception:
        resolved = annotations
    return [key for key, value in resolved.items() if typing.get_origin(value) is Annotated]


def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Turn a state update (mapping or object with attributes) into (key, value) pairs."""
    if isinstance(input, dict):
        return [(key, input[key]) for key in keys if key in input]
    pairs = []
    for key in keys:
        value = getattr(input, key, None)
        if value is not None:
            pairs.append((key, value))
    return pairs


_MISSING = re_export(globals(), [])
__getattr__ = module_getattr(__name__, globals(), _MISSING)


class PregelTask(NamedTuple):
    """A task in a :class:`StateSnapshot` (LangGraph ``langgraph.types.PregelTask``).

    ``id`` / ``name`` / ``path`` are positional, ``error`` / ``interrupts`` / ``state`` / ``result``
    are optional, matching LangGraph's named tuple.
    """

    id: str
    name: str
    path: tuple[Any, ...] = ()
    error: Exception | None = None
    interrupts: tuple[Any, ...] = ()
    state: Any = None
    result: Any = None
