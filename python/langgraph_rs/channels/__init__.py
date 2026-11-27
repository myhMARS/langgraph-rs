"""LangGraph ``langgraph.channels`` equivalents.

LangGraph lets you declare state as a mapping of channel *objects* instead of a TypedDict
(``StateGraph({"value": LastValue(int)})``). These classes mirror LangGraph's constructors and
carry a ``__channel_spec__`` that the extension maps onto its own channel kinds, so the resulting
graph structure (channel names and reduction semantics) stays identical to LangGraph's.

Unsupported channels raise when a graph is built instead of silently degrading: state reduction is
exactly the part that must not be approximated.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Sequence, TypeVar

__all__ = [
    "AnyValue",
    "BaseChannel",
    "BinaryOperatorAggregate",
    "DeltaChannel",
    "EphemeralValue",
    "LastValue",
    "LastValueAfterFinish",
    "NamedBarrierValue",
    "NamedBarrierValueAfterFinish",
    "Topic",
    "UntrackedValue",
]

Value = TypeVar("Value")

#: Sentinel for "no checkpoint yet" (LangGraph's `MISSING`).
_MISSING: Any = object()


class BaseChannel(Generic[Value]):
    """Base class of every channel (LangGraph's `BaseChannel`)."""

    __channel_spec__: Any = "last_value"
    #: LangGraph's runtime type for the channel, kept for introspection.
    typ: Any = None
    key: str = ""

    def __init__(self, typ: Any = None, key: str = "") -> None:
        self.typ = typ
        self.key = key

    #: Type variables LangGraph exposes on the class for annotations.
    ValueType: Any = None
    UpdateType: Any = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({getattr(self.typ, '__name__', self.typ)!r})"

    # -- serialization protocol (LangGraph's `BaseChannel`) ---------------------------------
    def copy(self) -> "BaseChannel[Value]":
        """A copy of the channel, via `checkpoint()` / `from_checkpoint()`."""
        return type(self).from_checkpoint(self.checkpoint())

    def checkpoint(self) -> Any:
        """Serializable form of the channel's current state (`MISSING` when empty)."""
        try:
            return self.get()
        except Exception:  # an empty channel checkpoints as MISSING, like LangGraph
            return _MISSING

    @classmethod
    def from_checkpoint(cls, checkpoint: Any) -> "BaseChannel[Value]":
        """Rebuild a channel from `checkpoint()`."""
        channel = cls()
        if hasattr(channel, "update") and checkpoint is not None:
            channel.update([checkpoint])
        return channel

    def get(self) -> Value:
        """The channel's current value; raises `EmptyChannelError` when it has none."""
        from ..errors import EmptyChannelError

        raise EmptyChannelError()

    def is_available(self) -> bool:
        """Whether the channel holds a value."""
        try:
            self.get()
            return True
        except Exception:
            return False

    def update(self, values: Sequence[Any]) -> bool:
        """Apply writes; returns whether the value changed."""
        return False

    def consume(self) -> bool:
        """Whether the channel was updated in this step (and reset that flag)."""
        return False

    def finish(self) -> None:
        """Called after a step; channels may reset themselves here."""
        return None


class LastValue(BaseChannel[Value]):
    """Stores the last value written to the channel."""

    __channel_spec__ = "last_value"


class LastValueAfterFinish(BaseChannel[Value]):
    """Stores the last value written, but only exposes it after the graph finishes."""

    __channel_spec__ = "last_value_after_finish"


class AnyValue(BaseChannel[Value]):
    """Stores the last value with no type or reducer validation."""

    __channel_spec__ = "any_value"


class EphemeralValue(BaseChannel[Value]):
    """Value written in one step and cleared in the next."""

    __channel_spec__ = "ephemeral_value"

    def __init__(self, typ: Any = None, guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard


class UntrackedValue(BaseChannel[Value]):
    """Value that is never checkpointed."""

    __channel_spec__ = "untracked_value"

    def __init__(self, typ: Any = None, guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard


class Topic(BaseChannel[list[Value]]):
    """Accumulates values; with ``accumulate=False`` only the current step is visible."""

    __channel_spec__ = "topic"

    def __init__(self, typ: Any = None, accumulate: bool = False) -> None:
        super().__init__(typ)
        self.accumulate = accumulate

    @property
    def __channel_spec__(self) -> str:  # type: ignore[override]
        return "topic_accumulate" if self.accumulate else "topic"

    def __repr__(self) -> str:
        return f"Topic({getattr(self.typ, '__name__', self.typ)!r}, accumulate={self.accumulate})"


class BinaryOperatorAggregate(BaseChannel[Value]):
    """Reduces writes with a binary operator (``operator.add``, ``max``, ...)."""

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]) -> None:
        super().__init__(typ)
        self.operator = operator

    @property
    def __channel_spec__(self) -> Callable[[Any, Any], Any]:  # type: ignore[override]
        return self.operator

    def __repr__(self) -> str:
        return f"BinaryOperatorAggregate({getattr(self.typ, '__name__', self.typ)!r}, {self.operator!r})"


class DeltaChannel(BaseChannel[Value]):
    """Accumulates deltas and snapshots every `snapshot_frequency` steps."""

    __channel_spec__ = "delta"

    def __init__(
        self,
        reducer: Callable[[Any, Any], Any],
        typ: type[Value] | None = None,
        *,
        snapshot_frequency: int = 100,
    ) -> None:
        super().__init__(typ)
        self.reducer = reducer
        self.snapshot_frequency = snapshot_frequency


class NamedBarrierValue(BaseChannel[Value]):
    """Waits until every named node has written the channel."""

    __channel_spec__ = None  # not supported by the extension

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        super().__init__(typ)
        self.names = set(names)


class NamedBarrierValueAfterFinish(NamedBarrierValue[Value]):
    """Barrier variant that only fires after the graph finishes."""


def _unsupported_channel(channel: BaseChannel) -> str:
    return (
        f"{type(channel).__name__} is not implemented by the langgraph_rs extension "
        "(barrier channels need engine support). Use LastValue, AnyValue, EphemeralValue, "
        "UntrackedValue, Topic, BinaryOperatorAggregate or DeltaChannel instead."
    )
# LangGraph's package re-exports its submodules, so `channels.topic.Topic` resolves too.
from . import (  # noqa: E402,F401
    any_value,
    base,
    binop,
    delta,
    ephemeral_value,
    last_value,
    named_barrier_value,
    topic,
    untracked_value,
)