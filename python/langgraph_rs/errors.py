"""LangGraph ``langgraph.errors`` equivalents.

The control-flow signals (``GraphBubbleUp``, ``GraphInterrupt``) come from the Rust extension;
everything else is defined here with LangGraph's exact constructor signatures and messages so
``except`` clauses and error text keep working after a drop-in switch.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Sequence

from ._compat import module_getattr, re_export

__all__ = [
    "Command",
    "EmptyChannelError",
    "EmptyInputError",
    "ErrorCode",
    "GraphBubbleUp",
    "GraphDrained",
    "GraphInterrupt",
    "GraphRecursionError",
    "Interrupt",
    "InvalidUpdateError",
    "LangGraphDeprecatedSinceV10",
    "NodeCancelledError",
    "NodeError",
    "NodeInterrupt",
    "NodeTimeoutError",
    "ParentCommand",
    "TaskNotFound",
    "UnknownInterruptError",
    "create_error_message",
    "is_invalid_resume",
]

_MISSING = re_export(
    globals(),
    ["Command", "GraphBubbleUp", "GraphInterrupt", "GraphRecursionError", "Interrupt"],
)


class EmptyInputError(Exception):
    """Raised when a graph is invoked without input."""


class ParentCommand(GraphBubbleUp):  # type: ignore[misc,valid-type]
    """Raised by a subgraph that wants the parent graph to run the command."""

    def __init__(self, command: Any) -> None:
        super().__init__("ParentCommand")
        self.command = command


def create_error_message(*, message: str, error_code: "ErrorCode") -> str:
    """Build LangGraph's error text, including the troubleshooting URL."""
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/"
        f"errors/{error_code.value}"
    )


_INVALID_RESUME = "_langgraph_invalid_resume"


def is_invalid_resume(error: BaseException) -> bool:
    """Tell an interrupt-related validation failure apart from invalid tool arguments."""
    return (
        getattr(error, _INVALID_RESUME, False) is True
        or getattr(error, "_invalid_resume", False) is True
    )


class EmptyChannelError(Exception):
    """Raised when a channel has no value yet."""


class InvalidUpdateError(Exception):
    """Raised when a node returns an update the state schema rejects."""


class TaskNotFound(Exception):
    """Raised when a pending task cannot be found in the checkpoint."""


class NodeTimeoutError(Exception):
    """Raised when a node exceeds its configured idle or run timeout."""

    def __init__(
        self,
        node: str,
        elapsed: float,
        *,
        kind: Literal["idle", "run"],
        idle_timeout: float | None = None,
        run_timeout: float | None = None,
    ) -> None:
        if kind == "idle":
            timeout = idle_timeout if idle_timeout is not None else 0.0
            message = (
                f"Node {node!r} exceeded its idle timeout of {timeout:.3f}s without making "
                f"progress (elapsed: {elapsed:.3f}s)."
            )
        else:
            timeout = run_timeout if run_timeout is not None else 0.0
            message = (
                f"Node {node!r} exceeded its run timeout of {timeout:.3f}s "
                f"(elapsed: {elapsed:.3f}s)."
            )
        super().__init__(message)
        self.node = node
        self.elapsed = elapsed
        self.kind = kind
        self.idle_timeout = idle_timeout
        self.run_timeout = run_timeout


class NodeCancelledError(Exception):
    """Raised when a node is cancelled by the run control."""


class UnknownInterruptError(Exception):
    """Raised when a resume value does not match any pending interrupt."""

    def __init__(
        self,
        message: str,
        *,
        reason: Literal["not_pending", "empty_map", "ambiguous"],
        interrupt_ids: Sequence[str] = (),
        pending_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.interrupt_ids = tuple(interrupt_ids)
        self.pending_ids = tuple(pending_ids)


class _ParentCommandBase(Exception):
    """Placeholder base replaced below by `GraphBubbleUp` when the extension provides it."""


def _bubble_up_base() -> type:
    base = globals().get("GraphBubbleUp")
    return base if isinstance(base, type) else Exception


def _graph_interrupt_base() -> type:
    base = globals().get("GraphInterrupt")
    return base if isinstance(base, type) else Exception


class GraphDrained(_bubble_up_base()):  # type: ignore[misc,valid-type]
    """Raised when the run loop drains because the runtime is shutting down."""

    def __init__(self, reason: str = "shutdown") -> None:
        super().__init__(f"Graph drained: {reason}")
        self.reason = reason


class NodeInterrupt(_graph_interrupt_base()):  # type: ignore[misc,valid-type]
    """Deprecated alias for :class:`Interrupt` raised from a node."""

    def __init__(self, value: Any, id: str | None = None) -> None:
        super().__init__(value)
        self.value = value
        self.id = id


class NodeError:
    """Wraps the exception a node raised, keeping the node name for the caller."""

    __slots__ = ("node", "error")

    def __init__(self, node: str, error: BaseException) -> None:
        self.node = node
        self.error = error

    def __repr__(self) -> str:
        return f"NodeError(node={self.node!r}, error={self.error!r})"


class ErrorCode(Enum):
    """Codes LangGraph uses in its ``InvalidUpdateError``-style messages."""

    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"


def _deprecated_warning_class() -> type:
    from .warnings import LangGraphDeprecatedSinceV10

    return LangGraphDeprecatedSinceV10


LangGraphDeprecatedSinceV10 = _deprecated_warning_class()

__getattr__ = module_getattr(__name__, globals(), _MISSING)


def _register_native_error_classes() -> None:
    """Hand the exception classes to the extension.

    Rust must not import this package, so it raises these classes only after they are registered
    here (the shim imports `langgraph_rs.errors` early in its `__init__`).
    """
    try:
        from . import langgraph_rs as _native
    except Exception:  # pragma: no cover - the extension is always present at runtime
        return
    register = getattr(_native, "_register_error_classes", None)
    if register is None:  # pragma: no cover - older builds
        return
    try:
        register(
            {
                "InvalidUpdateError": InvalidUpdateError,
                "NodeTimeoutError": NodeTimeoutError,
                "EmptyInputError": EmptyInputError,
                "EmptyChannelError": EmptyChannelError,
                "TaskNotFound": TaskNotFound,
            }
        )
    except Exception:  # pragma: no cover - defensive
        pass


_register_native_error_classes()
