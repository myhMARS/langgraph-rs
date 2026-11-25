"""Type variables shared by the graph APIs (LangGraph ``langgraph.typing``)."""

from __future__ import annotations

from typing import Any

try:  # `default=` requires typing_extensions before Python 3.13
    from typing import TypeVar
except ImportError:  # pragma: no cover - Python < 3.13 fallback
    from typing_extensions import TypeVar  # type: ignore[assignment]

__all__ = (
    "NodeInputT",
    "NodeInputT_contra",
    "StateLike",
    "StateT",
    "StateT_co",
    "StateT_contra",
    "InputT",
    "OutputT",
    "ContextT",
    "ContextT_contra",
)

StateT = TypeVar("StateT")
"""Type variable for the graph state."""

StateT_co = TypeVar("StateT_co", covariant=True)
StateT_contra = TypeVar("StateT_contra", contravariant=True)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

ContextT = TypeVar("ContextT")
"""Type variable for graph run scoped context (defaults to `None` upstream)."""

ContextT_contra = TypeVar("ContextT_contra", contravariant=True)
NodeInputT = TypeVar("NodeInputT")
NodeInputT_contra = TypeVar("NodeInputT_contra", contravariant=True)

StateLike = Any
"""Anything usable as graph state (a dict, TypedDict, pydantic model or dataclass)."""