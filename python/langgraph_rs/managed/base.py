"""Managed value base types (LangGraph ``langgraph.managed.base``)."""

from __future__ import annotations

from typing import Any, TypeVar

# LangGraph parameterises its managed-value helpers with these type variables.
U = TypeVar("U")
V = TypeVar("V")

__all__ = [
    "U",
    "V",
    "is_managed_value",
    "ManagedValue",
    "ManagedValueMapping",
    "ManagedValueSpec",
    "PregelScratchpad",
]


class PregelScratchpad:
    """State a managed value sees when it is computed."""

    step: int = 0
    stop: int = 0


class ManagedValue:
    """Base marker for managed channel values."""

    managed_name: str = ""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> Any:  # pragma: no cover - overridden upstream
        raise NotImplementedError


ManagedValueSpec = Any
"""A managed value spec: the marker class or an annotation carrying one."""

def is_managed_value(value: Any) -> bool:
    """Whether `value` is a managed value spec (a `ManagedValue` subclass)."""
    return isinstance(value, type) and issubclass(value, ManagedValue)


ManagedValueMapping = dict[str, ManagedValueSpec]
"""Mapping of channel name to managed value spec."""
