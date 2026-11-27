"""``langgraph.channels.base`` equivalents (re-exported for import compatibility).

LangGraph's channel classes are generic over ``Value`` / ``Update`` / ``Checkpoint``; the type
variables are re-exported so annotations written against LangGraph keep working.
"""

from __future__ import annotations

from typing import TypeVar

from . import BaseChannel

Value = TypeVar("Value")
Update = TypeVar("Update")
Checkpoint = TypeVar("Checkpoint")

__all__ = ["BaseChannel", "Checkpoint", "Update", "Value"]

# LangGraph re-exports the error here for custom channels.
from ..errors import EmptyChannelError  # noqa: E402,F401
if "EmptyChannelError" not in __all__:
    __all__.append("EmptyChannelError")
