"""`is_last_step` / `remaining_steps` annotations (LangGraph ``langgraph.managed.is_last_step``).

The extension tracks both values in its engine; these annotations exist so state schemas written
against LangGraph keep importing.
"""

from __future__ import annotations

from typing import Annotated

__all__ = (
    "IsLastStep",
    "IsLastStepManager",
    "ManagedValue",
    "PregelScratchpad",
    "RemainingSteps",
    "RemainingStepsManager",
)


class ManagedValue:
    """Marker base class for managed channel values (LangGraph parity)."""

    managed_name: str = ""


class PregelScratchpad:
    """Scratchpad passed to managed values upstream; kept for attribute compatibility."""

    step: int = 0
    stop: int = 0


class IsLastStepManager:
    """Marker for the `is_last_step` managed channel."""

    managed_name = "is_last_step"


class RemainingStepsManager:
    """Marker for the `remaining_steps` managed channel."""

    managed_name = "remaining_steps"


IsLastStep = Annotated[bool, IsLastStepManager]
RemainingSteps = Annotated[int, RemainingStepsManager]
