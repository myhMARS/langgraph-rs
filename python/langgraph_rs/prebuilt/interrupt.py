"""Human-in-the-loop interrupt payload types (LangGraph ``langgraph.prebuilt.interrupt``).

These moved to `langchain.agents.interrupt` upstream and are deprecated there; the extension keeps
them importable with the same shapes.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from ..warnings import LangGraphDeprecatedSinceV10  # noqa: F401

__all__ = ["ActionRequest", "HumanInterrupt", "HumanInterruptConfig", "HumanResponse"]


class HumanInterruptConfig(TypedDict):
    """Which actions a human may take at an interrupt."""

    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    """The action a human is asked to approve."""

    action: str
    args: dict


class HumanInterrupt(TypedDict):
    """An interrupt that requires human intervention."""

    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None


class HumanResponse(TypedDict):
    """The response provided when the graph resumes."""

    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
