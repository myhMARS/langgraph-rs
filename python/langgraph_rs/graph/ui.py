"""UI message helpers (LangGraph ``langgraph.graph.ui``).

`push_ui_message` / `delete_ui_message` emit the UI event on the custom stream (LangGraph also
writes it into a state key through `CONFIG_KEY_SEND`, which the extension does not expose yet), and
`ui_message_reducer` applies `remove-ui` markers when merging the UI list.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from typing_extensions import TypedDict

from ..config import CONF, get_config, get_stream_writer

try:  # pragma: no cover - langchain-core is optional
    from langchain_core.messages import AnyMessage  # noqa: F401
except Exception:  # pragma: no cover
    AnyMessage = Any

__all__ = (
    "UIMessage",
    "RemoveUIMessage",
    "AnyUIMessage",
    "push_ui_message",
    "delete_ui_message",
    "ui_message_reducer",
)


class UIMessage(TypedDict):
    """A UI update rendered by the client."""

    type: Literal["ui"]
    id: str
    name: str
    props: dict[str, Any]
    metadata: dict[str, Any]


class RemoveUIMessage(TypedDict):
    """A request to remove a UI component from the current state."""

    type: Literal["remove-ui"]
    id: str


AnyUIMessage = UIMessage | RemoveUIMessage


def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: Any = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage:
    """Emit a UI message and return it."""
    writer = get_stream_writer()
    config = get_config() or {}

    message_id = None
    if message is not None:
        if isinstance(message, dict):
            message_id = message.get("id")
        else:
            message_id = getattr(message, "id", None)

    event: UIMessage = {
        "type": "ui",
        "id": id or str(uuid4()),
        "name": name,
        "props": props,
        "metadata": {
            "merge": merge,
            "run_id": config.get("run_id", None),
            "tags": config.get("tags", None),
            "name": config.get("run_name", None),
            **(metadata or {}),
            **({"message_id": message_id} if message_id else {}),
        },
    }

    writer(event)
    return event


def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """Emit a `remove-ui` message and return it."""
    writer = get_stream_writer()
    event: RemoveUIMessage = {"type": "remove-ui", "id": id}
    writer(event)
    return event


def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage | None,
    right: list[AnyUIMessage] | AnyUIMessage | None,
) -> list[AnyUIMessage]:
    """Merge two UI message lists, applying `remove-ui` markers."""
    left = left or []
    right = right or []
    if isinstance(left, dict):
        left = [left]
    if isinstance(right, dict):
        right = [right]

    merged: list[AnyUIMessage] = list(left)
    for item in right:
        if item.get("type") != "remove-ui":
            merged.append(item)
            continue
        merged = [existing for existing in merged if existing.get("id") != item["id"]]
    return merged
