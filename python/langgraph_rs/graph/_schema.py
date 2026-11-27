"""Schema normalisation done in Python before the builder sees a dict.

The extension used to embed this code as a Python string and call back into it; the normalisation
now happens here so the native builder only ever receives a plain ``{channel: spec}`` mapping.
"""

from __future__ import annotations

import typing
from typing import Any


class MessagesState(dict):
    """LangGraph's built-in single-channel state (an append-only message list)."""

    __annotations__ = {"messages": typing.Annotated[list, "add_messages"]}


def _schema_from_class(cls: Any) -> dict[str, Any]:
    """Turn a ``TypedDict``/dataclass schema into ``{channel: reducer}``.

    ``typing.get_type_hints`` resolves PEP 563 string annotations (``from __future__ import
    annotations`` maps every annotation to a string), which a plain ``__annotations__`` walk would
    silently treat as a last-value channel.
    """
    try:
        annotations = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        annotations = {}
        for base in reversed(getattr(cls, "__mro__", [cls])):
            annotations.update(getattr(base, "__annotations__", {}))
    out: dict[str, Any] = {}
    for name, annotation in annotations.items():
        reducer: Any = "last_value"
        if getattr(annotation, "__name__", None) == "IsLastStep":
            reducer = "is_last_step"
        elif getattr(annotation, "__name__", None) == "RemainingSteps":
            reducer = "remaining_steps"
        elif typing.get_origin(annotation) is typing.Annotated:
            args = typing.get_args(annotation)
            reducer = args[1]
        out[name] = reducer
    return out


def _is_custom_channel(spec: Any) -> bool:
    """Whether ``spec`` is a `BaseChannel` subclass that overrides `update`/`get`."""
    try:
        from ..channels import BaseChannel
    except Exception:  # pragma: no cover - channels always import in practice
        return False
    if not (isinstance(spec, type) and issubclass(spec, BaseChannel)):
        if not isinstance(spec, BaseChannel):
            return False
        kind = type(spec)
    else:
        kind = spec
    for name in ("update", "get"):
        if getattr(kind, name, None) is not getattr(BaseChannel, name, None):
            return True
    return False


def normalize_schema(schema: Any) -> Any:
    """Return the dict the native builder expects, marking custom channels.

    ``None`` stays ``None`` (a dynamic schema); a class schema is expanded here; a channel that
    implements its own merge is wrapped in ``{"__custom_channel__": channel}`` so the engine does
    not have to know about the shim's base class.
    """
    if schema is None:
        return None
    if not isinstance(schema, dict):
        schema = _schema_from_class(schema)
    normalized: dict[str, Any] = {}
    for name, spec in schema.items():
        # A reducer function may name its engine channel (`add_messages` -> "messages").
        marker = getattr(spec, "__channel_spec__", None)
        if marker is not None:
            spec = marker
        normalized[name] = {"__custom_channel__": spec} if _is_custom_channel(spec) else spec
    return normalized
