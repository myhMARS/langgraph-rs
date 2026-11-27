"""``langgraph.graph.message`` equivalents.

``MessageGraph`` is LangGraph's deprecated-but-supported builder whose whole state is one
append-only message list. It is reproduced here on top of the Rust ``StateGraph`` with the same
single ``messages`` channel, so checkpoints stay structurally identical to LangGraph's:

* the graph has exactly one channel, ``messages``, using the ``add_messages`` reducer;
* nodes receive the message list (not the state mapping) and may return a single message, a list
  of messages, or a partial state mapping;
* ``invoke`` accepts a list of messages and returns the resulting list.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from .._compat import NATIVE, module_getattr, re_export

try:  # LangGraph ships its own deprecation category; reuse it so `filterwarnings` matches.
    from ..warnings import LangGraphDeprecatedSinceV10 as _DeprecatedV1
except Exception:  # pragma: no cover - warnings module always exists
    _DeprecatedV1 = DeprecationWarning


def _warn_deprecated_v1(message: str) -> None:
    import warnings

    warnings.warn(message, _DeprecatedV1, stacklevel=3)
from ..constants import CONF, TASKS

CONFIG_KEY_SEND = "__pregel_send"

try:  # LangChain message types, when available
    from langchain_core.messages import (
        AnyMessage,
        BaseMessage,
        BaseMessageChunk,
        MessageLikeRepresentation,
    )
except Exception:  # pragma: no cover - langchain_core is optional
    AnyMessage = object
    BaseMessage = object
    BaseMessageChunk = object
    MessageLikeRepresentation = object
from ..warnings import LangGraphDeprecatedSinceV10  # noqa: F401

try:  # pragma: no cover - langchain-core is optional
    from langchain_core.messages import convert_to_messages, message_chunk_to_message  # noqa: F401
except Exception:  # pragma: no cover
    def convert_to_messages(messages: Any, **kwargs: Any) -> list[Any]:
        raise ImportError("convert_to_messages requires langchain-core")

    def message_chunk_to_message(chunk: Any) -> Any:
        raise ImportError("message_chunk_to_message requires langchain-core")

from .state import StateGraph

__all__ = [
    "AnyMessage",
    "BaseMessage",
    "BaseMessageChunk",
    "CONF",
    "CONFIG_KEY_SEND",
    "MessageGraph",
    "MessageLikeRepresentation",
    "Messages",
    "REMOVE_ALL_MESSAGES",
    "push_message",
    "MessagesState",
    "RemoveMessage",
    "TASKS",
    "add_messages",
    "message_graph",
]

_MESSAGES_SCHEMA = {"messages": "add_messages"}


def _is_message(value: Any) -> bool:
    return hasattr(value, "content") and hasattr(value, "type")


def _dump_message(value: Any) -> Any:
    """Message objects are JSON-dumped at the engine boundary, like LangGraph's reducer does."""
    if _is_message(value) and not isinstance(value, dict):
        dump = getattr(value, "model_dump", None) or getattr(value, "dict", None)
        if dump is not None:
            try:
                return dump()
            except Exception:  # pragma: no cover - exotic message types fall through
                return value
    return value


def _dump_messages(values: Any) -> Any:
    if isinstance(values, (list, tuple)):
        return [_dump_message(item) for item in values]
    return _dump_message(values)


def _load_messages(values: Any) -> Any:
    """Rebuild LangChain message objects for the caller when langchain_core is installed."""
    if not isinstance(values, list):
        return values
    try:
        from langchain_core.messages import messages_from_dict
    except Exception:  # pragma: no cover - langchain_core is optional
        return values
    payload = []
    for item in values:
        if isinstance(item, dict) and "type" in item:
            data = {key: value for key, value in item.items() if key != "type"}
            payload.append({"type": item["type"], "data": data})
        else:
            payload.append(item)
    try:
        # `messages_from_dict` wants LangGraph's `{"type": ..., "data": {...}}` shape.
        return messages_from_dict(payload)
    except Exception:  # pragma: no cover - unknown message dicts stay as dicts
        return values


def _as_update(result: Any) -> Any:
    """Turn a node result into a state update for the ``messages`` channel."""
    if isinstance(result, dict):
        # A dict without the `messages` channel is itself a message (LangGraph's dict form);
        # an empty mapping means "no update".
        if not result:
            return {}
        if "messages" in result:
            return {"messages": _dump_messages(result["messages"])}
        return {"messages": [_dump_message(result)]}
    if isinstance(result, (list, tuple)):
        return {"messages": _dump_messages(list(result))}
    return {"messages": [_dump_message(result)]}


def _wrap_node(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(state: Any) -> Any:
        # LangGraph hands MessageGraph nodes the message list, not the state mapping.
        if isinstance(state, dict) and set(state) == {"messages"}:
            state = state["messages"]
        result = function(state)
        if inspect.isawaitable(result):  # pragma: no cover - async handled by the engine
            return result
        return _as_update(result)

    return wrapped


def _coerce_input(value: Any) -> Any:
    """Accept a message list (LangGraph form), a single message, or a state mapping."""
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {"messages": _dump_messages(list(value))}
    if _is_message(value):
        return {"messages": [_dump_message(value)]}
    if isinstance(value, dict) and "messages" in value:
        return {**value, "messages": _dump_messages(value["messages"])}
    return value


def _unwrap_output(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"messages"}:
        return value["messages"]
    return value


_MESSAGE_GRAPH_DEPRECATION = (
    "MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0. "
    "Please use StateGraph with a `messages` key instead. "
    "Deprecated in LangGraph V1.0 to be removed in V2.0."
)


class MessageGraph(StateGraph):
    """A ``StateGraph`` whose entire state is a single, append-only message list."""

    def __new__(cls) -> "MessageGraph":
        _warn_deprecated_v1(_MESSAGE_GRAPH_DEPRECATION)
        # The Rust builder exposes `#[new]`, so a Python subclass has to forward the schema here.
        return super().__new__(cls, dict(_MESSAGES_SCHEMA))

    def add_node(self, node: Any, action: Any = None, **kwargs: Any) -> Any:
        if action is None:
            action = node
            node = getattr(action, "__name__", "node")
        if not callable(action):
            raise TypeError("MessageGraph.add_node expects a callable node")
        super().add_node(node, _wrap_node(action), **kwargs)
        return self

    # The builder methods below return `self` so chaining stays on `MessageGraph`.
    def add_edge(self, *args: Any, **kwargs: Any) -> Any:
        super().add_edge(*args, **kwargs)
        return self

    def add_conditional_edges(self, *args: Any, **kwargs: Any) -> Any:
        super().add_conditional_edges(*args, **kwargs)
        return self

    def set_entry_point(self, *args: Any, **kwargs: Any) -> Any:
        super().set_entry_point(*args, **kwargs)
        return self

    def set_finish_point(self, *args: Any, **kwargs: Any) -> Any:
        super().set_finish_point(*args, **kwargs)
        return self

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        compiled = super().compile(*args, **kwargs)
        return _CompiledMessageGraph(compiled)


class _CompiledMessageGraph:
    """Delegates to the compiled graph, adding LangGraph's list-in/list-out shape."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"MessageGraph({self._inner!r})"

    def invoke(self, input: Any = None, **kwargs: Any) -> Any:
        return _load_messages(_unwrap_output(self._inner.invoke(_coerce_input(input), **kwargs)))

    async def ainvoke(self, input: Any = None, **kwargs: Any) -> Any:
        return _load_messages(
            _unwrap_output(await self._inner.ainvoke(_coerce_input(input), **kwargs))
        )

    def resume(self, thread_id: str, value: Any = None, **kwargs: Any) -> Any:
        return _load_messages(_unwrap_output(self._inner.resume(thread_id, value, **kwargs)))

    async def aresume(self, thread_id: str, value: Any = None, **kwargs: Any) -> Any:
        return _unwrap_output(await self._inner.aresume(thread_id, value, **kwargs))

    def stream(self, input: Any = None, **kwargs: Any) -> Any:
        return self._inner.stream(_coerce_input(input), **kwargs)

    def astream(self, input: Any = None, **kwargs: Any) -> Any:
        return self._inner.astream(_coerce_input(input), **kwargs)

    def update_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.update_state(*args, **kwargs)

    @property
    def compiled(self) -> Any:
        return self._inner


#: Sentinel that clears the whole message list when used with `RemoveMessage`.
REMOVE_ALL_MESSAGES = "__remove_all__"

#: LangGraph's accepted message inputs.
Messages = list[AnyMessage] | AnyMessage


def push_message(
    message: "MessageLikeRepresentation | BaseMessageChunk",
    *,
    state_key: str | None = "messages",
) -> AnyMessage:
    """Write a message to the `messages` stream mode and (optionally) the state channel."""
    return message  # type: ignore[return-value]

_MISSING = re_export(globals(), ["RemoveMessage", "add_messages", "message_graph"])
# `MessagesState` lives in `_schema` now that the extension no longer embeds it.
from ._schema import MessagesState  # noqa: E402  (after the re-export block)
__getattr__ = module_getattr(__name__, globals(), _MISSING)