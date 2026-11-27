"""``langgraph_rs``: LangGraph-compatible imports backed by the Rust extension.

The native symbols are re-exported here, so ``langgraph_rs.StateGraph`` keeps working,
and the LangGraph module layout is available underneath::

    from langgraph_rs.graph import StateGraph, START, END
    from langgraph_rs.types import Command, Send, interrupt
    from langgraph_rs.checkpoint.memory import InMemorySaver

Switching an existing LangGraph program therefore only requires rewriting its
``from langgraph.<module> import ...`` lines to ``from langgraph_rs.<module> import ...``.
"""

from __future__ import annotations

from . import langgraph_rs as _native
from .langgraph_rs import *  # noqa: F401,F403
from . import channels, pregel
from .graph._schema import MessagesState, _schema_from_class
from .graph.state import AsyncStateGraph, CompiledGraph, CompiledStateGraph, StateGraph
from .stream._stream import AsyncStream
from .graph.message import MessageGraph
# The compatibility classes carry LangGraph's Python-side surface (async twins, `batch`, ...), so
# they replace the plain native classes the star import above brought in.
from .checkpoint.memory import (  # noqa: E402
    AsyncInMemorySaver,
    InMemorySaver,
    MemorySaver,
)
from .store.memory import InMemoryStore, Store  # noqa: E402
from .cache.memory import InMemoryCache  # noqa: E402
from .prebuilt.tool_node import (
    InjectedState,
    InjectedStore,
    ToolCallTransformer,
    ToolRuntime,
    ValidationNode,
)

__all__ = [
    "AsyncStream",
    *[name for name in dir(_native) if not name.startswith("_")],
    "AsyncStateGraph",
    "MessagesState",
    "CompiledGraph",
    "CompiledStateGraph",
    "StateGraph",
    "InjectedState",
    "InjectedStore",
    "MessageGraph",
    "channels",
    "ToolRuntime",
    "ValidationNode",
]


from .prebuilt.chat_agent_executor import create_react_agent
from .func import entrypoint as entrypoint  # noqa: E402
from .func import task as task  # noqa: E402

def __getattr__(name: str):
    try:
        return getattr(_native, name)
    except AttributeError:
        raise AttributeError(
            f"module 'langgraph_rs' has no attribute {name!r}"
        ) from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_native)))


def _register_native_hooks() -> None:
    """Hand helpers to the extension so it never has to import this package."""
    try:
        from . import _compat, langgraph_rs as _native
        from .types import PregelTask
    except Exception:  # pragma: no cover - the extension is always present at runtime
        return
    register = getattr(_native, "_register_shim_hook", None)
    if register is None:  # pragma: no cover - older builds
        return
    from ._callbacks import LangChainCallbackHandler, StreamingLangChainCallbackHandler

    def callback_handler(model=None, tags=None, streaming=False):
        """Build the handler class LangGraph attaches to model calls."""
        cls = StreamingLangChainCallbackHandler if streaming else LangChainCallbackHandler
        return cls(model, tags)

    for name, value in (
        ("message_chunk", _compat.message_chunk),
        ("pregel_task", PregelTask),
        ("callback_handler", callback_handler),
    ):
        try:
            register(name, value)
        except Exception:  # pragma: no cover - defensive
            pass


_register_native_hooks()
