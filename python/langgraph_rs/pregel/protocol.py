"""``langgraph.pregel.protocol`` equivalents.

``PregelProtocol`` is the interface LangGraph's compiled graph implements (``invoke`` / ``stream`` /
``get_state`` / ``update_state`` and their async twins); ``StreamProtocol`` is the small callable
holder the engine uses when dispatching stream chunks. The extension's compiled graph is registered
as a virtual subclass, so ``isinstance(graph, PregelProtocol)`` behaves like LangGraph.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar

try:  # pragma: no cover - langchain-core is optional
    from langchain_core.runnables import Runnable, RunnableConfig
except Exception:  # pragma: no cover - minimal fallbacks so imports keep working
    class Runnable:  # type: ignore[no-redef]
        """Fallback base when langchain-core is unavailable."""

    RunnableConfig = dict  # type: ignore[misc,assignment]

StateT = TypeVar("StateT")
ContextT = TypeVar("ContextT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

__all__ = ("PregelProtocol", "StreamProtocol")


class PregelProtocol(Runnable, Generic[StateT, ContextT, InputT, OutputT]):
    """Interface of a compiled graph (mirrors LangGraph's ``PregelProtocol``)."""

    @abstractmethod
    def with_config(self, config: Any = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    def get_graph(self, config: Any = None, *, xray: Any = False) -> Any: ...

    @abstractmethod
    async def aget_graph(self, config: Any = None, *, xray: Any = False) -> Any: ...

    @abstractmethod
    def get_state(self, config: Any, *, subgraphs: bool = False) -> Any: ...

    @abstractmethod
    async def aget_state(self, config: Any, *, subgraphs: bool = False) -> Any: ...

    @abstractmethod
    def get_state_history(self, config: Any, **kwargs: Any) -> Iterator[Any]: ...

    @abstractmethod
    def aget_state_history(self, config: Any, **kwargs: Any) -> AsyncIterator[Any]: ...

    @abstractmethod
    def bulk_update_state(self, config: Any, supersteps: Any) -> Any: ...

    @abstractmethod
    async def abulk_update_state(self, config: Any, supersteps: Any) -> Any: ...

    @abstractmethod
    def update_state(self, config: Any, values: Any, as_node: Any = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def aupdate_state(self, config: Any, values: Any, as_node: Any = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    def stream(self, input: Any = None, config: Any = None, **kwargs: Any) -> Iterator[Any]: ...

    @abstractmethod
    def astream(self, input: Any = None, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]: ...

    @abstractmethod
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


class StreamProtocol:
    """Callable plus the stream modes it handles (mirrors LangGraph's ``StreamProtocol``)."""

    __slots__ = ("modes", "__call__")

    modes: set[Any]
    __call__: Callable[[Any], None]

    def __init__(self, __call__: Callable[[Any], None], modes: set[Any]) -> None:
        self.__call__ = __call__
        self.modes = modes


def _register_compiled_graphs() -> None:
    """Register the extension's compiled graphs as virtual ``PregelProtocol`` subclasses."""
    try:
        import langgraph_rs
    except Exception:  # pragma: no cover - import cycle during startup
        return
    for name in ("CompiledGraph", "CompiledStateGraph", "AsyncCompiledGraph"):
        graph = getattr(langgraph_rs, name, None)
        if graph is not None:
            try:
                PregelProtocol.register(graph)
            except Exception:  # pragma: no cover - already registered
                pass


_register_compiled_graphs()
