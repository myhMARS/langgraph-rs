"""``langgraph.graph.state`` equivalents."""

from __future__ import annotations

from .._compat import module_getattr, re_export
from ..channels import BaseChannel, BinaryOperatorAggregate
from ..types import All, CachePolicy

__all__ = [
    "All",
    "BaseChannel",
    "BinaryOperatorAggregate",
    "BranchSpec",
    "CachePolicy",
    "ChannelRead",
    "CompiledGraph",
    "CompiledStateGraph",
    "END",
    "START",
    "StateGraph",
]


class ChannelRead:
    """Marker telling the engine which channels a branch reads (LangGraph internal)."""

    __slots__ = ("channel",)

    def __init__(self, channel: str = "") -> None:
        self.channel = channel

    def __repr__(self) -> str:
        return f"ChannelRead({self.channel!r})"


class BranchSpec:
    """A conditional edge specification (LangGraph internal)."""

    __slots__ = ("path", "ends")

    def __init__(self, path: Any = None, ends: Any = None) -> None:
        self.path = path
        self.ends = ends

    def __repr__(self) -> str:
        return f"BranchSpec(path={self.path!r}, ends={self.ends!r})"


_MISSING = re_export(globals(), ["CompiledGraph", "CompiledStateGraph", "END", "START", "StateGraph"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)
