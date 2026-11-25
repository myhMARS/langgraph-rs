"""``langgraph.prebuilt.tool_node`` equivalents.

Besides ``ToolNode`` this exposes LangGraph's argument-injection markers. Parameters annotated
with them are filled by the runtime instead of by the model, and LangChain excludes them from the
tool schema the model sees (they subclass ``InjectedToolArg`` for exactly that reason).
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from .._compat import module_getattr, re_export

__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolNode",
    "ToolRuntime",
    "tools_condition",
]

try:  # pragma: no cover - langchain_core is optional
    from langchain_core.tools.base import InjectedToolArg as _InjectedToolArg
except Exception:  # pragma: no cover - fall back to a plain marker base
    class _InjectedToolArg:  # type: ignore[no-redef]
        """Base marker for tool arguments the model must not provide."""


class InjectedState(_InjectedToolArg):
    """Annotation for injecting the graph state into a tool argument."""


class InjectedStore(_InjectedToolArg):
    """Annotation for injecting the long-term store into a tool argument."""


_StateT = TypeVar("_StateT")


class ToolRuntime(_InjectedToolArg, Generic[_StateT]):
    """Runtime handle LangGraph injects into tools (`state`, `store`, `config`, ...).

    The extension currently injects ``InjectedState``/``InjectedStore``; annotating a parameter
    with ``ToolRuntime`` would therefore leave it unfilled, so it raises instead of passing None.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "ToolRuntime injection is not implemented by the langgraph_rs extension; "
            "annotate parameters with InjectedState or InjectedStore instead."
        )


_MISSING = re_export(globals(), ["ToolNode", "tools_condition"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)
