"""LangGraph ``langgraph.config`` equivalents."""

from __future__ import annotations

from ._compat import module_getattr, re_export
from contextvars import ContextVar

from . import langgraph_rs as _native
from .runtime import CONFIG_KEY_RUNTIME, StreamWriter
from .store.base import BaseStore

CONF = "configurable"
# LangGraph stores the runnable config in this context variable.
var_child_runnable_config: ContextVar = ContextVar("var_child_runnable_config", default=None)

def _get_store_or_raise():
    """The store of the current run.

    Outside a run LangGraph raises (`Called get_config outside of a runnable context`); inside a run
    without a configured store it returns `None`, so both cases are reproduced here.
    """
    if _native.current_runtime() is None:
        raise RuntimeError("Called get_config outside of a runnable context")
    store = _native.get_store()
    if store is None:
        return None
    from .runtime import _StoreBridge

    return _StoreBridge(store)


__all__ = [
    "BaseStore",
    "CONF",
    "CONFIG_KEY_RUNTIME",
    "StreamWriter",
    "current_context",
    "current_namespace",
    "current_node",
    "current_resume",
    "current_runtime",
    "current_step",
    "current_thread_id",
    "custom",
    "get_config",
    "get_store",
    "get_stream_writer",
    "message",
    "var_child_runnable_config",
    "record_tokens",
    "record_usage",
]

_MISSING = re_export(globals(), __all__)
# `re_export` copies the native `get_store` (which returns `None` outside a run); LangGraph raises,
# so bind the raising wrapper afterwards.
get_store = _get_store_or_raise
__getattr__ = module_getattr(__name__, globals(), _MISSING)


try:  # LangGraph re-exports the config type here; it comes from langchain_core.
    from langchain_core.runnables import RunnableConfig  # noqa: F401
except Exception:  # pragma: no cover - langchain_core is optional
    from .types import RunnableConfig  # type: ignore[assignment]  # noqa: F401
