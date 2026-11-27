"""LangGraph ``langgraph.config`` equivalents."""

from __future__ import annotations

from ._compat import module_getattr, re_export
from contextvars import ContextVar

from .runtime import CONFIG_KEY_RUNTIME, StreamWriter
from .store.base import BaseStore

CONF = "configurable"
# LangGraph stores the runnable config in this context variable.
var_child_runnable_config: ContextVar = ContextVar("var_child_runnable_config", default=None)

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
__getattr__ = module_getattr(__name__, globals(), _MISSING)


try:  # LangGraph re-exports the config type here; it comes from langchain_core.
    from langchain_core.runnables import RunnableConfig  # noqa: F401
except Exception:  # pragma: no cover - langchain_core is optional
    from .types import RunnableConfig  # type: ignore[assignment]  # noqa: F401
