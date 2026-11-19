"""``langgraph.prebuilt.chat_agent_executor`` equivalents.

The Rust implementation accepts ``(model, tools, messages_key=None, checkpointer=None)``;
LangGraph's wider signature (``prompt``, ``response_format``, ``state_schema``, ``store``,
hooks, ...) is not supported.
"""

from __future__ import annotations

from .._compat import module_getattr, re_export

__all__ = ["create_react_agent"]

_MISSING = re_export(globals(), __all__)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
