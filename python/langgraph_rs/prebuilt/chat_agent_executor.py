"""``langgraph.prebuilt.chat_agent_executor`` equivalents.

Besides ``create_react_agent`` this module exposes LangGraph's agent state schemas so code that
passes ``state_schema=AgentState`` (or the structured-response variant) keeps working.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from .._compat import module_getattr, re_export
from ..errors import ErrorCode, create_error_message
from ..graph.message import add_messages
from ..runtime import Runtime
from .tool_node import ToolCallWithContext
from ..managed import IsLastStep, RemainingSteps

try:  # LangChain message schema, when available
    from langchain_core.messages import AnyMessage
except Exception:  # pragma: no cover - langchain_core is optional
    AnyMessage = Any

__all__ = [
    "AgentState",
    "AgentStatePydantic",
    "AgentStateWithStructuredResponse",
    "AgentStateWithStructuredResponsePydantic",
    "ErrorCode",
    "LanguageModelLike",
    "Runtime",
    "ToolCallWithContext",
    "create_error_message",
    "Prompt",
    "StateSchema",
    "StateSchemaType",
    "StructuredResponse",
    "create_react_agent",
    "create_tool_calling_executor",
]


class AgentState(TypedDict):
    """LangGraph's default agent state."""

    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep


class AgentStateWithStructuredResponse(TypedDict):
    """Agent state with the parsed `structured_response` channel."""

    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep
    structured_response: Any


class AgentStatePydantic:
    """Pydantic flavour of :class:`AgentState` (kept for annotations)."""

    def __init__(self, **values: Any) -> None:
        self.messages = values.get("messages", [])
        self.remaining_steps = values.get("remaining_steps")
        self.is_last_step = values.get("is_last_step")


class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
    """Pydantic flavour of :class:`AgentStateWithStructuredResponse`."""

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self.structured_response = values.get("structured_response")


try:  # pragma: no cover - pydantic is normally available through langchain_core
    from pydantic import BaseModel as _BaseModel
except Exception:  # pragma: no cover
    _BaseModel = object

#: LangGraph's structured-response type (`dict` or a pydantic model).
StructuredResponse = dict | _BaseModel

#: Accepted `prompt` / `model` shapes (kept as aliases for annotations).
Prompt = Any
LanguageModelLike = Any
StateSchema = Any
StateSchemaType = Any

_MISSING = re_export(globals(), ["create_react_agent"])
#: Deprecated LangGraph name for `create_react_agent`.
create_tool_calling_executor = globals().get("create_react_agent")
__getattr__ = module_getattr(__name__, globals(), _MISSING)