"""``langgraph.prebuilt.tool_node`` equivalents.

Besides ``ToolNode`` this module exposes LangGraph's argument-injection markers. Parameters
annotated with them are filled by the runtime instead of by the model, and LangChain excludes them
from the tool schema the model sees (they subclass ``InjectedToolArg`` for exactly that reason).

`ValidationNode` validates the pending tool calls without running them: each call is checked
against the tool's argument schema and answered with a ``ToolMessage`` that carries the validated
arguments (or the validation error), which is LangGraph's pattern for structured extraction.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from .._compat import module_getattr, re_export

__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolNode",
    "ToolCallTransformer",
    "ToolRuntime",
    "ValidationNode",
    "tools_condition",
]

try:  # pragma: no cover - langchain_core is optional
    from langchain_core.tools.base import InjectedToolArg as _InjectedToolArg
except Exception:  # pragma: no cover - fall back to a plain marker base
    class _InjectedToolArg:  # type: ignore[no-redef]
        """Base marker for tool arguments the model must not provide."""

_StateT = TypeVar("_StateT")
_ContextT = TypeVar("_ContextT")


class InjectedState(_InjectedToolArg):
    """Annotation for injecting the graph state into a tool argument."""


class InjectedStore(_InjectedToolArg):
    """Annotation for injecting the long-term store into a tool argument."""


@dataclass
class ToolRuntime(Generic[_StateT, _ContextT]):
    """Runtime handle LangGraph injects into tools.

    The extension fills the fields it knows about at call time (`state`, `store`, `context`,
    `stream_writer`, `config`); `tool_call_id`, `execution_info` and `server_info` stay `None`
    because the engine does not track them for tool calls.
    """

    state: Any = None
    context: Any = None
    config: Any = None
    stream_writer: Any = None
    tool_call_id: str | None = None
    store: Any = None
    tools: list[Any] = field(default_factory=list)
    execution_info: Any = None
    server_info: Any = None


def _validation_error_text(error: Exception) -> str:
    return str(error)


def _validate_call(schema: Any, arguments: dict[str, Any]) -> tuple[bool, str]:
    """Check `arguments` against a tool / pydantic model / callable schema."""
    try:
        if isinstance(schema, type) and hasattr(schema, "model_validate"):
            schema.model_validate(arguments)
            return True, ""
        if callable(schema) and not isinstance(schema, type):
            inspect.signature(schema).bind(**arguments)
            return True, ""
        if hasattr(schema, "args_schema") and schema.args_schema is not None:
            return _validate_call(schema.args_schema, arguments)
        if isinstance(schema, type) and hasattr(schema, "__init__"):
            schema(**arguments)
            return True, ""
    except Exception as error:  # ValidationError, TypeError for bad signatures
        return False, _validation_error_text(error)
    return True, ""


class ValidationNode:
    """Validate pending tool calls instead of executing them.

    LangGraph's `ValidationNode` answers every tool call with a ``ToolMessage`` containing the
    validated arguments as compact JSON (or the validation error), which lets a graph enforce a
    schema without running the tools.
    """

    def __init__(
        self,
        schemas: Sequence[Any],
        *,
        format_error: Callable[[Exception, dict, Any], str] | None = None,
        name: str = "validation",
        tags: list[str] | None = None,
    ) -> None:
        self.schemas = list(schemas)
        self.format_error = format_error
        self.name = name
        self.tags = list(tags or [])
        self._by_name: dict[str, Any] = {}
        for schema in self.schemas:
            key = getattr(schema, "name", None) or getattr(schema, "__name__", None)
            if key:
                self._by_name[key] = schema

    def __repr__(self) -> str:
        return f"ValidationNode(schemas={list(self._by_name)})"

    def __call__(self, state: Any) -> Any:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return {"messages": []}
        last = messages[-1]
        calls = last.get("tool_calls") if isinstance(last, dict) else getattr(last, "tool_calls", None)
        if not calls:
            return {"messages": []}
        results = []
        for call in calls:
            name = call.get("name")
            arguments = call.get("args") or {}
            call_id = call.get("id") or ""
            schema = self._by_name.get(name)
            if schema is None:
                content = f"Error: {name} is not a valid tool, try one of {list(self._by_name)}."
            else:
                valid, error = _validate_call(schema, arguments)
                if valid:
                    content = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
                elif self.format_error is not None:
                    content = self.format_error(ValueError(error), call, schema)
                else:
                    content = error
            results.append(
                {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}
            )
        return {"messages": results}


class ToolCallTransformer:
    """LangGraph middleware that projects `tools` channel events into `ToolCallStream` handles.

    It is part of LangGraph's streaming middleware layer, which the extension does not implement;
    constructing it raises instead of silently doing nothing.
    """

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        raise NotImplementedError(
            "ToolCallTransformer is not implemented by the langgraph_rs extension "
            "(it needs LangGraph's streaming middleware layer)."
        )


_MISSING = re_export(globals(), ["ToolNode", "tools_condition"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)