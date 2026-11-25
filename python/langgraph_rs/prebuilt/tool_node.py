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
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

from .._compat import module_getattr, re_export

__all__ = [
    "INVALID_TOOL_NAME_ERROR_TEMPLATE",
    "TOOL_CALL_ERROR_TEMPLATE",
    "TOOL_EXECUTION_ERROR_TEMPLATE",
    "TOOL_INVOCATION_ERROR_TEMPLATE",
    "TOOL_MESSAGE_BLOCK_TYPES",
    "ToolCallRequest",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolInvocationError",
    "create_tool",
    "get_all_basemodel_annotations",
    "msg_content_output",
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


TOOL_MESSAGE_BLOCK_TYPES = (
    "text",
    "image_url",
    "image",
    "json",
    "search_result",
    "custom_tool_call_output",
    "document",
    "file",
)

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
TOOL_EXECUTION_ERROR_TEMPLATE = (
    "Error executing tool '{tool_name}' with kwargs {tool_kwargs} with error:\n {error}\n "
    "Please fix the error and try again."
)
TOOL_INVOCATION_ERROR_TEMPLATE = (
    "Error invoking tool '{tool_name}' with kwargs {tool_kwargs} with error:\n {error}\n "
    "Please fix the error and try again."
)

try:  # LangChain's tool exception, when available
    from langchain_core.tools import ToolException as _ToolException
except Exception:  # pragma: no cover - langchain_core is optional

    class _ToolException(Exception):  # type: ignore[no-redef]
        """Placeholder for LangChain's ToolException."""


class ToolInvocationError(_ToolException):
    """Raised when a tool cannot be invoked with the given arguments."""


@dataclass
class ToolCallRequest:
    """A tool call plus the context the wrapper receives."""

    tool_call: Any
    tool: Any = None
    state: Any = None
    runtime: Any = None


class ToolCallWithContext:
    """A tool call paired with the graph state (LangGraph's context form)."""

    __slots__ = ("tool_call", "state", "__type")

    def __init__(self, tool_call: Any = None, state: Any = None) -> None:
        self.tool_call = tool_call
        self.state = state
        self.__type = "tool_call_with_context"

    def __repr__(self) -> str:
        return f"ToolCallWithContext(tool_call={self.tool_call!r})"


class ToolCallWrapper:
    """Callable that can intercept a tool call before execution."""

    def __call__(self, request: ToolCallRequest, execute: Any) -> Any:  # pragma: no cover
        return execute(request)


def msg_content_output(output: Any) -> Any:
    """Convert tool output to `ToolMessage` content format."""
    if isinstance(output, str) or (
        isinstance(output, list)
        and all(
            isinstance(item, dict) and item.get("type") in TOOL_MESSAGE_BLOCK_TYPES
            for item in output
        )
    ):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:
        return str(output)


def get_all_basemodel_annotations(cls: Any, *, default_to_bound: bool = True) -> dict:
    """Collect pydantic field annotations across the MRO (LangGraph helper)."""
    annotations: dict[str, Any] = {}
    for base in reversed(getattr(cls, "__mro__", [cls])):
        fields = getattr(base, "model_fields", None)
        if not fields:
            annotations.update(getattr(base, "__annotations__", {}) or {})
            continue
        for name, field in fields.items():
            annotation = getattr(field, "annotation", None)
            annotations[name] = annotation
    return annotations


def create_tool(
    name_or_callable: Any = None,
    runnable: Any = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: Any = None,
    infer_schema: bool = True,
    response_format: str = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a LangChain tool from a function or Runnable (decorator-friendly)."""
    from langchain_core.tools import StructuredTool, tool

    def build(target: Any) -> Any:
        if runnable is not None or (target is not None and hasattr(target, "invoke")):
            name = name_or_callable if isinstance(name_or_callable, str) else None
            if not name:
                raise TypeError("create_tool with a Runnable requires a string name")
            target = target if runnable is None else runnable

            def call(**tool_kwargs: Any) -> Any:
                return target.invoke(tool_kwargs)

            return StructuredTool.from_function(
                func=call,
                name=name,
                description=description or getattr(target, "name", None) or name,
                return_direct=return_direct,
                args_schema=args_schema,
            )
        if isinstance(target, str):
            def decorate(function: Any) -> Any:
                return create_tool(
                    function,
                    *args,
                    description=description,
                    return_direct=return_direct,
                    args_schema=args_schema,
                    infer_schema=infer_schema,
                    response_format=response_format,
                    parse_docstring=parse_docstring,
                    error_on_invalid_docstring=error_on_invalid_docstring,
                    extras=extras,
                    **kwargs,
                )

            return decorate
        inferred = args_schema
        if inferred is None and infer_schema:
            try:
                from langchain_core.tools import create_schema_from_function

                inferred = create_schema_from_function(
                    getattr(target, "__name__", "tool"), target
                )
            except Exception:  # pragma: no cover - schema inference is best effort
                inferred = None
        return tool(
            target,
            *args,
            description=description,
            return_direct=return_direct,
            args_schema=inferred if inferred is not None else "auto",
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
            **({"extras": extras} if extras else {}),
            **kwargs,
        )

    if name_or_callable is None:
        return build
    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        return build(name_or_callable)
    return build(name_or_callable)
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