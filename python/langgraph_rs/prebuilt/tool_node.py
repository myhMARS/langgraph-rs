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
    "ContextT",
    "StateT",
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
    from langchain_core.tools.base import (
        _DirectlyInjectedToolArg as _DirectInjectedToolArg,
    )
except Exception:  # pragma: no cover - fall back to a plain marker base
    class _InjectedToolArg:  # type: ignore[no-redef]
        """Base marker for tool arguments the model must not provide."""

    _DirectInjectedToolArg = _InjectedToolArg  # type: ignore[assignment,misc]

# LangGraph parameterises ``ToolRuntime`` as ``Generic[ContextT, StateT]``.
try:  # ``default=`` needs Python 3.13 / typing_extensions.TypeVar
    from typing_extensions import TypeVar as _ExtTypeVar

    StateT = _ExtTypeVar("StateT", default=dict)
    ContextT = _ExtTypeVar("ContextT", default=None)
except Exception:  # pragma: no cover - plain TypeVar is enough for annotations
    StateT = TypeVar("StateT")
    ContextT = TypeVar("ContextT")
_StateT = StateT
_ContextT = ContextT


class InjectedState(_InjectedToolArg):
    """Annotation for injecting the graph state into a tool argument."""


class InjectedStore(_InjectedToolArg):
    """Annotation for injecting the long-term store into a tool argument."""


# The runtime handle is a native class: the engine builds it when a tool declares a
# `ToolRuntime` parameter. The Python subclass adds langchain's injected-argument marker so
# `@tool` leaves `runtime` out of the model-facing schema, exactly as LangGraph's does.
from ..langgraph_rs import ToolRuntime as _NativeToolRuntime  # noqa: E402


class ToolRuntime(_NativeToolRuntime, _DirectInjectedToolArg):
    """Runtime context LangGraph injects into tools (`runtime: ToolRuntime`).

    The engine builds the handle in Rust (it owns the run's state, store and config), so the
    subclass accepts the native instance here instead of constructing a second copy.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        try:
            from pydantic_core import core_schema

            return core_schema.is_instance_schema(_NativeToolRuntime)
        except Exception:  # pragma: no cover - pydantic_core ships with pydantic
            return handler(source)


def _validation_error_text(error: Exception) -> str:
    return str(error)


def _signature_model(schema: Any, signature: inspect.Signature) -> Any:
    """Pydantic model mirroring a callable's parameters, as LangGraph builds internally."""
    try:
        from pydantic import create_model
    except Exception:  # pragma: no cover - pydantic is a LangGraph dependency
        return None
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (annotation, default)
    try:
        return create_model("Validation", **fields)
    except Exception:  # pragma: no cover - unusual annotations
        return None


def _validate_call(schema: Any, arguments: dict[str, Any]) -> tuple[bool, str]:
    """Check `arguments` against a tool / pydantic model / callable schema."""
    try:
        if isinstance(schema, type) and hasattr(schema, "model_validate"):
            schema.model_validate(arguments)
            return True, ""
        if callable(schema) and not isinstance(schema, type):
            # LangGraph builds a pydantic model from the signature (named `Validation`), so the
            # annotations are enforced, not just the parameter names.
            signature = inspect.signature(schema)
            signature.bind(**arguments)
            model = _signature_model(schema, signature)
            if model is not None:
                model.model_validate(arguments)
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
                # LangGraph looks the schema up by name and lets the `KeyError` fail the node.
                raise KeyError(name)
            else:
                valid, error = _validate_call(schema, arguments)
                if valid:
                    content = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
                elif self.format_error is not None:
                    content = self.format_error(ValueError(error), call, schema)
                else:
                    content = error
            results.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": call_id,
                    "name": name,
                    # A validated call is reported as a successful tool message.
                    "status": "success",
                }
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
from ._tool_call_transformer import ToolCallTransformer  # noqa: E402

from ..langgraph_rs import ToolNode as _NativeToolNode  # noqa: E402  (native implementation)


def _tool_message(message: Any) -> Any:
    """Turn the native node's message dict into the ``ToolMessage`` LangGraph hands wrappers."""
    try:
        from langchain_core.messages import ToolMessage
    except Exception:  # pragma: no cover - langchain_core is optional
        return message
    if not isinstance(message, dict):
        return message
    kwargs = {
        "content": message.get("content", ""),
        "tool_call_id": message.get("tool_call_id", ""),
        "name": message.get("name"),
    }
    if message.get("status"):
        kwargs["status"] = message["status"]
    return ToolMessage(**kwargs)


def _message_dict(message: Any) -> dict[str, Any]:
    """Normalise a wrapper's return value (``ToolMessage``/dict/``Command``) into a dict."""
    if isinstance(message, dict):
        return message
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id is None:
        return message
    data = {
        "role": "tool",
        "content": getattr(message, "content", ""),
        "tool_call_id": tool_call_id,
        "name": getattr(message, "name", None),
    }
    status = getattr(message, "status", None)
    if status:
        data["status"] = status
    return data


class ToolNode:
    """LangGraph's ``ToolNode`` including its constructor surface.

    The native node executes the calls; ``wrap_tool_call`` / ``awrap_tool_call`` run in Python
    around it, exactly as LangGraph's middleware does.
    """

    def __init__(
        self,
        tools: Sequence[Any],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: Any = None,
        messages_key: str = "messages",
        wrap_tool_call: Callable[..., Any] | None = None,
        awrap_tool_call: Callable[..., Any] | None = None,
    ) -> None:
        self.tools = list(tools)
        self.tools_by_name: dict[str, Any] = {}
        for tool in self.tools:
            key = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if key:
                self.tools_by_name[key] = tool
        self.name = name
        self.tags = list(tags or [])
        self.messages_key = messages_key
        self._wrap = wrap_tool_call
        self._awrap = awrap_tool_call
        kwargs: dict[str, Any] = {"messages_key": messages_key}
        if handle_tool_errors is not None:
            kwargs["handle_tool_errors"] = handle_tool_errors
        self._inner = _NativeToolNode(self.tools, **kwargs)

    def get_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"ToolNode({list(self.tools_by_name)})"

    def _calls(self, state: Any) -> list[Any]:
        messages = state.get(self.messages_key, []) if isinstance(state, dict) else []
        if not messages:
            return []
        last = messages[-1]
        calls = (
            last.get("tool_calls") if isinstance(last, dict) else getattr(last, "tool_calls", None)
        )
        return list(calls or [])

    def _execute_one(self, tool_call: Any) -> Any:
        # LangGraph's handler receives the request object, not the bare tool call.
        tool_call = getattr(tool_call, "tool_call", tool_call)
        output = self._inner(
            {self.messages_key: [{"role": "ai", "content": "", "tool_calls": [tool_call]}]}
        )
        messages = output.get(self.messages_key) or [{}]
        return _tool_message(messages[0])

    def _request(self, tool_call: dict[str, Any], state: Any) -> Any:
        return ToolCallRequest(
            tool_call=tool_call,
            tool=self.tools_by_name.get(tool_call.get("name")),
            state=state,
            runtime=None,
        )

    def __call__(self, state: Any) -> Any:
        if self._wrap is None:
            return self._inner(state)
        results = [
            _message_dict(self._wrap(self._request(call, state), self._execute_one))
            for call in self._calls(state)
        ]
        return {self.messages_key: results}

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self(input)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        if self._awrap is None:
            return self(input)
        results = []
        for call in self._calls(input):
            request = self._request(call, input)
            produced = await self._awrap(request, self._aexecute)
            results.append(_message_dict(produced))
        return {self.messages_key: results}

    async def _aexecute(self, request: Any) -> Any:
        return self._execute_one(request)


_MISSING = re_export(globals(), ["tools_condition"])
__getattr__ = module_getattr(__name__, globals(), _MISSING)