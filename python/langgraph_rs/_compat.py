"""Shared helpers for the compatibility modules."""

from __future__ import annotations

from . import langgraph_rs as _native

NATIVE = _native


def re_export(namespace: dict, names: list[str]) -> list[str]:
    """Copy `names` from the extension into `namespace`; return the missing ones."""
    missing: list[str] = []
    for name in names:
        try:
            namespace[name] = getattr(_native, name)
        except AttributeError:
            missing.append(name)
    return missing


def unsupported(name: str, hint: str = "") -> type:
    """Build a placeholder that fails loudly the moment it is used.

    Importing stays possible -- so a ``from langgraph_rs...import X`` line does not break a
    module import chain -- but instantiating or subclassing raises ``NotImplementedError``
    with the reason instead of silently misbehaving.
    """

    def _raise(self, *args, **kwargs):
        raise NotImplementedError(f"{name} is not implemented by the langgraph_rs extension. {hint}")

    def _init_subclass(cls, **kwargs):
        raise NotImplementedError(
            f"{name} cannot be subclassed: the langgraph_rs extension does not implement it. {hint}"
        )

    return type(
        name,
        (),
        {
            "__doc__": f"Unsupported placeholder for LangGraph's {name}. {hint}".strip(),
            "__init__": _raise,
            "__init_subclass__": classmethod(_init_subclass),
        },
    )


def fill_unsupported(namespace: dict, names: list[str], hint: str = "") -> list[str]:
    """Replace every absent name with an :func:`unsupported` placeholder."""
    missing = [name for name in names if name not in namespace]
    for name in missing:
        namespace[name] = unsupported(name, hint)
    return missing


def module_getattr(module: str, namespace: dict, missing: list[str]):
    """Build a ``__getattr__`` that explains which symbols the shim cannot provide."""

    def __getattr__(name: str):
        try:
            return getattr(_native, name)
        except AttributeError:
            hint = ""
            if name in missing:
                hint = (
                    f" LangGraph provides {name!r} here, but the Rust extension does not "
                    "implement it; see docs/langgraph-compat.md for the supported surface."
                )
            raise AttributeError(
                f"module {module!r} has no attribute {name!r}.{hint}"
            ) from None

    return __getattr__

def jsonschema_of(schema: Any) -> Any:
    """Normalise a declared context schema into a JSON schema.

    LangGraph accepts a pydantic model, a ``TypedDict`` or a plain dict here; the extension stores
    whatever the caller passed, so this turns each form into a JSON schema dict.
    """
    if schema is None:
        return None
    if isinstance(schema, dict):
        if "properties" in schema or "type" in schema:
            return schema
        return {
            "title": "Context",
            "type": "object",
            "properties": {name: {} for name in schema},
        }
    for attribute in ("model_json_schema",):
        method = getattr(schema, attribute, None)
        if callable(method):
            return method()
    annotations = getattr(schema, "__annotations__", None)
    if isinstance(annotations, dict):
        return {
            "title": getattr(schema, "__name__", "Context"),
            "type": "object",
            "properties": {name: {} for name in annotations},
        }
    return schema


# --- LangChain object codec ---------------------------------------------------------------
#
# LangGraph serialises objects that JSON cannot represent with `JsonPlusSerializer`, which emits
# `{"lc": 1, "type": "constructor", "id": ["module", "attr", ...], "kwargs": {...}}` and revives
# them by importing `id`. The extension uses the same shape, so checkpoints are interchangeable.

def _import_path(path: list[str]) -> Any:
    module = __import__(".".join(path[:-1]), fromlist=[path[-1]])
    return getattr(module, path[-1])


def encode_value(value: Any) -> Any:
    """Encode a LangChain object as LangGraph's JSON-plus constructor dict."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        kind = type(value)
        return {
            "lc": 1,
            "type": "constructor",
            "id": [kind.__module__, kind.__name__],
            "kwargs": model_dump(),
        }
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        kind = type(value)
        payload = to_json()
        return {
            "lc": 1,
            "type": "constructor",
            "id": [kind.__module__, kind.__name__],
            "kwargs": payload if isinstance(payload, dict) else {"value": payload},
        }
    return None


def decode_value(value: Any) -> Any:
    """Revive a LangGraph JSON-plus constructor dict, leaving anything else untouched."""
    if not isinstance(value, dict) or value.get("lc") != 1:
        return value
    if value.get("type") != "constructor":
        return value
    path = value.get("id")
    if not isinstance(path, list) or not path:
        return value
    try:
        factory = _import_path(path)
    except Exception:
        return value
    kwargs = value.get("kwargs") or {}
    try:
        return factory(**kwargs)
    except Exception:
        return value


def message_chunk(payload: Any) -> Any:
    """Rebuild a LangGraph message chunk from the engine's JSON payload.

    ``stream_mode="messages"`` yields ``(message_chunk, metadata)`` in LangGraph; the engine carries
    the payload as JSON, so this turns it back into an ``AIMessageChunk`` (falling back to the dict
    when langchain_core is unavailable).
    """
    if not isinstance(payload, dict):
        return payload
    try:
        from langchain_core.messages import (
            AIMessage,
            AIMessageChunk,
            HumanMessage,
            HumanMessageChunk,
            SystemMessage,
            SystemMessageChunk,
        )
    except Exception:  # pragma: no cover - langchain_core is optional at runtime
        return payload
    role = str(payload.get("type") or "ai").lower()
    # LangGraph yields the plain message class when the model was not streamed token by token.
    streamed = payload.get("chunk", True)
    if streamed:
        factory = {
            "human": HumanMessageChunk,
            "system": SystemMessageChunk,
            "ai": AIMessageChunk,
            "assistant": AIMessageChunk,
        }.get(role, AIMessageChunk)
    else:
        factory = {
            "human": HumanMessage,
            "system": SystemMessage,
            "ai": AIMessage,
            "assistant": AIMessage,
        }.get(role, AIMessage)
    kwargs: dict[str, Any] = {"content": payload.get("content") or ""}
    if payload.get("id"):
        kwargs["id"] = payload["id"]
    if payload.get("tool_call_chunks"):
        kwargs["tool_call_chunks"] = payload["tool_call_chunks"]
    if payload.get("additional_kwargs"):
        kwargs["additional_kwargs"] = payload["additional_kwargs"]
    if payload.get("response_metadata"):
        kwargs["response_metadata"] = payload["response_metadata"]
    try:
        return factory(**kwargs)
    except Exception:  # pragma: no cover - malformed payload
        return payload
