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
