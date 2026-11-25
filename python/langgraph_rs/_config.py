"""Config helpers (LangGraph ``langgraph._internal._config`` equivalents).

LangGraph keeps a full config-merging implementation here; the legacy helpers only need the same
shape: defaults filled in, ambient config inherited, `configurable` merged.
"""

from __future__ import annotations

from typing import Any

from .config import CONF, var_child_runnable_config

__all__ = ["ensure_config", "patch_configurable"]

DEFAULT_RECURSION_LIMIT = 25


def ensure_config(*configs: dict[str, Any] | None) -> dict[str, Any]:
    """Return a config with every key present, merging `configs` and the ambient config."""
    merged: dict[str, Any] = {
        "tags": [],
        "metadata": {},
        "callbacks": None,
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        CONF: {},
    }
    ambient = var_child_runnable_config.get()
    if ambient:
        for key, value in ambient.items():
            if value in (None, [], {}, ""):
                continue
            merged[key] = dict(value) if isinstance(value, dict) else value
        merged["tags"] = list(merged.get("tags") or [])
    for config in configs:
        if not config:
            continue
        for key, value in config.items():
            if key == CONF and isinstance(value, dict):
                merged[CONF].update(value)
            elif key == "tags":
                merged["tags"] = list({*(merged.get("tags") or []), *value})
            elif key == "metadata" and isinstance(value, dict):
                merged["metadata"] = {**(merged.get("metadata") or {}), **value}
            else:
                merged[key] = value
    return merged


def patch_configurable(config: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Return `config` with `values` merged into its `configurable` dict."""
    patched = dict(config or {})
    configurable = dict(patched.get(CONF) or {})
    configurable.update(values or {})
    patched[CONF] = configurable
    return patched
