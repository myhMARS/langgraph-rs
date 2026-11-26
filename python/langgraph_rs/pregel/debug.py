"""``langgraph.pregel.debug`` equivalents.

LangGraph formats ``stream_mode="debug"`` payloads with these helpers. The extension builds the
same payload shapes inside its engine, so the helpers here are provided for import compatibility and
for code that formats debug output itself; they operate on plain mappings (and on LangGraph- or
extension-shaped task objects), never on engine internals.

Two of them are best-effort by nature: ``map_debug_checkpoint`` and ``tasks_w_writes`` read task and
pending-write data the way LangGraph does, but the extension has no ``PregelExecutableTask`` —
anything that provides the same attributes works.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

__all__ = [
    "COLOR_MAPPING",
    "EXCLUDED_METADATA_KEYS",
    "get_bolded_text",
    "get_colored_text",
    "is_multiple_channel_write",
    "map_debug_checkpoint",
    "map_debug_task_results",
    "map_debug_tasks",
    "map_task_result_writes",
    "rm_pregel_keys",
    "tasks_w_writes",
]

# LangGraph's write channels (`langgraph._internal._constants`).
RETURN = "__return__"
ERROR = "__error__"
INTERRUPT = "__interrupt__"
TAG_HIDDEN = "langsmith:hidden"

# Keys LangGraph drops from debug payload metadata because the task fields already carry them.
EXCLUDED_METADATA_KEYS = {
    "thread_id",
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_step",
    "langgraph_node",
    "langgraph_triggers",
    "langgraph_path",
    "langgraph_checkpoint_ns",
}

COLOR_MAPPING = {
    "black": "0;30",
    "red": "0;31",
    "green": "0;32",
    "yellow": "0;33",
    "blue": "0;34",
    "magenta": "0;35",
    "cyan": "0;36",
    "white": "0;37",
    "gray": "1;30",
}


def _get(task: Any, name: str, default: Any = None) -> Any:
    """Read `name` from an object or a mapping."""
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _config_of(task: Any) -> Mapping[str, Any] | None:
    config = _get(task, "config")
    return config if isinstance(config, Mapping) else None


def _filter_to_user_tags(tags: Any) -> list[str] | None:
    """Drop LangGraph's internal tags, keeping user-supplied ones."""
    if not tags:
        return None
    kept = [tag for tag in tags if not tag.startswith("langgraph:") and tag != TAG_HIDDEN]
    return kept or None


def map_debug_tasks(tasks: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Produce ``task`` events for ``stream_mode="debug"``."""
    for task in tasks:
        config = _config_of(task)
        if config is not None and TAG_HIDDEN in (config.get("tags") or []):
            continue
        payload: dict[str, Any] = {
            "id": _get(task, "id"),
            "name": _get(task, "name"),
            "input": _get(task, "input"),
            "triggers": _get(task, "triggers", []),
        }
        if config is not None:
            metadata = {
                key: value
                for key, value in (config.get("metadata") or {}).items()
                if key not in EXCLUDED_METADATA_KEYS
            }
            filtered_tags = _filter_to_user_tags(config.get("tags"))
            if filtered_tags is not None:
                metadata["tags"] = filtered_tags
            if metadata:
                payload["metadata"] = metadata
        yield payload


def is_multiple_channel_write(value: Any) -> bool:
    """Whether the payload already wraps multiple writes from the same channel."""
    return isinstance(value, dict) and "$writes" in value and isinstance(value["$writes"], list)


def map_task_result_writes(writes: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Fold task writes into a result dict, aggregating repeated channel writes.

    A channel written once maps to the value; a channel written several times maps to
    ``{"$writes": [first, second, ...]}`` (LangGraph's shape).
    """
    result: dict[str, Any] = {}
    for channel, value in writes:
        existing = result.get(channel)
        if existing is not None:
            channel_writes = (
                existing["$writes"] if is_multiple_channel_write(existing) else [existing]
            )
            channel_writes.append(value)
            result[channel] = {"$writes": channel_writes}
        else:
            result[channel] = value
    return result


def map_debug_task_results(
    task_tup: tuple[Any, Sequence[tuple[str, Any]]],
    stream_keys: str | Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Produce ``task_result`` events for ``stream_mode="debug"``."""
    stream_channels = [stream_keys] if isinstance(stream_keys, str) else list(stream_keys)
    task, writes = task_tup
    yield {
        "id": _get(task, "id"),
        "name": _get(task, "name"),
        "error": next((value for channel, value in writes if channel == ERROR), None),
        "result": map_task_result_writes(
            [(channel, value) for channel, value in writes if channel in stream_channels or channel == RETURN]
        ),
        "interrupts": [
            asdict(value) if is_dataclass(value) else value
            for channel, value in writes
            if channel == INTERRUPT
            for value in (value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value])
        ],
    }


def rm_pregel_keys(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Remove the engine's private ``__pregel_*`` keys from a config."""
    if config is None:
        return config
    return {
        "configurable": {
            key: value
            for key, value in (config.get("configurable") or {}).items()
            if not key.startswith("__pregel_")
        }
    }


def map_debug_checkpoint(
    config: Mapping[str, Any],
    channels: Mapping[str, Any],
    stream_channels: str | Sequence[str],
    metadata: Mapping[str, Any],
    tasks: Iterable[Any],
    pending_writes: Sequence[tuple[str, str, Any]] | None,
    parent_config: Mapping[str, Any] | None,
    output_keys: str | Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Produce ``checkpoint`` events for ``stream_mode="debug"``.

    ``channels`` may be channel objects (anything exposing ``checkpoint()``) or plain values; the
    values shown follow ``stream_channels``.
    """
    names = [stream_channels] if isinstance(stream_channels, str) else list(stream_channels)
    values: dict[str, Any] = {}
    for name in names:
        channel = channels.get(name) if isinstance(channels, Mapping) else None
        if channel is None:
            continue
        checkpointing = getattr(channel, "checkpoint", None)
        values[name] = checkpointing() if callable(checkpointing) else channel
    tasks_list = list(tasks)
    yield {
        "config": rm_pregel_keys(config),
        "parent_config": rm_pregel_keys(parent_config),
        "values": values,
        "metadata": dict(metadata),
        "next": [str(_get(task, "name")) for task in tasks_list],
        "tasks": [
            {
                "id": _get(task, "id"),
                "name": _get(task, "name"),
                "interrupts": tuple(
                    asdict(item) if is_dataclass(item) else item
                    for item in (_get(task, "interrupts", ()) or ())
                ),
                "state": _get(task, "state"),
                **({"error": _get(task, "error")} if _get(task, "error") else {}),
                **(
                    {"result": _get(task, "result")}
                    if _get(task, "result") and not _get(task, "error")
                    else {}
                ),
            }
            for task in tasks_list
        ],
    }


def tasks_w_writes(
    tasks: Iterable[Any],
    pending_writes: Sequence[tuple[str, str, Any]] | None,
    states: Mapping[str, Any] | None,
    output_keys: str | Sequence[str],
    *,
    live: bool = False,
) -> tuple[Any, ...]:
    """Attach write results (and subgraph states) to tasks for a ``StateSnapshot``.

    ``pending_writes`` are ``(task_id, channel, value)`` triples, the shape both LangGraph and this
    extension use for pending writes.
    """
    pending_by_task: dict[str, list[tuple[str, Any]]] = {}
    for task_id, channel, value in pending_writes or ():
        pending_by_task.setdefault(task_id, []).append((channel, value))
    names = [output_keys] if isinstance(output_keys, str) else list(output_keys)

    from ..types import PregelTask

    out: list[Any] = []
    for task in tasks:
        task_id = str(_get(task, "id"))
        writes = pending_by_task.get(task_id, [])
        returned = next((value for channel, value in writes if channel == RETURN), None)
        if returned is not None:
            result: Any = returned
        else:
            mapped = map_task_result_writes(
                [(channel, value) for channel, value in writes if channel in names]
            )
            if isinstance(output_keys, str):
                result = mapped.get(output_keys)
            else:
                result = mapped or {}
        out.append(
            PregelTask(
                task_id,
                str(_get(task, "name", "")),
                tuple(_get(task, "path", ()) or ()),
                _get(task, "error"),
                tuple(_get(task, "interrupts", ()) or ()) if not live else tuple(),
                states.get(task_id) if states else _get(task, "state"),
                result,
            )
        )
    return tuple(out)


def get_colored_text(text: str, color: str) -> str:
    """Wrap `text` in the ANSI color LangGraph's debug printer uses."""
    return f"\033[1;3{COLOR_MAPPING[color]}m{text}\033[0m"


def get_bolded_text(text: str) -> str:
    """Wrap `text` in ANSI bold."""
    return f"\033[1m{text}\033[0m"