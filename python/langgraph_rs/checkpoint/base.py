"""``langgraph.checkpoint.base`` equivalents.

The checkpoint *protocol* classes live in the extension; this module provides the small helper
functions and constants LangGraph exports next to them, plus the serializer re-exports. Pass a
plain object implementing ``put``/``put_writes``/``get_tuple``/``list`` to
``compile(checkpointer=...)`` to use a custom saver -- subclassing is not required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .._compat import fill_unsupported, module_getattr
from ..errors import EmptyChannelError
from typing import Union

ChannelVersions = dict[str, Union[str, int, float]]
DeltaChannelHistory = list[tuple[str, Any]]
ChannelProtocol = Any
from ..checkpoint.serde import EncryptedSerializer, JsonPlusSerializer, SerializerProtocol

ERROR = "__error__"
INTERRUPT = "__interrupt__"
RESUME = "__resume__"
SCHEDULED = "__scheduled__"
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}
LATEST_VERSION = 2
EXCLUDED_METADATA_KEYS = {
    "langgraph_node",
    "langgraph_step",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_checkpoint_ns",
    "langgraph_triggers",
    "langgraph_path",
    "checkpoint_id",
    "thread_id",
}

__all__ = [
    "BaseCheckpointSaver",
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointTuple",
    "ChannelProtocol",
    "ChannelVersions",
    "DeltaChannelHistory",
    "EmptyChannelError",
    "EncryptedSerializer",
    "ERROR",
    "EXCLUDED_METADATA_KEYS",
    "INTERRUPT",
    "JsonPlusSerializer",
    "LATEST_VERSION",
    "PendingWrite",
    "RESUME",
    "SCHEDULED",
    "SerializerProtocol",
    "WRITES_IDX_MAP",
    "copy_checkpoint",
    "create_checkpoint",
    "empty_checkpoint",
    "get_checkpoint_id",
    "get_checkpoint_metadata",
    "get_serializable_checkpoint_metadata",
    "writes_sort_key",
]


def empty_checkpoint() -> dict[str, Any]:
    """A checkpoint with no channel values (LangGraph shape)."""
    return {
        "v": LATEST_VERSION,
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
    }


def get_checkpoint_id(config: Mapping[str, Any]) -> str | None:
    """Checkpoint id stored in a run config."""
    return config.get("configurable", {}).get("checkpoint_id")


def get_checkpoint_metadata(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Strip LangGraph-internal keys from node metadata before persisting it."""
    configurable = config.get("configurable", {})
    kept = dict(metadata)
    for key in EXCLUDED_METADATA_KEYS:
        kept.pop(key, None)
    kept.setdefault("thread_id", configurable.get("thread_id"))
    kept.setdefault("checkpoint_ns", configurable.get("checkpoint_ns", ""))
    kept.setdefault("checkpoint_id", configurable.get("checkpoint_id"))
    return {key: value for key, value in kept.items() if value is not None}


def copy_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow copy of a checkpoint mapping (LangGraph helper)."""
    return dict(checkpoint)


def create_checkpoint(
    checkpoint: Mapping[str, Any],
    channels: Any = None,
    versions: Any = None,
) -> dict[str, Any]:
    """Create a new checkpoint from an existing one, bumping `id` and `ts`."""
    created = empty_checkpoint()
    created.update({key: value for key, value in checkpoint.items() if key not in {"id", "ts"}})
    if versions:
        created["channel_versions"] = dict(versions)
    return created


def get_serializable_checkpoint_metadata(
    config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Metadata with LangGraph-internal keys removed."""
    return get_checkpoint_metadata(config, metadata)


def writes_sort_key(task_path: str, task_id: str = "", idx: int = 0) -> tuple[str, str, int]:
    """Sort key for the writes of one super-step."""
    return (task_path, task_id, idx)


_MISSING = fill_unsupported(
    globals(),
    ["BaseCheckpointSaver", "Checkpoint", "CheckpointMetadata", "CheckpointTuple", "PendingWrite"],
    hint="Use a concrete saver, or pass an object implementing the saver protocol to compile().",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)
