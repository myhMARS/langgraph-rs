"""``langgraph.checkpoint.base`` equivalents.

The checkpoint *protocol* classes live in the extension; this module provides the small helper
functions and constants LangGraph exports next to them, plus the serializer re-exports. Pass a
plain object implementing ``put``/``put_writes``/``get_tuple``/``list`` to
``compile(checkpointer=...)`` to use a custom saver -- subclassing is not required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, NamedTuple, Sequence, TypedDict

from .._compat import fill_unsupported, module_getattr
from ..errors import EmptyChannelError
from typing import Union

ChannelVersions = dict[str, Union[str, int, float]]
class DeltaChannelHistory(TypedDict):
    """Per-channel write history (`get_delta_channel_history` results)."""

    writes: list[tuple[str, str, Any]]
    seed: int
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
    "maybe_add_typed_methods",
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


def maybe_add_typed_methods(cls: Any) -> Any:
    """Decorator LangGraph uses to add typed convenience methods; a no-op here."""
    return cls


def writes_sort_key(task_path: str, task_id: str = "", idx: int = 0) -> tuple[str, str, int]:
    """Sort key for the writes of one super-step."""
    return (task_path, task_id, idx)


class Checkpoint(TypedDict, total=False):
    """LangGraph's checkpoint body."""

    v: int
    id: str
    ts: str
    channel_values: dict[str, Any]
    channel_versions: ChannelVersions
    versions_seen: dict[str, ChannelVersions]
    updated_channels: list[str] | None


class CheckpointMetadata(TypedDict, total=False):
    """LangGraph's checkpoint metadata."""

    source: str
    step: int
    parents: dict[str, str]
    run_id: str


#: `(task_id, channel, value)` triple as stored by a saver.
PendingWrite = tuple[str, str, Any]


class CheckpointTuple(NamedTuple):
    """LangGraph's `CheckpointTuple`: attribute *and* item access, as there."""

    config: dict[str, Any]
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: dict[str, Any] | None
    pending_writes: list[PendingWrite] | None


_MISSING = fill_unsupported(
    globals(),
    ["BaseCheckpointSaver"],
    hint="Use a concrete saver, or pass an object implementing the saver protocol to compile().",
)
__getattr__ = module_getattr(__name__, globals(), _MISSING)


try:  # LangGraph re-exports the config type here.
    from langchain_core.runnables import RunnableConfig  # noqa: F401
except Exception:  # pragma: no cover - langchain_core is optional
    from ..types import RunnableConfig  # type: ignore[assignment]  # noqa: F401


class BaseCheckpointSaver:
    """LangGraph's checkpointer protocol.

    Subclasses implement `get_tuple` / `put` / `put_writes` / `list`; the rest of the surface
    (`get`, the async twins, `get_next_version`, `with_allowlist`) is provided here so a saver
    written for LangGraph works unchanged.
    """

    serde = JsonPlusSerializer()
    config_specs: list[Any] = []

    # -- sync protocol -----------------------------------------------------------------------
    def get(self, config: Any) -> Any:
        """Fetch the checkpoint the config addresses."""
        if value := self.get_tuple(config):
            return value.checkpoint
        return None

    def get_tuple(self, config: Any) -> Any:
        raise NotImplementedError

    def list(
        self,
        config: Any,
        *,
        filter: dict[str, Any] | None = None,
        before: Any = None,
        limit: int | None = None,
    ) -> Any:
        raise NotImplementedError

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        raise NotImplementedError

    def put_writes(
        self,
        config: Any,
        writes: Sequence[Any],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError

    def delete_thread(self, thread_id: str) -> None:
        raise NotImplementedError

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        raise NotImplementedError

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        raise NotImplementedError

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError

    def get_delta_channel_history(self, *, config: Any, channels: Sequence[str]) -> Any:
        raise NotImplementedError

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        """Channel versions default to integers, one greater than the current value."""
        if isinstance(current, str):
            raise NotImplementedError
        if current is None:
            return 1
        return current + 1

    def with_allowlist(self, extra_allowlist: Any) -> "BaseCheckpointSaver":
        """LangGraph returns a shallow clone with a wider serializer allowlist."""
        return self

    # -- async twins -------------------------------------------------------------------------
    async def aget(self, config: Any) -> Any:
        return self.get(config)

    async def aget_tuple(self, config: Any) -> Any:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Any,
        *,
        filter: dict[str, Any] | None = None,
        before: Any = None,
        limit: int | None = None,
    ) -> Any:
        return self.list(config, filter=filter, before=before, limit=limit)

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Any,
        writes: Sequence[Any],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        return self.delete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return self.copy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        return self.prune(thread_ids, strategy=strategy)

    async def aget_delta_channel_history(self, *, config: Any, channels: Sequence[str]) -> Any:
        return self.get_delta_channel_history(config=config, channels=channels)
