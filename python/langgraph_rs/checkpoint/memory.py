"""``langgraph.checkpoint.memory`` equivalents."""

from __future__ import annotations

from typing import Any, Iterator

from .._compat import module_getattr, re_export
from .base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from .serde import SerializerProtocol

ChannelVersions = dict
DeltaChannelHistory = list


class PersistentDict(dict):
    """Dict that survives process restarts in LangGraph; here a plain ``dict`` subclass."""

__all__ = [
    "AsyncInMemorySaver",
    "BaseCheckpointSaver",
    "Checkpoint",
    "CheckpointMetadata",
    "ChannelVersions",
    "CheckpointTuple",
    "DeltaChannelHistory",
    "PersistentDict",
    "InMemorySaver",
    "MemorySaver",
    "PendingWrite",
    "SerializerProtocol",
    "WRITES_IDX_MAP",
    "get_checkpoint_id",
    "get_checkpoint_metadata",
    "writes_sort_key",
]

_MISSING = re_export(globals(), ["AsyncInMemorySaver", "InMemorySaver"])
_NATIVE_SAVER = globals()["InMemorySaver"]
_NATIVE_ASYNC_SAVER = globals()["AsyncInMemorySaver"]


def _config_parts(config: Any) -> tuple[str, str, str | None]:
    configurable = (config or {}).get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if not thread_id:
        raise ValueError("Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id")
    return thread_id, configurable.get("checkpoint_ns") or "", configurable.get("checkpoint_id")


class InMemorySaver(_NATIVE_SAVER):
    """The engine's in-memory saver with LangGraph's ``BaseCheckpointSaver`` read API.

    The storage lives in Rust; these methods shape it the way LangGraph's savers do, so
    ``get_tuple`` / ``list`` / ``delete_thread`` behave as they do there.
    """

    def get_tuple(self, config: Any) -> CheckpointTuple | None:
        thread_id, checkpoint_ns, checkpoint_id = _config_parts(config)
        raw = self._checkpoint_tuple(thread_id, checkpoint_ns or None, checkpoint_id)
        return None if raw is None else CheckpointTuple(**raw)

    def get(self, config: Any) -> Checkpoint | None:
        found = self.get_tuple(config)
        return None if found is None else found.checkpoint

    def list(
        self,
        config: Any,
        *,
        before: Any = None,
        limit: int | None = None,
        filter: Any = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id, checkpoint_ns, checkpoint_id = _config_parts(config)
        before_id = before
        if isinstance(before, dict):
            before_id = ((before.get("configurable") or {}).get("checkpoint_id"))
        entries = self._checkpoint_tuples(thread_id, checkpoint_ns or None, before_id, limit)
        for raw in entries:
            tuple_value = CheckpointTuple(**raw)
            if filter and not all(
                tuple_value.metadata.get(key) == value
                for key, value in dict(filter).items()
            ):
                continue
            yield tuple_value

    async def aget_tuple(self, config: Any) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Any,
        *,
        before: Any = None,
        limit: int | None = None,
        filter: Any = None,
    ) -> Any:
        for entry in self.list(config, before=before, limit=limit, filter=filter):
            yield entry

    def copy_thread(
        self,
        source_thread_id: str,
        target_thread_id: str | None = None,
        *,
        checkpoint_id: str | None = None,
    ) -> None:
        """Copy a thread's checkpoints (LangGraph's `copy_thread`)."""
        super().copy_thread(source_thread_id, target_thread_id, checkpoint_id)

    async def acopy_thread(self, *args: Any, **kwargs: Any) -> None:
        self.copy_thread(*args, **kwargs)

    async def aput(self, *args: Any, **kwargs: Any) -> None:
        return self.put(*args, **kwargs)

    async def aput_writes(self, *args: Any, **kwargs: Any) -> None:
        return self.put_writes(*args, **kwargs)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    async def aget(self, config: Any) -> Checkpoint | None:
        return self.get(config)

    async def aput(self, *args: Any, **kwargs: Any) -> None:
        return self.put(*args, **kwargs)

    async def aput_writes(self, *args: Any, **kwargs: Any) -> None:
        return self.put_writes(*args, **kwargs)

    # LangGraph's `InMemorySaver` does not support these; the surface exists so callers get the
    # same error instead of `AttributeError`.
    def prune(self, thread_ids: Any, *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError

    async def aprune(self, thread_ids: Any, *, strategy: str = "keep_latest") -> None:
        raise NotImplementedError

    def delete_for_runs(self, run_ids: Any) -> None:
        raise NotImplementedError

    async def adelete_for_runs(self, run_ids: Any) -> None:
        raise NotImplementedError

    def get_delta_channel_history(
        self, *, config: Any, channels: Any
    ) -> dict[str, Any]:
        """Per-channel write log, shaped like LangGraph's delta history."""
        found = self.get_tuple(config)
        writes = list(getattr(found, "pending_writes", None) or [])
        history: dict[str, Any] = {}
        for channel in channels:
            entries = [
                (task_id, name, value)
                for task_id, name, value in writes
                if name == channel
            ]
            history[channel] = {"writes": entries, "seed": 0}
        return history

    async def aget_delta_channel_history(
        self, *, config: Any, channels: Any
    ) -> dict[str, Any]:
        return self.get_delta_channel_history(config=config, channels=channels)

    @property
    def config_specs(self) -> list[Any]:
        return []

    @property
    def serde(self) -> Any:
        from .serde.jsonplus import JsonPlusSerializer

        if getattr(self, "_serde_instance", None) is None:
            self._serde_instance = JsonPlusSerializer()
        return self._serde_instance

    def with_allowlist(self, extra_allowlist: Any) -> Any:
        """Return this saver; the engine's serializer already accepts every type."""
        return self


class AsyncInMemorySaver(_NATIVE_ASYNC_SAVER):
    """Async flavour: the engine's async saver with the same read API."""

    def get_tuple(self, config: Any) -> CheckpointTuple | None:
        return InMemorySaver.get_tuple(self, config)

    def list(self, config: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        return InMemorySaver.list(self, config, **kwargs)

    async def aget_tuple(self, config: Any) -> CheckpointTuple | None:
        return InMemorySaver.get_tuple(self, config)

    async def alist(self, config: Any, **kwargs: Any) -> Any:
        for entry in InMemorySaver.list(self, config, **kwargs):
            yield entry


# LangGraph's legacy name for the in-memory saver.
MemorySaver = InMemorySaver
__getattr__ = module_getattr(__name__, globals(), _MISSING)
