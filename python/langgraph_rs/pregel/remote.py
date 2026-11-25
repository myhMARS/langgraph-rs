"""Client for graphs served by the LangGraph Server API (``langgraph.pregel.remote``).

`RemoteGraph` talks HTTP to a deployment through `langgraph_sdk`, so a graph running elsewhere can be
used like a local one (including as a node inside another graph). Only the surface that maps onto the
SDK one-to-one is implemented; the SDK is an optional dependency and a missing client raises.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

__all__ = ["RemoteException", "RemoteGraph"]

# Names LangGraph's module also exposes at module level (its own imports, kept importable).
from ..constants import CONF  # noqa: E402,F811
from ..types import Command, Interrupt, StateSnapshot, StreamMode  # noqa: E402,F401

try:  # pragma: no cover - langgraph_sdk is optional
    from langgraph_sdk import get_client, get_sync_client  # noqa: E402,F401
except Exception:  # pragma: no cover
    def get_client(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("langgraph_sdk is not installed")

    def get_sync_client(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("langgraph_sdk is not installed")

__all__.extend(["Command", "Interrupt", "StateSnapshot", "StreamMode", "get_client", "get_sync_client"])

CONF = "configurable"
CONFIG_KEY_CHECKPOINT_ID = "checkpoint_id"
CONFIG_KEY_CHECKPOINT_NS = "checkpoint_ns"
CHECKPOINT_NS_SEPARATOR = "|"


class RemoteException(Exception):
    """An error reported by the remote graph."""


class RemoteGraph:
    """Call a graph deployed behind the LangGraph Server API.

    Either pass `url` (plus optional `api_key` / `headers`) so the SDK clients are created, or inject
    `client` (async) / `sync_client` yourself — which is also how this class is tested.
    """

    assistant_id: str
    name: str | None

    def __init__(
        self,
        assistant_id: str,
        /,
        *,
        url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: Any = None,
        sync_client: Any = None,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
        distributed_tracing: bool = False,
    ) -> None:
        self.assistant_id = assistant_id
        self.name = assistant_id if name is None else name
        self.config = dict(config or {})
        self.distributed_tracing = distributed_tracing

        if client is None and url is not None:
            from langgraph_sdk import get_client

            client = get_client(url=url, api_key=api_key, headers=headers)
        self.client = client

        if sync_client is None and url is not None:
            from langgraph_sdk import get_sync_client

            sync_client = get_sync_client(url=url, api_key=api_key, headers=headers)
        self.sync_client = sync_client

    # -- construction helpers -------------------------------------------

    def _validate_client(self) -> Any:
        if self.client is None:
            raise ValueError(
                "Asynchronous client not configured: pass `url` or an async `client` to RemoteGraph."
            )
        return self.client

    def _validate_sync_client(self) -> Any:
        if self.sync_client is None:
            raise ValueError(
                "Synchronous client not configured: pass `url` or a `sync_client` to RemoteGraph."
            )
        return self.sync_client

    def copy(self, update: dict[str, Any]) -> RemoteGraph:
        """Return a copy with `update` applied to this instance's attributes."""
        attrs = dict(self.__dict__)
        attrs.update(update)
        clone = type(self).__new__(type(self))
        clone.__dict__.update(attrs)
        return clone

    def with_config(self, config: Mapping[str, Any] | None = None, **kwargs: Any) -> RemoteGraph:
        """Return a copy with `config` merged in."""
        merged = {**self.config, **(config or {}), **kwargs}
        return self.copy({"config": merged})

    # -- config plumbing -------------------------------------------------

    def _merged_config(self, config: Mapping[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {**self.config}
        incoming = dict(config or {})
        merged.update({key: value for key, value in incoming.items() if key != CONF})
        merged[CONF] = {**(self.config.get(CONF) or {}), **(incoming.get(CONF) or {})}
        return merged

    @staticmethod
    def _thread_id(config: Mapping[str, Any] | None) -> str | None:
        return ((config or {}).get(CONF) or {}).get("thread_id")

    @staticmethod
    def _checkpoint_id(config: Mapping[str, Any] | None) -> str | None:
        return ((config or {}).get(CONF) or {}).get(CONFIG_KEY_CHECKPOINT_ID)

    # -- introspection ---------------------------------------------------

    def get_graph(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        xray: int | bool = False,
        headers: dict[str, str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the remote graph's drawable structure."""
        return self._validate_sync_client().assistants.get_graph(
            self.assistant_id, xray=xray, headers=headers
        )

    async def aget_graph(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        xray: int | bool = False,
        headers: dict[str, str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Async variant of `get_graph`."""
        return await self._validate_client().assistants.get_graph(
            self.assistant_id, xray=xray, headers=headers
        )

    # -- state -----------------------------------------------------------

    def _state_kwargs(self, config: Mapping[str, Any] | None, subgraphs: bool) -> dict[str, Any]:
        merged = self._merged_config(config)
        checkpoint_id = self._checkpoint_id(merged)
        kwargs: dict[str, Any] = {"subgraphs": subgraphs}
        if checkpoint_id:
            kwargs["checkpoint_id"] = checkpoint_id
        return kwargs

    def get_state(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        subgraphs: bool = False,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Return the thread's latest (or checkpointed) state."""
        sync_client = self._validate_sync_client()
        thread_id = self._thread_id(self._merged_config(config))
        return sync_client.threads.get_state(
            thread_id, headers=headers, **self._state_kwargs(config, subgraphs)
        )

    async def aget_state(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        subgraphs: bool = False,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Async variant of `get_state`."""
        client = self._validate_client()
        thread_id = self._thread_id(self._merged_config(config))
        return await client.threads.get_state(
            thread_id, headers=headers, **self._state_kwargs(config, subgraphs)
        )

    def get_state_history(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        """Return the thread's checkpoint history."""
        sync_client = self._validate_sync_client()
        thread_id = self._thread_id(self._merged_config(config))
        return sync_client.threads.get_history(
            thread_id, filter=filter, before=before, limit=limit, headers=headers
        )

    async def aget_state_history(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: Mapping[str, Any] | None = None,
        limit: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        """Async variant of `get_state_history`."""
        client = self._validate_client()
        thread_id = self._thread_id(self._merged_config(config))
        return await client.threads.get_history(
            thread_id, filter=filter, before=before, limit=limit, headers=headers
        )

    def update_state(
        self,
        config: Mapping[str, Any] | None,
        values: dict[str, Any] | Any,
        *,
        as_node: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Write `values` into the thread state and return the new checkpoint config."""
        sync_client = self._validate_sync_client()
        merged = self._merged_config(config)
        return sync_client.threads.update_state(
            self._thread_id(merged),
            values,
            as_node=as_node,
            checkpoint_id=self._checkpoint_id(merged),
            headers=headers,
        )

    async def aupdate_state(
        self,
        config: Mapping[str, Any] | None,
        values: dict[str, Any] | Any,
        *,
        as_node: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async variant of `update_state`."""
        client = self._validate_client()
        merged = self._merged_config(config)
        return await client.threads.update_state(
            self._thread_id(merged),
            values,
            as_node=as_node,
            checkpoint_id=self._checkpoint_id(merged),
            headers=headers,
        )

    # -- streaming -------------------------------------------------------

    @staticmethod
    def _modes(stream_mode: Any) -> list[str]:
        if stream_mode is None:
            return ["values"]
        if isinstance(stream_mode, str):
            return [stream_mode]
        return list(stream_mode)

    def _run_args(
        self,
        input: Any,
        config: Mapping[str, Any] | None,
        stream_mode: Any,
        *,
        subgraphs: bool,
        context: Any,
        interrupt_before: Any,
        interrupt_after: Any,
        headers: dict[str, str] | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        merged = self._merged_config(config)
        modes = self._modes(stream_mode)
        args: dict[str, Any] = {
            "thread_id": self._thread_id(merged),
            "assistant_id": self.assistant_id,
            "input": input,
            "stream_mode": modes,
            "stream_subgraphs": subgraphs,
            "config": {k: v for k, v in merged.items() if k != CONF} or None,
            "context": merged.get(CONF, {}).get("context", context),
            "interrupt_before": interrupt_before,
            "interrupt_after": interrupt_after,
            "headers": headers,
        }
        checkpoint_id = self._checkpoint_id(merged)
        if checkpoint_id:
            args["checkpoint_id"] = checkpoint_id
        args.update(kwargs)
        return {key: value for key, value in args.items() if value is not None}, modes

    @staticmethod
    def _chunk_mode(chunk: Any, modes: Sequence[str]) -> str:
        event = getattr(chunk, "event", None) or ""
        for mode in modes:
            if event.startswith(mode):
                return mode
        return modes[0]

    @staticmethod
    def _chunk_namespace(chunk: Any) -> tuple[str, ...]:
        """Checkpoint namespace encoded in the SDK event name (`"values|sub:1|inner:2"`)."""
        event = getattr(chunk, "event", None) or ""
        if CHECKPOINT_NS_SEPARATOR not in event:
            return ()
        namespace = event.split(CHECKPOINT_NS_SEPARATOR, 1)[1]
        return tuple(part for part in namespace.split(CHECKPOINT_NS_SEPARATOR) if part)

    def stream(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stream_mode: Any = None,
        subgraphs: bool = False,
        context: Any = None,
        interrupt_before: Any = None,
        interrupt_after: Any = None,
        headers: dict[str, str] | None = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Yield the remote run's chunks.

        `version="v2"` yields `StreamPart` dicts (`type` / `ns` / `data` / `interrupts`), matching the
        extension's own stream; `version="v1"` yields the SDK chunk payloads.
        """
        sync_client = self._validate_sync_client()
        args, modes = self._run_args(
            input,
            config,
            stream_mode,
            subgraphs=subgraphs,
            context=context,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            kwargs=kwargs,
        )
        args["version"] = "v2" if version == "v2" else "v1"
        single = isinstance(stream_mode, str)

        for chunk in sync_client.runs.stream(**args):
            event = getattr(chunk, "event", "") or ""
            if event.startswith("error"):
                raise RemoteException(getattr(chunk, "data", chunk))
            mode = self._chunk_mode(chunk, modes)
            namespace = self._chunk_namespace(chunk)
            data = getattr(chunk, "data", chunk)
            if version == "v2":
                yield {
                    "type": mode,
                    "ns": namespace,
                    "data": data,
                    "interrupts": (),
                }
            elif subgraphs:
                yield (namespace, mode, data) if not single else (namespace, data)
            elif single:
                yield data
            else:
                yield (mode, data) if namespace == () else (namespace, mode, data)

    async def astream(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
        *,
        stream_mode: Any = None,
        subgraphs: bool = False,
        context: Any = None,
        interrupt_before: Any = None,
        interrupt_after: Any = None,
        headers: dict[str, str] | None = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async variant of `stream`."""
        client = self._validate_client()
        args, modes = self._run_args(
            input,
            config,
            stream_mode,
            subgraphs=subgraphs,
            context=context,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            kwargs=kwargs,
        )
        args["version"] = "v2" if version == "v2" else "v1"
        single = isinstance(stream_mode, str)

        async for chunk in client.runs.stream(**args):
            event = getattr(chunk, "event", "") or ""
            if event.startswith("error"):
                raise RemoteException(getattr(chunk, "data", chunk))
            mode = self._chunk_mode(chunk, modes)
            namespace = self._chunk_namespace(chunk)
            data = getattr(chunk, "data", chunk)
            if version == "v2":
                yield {"type": mode, "ns": namespace, "data": data, "interrupts": ()}
            elif subgraphs:
                yield (namespace, mode, data) if not single else (namespace, data)
            elif single:
                yield data
            else:
                yield (mode, data) if namespace == () else (namespace, mode, data)

    # -- invoke ----------------------------------------------------------

    def invoke(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
        *,
        context: Any = None,
        interrupt_before: Any = None,
        interrupt_after: Any = None,
        headers: dict[str, str] | None = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> Any:
        """Run the remote graph and return its final state."""
        chunk: Any = None
        for chunk in self.stream(
            input,
            config=config,
            stream_mode="values",
            context=context,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            version=version,
            **kwargs,
        ):
            pass
        if chunk is None:
            return None
        return chunk["data"] if version == "v2" else chunk

    async def ainvoke(
        self,
        input: Any,
        config: Mapping[str, Any] | None = None,
        *,
        context: Any = None,
        interrupt_before: Any = None,
        interrupt_after: Any = None,
        headers: dict[str, str] | None = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> Any:
        """Async variant of `invoke`."""
        chunk: Any = None
        async for chunk in self.astream(
            input,
            config=config,
            stream_mode="values",
            context=context,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            version=version,
            **kwargs,
        ):
            pass
        if chunk is None:
            return None
        return chunk["data"] if version == "v2" else chunk

    def __repr__(self) -> str:
        return f"RemoteGraph({self.assistant_id!r}, name={self.name!r})"