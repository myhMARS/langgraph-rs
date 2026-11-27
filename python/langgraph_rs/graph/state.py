"""``langgraph.graph.state`` equivalents."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABCMeta
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.runnables import Runnable

from .._compat import _native, module_getattr, re_export
from ..channels import BaseChannel, BinaryOperatorAggregate
from ..types import All, CachePolicy

__all__ = [
    "All",
    "AsyncStateGraph",
    "BaseChannel",
    "BinaryOperatorAggregate",
    "BranchSpec",
    "CachePolicy",
    "ChannelRead",
    "CompiledGraph",
    "CompiledStateGraph",
    "END",
    "START",
    "StateGraph",
]


class ChannelRead:
    """Marker telling the engine which channels a branch reads (LangGraph internal)."""

    __slots__ = ("channel",)

    def __init__(self, channel: str = "") -> None:
        self.channel = channel

    def __repr__(self) -> str:
        return f"ChannelRead({self.channel!r})"


class BranchSpec:
    """A conditional edge specification (LangGraph internal)."""

    __slots__ = ("path", "ends")

    def __init__(self, path: Any = None, ends: Any = None) -> None:
        self.path = path
        self.ends = ends

    def __repr__(self) -> str:
        return f"BranchSpec(path={self.path!r}, ends={self.ends!r})"


_MISSING = re_export(globals(), ["END", "START", "StateGraph"])

_NATIVE_COMPILED = _native.CompiledGraph

# LangGraph's `invoke`/`stream` keywords that the extension cannot honour per run. They are
# rejected loudly instead of being silently dropped: each of them changes run behaviour.
_UNSUPPORTED_RUN_KWARGS = {
    "interrupt_before": "declare `interrupt_before` when compiling the graph",
    "interrupt_after": "declare `interrupt_after` when compiling the graph",
    "output_keys": "select the outgoing state in the graph's schema",
}

# Keyword arguments that only matter to LangGraph's internal streaming/printing and do not change
# the result, so they are accepted and ignored.
_IGNORED_RUN_KWARGS = {"print_mode", "debug"}

# Per-method allow-lists: the native graph accepts these (plus its own `thread_id` convenience).
_COMMON_KWARGS = ("thread_id", "context", "control", "checkpoint_ns")
_INVOKE_KWARGS = _COMMON_KWARGS
_STREAM_KWARGS = _COMMON_KWARGS + ("stream_mode", "subgraphs", "version", "live_sink")
_BATCH_KWARGS = ("thread_id", "context")


def _check_kwargs(kwargs: dict[str, Any]) -> None:
    for name, guidance in _UNSUPPORTED_RUN_KWARGS.items():
        if kwargs.get(name) is not None:
            kwargs.pop(name)
            raise NotImplementedError(
                f"`{name}` is not supported on a run; {guidance} instead"
            )


def _with_durability(config: Any, durability: Any) -> Any:
    if durability is None:
        return config
    merged = dict(config or {})
    if isinstance(durability, str):
        merged["durability"] = durability
        return merged
    # LangGraph accepts a `Durability`-like object; only its name is meaningful here.
    name = getattr(durability, "value", None) or getattr(durability, "name", None)
    if isinstance(name, str):
        merged["durability"] = name
    return merged


def _call_kwargs(
    config: Any,
    durability: Any,
    kwargs: dict[str, Any],
    allowed: Sequence[str] = _COMMON_KWARGS,
) -> dict[str, Any]:
    """Split LangGraph's keyword arguments into those the extension forwards and those it drops."""
    _check_kwargs(kwargs)
    call: dict[str, Any] = {"config": _with_durability(config, durability)}
    for name in allowed:
        if name in kwargs:
            call[name] = kwargs.pop(name)
    for name in list(kwargs):
        if name in _IGNORED_RUN_KWARGS:
            kwargs.pop(name)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword argument(s): {unknown}")
    return call


def _thread_id_of(config: Any) -> str | None:
    """Read ``configurable.thread_id`` from a LangGraph config dict."""
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable") or {}
    value = configurable.get("thread_id") if isinstance(configurable, dict) else None
    return value if isinstance(value, str) else None


def _dispatch_lifecycle(graph: Any, config: Any) -> None:
    """Drain the run's recorded transitions and hand them to ``GraphCallbackHandler``s.

    The engine records pause/resume events into a queue (LangGraph's loop does the same); this is
    the single place that turns them into callbacks, once per run and with a run id.
    """
    try:
        events = graph.take_lifecycle_events()
    except Exception:  # pragma: no cover - a graph without the queue
        return
    if not events:
        return
    from uuid import uuid4

    from ..callbacks import (
        GraphInterruptEvent,
        GraphResumeEvent,
        get_sync_graph_callback_manager_for_config,
    )

    try:
        manager = get_sync_graph_callback_manager_for_config(config or {}, run_id=uuid4())
    except Exception:  # pragma: no cover - a broken config must not break the run
        return
    if not manager.handlers:
        return
    for event in events:
        namespace = tuple(part for part in str(event.get("namespace") or "").split("|") if part)
        if event.get("kind") == "resume":
            payload = GraphResumeEvent(
                run_id=manager.run_id,
                status=event.get("status") or "pending",
                checkpoint_id=event.get("checkpoint_id") or "",
                checkpoint_ns=namespace,
            )
            try:
                manager.on_resume(payload)
            except Exception:  # pragma: no cover - a broken handler must not break the run
                pass
        else:
            payload = GraphInterruptEvent(
                run_id=manager.run_id,
                status=event.get("status") or "pending",
                checkpoint_id=event.get("checkpoint_id") or "",
                checkpoint_ns=namespace,
                interrupts=tuple(event.get("interrupts") or ()),
            )
            try:
                manager.on_interrupt(payload)
            except Exception:  # pragma: no cover
                pass


class StateGraph(_native.StateGraph):
    """Rust builder whose ``compile()`` returns the ``Runnable`` facade.

    The native method returns the engine object; wrapping happens here so the extension never has
    to call back into Python to build the facade. The schema is normalised here too: class schemas
    are expanded and custom channels marked, so the engine only sees plain data.
    """

    def __new__(
        cls,
        state_schema: Any = None,
        input_schema: Any = None,
        output_schema: Any = None,
        private_schema: Any = None,
        managed_values: bool = False,
        context_schema: Any = None,
    ) -> Any:
        # The native constructor is `tp_new`, so normalisation has to happen here (before the
        # engine sees the schema); a subclass `__init__` would run too late.
        from ._schema import normalize_schema

        instance = super().__new__(
            cls,
            normalize_schema(state_schema),
            input_schema,
            output_schema,
            private_schema,
            managed_values,
        )
        # Kept for the compiled graph's introspection surface (`schema_to_mapper`, ...).
        instance._builder_schema = state_schema
        instance._builder_outputs = list(output_schema) if output_schema else None
        instance._builder_handlers = {}
        # LangGraph takes the runtime context schema on the builder and carries it into `compile`.
        instance._builder_context_schema = context_schema
        return instance

    def add_node(self, name: Any, action: Any = None, **kwargs: Any) -> Any:
        """Register nodes, translating LangGraph's callable `error_handler` into a handler node.

        LangGraph synthesises `__error_handler__<node>` and routes the failure there, so the
        handler sees the state and its update replaces the node's result; the branch then ends.
        """
        # LangGraph rejects a duplicate name as soon as it is registered.
        names = self.__dict__.setdefault("_node_names", set())
        key = str(name)
        if key in names:
            raise ValueError(f"Node `{key}` already present.")
        names.add(key)
        action = _with_runtime(action)
        # `set_node_defaults` applies to nodes registered afterwards.
        defaults = getattr(self, "_node_defaults", None)
        if defaults:
            for key, value in defaults.items():
                if value is not None and key not in kwargs:
                    kwargs[key] = value
        timeout = kwargs.get("timeout")
        if timeout is not None:
            # LangGraph accepts seconds / `timedelta` / `TimeoutPolicy`; the engine takes the
            # policy object, so plain durations are wrapped here.
            if not isinstance(timeout, _native.TimeoutPolicy):
                kwargs["timeout"] = _native.TimeoutPolicy(run_timeout=timeout)
            if not inspect.iscoroutinefunction(action):
                self.__dict__.setdefault("_sync_timeouts", {})[str(name)] = True
        handler = kwargs.get("error_handler")
        handlers = getattr(self, "_builder_handlers", None)
        if handler is not None and not isinstance(handler, str) and callable(handler):
            hidden = f"__error_handler__{name}"
            super().add_node(hidden, handler)
            if handlers is not None:
                handlers[str(name)] = handler
            return super().add_node(name, action, **{**kwargs, "error_handler": hidden})
        if handlers is not None and handler is not None:
            handlers[str(name)] = handler
        return super().add_node(name, action, **kwargs)

    def add_edge(self, start_key: Any, end_key: Any) -> Any:
        """Record edge declaration order so `get_graph()` matches LangGraph's ordering."""
        result = super().add_edge(start_key, end_key)
        self.__dict__.setdefault("_edge_order", []).append((start_key, end_key, False))
        return result

    def add_conditional_edges(self, source: Any, path: Any, path_map: Any = None) -> Any:
        """Record branch declaration order alongside the plain edges."""
        result = super().add_conditional_edges(source, path, path_map)
        targets: list[Any] = []
        if isinstance(path_map, dict):
            for value in path_map.values():
                targets.extend(value if isinstance(value, (list, tuple)) else [value])
        elif isinstance(path_map, (list, tuple)):
            targets.extend(path_map)
        if targets:
            for target in targets:
                self.__dict__.setdefault("_edge_order", []).append((source, target, True))
        else:
            self.__dict__.setdefault("_edge_order", []).append((source, None, True))
        return result

    def add_sequence(self, nodes: Any) -> Any:
        """Add a chain of nodes, each wired to the next, like LangGraph's `add_sequence`."""
        items = list(nodes)
        names: list[str] = []
        for item in items:
            if isinstance(item, tuple):
                name, action = item
                self.add_node(name, action)
                names.append(str(name))
            else:
                action = item
                name = getattr(action, "__name__", f"node_{len(names)}")
                self.add_node(name, action)
                names.append(str(name))
        for source, target in zip(names, names[1:]):
            self.add_edge(source, target)
        return self

    def set_node_defaults(
        self,
        *,
        retry_policy: Any = None,
        cache_policy: Any = None,
        error_handler: Any = None,
        timeout: Any = None,
    ) -> Any:
        """Apply defaults to nodes added afterwards (LangGraph's `set_node_defaults`)."""
        self._node_defaults = {
            "retry_policy": retry_policy,
            "cache_policy": cache_policy,
            "error_handler": error_handler,
            "timeout": timeout,
        }
        return self

    def validate(self, interrupt: Any = None) -> Any:
        """Compile-time validation without running: raises `ValueError` when the graph is invalid."""
        del interrupt
        self.compile()
        return self

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        # LangGraph refuses a timeout on a synchronous node: the thread cannot be cancelled, so a
        # timeout would abandon work instead of stopping it.
        sync_timeouts = getattr(self, "_sync_timeouts", None) or {}
        if sync_timeouts:
            name = sorted(sync_timeouts)[0]
            raise ValueError(
                "Node timeouts are only supported for async nodes because sync Python execution "
                f"cannot be safely cancelled in-process. Node {name!r} is sync."
            )
        if kwargs.get("context_schema") is None:
            declared = getattr(self, "_builder_context_schema", None)
            if declared is not None:
                kwargs["context_schema"] = declared
        return CompiledStateGraph(
            super().compile(*args, **kwargs),
            schema_class=getattr(self, "_builder_schema", None),
            output_names=getattr(self, "_builder_outputs", None),
            error_handlers=getattr(self, "_builder_handlers", None),
            edge_order=getattr(self, "_edge_order", None),
        )


class AsyncStateGraph(_native.AsyncStateGraph):
    """Async counterpart of :class:`StateGraph`."""

    def __new__(
        cls,
        state_schema: Any = None,
        input_schema: Any = None,
        output_schema: Any = None,
        private_schema: Any = None,
        managed_values: bool = False,
    ) -> Any:
        # The native constructor is `tp_new`, so normalisation has to happen here (before the
        # engine sees the schema); a subclass `__init__` would run too late.
        from ._schema import normalize_schema

        # The async builder takes the schema only; the channel views are a sync-builder feature.
        return super().__new__(cls, normalize_schema(state_schema))

    def add_node(self, name: Any, action: Any = None, **kwargs: Any) -> Any:
        """Wrap plain durations in `TimeoutPolicy`, as LangGraph accepts both."""
        timeout = kwargs.get("timeout")
        if timeout is not None and not isinstance(timeout, _native.TimeoutPolicy):
            kwargs["timeout"] = _native.TimeoutPolicy(run_timeout=timeout)
        return super().add_node(name, action, **kwargs)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        return CompiledStateGraph(super().compile(*args, **kwargs))


class _CompiledMeta(ABCMeta):
    """Forward class-level lookups (``hasattr(CompiledStateGraph, ...)``) to the native class."""

    def __getattr__(cls, name: str) -> Any:
        return getattr(_NATIVE_COMPILED, name)



def _normalize_supersteps(supersteps: Any) -> Any:
    """Accept LangGraph's `(values, as_node, task_id)` tuples as well as update dicts."""
    if not supersteps:
        return supersteps
    normalized = []
    for step in supersteps:
        entries = []
        for entry in step:
            if isinstance(entry, dict):
                entries.append(
                    (entry.get("values"), entry.get("as_node"), entry.get("task_id"))
                )
            else:
                entries.append(tuple(entry))
        normalized.append(entries)
    return normalized



def _accepts_runtime(action: Any) -> bool:
    """Does this node declare a `runtime` parameter (LangGraph `Runtime` injection)?"""
    if not callable(action) or inspect.isclass(action):
        return False
    if getattr(action, "_langgraph_runtime_wrapped", False):
        return False
    try:
        parameters = inspect.signature(action).parameters
    except (TypeError, ValueError):
        return False
    if "runtime" not in parameters:
        return False
    return parameters["runtime"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _current_runtime() -> Any:
    """Build the `Runtime` handed to a node, from the running task's context."""
    from ..runtime import Runtime

    try:
        context = _native.current_context()
    except Exception:  # pragma: no cover - no active run
        context = None
    try:
        store = _native.get_store()
    except Exception:  # pragma: no cover - no store configured
        store = None

    try:
        writer = _native.get_stream_writer()
    except Exception:  # pragma: no cover - no stream writer
        writer = None
    return Runtime(context=context, store=store, stream_writer=writer)


def _with_runtime(action: Any) -> Any:
    """Wrap a node that asked for a `Runtime` so the engine can keep calling it with the state."""
    if not _accepts_runtime(action):
        return action
    if inspect.iscoroutinefunction(action):

        async def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("runtime", _current_runtime())
            return await action(state, *args, **kwargs)

    else:

        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("runtime", _current_runtime())
            return action(state, *args, **kwargs)

    wrapper._langgraph_runtime_wrapped = True  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(action, "__name__", "node")
    wrapper.__doc__ = getattr(action, "__doc__", None)
    return wrapper


def _state_snapshot(raw: Any, requested: Any = None) -> Any:
    """Turn the engine's snapshot dict into LangGraph's `StateSnapshot` named tuple."""
    if raw is None:
        return None
    from ..types import StateSnapshot

    snapshot = StateSnapshot.from_native(raw)
    if snapshot is None:
        return None
    if not snapshot.config and requested is not None:
        # LangGraph echoes the requested config for a thread with no checkpoints.
        configurable = {}
        if isinstance(requested, dict):
            configurable = dict(requested.get("configurable") or {})
        elif isinstance(requested, str):
            configurable = {"thread_id": requested}
        snapshot = snapshot._replace(config={"configurable": configurable})
    if snapshot.tasks:
        tasks = []
        for task in snapshot.tasks:
            state = getattr(task, "state", None)
            if isinstance(state, dict) and "values" in state:
                state = StateSnapshot.from_native(state)
                try:
                    task = task._replace(state=state)
                except AttributeError:
                    pass
            tasks.append(task)
        if tuple(tasks) != snapshot.tasks:
            snapshot = snapshot._replace(tasks=tuple(tasks))
    return snapshot


from ..pregel import Pregel as _Pregel


class CompiledStateGraph(Runnable, metaclass=_CompiledMeta):
    """The compiled graph, exposing LangGraph's ``Runnable`` API.

    LangGraph's compiled graph is a ``Runnable``; the extension's engine lives in Rust, so this
    facade forwards the run methods to the native object while inheriting the ``Runnable`` helpers
    (``bind``, ``with_config``, ``pipe``, ``map``, ``as_tool``, ``batch_as_completed``,
    ``astream_log`` and the schema introspection helpers).
    """

    def __init__(
        self,
        inner: Any,
        *,
        schema_class: Any = None,
        output_names: list[str] | None = None,
        error_handlers: dict[str, Any] | None = None,
        edge_order: list[Any] | None = None,
    ) -> None:
        self._inner = inner
        self._schema_class = schema_class
        self._output_names = output_names
        self._error_handlers = dict(error_handlers or {})
        # Declaration order of the builder's edges, used by `get_graph()`.
        self._edge_order = list(edge_order or [])

    # -- LangGraph introspection surface -----------------------------------------------------
    @property
    def input_channels(self) -> str:
        """LangGraph's single entry channel."""
        return "__start__"

    def _channel_names(self) -> list[str]:
        return sorted(getattr(self._inner, "channels", {}) or {})

    @property
    def output_channels(self) -> list[str]:
        return list(self._output_names) if self._output_names else self._channel_names()

    @property
    def stream_channels(self) -> list[str]:
        return self.output_channels

    @property
    def stream_channels_list(self) -> list[str]:
        return self.stream_channels

    @property
    def stream_channels_asis(self) -> list[str]:
        return self.stream_channels

    @property
    def node_error_handler_map(self) -> dict[str, Any]:
        return dict(self._error_handlers)

    @property
    def schema_to_mapper(self) -> dict[Any, Any]:
        """LangGraph maps each schema class to its output mapper; the extension maps to `None`."""
        return {} if self._schema_class is None else {self._schema_class: None}

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Render the graph in notebooks (PNG, as LangGraph does)."""
        bundle: dict[str, Any] = {"text/plain": repr(self)}
        try:
            png = self._inner.draw_png()
        except Exception:  # pragma: no cover - graphviz is optional
            return bundle
        if png:
            bundle["image/png"] = png
        return bundle

    # -- Runnable plumbing ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Everything the facade does not define (resume, stream_live, get_subgraphs, ...) is
        # answered by the native graph.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __dir__(self) -> list[str]:
        # `dir()` should list the forwarded native methods too, so the object introspects like
        # LangGraph's compiled graph.
        return sorted(set(object.__dir__(self)) | set(dir(self._inner)))

    def __repr__(self) -> str:
        return f"CompiledStateGraph({self.get_name()!r})"

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        base = name or self._inner._graph_name() or "LangGraph"
        return f"{base}{suffix}" if suffix else base

    @property
    def name(self) -> str:
        return self.get_name()

    @property
    def input_schema(self) -> dict[str, Any]:
        channels = sorted(getattr(self._inner, "channels", {}) or {})
        return {"title": self.get_name(), "type": "object", "properties": {name: {} for name in channels}}

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.input_schema

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "title": f"{self.get_name()}Config",
            "type": "object",
            "properties": {
                "configurable": {"type": "object", "properties": {"thread_id": {"type": "string"}}},
            },
        }

    def get_state(self, thread_id: Any = None, checkpoint_ns: Any = None, subgraphs: bool = False, config: Any = None) -> Any:
        """LangGraph's `get_state`, returning a `StateSnapshot` named tuple."""
        return _state_snapshot(
            self._inner.get_state(
                thread_id, checkpoint_ns=checkpoint_ns, subgraphs=subgraphs
            ),
            thread_id if config is None else config,
        )

    def get_state_history(self, thread_id: Any, before: Any = None, limit: Any = None, checkpoint_ns: Any = None, filter: Any = None) -> Any:
        """LangGraph's `get_state_history`, newest first."""
        # LangGraph returns an iterator; a list keeps it iterable and indexable for callers that
        # relied on the extension's earlier behaviour.
        try:
            entries = self._inner.get_state_history(
                thread_id, before=before, limit=limit, checkpoint_ns=checkpoint_ns, filter=filter
            )
        except TypeError:
            # Async graphs have no server-side filter; apply it here.
            entries = self._inner.get_state_history(
                thread_id, before=before, limit=limit, checkpoint_ns=checkpoint_ns
            )
        snapshots = [_state_snapshot(entry) for entry in entries]
        if filter:
            wanted = dict(filter)
            snapshots = [
                snapshot
                for snapshot in snapshots
                if all(
                    (snapshot.metadata or {}).get(key) == value
                    for key, value in wanted.items()
                )
            ]
        return snapshots

    async def aget_state(
        self,
        thread_id: Any = None,
        checkpoint_ns: Any = None,
        subgraphs: bool = False,
        config: Any = None,
    ) -> Any:
        if getattr(self._inner, "_native_async", False):
            return _state_snapshot(
                await self._inner.aget_state(
                    thread_id, checkpoint_ns=checkpoint_ns, subgraphs=subgraphs
                )
            )
        return await asyncio.to_thread(
            self._inner.get_state,
            thread_id,
            checkpoint_ns=checkpoint_ns,
            subgraphs=subgraphs,
            config=config,
        )

    async def aget_state_history(
        self,
        thread_id: Any,
        before: Any = None,
        limit: Any = None,
        checkpoint_ns: Any = None,
        filter: Any = None,
    ) -> Any:
        if getattr(self._inner, "_native_async", False):
            return await self._inner.aget_state_history(
                thread_id, before=before, limit=limit, checkpoint_ns=checkpoint_ns, filter=filter
            )
        return await asyncio.to_thread(
            lambda: list(
                self._inner.get_state_history(
                    thread_id,
                    before=before,
                    limit=limit,
                    checkpoint_ns=checkpoint_ns,
                    filter=filter,
                )
            )
        )

    def update_state(
        self,
        thread_id: Any,
        values: Any,
        as_node: Any = None,
        task_id: Any = None,
    ) -> Any:
        """Write values to a thread, returning the new checkpoint's config (LangGraph shape)."""
        snapshot = self._inner.update_state(
            thread_id, values, as_node=as_node, task_id=task_id
        )
        return getattr(snapshot, "config", snapshot)

    async def aupdate_state(
        self,
        thread_id: Any,
        values: Any,
        as_node: Any = None,
        task_id: Any = None,
    ) -> Any:
        if getattr(self._inner, "_native_async", False):
            snapshot = await self._inner.aupdate_state(
                thread_id, values, as_node=as_node, task_id=task_id
            )
        else:
            snapshot = await asyncio.to_thread(
                self._inner.update_state, thread_id, values, as_node=as_node, task_id=task_id
            )
        return getattr(snapshot, "config", snapshot)

    def bulk_update_state(
        self, config: Any = None, supersteps: Any = None, *, thread_id: Any = None
    ) -> Any:
        """Apply several supersteps of updates, returning the new checkpoint's config."""
        snapshot = self._inner.bulk_update_state(
            config, _normalize_supersteps(supersteps), thread_id=thread_id
        )
        if isinstance(snapshot, (list, tuple)):
            return [getattr(item, "config", item) for item in snapshot]
        return getattr(snapshot, "config", snapshot)

    async def abulk_update_state(
        self, config: Any = None, supersteps: Any = None, *, thread_id: Any = None
    ) -> Any:
        normalized = _normalize_supersteps(supersteps)
        if getattr(self._inner, "_native_async", False):
            snapshot = await self._inner.abulk_update_state(
                config, normalized, thread_id=thread_id
            )
        else:
            snapshot = await asyncio.to_thread(
                self._inner.bulk_update_state, config, normalized, thread_id=thread_id
            )
        if isinstance(snapshot, (list, tuple)):
            return [getattr(item, "config", item) for item in snapshot]
        return getattr(snapshot, "config", snapshot)

    async def aclear_cache(self, nodes: Any = None) -> Any:
        if getattr(self._inner, "_native_async", False):
            return await self._inner.aclear_cache(nodes)
        return await asyncio.to_thread(self._inner.clear_cache, nodes)

    def get_context_jsonschema(self) -> Any:
        """LangGraph's context JSON Schema, derived in Python from the declared schema."""
        from .._compat import jsonschema_of

        raw = self._inner.get_context_jsonschema()
        return None if raw is None else jsonschema_of(raw)

    async def aget_graph(self, config: Any = None, *, xray: Any = False) -> Any:
        return self._inner.get_graph(config, xray=xray)

    async def aget_subgraphs(self, *, namespace: Any = None, recurse: bool = False) -> Any:
        return list(self._inner.get_subgraphs(namespace=namespace, recurse=recurse))

    def get_graph(self, config: Any = None, *, xray: Any = False) -> Any:
        """Return the renderable graph.

        LangGraph returns a `langchain_core.runnables.graph.Graph`; when langchain-core is
        installed the native description is rebuilt into one so `nodes`, `edges` and the drawing
        helpers behave exactly as they do there. Without it the native graph object is returned.
        """
        native = self._inner.get_graph(config, xray=xray)
        try:
            from langchain_core.runnables.graph import Graph
        except Exception:  # pragma: no cover - langchain-core is optional
            return native
        description = native.to_dict()
        graph = Graph()
        nodes: dict[str, Any] = {}
        for item in description.get("nodes", ()):
            node_id = item["id"]
            nodes[node_id] = graph.add_node(None, id=node_id, metadata=item.get("metadata"))
        declared: list[tuple[Any, Any, bool]] = []
        for item in description.get("edges", ()):
            declared.append((item["source"], item["target"], bool(item.get("conditional"))))
        for item in description.get("conditional_edges", ()):
            if item.get("target") is not None:
                declared.append((item["source"], item["target"], True))
        # LangGraph keeps the declaration order; reconstruct it from the builder's call history.
        order = getattr(self, "_edge_order", None) or []
        rank = {edge: index for index, edge in enumerate(order)}
        fallback = len(rank)
        decorated = [
            (rank.get(edge, fallback + position), position, edge)
            for position, edge in enumerate(declared)
        ]
        decorated.sort(key=lambda item: (item[0], item[1]))
        declared = [edge for _, _, edge in decorated]
        for source, target, conditional in declared:
            if source not in nodes or target not in nodes:
                continue
            graph.add_edge(nodes[source], nodes[target], data=None, conditional=conditional)
        # Convenience kept from the native graph: the raw description LangGraph derives from the
        # same data via `draw_mermaid()` / `to_json()`.
        graph.to_dict = lambda: description  # type: ignore[method-assign]
        return graph

    # -- run methods ------------------------------------------------------------------------
    def _reject_resume_without_checkpointer(self, input: Any) -> None:
        """LangGraph refuses `Command(resume=...)` when the graph has no checkpointer."""
        try:
            from ..types import Command as _Command
        except Exception:  # pragma: no cover - types module always exists
            return
        if isinstance(input, _Command) and getattr(input, "resume", None) is not None:
            if not self._inner.has_checkpointer():
                raise RuntimeError("Cannot use Command(resume=...) without checkpointer")

    def invoke(self, input: Any, config: Any = None, *, durability: Any = None, **kwargs: Any) -> Any:
        self._reject_resume_without_checkpointer(input)
        call = _call_kwargs(config, durability, kwargs, _INVOKE_KWARGS)
        result = self._inner.invoke(input, **call)
        _dispatch_lifecycle(self._inner, config)
        return result

    async def ainvoke(self, input: Any, config: Any = None, *, durability: Any = None, **kwargs: Any) -> Any:
        if getattr(self._inner, "_native_async", False):
            call = _call_kwargs(config, durability, kwargs, _INVOKE_KWARGS)
            result = await self._inner.ainvoke(input, **call)
            _dispatch_lifecycle(self._inner, config)
            return result
        # Python drives Rust: the synchronous call runs on a worker thread so the loop stays free.
        return await asyncio.to_thread(
            self.invoke, input, config, durability=durability, **kwargs
        )

    def stream(self, input: Any, config: Any = None, *, durability: Any = None, **kwargs: Any) -> Iterator[Any]:
        self._reject_resume_without_checkpointer(input)
        call = _call_kwargs(config, durability, kwargs, _STREAM_KWARGS)
        try:
            chunks = self._inner.stream(input, **call)
        except TypeError as error:
            # The synchronous engine has no live sink; the parts still stream, just not live.
            if "live_sink" not in call or "live_sink" not in str(error):
                raise
            call.pop("live_sink")
            chunks = self._inner.stream(input, **call)
        _dispatch_lifecycle(self._inner, config)
        return chunks

    async def astream(
        self, input: Any, config: Any = None, *, durability: Any = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        if getattr(self._inner, "_native_async", False):
            call = _call_kwargs(config, durability, kwargs, _STREAM_KWARGS)
            try:
                stream = self._inner.astream(input, **call)
            except TypeError as error:
                if "live_sink" not in call or "live_sink" not in str(error):
                    raise
                call.pop("live_sink")
                stream = self._inner.astream(input, **call)
            from ..stream import AsyncStream

            _dispatch_lifecycle(self._inner, config)
            async for chunk in AsyncStream(stream):
                yield chunk
            return
        # Async generator over the synchronous stream, collected on a worker thread.
        chunks = await asyncio.to_thread(
            self.stream, input, config, durability=durability, **kwargs
        )
        for chunk in chunks:
            yield chunk

    def batch(
        self,
        inputs: Sequence[Any],
        config: Any = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        call = _call_kwargs(config, None, kwargs, _BATCH_KWARGS)
        return self._inner.batch(inputs, return_exceptions=return_exceptions, **call)

    async def abatch(
        self,
        inputs: Sequence[Any],
        config: Any = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        if getattr(self._inner, "_native_async", False):
            call = _call_kwargs(config, None, kwargs, _BATCH_KWARGS)
            return await self._inner.abatch(
                inputs, return_exceptions=return_exceptions, **call
            )
        return await asyncio.to_thread(
            self.batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # `langchain_core.Runnable` also defines these two, but its defaults reject compiled graphs;
    # the extension's v3 event stream is the one that must be used. The native method takes
    # `thread_id` (not a config dict), so the thread is read out of the config here.
    def stream_events(
        self,
        input: Any = None,
        config: Any = None,
        *,
        version: str = "v3",
        transformers: Any = None,
        **kwargs: Any,
    ) -> Any:
        from ..stream._events import stream_events as _stream_events

        thread_id = _thread_id_of(config)
        if thread_id is not None:
            kwargs.setdefault("thread_id", thread_id)
        return _stream_events(
            self,
            input,
            version=version,
            transformers=transformers,
            compile_transformers=getattr(self._inner, "stream_transformers", None) or None,
            **kwargs,
        )

    def astream_events(
        self,
        input: Any = None,
        config: Any = None,
        *,
        version: str = "v3",
        transformers: Any = None,
        **kwargs: Any,
    ) -> Any:
        # LangGraph returns an awaitable for v3 and an async generator for v2, so this must stay a
        # plain `def` that hands back whichever the shim produces.
        from ..stream._events import astream_events as _astream_events

        thread_id = _thread_id_of(config)
        if thread_id is not None:
            kwargs.setdefault("thread_id", thread_id)
        return _astream_events(
            self,
            input,
            version=version,
            transformers=transformers,
            compile_transformers=getattr(self._inner, "stream_transformers", None) or None,
            **kwargs,
        )


CompiledGraph = CompiledStateGraph

# `langgraph_rs.pregel.protocol` registers the compiled graphs while the package is still being
# imported, i.e. before this facade exists; register it again now that it does.
try:  # pragma: no cover - import cycle during startup
    from ..pregel.protocol import PregelProtocol

    PregelProtocol.register(CompiledStateGraph)
except Exception:  # pragma: no cover
    pass

__getattr__ = module_getattr(__name__, globals(), _MISSING)


# LangGraph's compiled graph *is* a `Pregel`; registering keeps `isinstance` true without
# shadowing the native methods the facade forwards.
_Pregel.register(CompiledStateGraph)
