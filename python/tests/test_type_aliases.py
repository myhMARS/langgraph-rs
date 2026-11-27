"""Type aliases and module surfaces that must match LangGraph exactly.

`StreamMode` / `All` are `Literal` aliases and `StreamWriter` is a callable type — code that uses
them in annotations or compares them with LangGraph's own aliases depends on that. The module
surfaces (`langgraph.func`, `langgraph.middleware`, jsonplus constants) are import-compatibility
contracts.
"""

from __future__ import annotations

import importlib
import unittest

import langgraph_rs
import langgraph_rs.func as rs_func
import langgraph_rs.types as rs_types

try:
    import langgraph.func as lg_func
    import langgraph.types as lg_types
    from langgraph.checkpoint.serde import jsonplus as lg_jsonplus

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


class TypeAliasTest(unittest.TestCase):
    def test_stream_mode_is_the_same_literal(self):
        self.assertEqual(
            str(rs_types.StreamMode),
            "typing.Literal['values', 'updates', 'checkpoints', 'tasks', 'debug', 'messages', 'custom']",
        )

    def test_all_is_the_wildcard_literal(self):
        self.assertEqual(str(rs_types.All), "typing.Literal['*']")

    def test_stream_writer_is_a_callable_type(self):
        import collections.abc

        self.assertEqual(getattr(rs_types.StreamWriter, "__origin__", None), collections.abc.Callable)

    def test_deprecated_kwargs_and_missing_exist(self):
        self.assertIsInstance(rs_types.DeprecatedKwargs, type)
        self.assertIsNotNone(rs_types.MISSING)


class FuncSurfaceTest(unittest.TestCase):
    def test_get_runnable_for_entrypoint_invokes_the_function(self):
        runner = rs_func.get_runnable_for_entrypoint(lambda value: value * 2)
        self.assertEqual(runner.invoke(21), 42)

    def test_type_vars_are_exported(self):
        for name in ("ContextT", "P", "R", "S", "T"):
            self.assertTrue(hasattr(rs_func, name), name)

    def test_middleware_module_imports(self):
        module = importlib.import_module("langgraph_rs.middleware")
        self.assertEqual(module.__all__, [])


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class LangGraphParityTest(unittest.TestCase):
    def test_aliases_match_langgraph(self):
        self.assertEqual(rs_types.StreamMode, lg_types.StreamMode)
        self.assertEqual(rs_types.All, lg_types.All)
        self.assertEqual(rs_types.StreamWriter, lg_types.StreamWriter)

    def test_func_surface_covers_langgraph(self):
        lg_public = {name for name in dir(lg_func) if not name.startswith("_")}
        rs_public = {name for name in dir(rs_func) if not name.startswith("_")}
        self.assertEqual(sorted(lg_public - rs_public), [])

    def test_jsonplus_extension_ids_match(self):
        from langgraph_rs.checkpoint.serde import jsonplus as rs_jsonplus

        names = [name for name in dir(lg_jsonplus) if name.startswith("EXT_")]
        self.assertTrue(names)
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(rs_jsonplus, name), getattr(lg_jsonplus, name))

    def test_deprecated_kwargs_annotations_match(self):
        self.assertEqual(
            dict(rs_types.DeprecatedKwargs.__annotations__),
            dict(lg_types.DeprecatedKwargs.__annotations__),
        )


if __name__ == "__main__":
    unittest.main()

class PregelPlaceholderTest(unittest.TestCase):
    """Internal engine classes stay importable and raise with guidance when used."""

    def test_internal_classes_are_importable(self):
        from langgraph_rs.pregel import (
            ChannelWrite,
            ChannelWriteEntry,
            PregelNode,
            SyncAsyncFuture,
        )
        from langgraph_rs.pregel._write import ChannelWrite as PrivateChannelWrite

        self.assertIs(ChannelWrite, PrivateChannelWrite)
        for name in (ChannelWrite, ChannelWriteEntry, PregelNode, SyncAsyncFuture):
            self.assertTrue(callable(name))

    def test_using_them_raises_with_a_hint(self):
        from langgraph_rs.pregel import ChannelWrite

        with self.assertRaises(NotImplementedError) as raised:
            ChannelWrite([])
        self.assertIn("StateGraph.compile()", str(raised.exception))


class PregelProtocolSurfaceTest(unittest.TestCase):
    """Protocol classes, TypeVars and debug helpers exist and behave like LangGraph."""

    def test_protocol_registration(self):
        from langgraph_rs.pregel.protocol import PregelProtocol

        graph = langgraph_rs.StateGraph({"messages": "messages"})
        graph.add_node("node", lambda state: {})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        self.assertIsInstance(graph.compile(), PregelProtocol)
        self.assertTrue(PregelProtocol.__abstractmethods__)

    def test_stream_protocol_holder(self):
        from langgraph_rs.pregel.protocol import StreamProtocol

        seen: list[int] = []
        holder = StreamProtocol(seen.append, {"values"})
        holder(1)
        self.assertEqual(seen, [1])
        self.assertEqual(holder.modes, {"values"})

    def test_typevars_are_exported(self):
        from langgraph_rs.channels.base import Checkpoint, Update, Value
        from langgraph_rs.managed.base import U, V
        from langgraph_rs.prebuilt.tool_node import ContextT, StateT

        for variable in (Checkpoint, Update, Value, U, V, ContextT, StateT):
            self.assertTrue(hasattr(variable, "__name__"))

    def test_pregel_task_is_a_named_tuple(self):
        from langgraph_rs.types import PregelTask

        task = PregelTask("id1", "node", ())
        self.assertEqual(task._fields, ("id", "name", "path", "error", "interrupts", "state", "result"))
        self.assertEqual((task.id, task.name, task.result), ("id1", "node", None))

    def test_debug_helpers_match_langgraph(self):
        from langgraph_rs.pregel.debug import (
            get_bolded_text,
            get_colored_text,
            is_multiple_channel_write,
            map_task_result_writes,
            rm_pregel_keys,
        )

        self.assertTrue(is_multiple_channel_write({"$writes": [1]}))
        self.assertFalse(is_multiple_channel_write(1))
        self.assertEqual(map_task_result_writes([("a", 1)]), {"a": 1})
        self.assertEqual(
            map_task_result_writes([("a", 1), ("a", 2)]), {"a": {"$writes": [1, 2]}}
        )
        self.assertEqual(
            rm_pregel_keys({"configurable": {"__pregel_x": 1, "thread_id": "t"}}),
            {"configurable": {"thread_id": "t"}},
        )
        self.assertEqual(get_bolded_text("x"), "\033[1mx\033[0m")
        self.assertEqual(get_colored_text("x", "red"), "\033[1;30;31mx\033[0m")

    def test_internal_module_reexports(self):
        from langgraph_rs.pregel._write import ChannelWrite, ChannelWriteEntry, ChannelWriteTupleEntry
        from langgraph_rs.pregel.main import NodeBuilder, Pregel

        for name in (ChannelWrite, ChannelWriteEntry, ChannelWriteTupleEntry, NodeBuilder, Pregel):
            self.assertTrue(callable(name))


class GraphCallbacksModuleTest(unittest.TestCase):
    """``langgraph.callbacks`` surface: events, handler, and the two config managers."""

    def test_events_match_langgraph_fields(self):
        import dataclasses

        from langgraph_rs.callbacks import GraphInterruptEvent, GraphResumeEvent

        self.assertEqual(
            [field.name for field in dataclasses.fields(GraphInterruptEvent)],
            ["run_id", "status", "checkpoint_id", "checkpoint_ns", "interrupts"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(GraphResumeEvent)],
            ["run_id", "status", "checkpoint_id", "checkpoint_ns"],
        )

    def test_sync_manager_dispatches_to_graph_handlers(self):
        from uuid import uuid4

        from langgraph_rs.callbacks import (
            GraphCallbackHandler,
            GraphInterruptEvent,
            get_sync_graph_callback_manager_for_config,
        )

        seen = []

        class Handler(GraphCallbackHandler):
            def on_interrupt(self, event):
                seen.append((event.status, event.checkpoint_id))

        run_id = uuid4()
        manager = get_sync_graph_callback_manager_for_config(
            {"callbacks": [Handler(), object()]}, run_id=run_id
        )
        # Only `GraphCallbackHandler` subclasses are kept.
        self.assertEqual(len(manager.handlers), 1)
        self.assertEqual(manager.run_id, run_id)
        manager.on_interrupt(GraphInterruptEvent(None, "pending", "cp1", (), ()))
        self.assertEqual(seen, [("pending", "cp1")])

    def test_async_manager_is_async(self):
        import asyncio

        from langgraph_rs.callbacks import (
            GraphCallbackHandler,
            GraphResumeEvent,
            get_async_graph_callback_manager_for_config,
        )

        seen = []

        class Handler(GraphCallbackHandler):
            def on_resume(self, event):
                seen.append(event.checkpoint_id)

        manager = get_async_graph_callback_manager_for_config({"callbacks": [Handler()]})
        self.assertTrue(manager.is_async)
        asyncio.run(manager.on_resume(GraphResumeEvent(None, "done", "cp2", ())))
        self.assertEqual(seen, ["cp2"])


class GraphLifecycleDispatchTest(unittest.TestCase):
    """The engine dispatches `on_interrupt` / `on_resume` to `GraphCallbackHandler`s."""

    @staticmethod
    def _handler(seen):
        from langgraph_rs.callbacks import GraphCallbackHandler

        class Handler(GraphCallbackHandler):
            def on_interrupt(self, event):
                seen.append(("interrupt", event.status, bool(event.checkpoint_id), len(event.interrupts)))

            def on_resume(self, event):
                seen.append(("resume", event.status, bool(event.checkpoint_id)))

        return Handler()

    @staticmethod
    def _graph(state_type, interrupt):
        graph = state_type({"answer": "last"})

        def node(state):
            if state.get("answer") is None:
                return {"answer": interrupt({"q": "ok?"})}
            return {}

        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_invoke_dispatches_interrupt_and_resume(self):
        seen = []
        app = self._graph(langgraph_rs.StateGraph, langgraph_rs.interrupt)
        config = {"configurable": {"thread_id": "life-1"}, "callbacks": [self._handler(seen)]}

        app.invoke({}, config=config)
        self.assertEqual(seen, [("interrupt", "pending", True, 1)])
        app.invoke(langgraph_rs.Command(resume="yes"), config=config)
        self.assertEqual(seen[-1], ("resume", "pending", True))

    def test_stream_dispatches_interrupt_and_resume(self):
        seen = []
        app = self._graph(langgraph_rs.StateGraph, langgraph_rs.interrupt)
        config = {"configurable": {"thread_id": "life-2"}, "callbacks": [self._handler(seen)]}

        list(app.stream({}, config=config, stream_mode="values", version="v2"))
        self.assertEqual(seen, [("interrupt", "pending", True, 1)])
        list(app.stream(langgraph_rs.Command(resume="yes"), config=config, stream_mode="values", version="v2"))
        self.assertEqual(seen[-1], ("resume", "pending", True))

    def test_ainvoke_dispatches_interrupt_and_resume(self):
        import asyncio

        seen = []
        app = self._graph(langgraph_rs.AsyncStateGraph, langgraph_rs.interrupt)
        config = {"configurable": {"thread_id": "life-3"}, "callbacks": [self._handler(seen)]}

        async def scenario():
            await app.ainvoke({}, config=config)
            await app.ainvoke(langgraph_rs.Command(resume="yes"), config=config)

        asyncio.run(scenario())
        self.assertEqual(seen[0], ("interrupt", "pending", True, 1))
        self.assertEqual(seen[-1], ("resume", "pending", True))

    def test_runs_without_handlers_are_unaffected(self):
        app = self._graph(langgraph_rs.StateGraph, langgraph_rs.interrupt)
        result = app.invoke({}, config={"configurable": {"thread_id": "life-4"}})
        self.assertIn("__interrupt__", result)


class CompiledGraphAsyncSurfaceTest(unittest.TestCase):
    """LangGraph's compiled graph exposes sync and async APIs on one object."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"messages": "messages"})
        graph.add_node("node", lambda state: {"messages": []})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile()

    def test_async_methods_exist_on_the_sync_graph(self):
        app = self._app()
        for name in ("ainvoke", "astream", "astream_events"):
            self.assertTrue(hasattr(app, name), name)

    def test_ainvoke_returns_the_state(self):
        import asyncio

        result = asyncio.run(self._app().ainvoke({"messages": []}))
        self.assertEqual(sorted(result.keys()), ["messages"])

    def test_astream_is_an_async_generator(self):
        import asyncio
        import inspect

        stream = self._app().astream({"messages": []}, stream_mode="values", version="v2")
        self.assertTrue(inspect.isasyncgen(stream))

        async def count():
            return [part async for part in stream]

        self.assertEqual(len(asyncio.run(count())), 2)

    def test_astream_events_returns_a_run_stream(self):
        import asyncio

        # The facade builds the run stream in Python; the native graph only exposes `stream`.
        async def collect():
            app = self._app()
            run = await app.astream_events({"messages": []}, version="v3")
            return [value async for value in run.values]

        self.assertEqual(len(asyncio.run(collect())), 2)


class BatchSurfaceTest(unittest.TestCase):
    """``batch`` / ``abatch`` accept LangGraph's config list and ``return_exceptions``."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        graph.add_node("node", lambda state: {"n": (state.get("n") or 0) + 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile()

    def test_shared_config_runs_every_input(self):
        results = self._app().batch([{"messages": []}, {"messages": []}])
        self.assertEqual([result["n"] for result in results], [1, 1])

    def test_per_input_configs_are_used(self):
        results = self._app().batch(
            [{"n": 10}, {"n": 20}], config=[{"configurable": {}}, {"configurable": {}}]
        )
        self.assertEqual([result["n"] for result in results], [11, 21])

    def test_config_list_length_must_match(self):
        with self.assertRaises(TypeError):
            self._app().batch([{"n": 1}], config=[{"configurable": {}}, {"configurable": {}}])

    def test_return_exceptions_collects_failures(self):
        graph = langgraph_rs.StateGraph({"messages": "messages"})
        graph.add_node("boom", lambda state: 1 / 0)
        graph.set_entry_point("boom")
        graph.set_finish_point("boom")
        app = graph.compile()
        result = app.batch([{"messages": []}], return_exceptions=True)
        self.assertIsInstance(result[0], BaseException)

    def test_abatch_matches_batch(self):
        import asyncio

        app = self._app()
        results = asyncio.run(app.abatch([{"n": 1}, {"n": 2}]))
        self.assertEqual([result["n"] for result in results], [2, 3])


class StateApiSurfaceTest(unittest.TestCase):
    """Async state twins, `bulk_update_state`, `clear_cache` and `get_graph(config)`."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        graph.add_node("node", lambda state: {"n": (state.get("n") or 0) + 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        app.invoke({"messages": []}, thread_id="state-1")
        return app, {"configurable": {"thread_id": "state-1"}}

    def test_bulk_update_state_applies_supersteps(self):
        app, config = self._app()
        snapshot = app.bulk_update_state(config, [[{"values": {"n": 42}, "as_node": "node"}]])
        self.assertEqual(snapshot.values["n"], 42)

    def test_clear_cache_is_available(self):
        app, _ = self._app()
        self.assertIsNone(app.clear_cache())

    def test_get_graph_accepts_config_and_xray(self):
        app, config = self._app()
        self.assertIsNotNone(app.get_graph(config, xray=True))

    def test_async_state_methods(self):
        import asyncio

        app, config = self._app()

        async def scenario():
            state = await app.aget_state(config)
            history = await app.aget_state_history(config)
            updated = await app.aupdate_state(config, {"n": 5})
            bulk = await app.abulk_update_state(config, [[{"values": {"n": 7}, "as_node": "node"}]])
            await app.aclear_cache()
            return state.values["n"], len(history), updated.values["n"], bulk.values["n"]

        state_n, history_len, updated_n, bulk_n = asyncio.run(scenario())
        self.assertEqual((state_n, updated_n, bulk_n), (1, 5, 7))
        self.assertGreater(history_len, 0)


class GraphIntrospectionTest(unittest.TestCase):
    """`get_subgraphs`, `copy` and the channel/store attributes."""

    @staticmethod
    def _parent():
        child = langgraph_rs.StateGraph({"messages": "messages"})
        child.add_node("inner", lambda state: {})
        child.set_entry_point("inner")
        child.set_finish_point("inner")
        parent = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        parent.add_node("node", lambda state: {})
        parent.add_subgraph("child", child.compile())
        parent.set_entry_point("node")
        parent.set_finish_point("node")
        return parent.compile()

    def test_get_subgraphs_returns_the_embedded_graph(self):
        app = self._parent()
        found = list(app.get_subgraphs())
        self.assertEqual([name for name, _ in found], ["child"])
        self.assertEqual(type(found[0][1]).__name__, "CompiledGraph")

    def test_copy_returns_an_independent_object(self):
        app = self._parent()
        copied = app.copy()
        self.assertIsNot(copied, app)
        # `copy(update)` applies the given fields to the copy
        renamed = app.copy({"name": "renamed"})
        self.assertIsNot(renamed, app)

    def test_channels_and_store_attributes(self):
        app = self._parent()
        self.assertEqual(sorted(app.channels), ["messages", "n"])
        self.assertIsNone(app.store)


class RunnableFacadeTest(unittest.TestCase):
    """The compiled graph is a LangChain `Runnable`, like LangGraph's is."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        graph.add_node("node", lambda state: {"n": (state.get("n") or 0) + 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile()

    def test_compiled_graph_is_a_runnable(self):
        from langchain_core.runnables import Runnable

        app = self._app()
        self.assertIsInstance(app, Runnable)
        self.assertEqual(type(app).__name__, "CompiledStateGraph")

    def test_runnable_helpers_work(self):
        app = self._app()
        self.assertEqual(type(app.bind()).__name__, "RunnableBinding")
        self.assertEqual(type(app.with_config({"tags": ["t"]})).__name__, "RunnableBinding")
        self.assertEqual(type(app | (lambda value: value)).__name__, "RunnableSequence")
        self.assertIsNotNone(app.get_input_schema())

    def test_pipe_runs_the_graph(self):
        app = self._app() | (lambda state: state["n"])
        self.assertEqual(app.invoke({"n": 1}), 2)

    def test_native_methods_win_over_runnable_defaults(self):
        app = self._app()
        # `get_graph` must return the real graph (a langchain-core `Graph`), not the placeholder
        # langchain-core's `Runnable` default would build.
        graph = app.get_graph(xray=True)
        self.assertIn("node", graph.nodes)
        self.assertIn("node", graph.draw_mermaid())

    def test_stream_events_still_uses_the_v3_implementation(self):
        app = self._app()
        run = app.stream_events({"n": 1}, version="v3")
        self.assertEqual(len(list(run.values)), 2)

    def test_thread_id_keyword_is_forwarded(self):
        graph = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        graph.add_node("node", lambda state: {"n": 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        app.invoke({"messages": []}, thread_id="facade-1")
        state = app.get_state({"configurable": {"thread_id": "facade-1"}})
        self.assertEqual(state.values["n"], 1)

    def test_unsupported_run_keyword_is_rejected(self):
        app = self._app()
        with self.assertRaises(NotImplementedError):
            app.invoke({"messages": []}, interrupt_before=["node"])


class GraphIntrospectionAttributesTest(unittest.TestCase):
    """The engine-internal attributes LangGraph exposes, where they can be truthful."""

    @staticmethod
    def _app():
        from langgraph_rs.types import RetryPolicy

        graph = langgraph_rs.StateGraph({"messages": "messages", "n": "last"})
        graph.add_node("node", lambda state: {"n": 1}, retry_policy=RetryPolicy(max_attempts=3))
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile(interrupt_before=["node"], name="demo")

    def test_interrupt_sets_are_exposed(self):
        app = self._app()
        self.assertEqual(app.interrupt_before_nodes, ["node"])
        self.assertEqual(app.interrupt_after_nodes, [])

    def test_retry_policy_is_exposed_per_node(self):
        app = self._app()
        self.assertEqual(app.retry_policy["node"]["max_attempts"], 3)
        self.assertIn("backoff_factor", app.retry_policy["node"])

    def test_timeouts_and_cache_policies_exist(self):
        app = self._app()
        self.assertEqual(app.step_timeout, {})
        self.assertEqual(app.cache_policy, {})
        self.assertEqual(app.stream_transformers, [])

    def test_debug_and_name(self):
        app = self._app()
        self.assertIs(app.debug, False)
        self.assertEqual(app.get_name(), "demo")

    def test_async_graph_exposes_the_same_attributes(self):
        graph = langgraph_rs.AsyncStateGraph({"messages": "messages"})
        graph.add_node("node", lambda state: {})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile(interrupt_after=["node"])
        self.assertEqual(app.interrupt_after_nodes, ["node"])
        self.assertIs(app.debug, False)


class CustomChannelTest(unittest.TestCase):
    """A `BaseChannel` subclass that overrides `update`/`get` runs its own merge logic."""

    @staticmethod
    def _max_channel_class():
        from langgraph_rs.channels import BaseChannel

        class MaxValue(BaseChannel):
            __channel_spec__ = None

            def __init__(self):
                super().__init__()
                self._value = None

            def update(self, values):
                for value in values:
                    if value is None:
                        continue
                    self._value = value if self._value is None else max(self._value, value)
                return True

            def get(self):
                return self._value

        return MaxValue

    def test_custom_channel_merge(self):
        MaxValue = self._max_channel_class()
        graph = langgraph_rs.StateGraph({"best": MaxValue()})
        graph.add_node("a", lambda state: {"best": 5})
        graph.add_node("b", lambda state: {"best": 3})
        graph.add_node("c", lambda state: {"best": 9})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.set_finish_point("c")
        self.assertEqual(graph.compile().invoke({"best": None})["best"], 9)

    def test_builtin_channels_still_use_their_spec(self):
        from langgraph_rs.channels import LastValue

        graph = langgraph_rs.StateGraph({"value": LastValue(int)})
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        self.assertEqual(graph.compile().invoke({"value": 0})["value"], 1)


class BuilderAndSchemaAttributesTest(unittest.TestCase):
    """`builder`, `stream_mode`, `config`, `validate`, `get_context_jsonschema`."""

    @staticmethod
    def _app(**kwargs):
        graph = langgraph_rs.StateGraph({"messages": "messages"})
        graph.add_node("node", lambda state: {})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile(**kwargs)

    def test_builder_is_the_source_graph(self):
        app = self._app()
        self.assertEqual(type(app.builder).__name__, "StateGraph")

    def test_stream_defaults(self):
        app = self._app()
        self.assertEqual(app.stream_mode, "values")
        self.assertIs(app.stream_eager, False)
        self.assertEqual(app.config, {})

    def test_validate_returns_the_compiled_graph(self):
        app = self._app()
        self.assertIsNotNone(app.validate())

    def test_context_jsonschema(self):
        from typing import TypedDict

        class Context(TypedDict, total=False):
            user: str

        app = self._app(context_schema=Context)
        schema = app.get_context_jsonschema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("user", schema["properties"])
        self.assertIsNone(self._app().get_context_jsonschema())


class DurabilityPlacementTest(unittest.TestCase):
    """`durability` is honoured as the invoke keyword, like LangGraph."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"value": "last"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def _history(self, app, thread):
        return list(app.get_state_history({"configurable": {"thread_id": thread}}))

    def test_keyword_exit_writes_one_checkpoint(self):
        app = self._app()
        app.invoke({"value": 0}, {"configurable": {"thread_id": "dur-kw"}}, durability="exit")
        self.assertEqual(len(self._history(app, "dur-kw")), 1)

    def test_default_keeps_every_superstep(self):
        app = self._app()
        app.invoke({"value": 0}, {"configurable": {"thread_id": "dur-def"}})
        self.assertEqual(len(self._history(app, "dur-def")), 4)


class ParallelInterruptTest(unittest.TestCase):
    """A superstep that pauses several tasks surfaces every interrupt, like LangGraph."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"a": "last", "b": "last"})
        graph.add_node("na", lambda state: {"a": langgraph_rs.interrupt({"ask": "a"})})
        graph.add_node("nb", lambda state: {"b": langgraph_rs.interrupt({"ask": "b"})})
        graph.set_entry_point("na")
        graph.set_entry_point("nb")
        graph.set_finish_point("na")
        graph.set_finish_point("nb")
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_both_interrupts_are_surfaced(self):
        app = self._app()
        result = app.invoke({}, {"configurable": {"thread_id": "pi-1"}})
        payloads = sorted(item.value["ask"] for item in result["__interrupt__"])
        self.assertEqual(payloads, ["a", "b"])

    def test_resume_by_interrupt_id(self):
        app = self._app()
        config = {"configurable": {"thread_id": "pi-2"}}
        result = app.invoke({}, config)
        resume = {item.id: f"v{item.value['ask']}" for item in result["__interrupt__"]}
        resumed = app.invoke(langgraph_rs.Command(resume=resume), config)
        self.assertEqual((resumed["a"], resumed["b"]), ("va", "vb"))

    def test_each_interrupt_keeps_its_task(self):
        app = self._app()
        config = {"configurable": {"thread_id": "pi-3"}}
        app.invoke({}, config)
        snapshot = app.get_state(config)
        per_task = {task.name: len(task.interrupts) for task in snapshot.tasks}
        self.assertEqual(per_task, {"na": 1, "nb": 1})


class SubgraphUpdateChunkTest(unittest.TestCase):
    """A subgraph node reports its own `updates` chunk in the parent namespace."""

    @staticmethod
    def _app():
        sub = langgraph_rs.StateGraph({"value": "last"})
        sub.add_node("inner", lambda state: {"value": (state.get("value") or 0) + 1})
        sub.set_entry_point("inner")
        sub.set_finish_point("inner")
        parent = langgraph_rs.StateGraph({"value": "last"})
        parent.add_node("first", lambda state: {"value": 5})
        parent.add_node("sub", sub.compile())
        parent.add_node("after", lambda state: {"value": (state.get("value") or 0) + 10})
        parent.set_entry_point("first")
        parent.add_edge("first", "sub")
        parent.add_edge("sub", "after")
        parent.set_finish_point("after")
        return parent.compile()

    def test_parent_namespace_reports_the_subgraph_node(self):
        app = self._app()
        data = [part["data"] for part in app.stream({"value": 0}, stream_mode="updates", version="v2")]
        self.assertEqual(data, [{"first": {"value": 5}}, {"sub": {"value": 6}}, {"after": {"value": 16}}])

    def test_subgraph_namespace_reports_inner_nodes(self):
        app = self._app()
        parts = list(app.stream({"value": 0}, stream_mode="updates", subgraphs=True, version="v2"))
        named = [(tuple(item.split(":")[0] for item in part["ns"]), part["data"]) for part in parts]
        self.assertEqual(
            named,
            [
                ((), {"first": {"value": 5}}),
                (("sub",), {"inner": {"value": 6}}),
                ((), {"sub": {"value": 6}}),
                ((), {"after": {"value": 16}}),
            ],
        )


class LangChainObjectStateTest(unittest.TestCase):
    """LangChain objects (messages, documents) survive the state round trip."""

    @staticmethod
    def _graph(node):
        graph = langgraph_rs.StateGraph({"messages": "append", "value": "last"})
        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph

    def test_node_can_return_a_message_object(self):
        from langchain_core.messages import AIMessage

        app = self._graph(lambda state: {"messages": [AIMessage("hello")]}).compile()
        result = app.invoke({"messages": [], "value": 0})
        self.assertEqual(type(result["messages"][0]).__name__, "AIMessage")
        self.assertEqual(result["messages"][0].content, "hello")

    def test_message_objects_are_accepted_as_input(self):
        from langchain_core.messages import HumanMessage

        app = self._graph(lambda state: {"value": 1}).compile()
        result = app.invoke({"messages": [HumanMessage("hi")], "value": 0})
        self.assertEqual(type(result["messages"][0]).__name__, "HumanMessage")

    def test_tool_calls_survive_a_checkpoint(self):
        from langchain_core.messages import AIMessage

        app = self._graph(
            lambda state: {
                "messages": [
                    AIMessage("", tool_calls=[{"name": "t", "args": {"x": 1}, "id": "c1"}])
                ]
            }
        ).compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "msg-tool"}}
        app.invoke({"messages": [], "value": 0}, config)
        message = app.get_state(config).values["messages"][0]
        self.assertEqual(type(message).__name__, "AIMessage")
        self.assertEqual(message.tool_calls[0]["name"], "t")

    def test_plain_dicts_still_work(self):
        app = self._graph(
            lambda state: {"messages": [{"role": "assistant", "content": "hi"}]}
        ).compile()
        result = app.invoke({"messages": [], "value": 0})
        self.assertEqual(result["messages"][0], {"role": "assistant", "content": "hi"})


class MessageStreamShapeTest(unittest.TestCase):
    """`stream_mode="messages"` yields `(message_chunk, metadata)`, like LangGraph."""

    @staticmethod
    def _app():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        model = GenericFakeChatModel(messages=iter([AIMessage("Hello")]))
        graph = langgraph_rs.StateGraph({"messages": "append"})
        graph.add_node("node", lambda state: {"messages": [model.invoke(state["messages"])]})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile()

    def test_chunk_is_a_message_chunk(self):
        app = self._app()
        chunk, metadata = next(app.stream({"messages": []}, stream_mode="messages"))
        self.assertEqual(type(chunk).__name__, "AIMessageChunk")
        self.assertEqual(chunk.content, "Hello")
        self.assertEqual(metadata["langgraph_node"], "node")

    def test_run_stream_exposes_the_messages_projection(self):
        import asyncio

        app = self._app()

        async def collect():
            run = await app.astream_events({"messages": []}, version="v3")
            self.assertIn("messages", run.extensions)
            return len([item async for item in run.messages])

        self.assertGreaterEqual(asyncio.run(collect()), 1)

    def test_explicit_transformer_does_not_collide_with_the_default(self):
        from langgraph_rs.stream.transformers import MessagesTransformer

        app = self._app()
        run = app.stream_events({"messages": []}, version="v3", transformers=[MessagesTransformer])
        self.assertIn("messages", run.extensions)
        run.abort()


class MessageMetadataTest(unittest.TestCase):
    """Message metadata carries the node path and, after the first superstep, the triggers."""

    @staticmethod
    def _graph():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        model = GenericFakeChatModel(messages=iter([AIMessage("second")]))
        graph = langgraph_rs.StateGraph({"messages": "append"})
        graph.add_node("a", lambda state: {"messages": []})
        graph.add_node("b", lambda state: {"messages": [model.invoke(state["messages"])]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        return graph.compile()

    def test_path_is_reported(self):
        chunk, metadata = next(self._graph().stream({"messages": []}, stream_mode="messages"))
        self.assertEqual(metadata["langgraph_path"], ("__pregel_pull", "b"))

    def test_triggers_name_the_scheduling_channel(self):
        # LangGraph reports the trigger *channel* (`branch:to:<node>`), not the node that ran.
        chunk, metadata = next(self._graph().stream({"messages": []}, stream_mode="messages"))
        self.assertEqual(tuple(metadata["langgraph_triggers"]), ("branch:to:b",))

    def test_current_triggers_is_available_inside_a_node(self):
        seen = {}
        graph = langgraph_rs.StateGraph({"value": "last"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: (seen.update({"triggers": langgraph_rs.current_triggers()}) or {"value": 2}))
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        graph.compile().invoke({"value": 0})
        self.assertEqual(seen["triggers"], ["branch:to:b"])


class BulkUpdateAndHistoryParamsTest(unittest.TestCase):
    """`bulk_update_state` accepts LangGraph's tuples and history filters match LangGraph."""

    @staticmethod
    def _app(thread):
        graph = langgraph_rs.StateGraph({"value": "last", "note": "last"})
        graph.add_node("a", lambda state: {"value": (state.get("value") or 0) + 1})
        graph.add_node("b", lambda state: {"value": (state.get("value") or 0) + 10})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": thread}}
        app.invoke({"value": 0}, config)
        return app, config

    def test_bulk_update_accepts_langgraph_tuples(self):
        app, config = self._app("bulk-tuple")
        app.bulk_update_state(
            config,
            [
                [({"value": 100}, "a", None)],
                [({"note": "n1"}, "a", None), ({"value": 200}, "b", None)],
            ],
        )
        values = app.get_state(config).values
        self.assertEqual((values["value"], values["note"]), (200, "n1"))

    def test_bulk_update_writes_one_checkpoint_per_superstep(self):
        app, config = self._app("bulk-granularity")
        before = len(list(app.get_state_history(config)))
        app.bulk_update_state(
            config,
            [
                [({"value": 100}, "a", None)],
                [({"note": "n1"}, "a", None), ({"value": 200}, "b", None)],
            ],
        )
        self.assertEqual(len(list(app.get_state_history(config))) - before, 2)

    def test_bulk_update_unions_the_superstep_next(self):
        app, config = self._app("bulk-next")
        app.bulk_update_state(
            config,
            [
                [({"value": 100}, "a", None)],
                [({"note": "n1"}, "a", None), ({"value": 200}, "b", None)],
            ],
        )
        # `a` schedules `b`; `b` schedules nothing, so the union is `b`.
        self.assertEqual(list(app.get_state(config).next), ["b"])

    def test_bulk_update_keeps_the_dict_spelling(self):
        app, config = self._app("bulk-dict")
        app.bulk_update_state(config, [[{"values": {"value": 7}, "as_node": "a"}]])
        self.assertEqual(app.get_state(config).values["value"], 7)

    def test_bulk_update_rejects_unknown_shapes(self):
        app, config = self._app("bulk-bad")
        with self.assertRaises(TypeError):
            app.bulk_update_state(config, [[("not-a-dict",)]])

    def test_history_before_accepts_a_config(self):
        app, config = self._app("hist-before")
        history = list(app.get_state_history(config))
        older = list(app.get_state_history(config, before=history[1].config))
        self.assertEqual([snap.metadata["step"] for snap in older], [0, -1])

    def test_history_limit_and_filter(self):
        app, config = self._app("hist-limit")
        self.assertEqual(
            [snap.metadata["step"] for snap in app.get_state_history(config, limit=2)], [2, 1]
        )
        self.assertEqual(
            list(app.get_state_history(config, filter={"next": ("b",)})), []
        )


class LifecycleQueueTest(unittest.TestCase):
    """Pause/resume events travel through the run queue and carry a run id."""

    @staticmethod
    def _app(state_type, interrupt, node_fn=None):
        graph = state_type({"answer": "last"})

        def node(state):
            if state.get("answer") is None:
                return {"answer": interrupt({"q": "ok?"})}
            return {}

        graph.add_node("node", node_fn or node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    @staticmethod
    def _handler(seen):
        from langgraph_rs.callbacks import GraphCallbackHandler

        class Handler(GraphCallbackHandler):
            def on_interrupt(self, event):
                seen.append(("interrupt", event.status, event.run_id is not None))

            def on_resume(self, event):
                seen.append(("resume", event.status, event.run_id is not None))

        return Handler()

    def test_events_carry_a_run_id(self):
        seen = []
        app = self._app(langgraph_rs.StateGraph, langgraph_rs.interrupt)
        config = {"configurable": {"thread_id": "life-run-id"}, "callbacks": [self._handler(seen)]}
        app.invoke({}, config=config)
        self.assertEqual(seen, [("interrupt", "pending", True)])

    def test_queue_is_drained_between_runs(self):
        seen = []
        app = self._app(langgraph_rs.StateGraph, langgraph_rs.interrupt)
        config = {"configurable": {"thread_id": "life-drain"}, "callbacks": [self._handler(seen)]}
        app.invoke({}, config=config)
        first = list(seen)
        seen.clear()
        app.invoke(langgraph_rs.Command(resume="yes"), config=config)
        # A single resume event: the interrupt from the previous run is not replayed.
        self.assertEqual(first, [("interrupt", "pending", True)])
        self.assertEqual(seen, [("resume", "pending", True)])


class CallableErrorHandlerTest(unittest.TestCase):
    """LangGraph's callable `error_handler` runs through a synthesised `__error_handler__` node."""

    @staticmethod
    def _failing():
        def boom(state):
            raise ValueError("nope")

        return boom

    def test_handler_receives_the_state_and_ends_the_branch(self):
        graph = langgraph_rs.StateGraph({"value": "last_value", "note": "last_value"})
        graph.add_node(
            "boom", self._failing(), error_handler=lambda state: {"note": f"handled:{state}"}
        )
        graph.add_node("after", lambda state: {"value": 99})
        graph.set_entry_point("boom")
        graph.add_edge("boom", "after")
        graph.set_finish_point("after")
        result = graph.compile().invoke({"value": 0})
        # LangGraph's handler receives the state, and the failed node's edges do not run.
        self.assertEqual(result, {"value": 0, "note": "handled:{'value': 0}"})

    def test_handler_map_keeps_the_callable(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        handler = lambda state: {"value": 0}  # noqa: E731 - LangGraph users pass lambdas
        graph.add_node("boom", self._failing(), error_handler=handler)
        graph.set_entry_point("boom")
        graph.set_finish_point("boom")
        self.assertIs(graph.compile().node_error_handler_map["boom"], handler)


class IntrospectionSurfaceTest(unittest.TestCase):
    """The attribute surface LangGraph exposes for tooling (`input_channels`, ...)."""

    @staticmethod
    def _app(**schema):
        graph = langgraph_rs.StateGraph(schema or {"value": "last_value", "note": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        return graph.compile()

    def test_channel_attributes(self):
        app = self._app()
        self.assertEqual(app.input_channels, "__start__")
        self.assertEqual(app.output_channels, ["note", "value"])
        self.assertEqual(app.stream_channels, ["note", "value"])
        self.assertEqual(app.stream_channels_list, ["note", "value"])
        self.assertEqual(app.stream_channels_asis, ["note", "value"])

    def test_schema_to_mapper_keys_on_the_schema_class(self):
        class State(dict):
            __annotations__ = {"value": int}

        graph = langgraph_rs.StateGraph(State)
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        self.assertEqual(graph.compile().schema_to_mapper, {State: None})

    def test_repr_mimebundle_falls_back_to_text(self):
        bundle = self._app()._repr_mimebundle_()
        self.assertIn("text/plain", bundle)


class ConditionalPathMapFormsTest(unittest.TestCase):
    """LangGraph accepts a list of destinations as the conditional `path_map`."""

    def test_list_path_map(self):
        graph = langgraph_rs.StateGraph({"log": "append"})
        graph.add_node("work", lambda state: {"log": ["x"]})
        graph.add_conditional_edges(langgraph_rs.START, lambda state: ["work"], ["work"])
        graph.set_finish_point("work")
        self.assertEqual(graph.compile().invoke({"log": []}), {"log": ["x"]})

    def test_dict_path_map_still_works(self):
        graph = langgraph_rs.StateGraph({"log": "append"})
        graph.add_node("work", lambda state: {"log": ["y"]})
        graph.add_conditional_edges(langgraph_rs.START, lambda state: "k", {"k": "work"})
        graph.set_finish_point("work")
        self.assertEqual(graph.compile().invoke({"log": []}), {"log": ["y"]})


class CheckpointSaverProtocolTest(unittest.TestCase):
    """`InMemorySaver` implements LangGraph's read side of `BaseCheckpointSaver`."""

    @staticmethod
    def _saver_and_config(thread="saver-protocol"):
        from langgraph_rs.checkpoint.memory import InMemorySaver

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": thread}}
        graph.compile(checkpointer=saver).invoke({"value": 0}, config)
        return saver, config

    def test_get_tuple_returns_a_checkpoint_tuple(self):
        saver, config = self._saver_and_config()
        found = saver.get_tuple(config)
        self.assertEqual(type(found).__name__, "CheckpointTuple")
        # LangGraph's tuple is a NamedTuple: attribute access, and both halves present.
        self.assertIn("channel_values", found.checkpoint)
        self.assertEqual(found.metadata["source"], "loop")
        # Each call rebuilds the tuple from the stored checkpoint; the contents match.
        self.assertEqual(saver.get(config), found.checkpoint)

    def test_list_walks_the_history_with_metadata_filters(self):
        saver, config = self._saver_and_config()
        self.assertEqual(len(list(saver.list(config))), 3)
        self.assertEqual(len(list(saver.list(config, limit=2))), 2)
        self.assertEqual(
            [entry.metadata["source"] for entry in saver.list(config, filter={"source": "loop"})],
            ["loop", "loop"],
        )

    def test_delete_thread_and_version_counter(self):
        saver, config = self._saver_and_config()
        saver.delete_thread(config["configurable"]["thread_id"])
        self.assertIsNone(saver.get_tuple(config))
        version = saver.get_next_version(None, None)
        self.assertIsInstance(version, str)
        self.assertEqual(version.split(".")[0], "0" * 31 + "1")


class EventsVersionShapeTest(unittest.TestCase):
    """`astream_events` is awaitable for v3 and an async generator for v2 (as in LangGraph)."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        return graph.compile()

    def test_v2_yields_langchain_chain_events(self):
        import asyncio

        async def collect():
            stream = self._app().astream_events({"value": 0}, version="v2")
            self.assertTrue(hasattr(stream, "__aiter__"))
            return [(event["event"], event["name"]) async for event in stream]

        events = asyncio.run(collect())
        self.assertEqual(
            events,
            [
                ("on_chain_start", "LangGraph"),
                ("on_chain_start", "a"),
                ("on_chain_stream", "a"),
                ("on_chain_end", "a"),
                ("on_chain_stream", "LangGraph"),
                ("on_chain_end", "LangGraph"),
            ],
        )

    def test_v3_stays_awaitable(self):
        import asyncio

        async def collect():
            run = await self._app().astream_events({"value": 0}, version="v3")
            self.assertIn("values", run.extensions)
            return [value async for value in run.values]

        self.assertEqual(len(asyncio.run(collect())), 2)


class TimeoutPolicySemanticsTest(unittest.TestCase):
    """LangGraph's timeout rules: async nodes only, reported as `NodeTimeoutError`."""

    def test_sync_node_timeout_is_rejected_at_compile(self):
        import time

        def slow(state):
            time.sleep(0.05)
            return {"value": 1}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("slow", slow, timeout=0.01)
        graph.set_entry_point("slow")
        graph.set_finish_point("slow")
        with self.assertRaises(ValueError) as raised:
            graph.compile()
        self.assertIn("only supported for async nodes", str(raised.exception))

    def test_async_node_timeout_raises_node_timeout_error(self):
        import asyncio

        from langgraph_rs.errors import NodeTimeoutError

        async def scenario():
            async def slow(state):
                await asyncio.sleep(0.2)
                return {"value": 1}

            graph = langgraph_rs.AsyncStateGraph({"value": "last_value"})
            # Plain seconds are accepted, as in LangGraph.
            graph.add_node("slow", slow, timeout=0.02)
            graph.set_entry_point("slow")
            graph.set_finish_point("slow")
            with self.assertRaises(NodeTimeoutError):
                await graph.compile().ainvoke({"value": 0})

        asyncio.run(scenario())


class IdleTimeoutTest(unittest.TestCase):
    """`idle_timeout` fires only when the node stops reporting progress."""

    def test_idle_node_raises_node_timeout_error(self):
        import asyncio

        from langgraph_rs.errors import NodeTimeoutError

        async def scenario():
            async def silent(state):
                await asyncio.sleep(0.3)
                return {"value": 1}

            graph = langgraph_rs.AsyncStateGraph({"value": "last_value"})
            graph.add_node("silent", silent, timeout=langgraph_rs.TimeoutPolicy(idle_timeout=0.05))
            graph.set_entry_point("silent")
            graph.set_finish_point("silent")
            with self.assertRaises(NodeTimeoutError) as raised:
                await graph.compile().ainvoke({"value": 0})
            self.assertIn("idle timeout", str(raised.exception))

        asyncio.run(scenario())

    def test_progress_keeps_the_watchdog_quiet(self):
        import asyncio

        from langgraph_rs.config import get_stream_writer

        async def scenario():
            async def busy(state):
                writer = get_stream_writer()
                for step in range(5):
                    await asyncio.sleep(0.02)
                    writer({"tick": step})
                return {"value": 1}

            graph = langgraph_rs.AsyncStateGraph({"value": "last_value"})
            graph.add_node("busy", busy, timeout=langgraph_rs.TimeoutPolicy(idle_timeout=0.05))
            graph.set_entry_point("busy")
            graph.set_finish_point("busy")
            self.assertEqual(await graph.compile().ainvoke({"value": 0}), {"value": 1})

        asyncio.run(scenario())


class ToolNodeErrorHandlingTest(unittest.TestCase):
    """`ToolNode(handle_tool_errors=...)` follows LangGraph's four shapes."""

    @staticmethod
    def _run(handle):
        from langchain_core.messages import AIMessage

        def boom(query: str) -> str:
            raise ValueError("kaboom")

        ai = AIMessage(
            content="",
            tool_calls=[{"name": "boom", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}],
        )
        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("tools", langgraph_rs.prebuilt.ToolNode([boom], handle_tool_errors=handle))
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        return graph.compile().invoke({"messages": [{"role": "user", "content": "go"}, ai]})

    def test_default_message(self):
        last = self._run(True)["messages"][-1]
        self.assertEqual(last["status"], "error")
        self.assertIn("Error: ValueError('kaboom')", last["content"])
        self.assertIn("Please fix your mistakes.", last["content"])

    def test_string_replaces_the_message(self):
        last = self._run("nope")["messages"][-1]
        self.assertEqual(last["content"], "nope")
        self.assertEqual(last["status"], "error")

    def test_callable_receives_the_error(self):
        last = self._run(lambda error: f"handled {error}")["messages"][-1]
        # LangGraph hands the handler the original exception, so `str(exc)` is the message.
        self.assertEqual(last["content"], "handled kaboom")

    def test_false_propagates(self):
        with self.assertRaises(Exception):
            self._run(False)


class TimeoutDurationsTest(unittest.TestCase):
    """`timedelta` is accepted wherever a timeout is (LangGraph takes both forms)."""

    def test_timeout_policy_accepts_timedelta(self):
        from datetime import timedelta

        policy = langgraph_rs.TimeoutPolicy(run_timeout=timedelta(milliseconds=250))
        self.assertAlmostEqual(policy.run_timeout, 0.25, places=3)

    def test_send_accepts_timedelta(self):
        from datetime import timedelta

        langgraph_rs.types.Send("work", {"v": 1}, timeout=timedelta(milliseconds=50))

    def test_node_timeout_accepts_timedelta(self):
        import asyncio

        from datetime import timedelta

        from langgraph_rs.errors import NodeTimeoutError

        async def scenario():
            async def slow(state):
                await asyncio.sleep(0.2)
                return {"value": 1}

            graph = langgraph_rs.AsyncStateGraph({"value": "last_value"})
            graph.add_node("slow", slow, timeout=timedelta(milliseconds=20))
            graph.set_entry_point("slow")
            graph.set_finish_point("slow")
            with self.assertRaises(NodeTimeoutError):
                await graph.compile().ainvoke({"value": 0})

        asyncio.run(scenario())


class ValidationNodeTest(unittest.TestCase):
    """`ValidationNode` answers every call with a validated `ToolMessage`."""

    @staticmethod
    def _node():
        def search(query: str) -> str:
            """Search for a query."""
            return "results"

        def need_int(count: int) -> str:
            """Needs an int."""
            return "done"

        return langgraph_rs.prebuilt.ValidationNode([search, need_int])

    def test_valid_call_reports_success(self):
        out = self._node()({"messages": [{"role": "ai", "content": "", "tool_calls": [
            {"name": "search", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}]}]})
        self.assertEqual(out["messages"][0]["status"], "success")
        self.assertEqual(out["messages"][0]["content"], '{"query":"x"}')

    def test_invalid_arguments_are_reported(self):
        out = self._node()({"messages": [{"role": "ai", "content": "", "tool_calls": [
            {"name": "need_int", "args": {"count": "nope"}, "id": "c2", "type": "tool_call"}]}]})
        self.assertIn("validation error", out["messages"][0]["content"])


class ToolNodeSurfaceTest(unittest.TestCase):
    """`ToolNode` exposes LangGraph's constructor surface and the call wrappers."""

    @staticmethod
    def _tool():
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        return add

    @staticmethod
    def _ai(name="add", args=None):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args or {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}
            ],
        )

    def test_constructor_surface(self):
        node = langgraph_rs.prebuilt.ToolNode([self._tool()], name="caller", tags=["x"])
        self.assertEqual(node.name, "caller")
        self.assertEqual(node.get_name(), "caller")
        self.assertEqual(node.tags, ["x"])
        self.assertEqual(list(node.tools_by_name), ["add"])

    def test_wrap_tool_call_receives_the_request(self):
        seen = []

        def wrap(request, handler):
            seen.append((type(request).__name__, request.tool_call["name"]))
            return handler(request)

        node = langgraph_rs.prebuilt.ToolNode([self._tool()], wrap_tool_call=wrap)
        message = node({"messages": [self._ai()]})["messages"][0]
        self.assertEqual(seen, [("ToolCallRequest", "add")])
        self.assertEqual(message["content"], "3")
        self.assertEqual(message["status"], "success")

    def test_awrap_tool_call_runs_async(self):
        import asyncio

        seen = []

        async def awrap(request, handler):
            seen.append(request.tool_call["name"])
            return await handler(request)

        node = langgraph_rs.prebuilt.ToolNode([self._tool()], awrap_tool_call=awrap)
        output = asyncio.run(node.ainvoke({"messages": [self._ai(args={"a": 2, "b": 3})]}))
        self.assertEqual(seen, ["add"])
        self.assertEqual(output["messages"][0]["content"], "5")

    def test_invoke_delegates(self):
        node = langgraph_rs.prebuilt.ToolNode([self._tool()])
        self.assertEqual(node.invoke({"messages": [self._ai()]})["messages"][0]["content"], "3")


class TasksStreamShapeTest(unittest.TestCase):
    """`stream_mode="tasks"` payloads match LangGraph's `PregelTask` fields."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        return graph.compile()

    def test_payload_keys_and_types(self):
        chunks = list(self._app().stream({"value": 0}, stream_mode="tasks"))
        # Each task yields a start record (with `input`) and a finish record.
        starts = [chunk for chunk in chunks if "input" in chunk]
        self.assertEqual(len(starts), 2)
        self.assertEqual(sorted(starts[0]), ["id", "input", "name", "triggers"])
        self.assertEqual(starts[0]["input"], {"value": 0})
        self.assertEqual(starts[1]["input"], {"value": 1})
        self.assertIsInstance(starts[0]["triggers"], tuple)
        self.assertEqual(starts[0]["triggers"], ("branch:to:a",))


class SubgraphStreamShapeTest(unittest.TestCase):
    """`subgraphs=True` pairs every chunk with a namespace; the top level uses `()`."""

    @staticmethod
    def _app():
        child = langgraph_rs.StateGraph({"value": "last_value"})
        child.add_node("inner", lambda state: {"value": 7})
        child.add_edge(langgraph_rs.START, "inner")
        child.add_edge("inner", langgraph_rs.END)
        parent = langgraph_rs.StateGraph({"value": "last_value"})
        parent.add_node("sub", child.compile())
        parent.add_node("tail", lambda state: {"value": 9})
        parent.add_edge(langgraph_rs.START, "sub")
        parent.add_edge("sub", "tail")
        parent.add_edge("tail", langgraph_rs.END)
        return parent.compile()

    def test_updates_chunks_are_namespaced(self):
        chunks = list(self._app().stream({"value": 0}, stream_mode="updates", subgraphs=True))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks))
        namespace, payload = chunks[0]
        self.assertEqual(namespace[0].split(":")[0], "sub")
        self.assertEqual(payload, {"inner": {"value": 7}})
        self.assertEqual(chunks[1], ((), {"sub": {"value": 7}}))
        self.assertEqual(chunks[2], ((), {"tail": {"value": 9}}))

    def test_without_subgraphs_chunks_are_bare(self):
        chunks = list(self._app().stream({"value": 0}, stream_mode="updates"))
        self.assertIn({"sub": {"value": 7}}, chunks)


class MessagesStreamTokensTest(unittest.TestCase):
    """`stream_mode="messages"` streams model tokens, as LangGraph's handler does."""

    @staticmethod
    def _app():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello world")]))
        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("model", lambda state: {"messages": [model.invoke(state["messages"])]})
        graph.add_edge(langgraph_rs.START, "model")
        graph.add_edge("model", langgraph_rs.END)
        return graph.compile()

    def test_tokens_are_streamed_separately(self):
        chunks = list(
            self._app().stream(
                {"messages": [{"role": "user", "content": "hi"}]}, stream_mode="messages"
            )
        )
        self.assertEqual([chunk[0].content for chunk in chunks], ["hello", " ", "world"])
        metadata = chunks[0][1]
        # LangGraph's metadata keys for a model event inside a node.
        self.assertEqual(metadata["langgraph_node"], "model")
        self.assertEqual(metadata["langgraph_step"], 1)
        self.assertTrue(metadata["checkpoint_ns"].startswith("model:"))
        self.assertNotIn("tags", metadata)


class ReactAgentLoopTest(unittest.TestCase):
    """`create_react_agent` runs model -> tools -> model, as LangGraph does."""

    @staticmethod
    def _app():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        class _Fake(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = _Fake(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "double",
                                "args": {"x": 21},
                                "id": "c1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="the answer is 42"),
                ]
            )
        )

        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        return langgraph_rs.prebuilt.create_react_agent(model, [double])

    def test_full_loop(self):
        output = self._app().invoke({"messages": [{"role": "user", "content": "double 21"}]})
        types = [message.get("type") or message.get("role") for message in output["messages"]]
        # LangGraph: human -> ai(tool call) -> tool -> ai(answer).
        self.assertEqual(len(types), 4)
        self.assertEqual(types[1], "ai")
        self.assertEqual(output["messages"][1]["tool_calls"][0]["name"], "double")
        self.assertEqual(types[2], "tool")
        self.assertEqual(output["messages"][2]["content"], "42")
        self.assertEqual(output["messages"][3]["content"], "the answer is 42")


class StateSnapshotShapeTest(unittest.TestCase):
    """`StateSnapshot` fields use LangGraph's types (tuples for next/interrupts/tasks)."""

    @staticmethod
    def _snapshot():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "snapshot-shape"}}
        app.invoke({"value": 0}, config)
        return app.get_state(config)

    def test_container_types(self):
        snapshot = self._snapshot()
        self.assertIsInstance(snapshot.next, tuple)
        self.assertIsInstance(snapshot.interrupts, tuple)
        self.assertIsInstance(snapshot.tasks, tuple)
        self.assertIsInstance(snapshot.metadata, dict)

    def test_metadata_carries_parents(self):
        self.assertEqual(self._snapshot().metadata["parents"], {})


class SnapshotTaskObjectsTest(unittest.TestCase):
    """`snapshot.tasks` holds `PregelTask` named tuples and `interrupts` holds `Interrupt`."""

    @staticmethod
    def _snapshot():
        def ask(state):
            return {"value": langgraph_rs.interrupt("q?")}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("ask", ask)
        graph.add_edge(langgraph_rs.START, "ask")
        graph.add_edge("ask", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "task-objects"}}
        app.invoke({"value": 0}, config)
        return app.get_state(config)

    def test_task_is_a_named_tuple(self):
        task = self._snapshot().tasks[0]
        self.assertEqual(
            task._fields, ("id", "name", "path", "error", "interrupts", "state", "result")
        )
        self.assertEqual(task.name, "ask")
        self.assertEqual(task.path, ("__pregel_pull", "ask"))
        self.assertIsNone(task.error)

    def test_interrupt_is_an_object(self):
        interrupt = self._snapshot().interrupts[0]
        self.assertTrue(interrupt.id)
        self.assertEqual(interrupt.value, "q?")


class PolicySurfaceTest(unittest.TestCase):
    """`RetryPolicy` / `CachePolicy` / `TimeoutPolicy` match LangGraph's fields and reprs."""

    def test_retry_policy_takes_bool_jitter_and_exception_types(self):
        policy = langgraph_rs.RetryPolicy(jitter=False, retry_on=(ValueError, KeyError))
        text = repr(policy)
        self.assertIn("jitter=False", text)
        self.assertIn("'ValueError'", text)
        self.assertTrue(text.startswith("RetryPolicy(initial_interval="))

    def test_retry_policy_default_jitter_is_true(self):
        self.assertIn("jitter=True", repr(langgraph_rs.RetryPolicy()))

    def test_cache_policy_exposes_key_func(self):
        def key(state):
            return "k"

        policy = langgraph_rs.CachePolicy(ttl=30, key_func=key)
        self.assertIs(policy.key_func, key)
        self.assertIn("ttl=30", repr(policy))

    def test_timeout_policy_exposes_refresh_on(self):
        self.assertEqual(langgraph_rs.TimeoutPolicy().refresh_on, "auto")
        self.assertEqual(
            langgraph_rs.TimeoutPolicy(run_timeout=0.5, refresh_on="heartbeat").refresh_on,
            "heartbeat",
        )
