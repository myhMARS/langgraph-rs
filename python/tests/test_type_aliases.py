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
        # LangGraph returns the new config; the state is read back with `get_state`.
        written = app.bulk_update_state(config, [[({"n": 42}, "node", None)]])
        self.assertEqual(sorted(written), ["configurable"])
        self.assertEqual(app.get_state(config).values["n"], 42)

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
            await app.aupdate_state(config, {"n": 5})
            updated = await app.aget_state(config)
            await app.abulk_update_state(
                config, [[{"values": {"n": 7}, "as_node": "node"}]]
            )
            bulk = await app.aget_state(config)
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
        self.assertEqual(last.status, "error")
        self.assertIn("Error: ValueError('kaboom')", last.content)
        self.assertIn("Please fix your mistakes.", last.content)

    def test_string_replaces_the_message(self):
        last = self._run("nope")["messages"][-1]
        self.assertEqual(last.content, "nope")
        self.assertEqual(last.status, "error")

    def test_callable_receives_the_error(self):
        seen = []

        def handler(error):
            seen.append(type(error).__name__)
            return f"handled {error}"

        last = self._run(handler)["messages"][-1]
        # LangGraph hands the handler the original exception object.
        self.assertEqual(last.content, "handled kaboom")
        self.assertEqual(seen, ["ValueError"])

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


class StoreSurfaceTest(unittest.TestCase):
    """`InMemoryStore` matches LangGraph's batch/async surface and item shape."""

    def test_async_twins(self):
        import asyncio

        store = langgraph_rs.store.memory.InMemoryStore()

        async def scenario():
            await store.aput(("m",), "k", {"v": 1})
            item = await store.aget(("m",), "k")
            self.assertEqual(item.value, {"v": 1})
            self.assertEqual([entry.key for entry in await store.asearch(("m",))], ["k"])
            self.assertIn(("m",), await store.alist_namespaces())
            await store.adelete(("m",), "k")
            self.assertIsNone(await store.aget(("m",), "k"))

        asyncio.run(scenario())

    def test_batch_dispatch(self):
        from langgraph_rs.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

        store = langgraph_rs.store.memory.InMemoryStore()
        store.put(("a",), "k", {"v": 1})
        results = store.batch(
            [
                GetOp(namespace=("a",), key="k"),
                PutOp(namespace=("b",), key="k2", value={"v": 2}),
                SearchOp(namespace_prefix=("a",)),
                ListNamespacesOp(),
            ]
        )
        self.assertEqual(results[0].value, {"v": 1})
        self.assertIsNone(results[1])
        self.assertEqual([item.key for item in results[2]], ["k"])
        # Reads see the state the batch started from, so the `PutOp` above is not listed yet.
        self.assertEqual(results[3], [("a",)])

    def test_item_types(self):
        from datetime import datetime

        store = langgraph_rs.store.memory.InMemoryStore()
        store.put(("n",), "k", {"v": 1})
        item = store.get(("n",), "k")
        self.assertEqual(item.namespace, ("n",))
        self.assertIsInstance(item.created_at, datetime)
        self.assertIsInstance(item.updated_at, datetime)

    def test_ttl_flags(self):
        store = langgraph_rs.store.memory.InMemoryStore()
        self.assertFalse(store.supports_ttl)
        self.assertIsNone(store.ttl_config)


class CheckpointerSurfaceTest(unittest.TestCase):
    """The top-level saver is the compatibility class, with LangGraph's async twins."""

    def test_top_level_saver_is_the_shim_class(self):
        from langgraph_rs.checkpoint.memory import InMemorySaver as ShimSaver

        self.assertIs(langgraph_rs.InMemorySaver, ShimSaver)
        self.assertIs(langgraph_rs.MemorySaver, ShimSaver)

    def test_async_twins_exist(self):
        saver = langgraph_rs.InMemorySaver()
        for name in ("aget_tuple", "alist", "aput", "aput_writes", "adelete_thread", "acopy_thread"):
            self.assertTrue(hasattr(saver, name), name)

    def test_copy_thread(self):
        saver = langgraph_rs.InMemorySaver()
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=saver)
        app.invoke({"value": 0}, {"configurable": {"thread_id": "copy-src"}})
        saver.copy_thread("copy-src", "copy-dst")
        self.assertEqual(
            app.get_state({"configurable": {"thread_id": "copy-dst"}}).values, {"value": 1}
        )

    def test_store_is_the_shim_class(self):
        from langgraph_rs.store.memory import InMemoryStore as ShimStore

        self.assertIs(langgraph_rs.InMemoryStore, ShimStore)


class ErrorTypeParityTest(unittest.TestCase):
    """Error types match LangGraph's, not bare `RuntimeError`."""

    def test_concurrent_write_raises_invalid_update(self):
        from langgraph_rs.errors import InvalidUpdateError

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge(langgraph_rs.START, "b")
        graph.add_edge("a", langgraph_rs.END)
        graph.add_edge("b", langgraph_rs.END)
        with self.assertRaises(InvalidUpdateError) as raised:
            graph.compile().invoke({"value": 0})
        self.assertIn("At key 'value'", str(raised.exception))

    def test_command_on_a_fresh_thread_starts_the_run(self):
        def node(state):
            return {"value": 1, "seen": dict(state)}

        graph = langgraph_rs.StateGraph({"value": "last_value", "seen": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        # `resume` is ignored without a checkpoint; `update` becomes the input.
        self.assertEqual(
            app.invoke(langgraph_rs.Command(resume=5), {"configurable": {"thread_id": "fresh-1"}}),
            {"value": 1, "seen": {}},
        )
        self.assertEqual(
            app.invoke(
                langgraph_rs.Command(update={"value": 9}),
                {"configurable": {"thread_id": "fresh-2"}},
            ),
            {"value": 1, "seen": {"value": 9}},
        )


class InputCheckpointTest(unittest.TestCase):
    """The `source="input"` checkpoint holds no values, as in LangGraph."""

    def test_input_checkpoint_is_empty(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "input-checkpoint"}}
        app.invoke({"value": 0}, config)
        history = list(app.get_state_history(config))
        sources = [item.metadata["source"] for item in history]
        self.assertEqual(sources, ["loop", "loop", "loop", "input"])
        self.assertEqual(history[-1].values, {})
        self.assertEqual(history[0].values, {"value": 2})


class DeferAndCommandTest(unittest.TestCase):
    """`add_node(defer=True)` and `Command(update=..., goto=...)` follow LangGraph."""

    def test_defer_keyword_and_entry_node(self):
        graph = langgraph_rs.StateGraph({"log": "append", "value": "last_value"})
        graph.add_node("d", lambda state: {"value": 1}, defer=True)
        graph.add_node("e", lambda state: {"log": ["e"]})
        graph.add_edge(langgraph_rs.START, "d")
        graph.add_edge(langgraph_rs.START, "e")
        graph.add_edge("d", langgraph_rs.END)
        graph.add_edge("e", langgraph_rs.END)
        # The deferred node runs after the regular one and may have outgoing edges.
        self.assertEqual(
            graph.compile().invoke({"log": [], "value": 0}), {"log": ["e"], "value": 1}
        )

    def test_command_update_with_goto(self):
        def start_node(state):
            return langgraph_rs.Command(update={"value": 1}, goto="b")

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", start_node)
        graph.add_node("b", lambda state: {"value": (state.get("value") or 0) + 10})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("b", langgraph_rs.END)
        self.assertEqual(graph.compile().invoke({"value": 0}), {"value": 11})


class ToolInterruptTest(unittest.TestCase):
    """`interrupt()` inside a tool pauses the graph and resumes like LangGraph."""

    @staticmethod
    def _app():
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool
        from langgraph_rs.prebuilt import ToolNode

        @tool
        def ask(reason: str) -> str:
            """Ask the user."""
            answer = langgraph_rs.interrupt({"question": reason})
            return f"answer={answer}"

        def agent(state):
            last = state["messages"][-1]
            role = last.get("role") if isinstance(last, dict) else getattr(last, "type", None)
            if role == "tool":
                return {"messages": [AIMessage(content="done")]}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "ask", "args": {"reason": "why?"}, "id": "c1", "type": "tool_call"}
                        ],
                    )
                ]
            }

        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode([ask]))
        graph.add_edge(langgraph_rs.START, "agent")
        graph.add_conditional_edges(
            "agent",
            lambda state: "tools"
            if (
                state["messages"][-1].get("tool_calls")
                if isinstance(state["messages"][-1], dict)
                else getattr(state["messages"][-1], "tool_calls", None)
            )
            else langgraph_rs.END,
        )
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_interrupt_pauses_and_resumes(self):
        app = self._app()
        config = {"configurable": {"thread_id": "tool-interrupt"}}
        paused = app.invoke({"messages": [{"role": "user", "content": "go"}]}, config)
        self.assertEqual(
            [item.value for item in paused["__interrupt__"]], [{"question": "why?"}]
        )
        resumed = app.invoke(langgraph_rs.Command(resume="yes"), config)
        contents = [
            message.get("content") if isinstance(message, dict) else message.content
            for message in resumed["messages"]
        ]
        self.assertEqual(contents[-2:], ["answer=yes", "done"])


class InterruptSchemaTest(unittest.TestCase):
    """`interrupt(response_schema=...)` accepts a class and validates the resume value."""

    @staticmethod
    def _app():
        from typing import TypedDict

        class Answer(TypedDict):
            answer: str

        def ask(state):
            return {"value": langgraph_rs.interrupt({"q": "?"}, response_schema=Answer)}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("ask", ask)
        graph.add_edge(langgraph_rs.START, "ask")
        graph.add_edge("ask", langgraph_rs.END)
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_schema_is_a_class(self):
        app = self._app()
        config = {"configurable": {"thread_id": "schema-ok"}}
        paused = app.invoke({"value": None}, config)
        schema = paused["__interrupt__"][0].response_schema
        self.assertEqual(schema["title"], "Answer")
        resumed = app.invoke(langgraph_rs.Command(resume={"answer": "yes"}), config)
        self.assertEqual(resumed, {"value": {"answer": "yes"}})

    def test_invalid_resume_is_rejected(self):
        app = self._app()
        config = {"configurable": {"thread_id": "schema-bad"}}
        app.invoke({"value": None}, config)
        with self.assertRaises(Exception) as raised:
            app.invoke(langgraph_rs.Command(resume={"bad": 1}), config)
        # LangGraph raises pydantic's `ValidationError`; the engine reports the same problem as a
        # `RuntimeError` naming the missing property (documented difference).
        self.assertIn("answer", str(raised.exception))


class InvalidNodeOutputTest(unittest.TestCase):
    """A node returning a non-mapping raises `InvalidUpdateError`, as in LangGraph."""

    def test_bad_output(self):
        from langgraph_rs.errors import InvalidUpdateError

        def bad(state):
            return ["not-a-dict"]

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("b", bad)
        graph.add_edge(langgraph_rs.START, "b")
        graph.add_edge("b", langgraph_rs.END)
        with self.assertRaises(InvalidUpdateError) as raised:
            graph.compile().invoke({"value": 0})
        self.assertIn("Expected dict, got ['not-a-dict']", str(raised.exception))


class NodeExceptionPropagationTest(unittest.TestCase):
    """A failing node re-raises its original exception, as LangGraph does."""

    @staticmethod
    def _app(factory, **kwargs):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", factory, **kwargs)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        return graph.compile()

    def test_value_error_is_reraised(self):
        def boom(state):
            raise ValueError("kaboom")

        with self.assertRaises(ValueError) as raised:
            self._app(boom).invoke({"value": 0})
        self.assertEqual(str(raised.exception), "kaboom")

    def test_key_error_is_reraised(self):
        def boom(state):
            raise KeyError("missing")

        with self.assertRaises(KeyError):
            self._app(boom).invoke({"value": 0})

    def test_retries_keep_the_original_exception(self):
        attempts = []

        def boom(state):
            attempts.append(1)
            raise ValueError("kaboom")

        app = self._app(boom, retry_policy=langgraph_rs.RetryPolicy(max_attempts=2, initial_interval=0.01))
        with self.assertRaises(ValueError):
            app.invoke({"value": 0})
        self.assertEqual(len(attempts), 2)


class FailedTaskSnapshotTest(unittest.TestCase):
    """A failed run leaves the failing task's error on the snapshot, as in LangGraph."""

    def test_task_error_is_reported(self):
        def ok(state):
            return {"value": 1}

        def boom(state):
            raise ValueError("kaboom")

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", ok)
        graph.add_node("b", boom)
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "failed-run"}}
        with self.assertRaises(ValueError):
            app.invoke({"value": 0}, config)
        snapshot = app.get_state(config)
        self.assertEqual(snapshot.values, {"value": 1})
        self.assertEqual(snapshot.next, ("b",))
        self.assertEqual(snapshot.tasks[0].name, "b")
        self.assertEqual(snapshot.tasks[0].error, "ValueError('kaboom')")
        # The failure does not add a history entry.
        self.assertEqual(
            [item.metadata["step"] for item in app.get_state_history(config)], [1, 0, -1]
        )


class ToolErrorTypeFilterTest(unittest.TestCase):
    """`handle_tool_errors=<type or tuple>` only handles matching exceptions."""

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
        return graph.compile().invoke({"messages": [ai]})["messages"][-1]

    def test_matching_type_is_handled(self):
        last = self._run(ValueError)
        self.assertEqual(last.status, "error")
        self.assertIn("Error: ValueError('kaboom')", last.content)

    def test_tuple_of_types_is_handled(self):
        self.assertIn("Error: ValueError('kaboom')", self._run((ValueError, KeyError)).content)

    def test_non_matching_type_propagates(self):
        with self.assertRaises(ValueError):
            self._run(KeyError)


class GetConfigTest(unittest.TestCase):
    """`get_config()` inside a node mirrors LangGraph's RunnableConfig view."""

    def test_config_shape(self):
        from langgraph_rs.config import get_config

        seen = {}

        def node(state):
            seen.update(get_config())
            return {"value": 1}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        graph.compile().invoke(
            {"value": 0}, {"tags": ["t1"], "metadata": {"m": 1}}
        )
        self.assertEqual(sorted(seen), ["callbacks", "configurable", "metadata", "tags"])
        self.assertEqual(seen["tags"], ["t1"])
        metadata = seen["metadata"]
        self.assertEqual(metadata["m"], 1)
        self.assertEqual(metadata["langgraph_node"], "n")
        self.assertEqual(metadata["langgraph_step"], 1)
        self.assertIsInstance(metadata["langgraph_path"], tuple)
        self.assertEqual(metadata["langgraph_path"], ("__pregel_pull", "n"))
        self.assertIsInstance(metadata["langgraph_triggers"], tuple)
        self.assertEqual(metadata["langgraph_triggers"], ("branch:to:n",))
        self.assertTrue(metadata["langgraph_checkpoint_ns"].startswith("n:"))

    def test_runnable_config_is_exported(self):
        from langgraph_rs.config import RunnableConfig

        self.assertIsNotNone(RunnableConfig)


class ReactAgentHelperTest(unittest.TestCase):
    """`create_react_agent` keeps LangGraph's parameter names and message flow."""

    @staticmethod
    def _model():
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class Stub(BaseChatModel):
            replies: list
            index: int = 0

            @property
            def _llm_type(self):
                return "stub"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                reply = self.replies[min(self.index, len(self.replies) - 1)]
                self.index += 1
                return ChatResult(generations=[ChatGeneration(message=reply)])

        return Stub(
            replies=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="3"),
            ]
        )

    def test_signature_matches_langgraph(self):
        import inspect

        parameters = list(
            inspect.signature(langgraph_rs.prebuilt.create_react_agent).parameters
        )
        for name in (
            "model",
            "tools",
            "prompt",
            "response_format",
            "state_schema",
            "config_schema",
            "checkpointer",
            "store",
            "interrupt_before",
            "interrupt_after",
            "debug",
            "version",
            "name",
        ):
            self.assertIn(name, parameters)

    def test_agent_runs_tool_loop(self):
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        agent = langgraph_rs.prebuilt.create_react_agent(self._model(), [add])
        out = agent.invoke({"messages": [{"role": "user", "content": "1+2?"}]})
        contents = [
            message.get("content") if isinstance(message, dict) else message.content
            for message in out["messages"]
        ]
        self.assertEqual(contents[-2:], ["3", "3"])
        self.assertEqual(sorted(agent.get_graph().nodes), ["__end__", "__start__", "agent", "tools"])


class MessageObjectsTest(unittest.TestCase):
    """`add_messages` channels hand node code and callers LangChain message objects."""

    def test_messages_are_objects(self):
        from langchain_core.messages import AIMessage, HumanMessage

        seen = {}

        def node(state):
            seen["type"] = type(state["messages"][-1]).__name__
            # Attribute access is how LangGraph code reads messages.
            seen["content"] = state["messages"][-1].content
            return {"messages": [AIMessage(content="hello", id="m2")]}

        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        out = graph.compile().invoke({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(seen, {"type": "HumanMessage", "content": "hi"})
        self.assertIsInstance(out["messages"][0], HumanMessage)
        self.assertIsInstance(out["messages"][1], AIMessage)
        self.assertEqual([message.type for message in out["messages"]], ["human", "ai"])
        self.assertEqual([message.content for message in out["messages"]], ["hi", "hello"])

    def test_plain_channels_stay_plain(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        self.assertEqual(graph.compile().invoke({"value": 0}), {"value": 1})

    def test_switch_can_be_turned_off(self):
        def node(state):
            return {"messages": [{"role": "ai", "content": "hello", "id": "m2"}]}

        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        langgraph_rs._set_messages_objects(False)
        try:
            out = graph.compile().invoke({"messages": [{"role": "user", "content": "hi"}]})
            self.assertIsInstance(out["messages"][0], dict)
        finally:
            langgraph_rs._set_messages_objects(True)


class SaverSurfaceTest2(unittest.TestCase):
    """The rest of LangGraph's `BaseCheckpointSaver` surface."""

    @staticmethod
    def _saver():
        saver = langgraph_rs.InMemorySaver()
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": (state.get("value") or 0) + 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "surface"}}
        app.invoke({"value": 0}, config)
        return saver, config

    def test_get_returns_the_raw_checkpoint(self):
        saver, config = self._saver()
        checkpoint = saver.get(config)
        self.assertIsInstance(checkpoint, dict)
        for key in ("id", "channel_values", "channel_versions", "v", "ts"):
            self.assertIn(key, checkpoint)

    def test_delta_channel_history_shape(self):
        saver, config = self._saver()
        history = saver.get_delta_channel_history(config=config, channels=["value"])
        self.assertEqual(sorted(history), ["value"])
        self.assertEqual(sorted(history["value"]), ["seed", "writes"])
        self.assertEqual(history["value"]["seed"], 0)

    def test_config_specs_and_serde(self):
        saver, _ = self._saver()
        self.assertEqual(saver.config_specs, [])
        self.assertEqual(type(saver.serde).__name__, "JsonPlusSerializer")
        self.assertIs(saver.with_allowlist([("x",)]), saver)

    def test_unsupported_operations_raise(self):
        saver, _ = self._saver()
        with self.assertRaises(NotImplementedError):
            saver.prune(["surface"])
        with self.assertRaises(NotImplementedError):
            saver.delete_for_runs(["run"])


class UpdateStateShapeTest(unittest.TestCase):
    """`update_state` infers its author and returns the new config, as LangGraph does."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node("a", lambda state: {"value": 1, "log": ["a"]})
        graph.add_node("b", lambda state: {"value": 2, "log": ["b"]})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_infers_the_last_writer(self):
        app = self._app()
        config = {"configurable": {"thread_id": "infer"}}
        app.invoke({"value": 0, "log": []}, config)
        written = app.update_state(config, {"value": 99})
        # A config dict, not a snapshot.
        self.assertEqual(sorted(written), ["configurable"])
        snapshot = app.get_state(config)
        # `b` wrote last and has no outgoing edge besides END.
        self.assertEqual(snapshot.next, ())
        self.assertEqual(snapshot.values, {"value": 99, "log": ["a", "b"]})
        self.assertEqual(snapshot.metadata["source"], "update")

    def test_explicit_as_node_reschedules(self):
        app = self._app()
        config = {"configurable": {"thread_id": "as-node"}}
        app.invoke({"value": 0, "log": []}, config)
        app.update_state(config, {"value": 100}, as_node="a")
        self.assertEqual(app.get_state(config).next, ("b",))


class BulkUpdateStateShapeTest(unittest.TestCase):
    """`bulk_update_state` takes LangGraph's tuples and returns the new config."""

    @staticmethod
    def _app():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        return graph.compile(checkpointer=langgraph_rs.InMemorySaver())

    def test_tuple_supersteps(self):
        app = self._app()
        config = {"configurable": {"thread_id": "bulk-tuple"}}
        app.invoke({"value": 0}, config)
        written = app.bulk_update_state(config, [[({"value": 7}, "a", None)]])
        self.assertEqual(sorted(written), ["configurable"])
        snapshot = app.get_state(config)
        self.assertEqual(snapshot.values, {"value": 7})
        self.assertEqual(snapshot.next, ("b",))

    def test_dict_supersteps_still_work(self):
        app = self._app()
        config = {"configurable": {"thread_id": "bulk-dict"}}
        app.invoke({"value": 0}, config)
        app.bulk_update_state(config, [[{"values": {"value": 9}, "as_node": "a"}]])
        self.assertEqual(app.get_state(config).values, {"value": 9})


class RetryOnCallableTest(unittest.TestCase):
    """`RetryPolicy(retry_on=...)` accepts a callable, an exception type or a sequence."""

    def _run(self, policy):
        attempts = []

        def boom(state):
            attempts.append(1)
            raise ValueError("kaboom")

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("b", boom, retry_policy=policy)
        graph.add_edge(langgraph_rs.START, "b")
        graph.add_edge("b", langgraph_rs.END)
        with self.assertRaises(ValueError):
            graph.compile().invoke({"value": 0})
        return len(attempts)

    def test_callable_predicate_true_retries(self):
        self.assertEqual(
            self._run(
                langgraph_rs.RetryPolicy(
                    max_attempts=3, retry_on=lambda exc: isinstance(exc, ValueError)
                )
            ),
            3,
        )

    def test_callable_predicate_false_stops(self):
        self.assertEqual(
            self._run(langgraph_rs.RetryPolicy(max_attempts=3, retry_on=lambda exc: False)), 1
        )

    def test_exception_type_and_sequence(self):
        self.assertEqual(self._run(langgraph_rs.RetryPolicy(max_attempts=3, retry_on=ValueError)), 3)
        self.assertEqual(
            self._run(langgraph_rs.RetryPolicy(max_attempts=3, retry_on=[ValueError, KeyError])), 3
        )
        self.assertEqual(self._run(langgraph_rs.RetryPolicy(max_attempts=3, retry_on=KeyError)), 1)

    def test_repr_matches_langgraph(self):
        self.assertEqual(
            repr(langgraph_rs.RetryPolicy(retry_on=ValueError)),
            "RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, "
            "max_attempts=3, jitter=True, retry_on=<class 'ValueError'>)",
        )
        self.assertIn(
            "retry_on=[<class 'ValueError'>, <class 'KeyError'>]",
            repr(langgraph_rs.RetryPolicy(retry_on=[ValueError, KeyError])),
        )


class SubgraphDebugStreamTest(unittest.TestCase):
    """`stream_mode="debug", subgraphs=True` wraps the whole debug chunk."""

    @staticmethod
    def _app():
        child = langgraph_rs.StateGraph({"value": "last_value"})
        child.add_node("c", lambda state: {"value": 1})
        child.add_edge(langgraph_rs.START, "c")
        child.add_edge("c", langgraph_rs.END)
        parent = langgraph_rs.StateGraph({"value": "last_value"})
        parent.add_node("sub", child.compile())
        parent.add_edge(langgraph_rs.START, "sub")
        parent.add_edge("sub", langgraph_rs.END)
        return parent.compile()

    def test_chunks_keep_their_shape(self):
        chunks = list(self._app().stream({"value": 0}, stream_mode="debug", subgraphs=True))
        shapes = [(list(namespace), chunk.get("type")) for namespace, chunk in chunks]
        self.assertEqual(
            [kind for _, kind in shapes], ["task", "task", "task_result", "task_result"]
        )
        # Only the subgraph's own records carry a namespace.
        self.assertEqual(shapes[0][0], [])
        self.assertEqual(len(shapes[1][0]), 1)
        self.assertTrue(shapes[1][0][0].startswith("sub:"))
        # The wrapped chunk is the full debug record, not just its payload.
        for _, chunk in chunks:
            self.assertEqual(sorted(chunk), ["payload", "step", "timestamp", "type"])


from langgraph_rs.prebuilt import ToolRuntime as _ToolRuntime


class ToolRuntimeInjectionTest(unittest.TestCase):
    """A `runtime: ToolRuntime` parameter is hidden from the schema and injected filled."""

    @staticmethod
    def _tool():
        from langchain_core.tools import tool

        @tool
        def inspect_runtime(query: str, runtime: _ToolRuntime) -> str:
            """Read the runtime."""
            return (
                f"state={sorted(runtime.state)}"
                f"|cfg={bool(runtime.config)}"
                f"|writer={runtime.stream_writer is not None}"
            )

        return inspect_runtime

    def test_schema_hides_the_runtime(self):
        tool = self._tool()
        self.assertEqual(sorted(tool.tool_call_schema.model_json_schema()["properties"]), ["query"])

    def test_runtime_is_filled(self):
        from langchain_core.messages import AIMessage

        from langgraph_rs.prebuilt import ToolNode

        ai = AIMessage(
            content="",
            tool_calls=[{"name": "inspect_runtime", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}],
        )
        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("tools", ToolNode([self._tool()]))
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        content = graph.compile().invoke({"messages": [ai]})["messages"][-1].content
        self.assertIn("state=['messages']", content)
        self.assertIn("cfg=True", content)
        self.assertIn("writer=True", content)


class GraphValidationErrorTest(unittest.TestCase):
    """Graph-construction problems raise `ValueError` with LangGraph's wording."""

    def test_no_entry_point(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        with self.assertRaises(ValueError) as raised:
            graph.compile()
        self.assertEqual(
            str(raised.exception),
            "Graph must have an entrypoint: add at least one edge from START to another node",
        )

    def test_unknown_edge_target(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        graph.add_edge(langgraph_rs.START, "b")
        with self.assertRaises(ValueError) as raised:
            graph.compile()
        self.assertEqual(str(raised.exception), "Found edge ending at unknown node `b`")

    def test_unknown_branch_target(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_conditional_edges("a", lambda state: "missing", ["other"])
        with self.assertRaises(ValueError) as raised:
            graph.compile()
        self.assertEqual(
            str(raised.exception),
            "At 'a' node, 'condition' branch found unknown target 'other'",
        )

    def test_duplicate_node(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        graph.add_node("a", lambda state: {})
        with self.assertRaises(ValueError) as raised:
            graph.compile()
        self.assertEqual(str(raised.exception), "Node `a` already present.")


class DeprecationWarningTest(unittest.TestCase):
    """Deprecated helpers warn with LangGraph's category and wording."""

    def test_validation_node_warns(self):
        from langchain_core.tools import tool

        from langgraph_rs.warnings import LangGraphDeprecatedSinceV10

        @tool
        def t(a: int) -> int:
            """T."""
            return a

        with self.assertWarns(LangGraphDeprecatedSinceV10) as caught:
            langgraph_rs.prebuilt.ValidationNode([t])
        self.assertIn("ValidationNode is deprecated", str(caught.warning))

    def test_message_graph_warns(self):
        from langgraph_rs.warnings import LangGraphDeprecatedSinceV10

        with self.assertWarns(LangGraphDeprecatedSinceV10) as caught:
            langgraph_rs.MessageGraph()
        self.assertIn("MessageGraph is deprecated", str(caught.warning))
        self.assertIn("StateGraph with a `messages` key", str(caught.warning))


class CommandSurfaceTest(unittest.TestCase):
    """`Command` and the no-checkpointer paths behave like LangGraph."""

    def test_command_attributes(self):
        command = langgraph_rs.Command(resume=1, update={"a": 2}, goto="x")
        self.assertEqual(
            sorted(name for name in dir(command) if not name.startswith("_")),
            ["PARENT", "goto", "graph", "resume", "update"],
        )
        self.assertEqual(command.resume, 1)
        self.assertEqual(command.update, {"a": 2})
        self.assertEqual(command.goto, "x")
        self.assertIsNone(command.graph)
        self.assertEqual(langgraph_rs.Command.PARENT, "__parent__")

    def test_thread_id_without_checkpointer_is_ignored(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile()
        # LangGraph ignores the thread id when the graph cannot checkpoint.
        self.assertEqual(app.invoke({"value": 0}, {"configurable": {"thread_id": "t1"}}), {"value": 1})
        with self.assertRaises(ValueError) as raised:
            app.get_state({"configurable": {"thread_id": "t1"}})
        self.assertEqual(str(raised.exception), "No checkpointer set")

    def test_resume_without_checkpointer_is_rejected(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("ask", lambda state: {"value": langgraph_rs.interrupt("q")})
        graph.add_edge(langgraph_rs.START, "ask")
        graph.add_edge("ask", langgraph_rs.END)
        with self.assertRaises(RuntimeError) as raised:
            graph.compile().invoke(
                langgraph_rs.Command(resume=1), {"configurable": {"thread_id": "t2"}}
            )
        self.assertEqual(
            str(raised.exception), "Cannot use Command(resume=...) without checkpointer"
        )

    def test_cache_ttl_accepts_fractions(self):
        calls = []

        def node(state):
            calls.append(state["value"])
            return {"value": state["value"] + 1}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node, cache_policy=langgraph_rs.CachePolicy(ttl=0.2))
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile(cache=langgraph_rs.InMemoryCache())
        app.invoke({"value": 1})
        app.invoke({"value": 1})
        self.assertEqual(calls, [1])


class TypesSurfaceTest(unittest.TestCase):
    """`Send` / `Interrupt` / `TimeoutPolicy` expose LangGraph's attributes."""

    def test_send_attributes(self):
        send = langgraph_rs.Send("n", {"a": 1}, timeout=0.5)
        self.assertEqual(send.node, "n")
        self.assertEqual(send.arg, {"a": 1})
        # LangGraph normalises a plain number into a TimeoutPolicy.
        self.assertEqual(type(send.timeout).__name__, "TimeoutPolicy")
        self.assertEqual(send.timeout.run_timeout, 0.5)
        plain = langgraph_rs.Send("m")
        self.assertIsNone(plain.arg)
        self.assertIsNone(plain.timeout)

    def test_interrupt_namespace_helpers(self):
        interrupt = langgraph_rs.Interrupt("q")
        self.assertEqual(interrupt.interrupt_id, interrupt.id)
        first = langgraph_rs.Interrupt.from_ns("v", "ns")
        second = langgraph_rs.Interrupt.from_ns("v", "ns")
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.interrupt_id, first.id)

    def test_timeout_policy_repr(self):
        self.assertEqual(
            repr(langgraph_rs.TimeoutPolicy(run_timeout=0.5)),
            "TimeoutPolicy(run_timeout=0.5, idle_timeout=None, refresh_on='auto')",
        )


class BuilderHelpersTest(unittest.TestCase):
    """`add_sequence`, `set_node_defaults` and `validate` behave like LangGraph."""

    def test_add_sequence_chains_nodes(self):
        def first(state):
            return {"value": 1}

        def second(state):
            return {"value": 2}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_sequence([first, second])
        graph.add_edge(langgraph_rs.START, "first")
        graph.add_edge("second", langgraph_rs.END)
        self.assertEqual(graph.compile().invoke({"value": 0}), {"value": 2})

    def test_add_sequence_with_named_pairs(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_sequence([("a", lambda state: {"value": 1}), ("b", lambda state: {"value": 2})])
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("b", langgraph_rs.END)
        self.assertEqual(graph.compile().invoke({"value": 0}), {"value": 2})

    def test_validate_returns_self_and_rejects_missing_entry(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        with self.assertRaises(ValueError):
            graph.validate()
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        self.assertIs(graph.validate(), graph)

    def test_set_node_defaults_applies_retry_policy(self):
        attempts = []

        def boom(state):
            attempts.append(1)
            raise ValueError("x")

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.set_node_defaults(
            retry_policy=langgraph_rs.RetryPolicy(max_attempts=2, initial_interval=0.01)
        )
        graph.add_node("n", boom)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        with self.assertRaises(ValueError):
            graph.compile().invoke({"value": 0})
        self.assertEqual(len(attempts), 2)

    def test_store_index_attributes(self):
        plain = langgraph_rs.InMemoryStore()
        self.assertIsNone(plain.embeddings)
        self.assertIsNone(plain.index_config)

        def embed(texts):
            return [[1.0, 0.0] for _ in texts]

        store = langgraph_rs.InMemoryStore(index={"dims": 2, "embed": embed})
        self.assertEqual(store.index_config["dims"], 2)
        self.assertIs(store.embeddings, embed)


class MessagesModeClassTest(unittest.TestCase):
    """`stream_mode="messages"` yields a message (not a chunk) for non-streamed models."""

    @staticmethod
    def _app(content_calls=True):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class Stub(BaseChatModel):
            replies: list
            index: int = 0

            @property
            def _llm_type(self):
                return "stub"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                reply = self.replies[min(self.index, len(self.replies) - 1)]
                self.index += 1
                return ChatResult(generations=[ChatGeneration(message=reply)])

        model = Stub(replies=[AIMessage(content="hello")])

        def node(state):
            return {"messages": [model.invoke(state["messages"])]}

        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        return graph.compile()

    def test_complete_message_keeps_its_class(self):
        chunks = list(
            self._app().stream(
                {"messages": [{"role": "user", "content": "hi"}]}, stream_mode="messages"
            )
        )
        self.assertEqual([type(chunk).__name__ for chunk, _ in chunks], ["AIMessage"])
        self.assertEqual([chunk.content for chunk, _ in chunks], ["hello"])


class SaverWriteApiTest(unittest.TestCase):
    """`put` / `put_writes` behave like LangGraph's `BaseCheckpointSaver`."""

    @staticmethod
    def _app():
        saver = langgraph_rs.InMemorySaver()
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "writes"}}
        app.invoke({"value": 0}, config)
        return saver, config

    def test_put_keeps_the_checkpoint(self):
        saver, config = self._app()
        found = saver.get_tuple(config)
        written = saver.put(found.config, dict(found.checkpoint), dict(found.metadata), {})
        self.assertEqual(
            written["configurable"]["checkpoint_id"], found.checkpoint["id"]
        )
        self.assertEqual(len(list(saver.list(config))), 3)

    def test_put_writes_appends_pending_writes(self):
        saver, config = self._app()
        found = saver.get_tuple(config)
        saver.put_writes(found.config, [("value", 9)], "task-1")
        after = saver.get_tuple(config)
        self.assertEqual([(task, channel) for task, channel, _ in after.pending_writes], [("task-1", "value")])


class StateSnapshotTupleTest(unittest.TestCase):
    """`StateSnapshot` is LangGraph's named tuple, with the extension's helpers on top."""

    @staticmethod
    def _snapshot():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "tuple"}}
        app.invoke({"value": 0}, config)
        return app.get_state(config)

    def test_is_a_named_tuple(self):
        snapshot = self._snapshot()
        self.assertIsInstance(snapshot, tuple)
        self.assertEqual(len(snapshot), 8)
        self.assertEqual(
            snapshot._fields,
            ("values", "next", "config", "metadata", "created_at", "parent_config", "tasks", "interrupts"),
        )
        values, next_nodes = snapshot[0], snapshot[1]
        self.assertEqual(values, {"value": 1})
        self.assertEqual(next_nodes, ())

    def test_fields_and_helpers(self):
        snapshot = self._snapshot()
        self.assertEqual(snapshot.values, {"value": 1})
        self.assertEqual(snapshot.metadata["source"], "loop")
        self.assertFalse(snapshot.interrupted)
        self.assertEqual(snapshot.step, 1)
        self.assertEqual(snapshot.thread_id, "tuple")
        self.assertEqual(snapshot.checkpoint_ns, "")

    def test_mapping_style_still_works(self):
        snapshot = self._snapshot()
        self.assertEqual(snapshot["values"], {"value": 1})
        self.assertIn("values", snapshot)
        self.assertEqual(sorted(snapshot.keys()), sorted(snapshot._fields))
        self.assertEqual(snapshot.get("missing", "fallback"), "fallback")


class GraphDrawingOrderTest(unittest.TestCase):
    """`get_graph()` keeps LangGraph's declaration order for edges."""

    @staticmethod
    def _graph():
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_conditional_edges("a", lambda state: "b", ["b"])
        graph.add_edge("b", langgraph_rs.END)
        return graph.compile()

    def test_edge_order(self):
        drawn = self._graph().get_graph()
        self.assertEqual(
            [(edge.source, edge.target, edge.conditional) for edge in drawn.edges],
            [("__start__", "a", False), ("a", "b", True), ("b", "__end__", False)],
        )

    def test_mermaid_keeps_the_order(self):
        mermaid = self._graph().get_graph().draw_mermaid()
        self.assertIn("\t__start__ --> a;\n\ta -.-> b;\n\tb --> __end__;", mermaid)


class RuntimeInjectionTest(unittest.TestCase):
    """Nodes declaring `runtime: Runtime` receive one, like LangGraph."""

    def test_node_receives_runtime(self):
        from langgraph_rs.runtime import Runtime

        seen = {}

        def node(state, runtime: Runtime):
            seen["type"] = type(runtime).__name__
            seen["writer"] = callable(runtime.stream_writer)
            runtime.store.put(("ns",), "k", {"v": 2})
            seen["stored"] = runtime.store.get(("ns",), "k").value
            return {"value": 1}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile(store=langgraph_rs.InMemoryStore())
        self.assertEqual(app.invoke({"value": 0}), {"value": 1})
        self.assertEqual(seen["type"], "Runtime")
        self.assertTrue(seen["writer"])
        self.assertEqual(seen["stored"], {"v": 2})

    def test_plain_node_is_untouched(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        self.assertEqual(graph.compile().invoke({"value": 0}), {"value": 1})

    def test_async_node_receives_runtime(self):
        from langgraph_rs.runtime import Runtime

        seen = {}

        async def node(state, runtime: Runtime):
            seen["async"] = type(runtime).__name__
            return {"value": 2}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile()
        import asyncio

        self.assertEqual(asyncio.run(app.ainvoke({"value": 0})), {"value": 2})
        self.assertEqual(seen["async"], "Runtime")


class AsyncNodeInSyncGraphTest(unittest.TestCase):
    """LangGraph accepts async nodes in a sync builder and runs them under `ainvoke`."""

    @staticmethod
    def _app(node):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        return graph.compile()

    def test_async_node_runs_under_ainvoke(self):
        import asyncio

        async def node(state):
            return {"value": 1}

        self.assertEqual(
            asyncio.run(self._app(node).ainvoke({"value": 0})), {"value": 1}
        )

    def test_async_node_can_stream(self):
        import asyncio

        async def node(state):
            return {"value": 3}

        app = self._app(node)

        async def collect():
            return [chunk async for chunk in app.astream({"value": 0})]

        chunks = asyncio.run(collect())
        self.assertEqual(chunks[-1], {"value": 3})


class ContextSchemaTest(unittest.TestCase):
    """`context_schema` on the builder and `context=` at run time, like LangGraph."""

    @staticmethod
    def _app(seen):
        from dataclasses import dataclass

        from langgraph_rs.runtime import Runtime

        @dataclass
        class Ctx:
            user: str

        def node(state, runtime: Runtime):
            seen["context"] = runtime.context
            return {"value": 1}

        graph = langgraph_rs.StateGraph({"value": "last_value"}, context_schema=Ctx)
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        return graph.compile(), Ctx

    def test_context_object_reaches_the_node(self):
        seen = {}
        app, Ctx = self._app(seen)
        self.assertEqual(app.invoke({"value": 0}, context=Ctx(user="ann")), {"value": 1})
        self.assertEqual(seen["context"], Ctx(user="ann"))
        self.assertIs(type(seen["context"]), Ctx)

    def test_context_jsonschema_is_exposed(self):
        seen = {}
        app, _ = self._app(seen)
        schema = app.get_context_jsonschema()
        self.assertEqual(schema["properties"]["user"]["type"], "string")
        self.assertEqual(schema["required"], ["user"])


class PythonStateValuesTest(unittest.TestCase):
    """LangGraph state may hold arbitrary Python objects."""

    def test_dataclass_and_set_survive_a_run(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int

        def node(state):
            return {"tag": {"set", "frozenset"}}

        graph = langgraph_rs.StateGraph({"point": "last_value", "tag": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        out = graph.compile().invoke({"point": Point(1), "tag": None})
        self.assertEqual(out["point"], Point(1))
        self.assertIs(type(out["point"]), Point)
        self.assertEqual(out["tag"], {"set", "frozenset"})

    def test_objects_survive_a_checkpointer(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int

        graph = langgraph_rs.StateGraph({"point": "last_value"})
        graph.add_node("n", lambda state: {"point": state["point"]})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "objects"}}
        app.invoke({"point": Point(2)}, config)
        state = app.get_state(config)
        self.assertEqual(state.values["point"], Point(2))

    def test_same_object_keeps_one_placeholder(self):
        from dataclasses import dataclass

        @dataclass
        class Box:
            value: int

        box = Box(1)
        graph = langgraph_rs.StateGraph({"box": "last_value"})
        graph.add_node("n", lambda state: {"box": state["box"]})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile()
        first = app.invoke({"box": box})["box"]
        second = app.invoke({"box": box})["box"]
        self.assertIs(first, box)
        self.assertIs(second, box)

    def test_callables_are_still_rejected(self):
        graph = langgraph_rs.StateGraph({"fn": "last_value"})
        graph.add_node("n", lambda state: {})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        with self.assertRaises(TypeError):
            graph.compile().invoke({"fn": lambda: None})


class UnknownChannelTest(unittest.TestCase):
    """Keys the state does not declare are ignored, like LangGraph."""

    @staticmethod
    def _app(node):
        from typing import TypedDict

        class State(TypedDict, total=False):
            value: int

        graph = langgraph_rs.StateGraph(State)
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        return graph.compile()

    def test_unknown_input_keys_are_dropped(self):
        app = self._app(lambda state: {"value": 1})
        self.assertEqual(app.invoke({"value": 0, "typo": 5}), {"value": 1})

    def test_unknown_update_keys_are_dropped(self):
        app = self._app(lambda state: {"value": 1, "typo": 5})
        self.assertEqual(app.invoke({"value": 0}), {"value": 1})

    def test_python_objects_in_state_still_work(self):
        from dataclasses import dataclass
        from typing import TypedDict

        @dataclass
        class Point:
            x: int

        class State(TypedDict, total=False):
            point: object
            value: int

        graph = langgraph_rs.StateGraph(State)
        graph.add_node("n", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        out = graph.compile().invoke({"point": Point(4), "value": 0})
        self.assertEqual(out["point"], Point(4))
        self.assertEqual(out["value"], 1)


class InputCheckpointValuesTest(unittest.TestCase):
    """The `source="input"` checkpoint holds the thread's values before the new turn."""

    def test_input_checkpoints_carry_previous_values(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "inputs"}}
        for payload in ({"value": 0}, {"value": 10}, {"value": 20}):
            app.invoke(payload, config)
        inputs = [
            (snapshot.metadata["step"], snapshot.values)
            for snapshot in app.get_state_history(config)
            if snapshot.metadata.get("source") == "input"
        ]
        self.assertEqual(inputs, [(5, {"value": 11}), (2, {"value": 1}), (-1, {})])

    def test_history_shape_matches_langgraph(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        config = {"configurable": {"thread_id": "shape"}}
        app.invoke({"value": 0}, config)
        app.invoke({"value": 10}, config)
        rows = [
            (snapshot.metadata["step"], snapshot.metadata["source"], snapshot.values, snapshot.next)
            for snapshot in app.get_state_history(config)
        ]
        self.assertEqual(
            rows,
            [
                (4, "loop", {"value": 11}, ()),
                (3, "loop", {"value": 10}, ("a",)),
                (2, "input", {"value": 1}, ("__start__",)),
                (1, "loop", {"value": 1}, ()),
                (0, "loop", {"value": 0}, ("a",)),
                (-1, "input", {}, ("__start__",)),
            ],
        )


class ErrorShapeTest(unittest.TestCase):
    """Error texts and the empty-thread snapshot match LangGraph."""

    def test_invalid_node_output_message(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("bad", lambda state: "not a dict")
        graph.add_edge(langgraph_rs.START, "bad")
        graph.add_edge("bad", langgraph_rs.END)
        with self.assertRaises(Exception) as raised:
            graph.compile().invoke({"value": 0})
        lines = str(raised.exception).splitlines()
        self.assertEqual(lines[0], "Expected dict, got not a dict")
        # LangGraph appends its troubleshooting link with the node-return code.
        self.assertIn("INVALID_GRAPH_NODE_RETURN_VALUE", lines[1])

    def test_concurrent_update_code(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": 2})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge(langgraph_rs.START, "b")
        graph.add_edge("a", langgraph_rs.END)
        graph.add_edge("b", langgraph_rs.END)
        with self.assertRaises(Exception) as raised:
            graph.compile().invoke({"value": 0})
        self.assertIn("INVALID_CONCURRENT_GRAPH_UPDATE", str(raised.exception))

    def test_unknown_thread_echoes_the_requested_config(self):
        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", lambda state: {})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile(checkpointer=langgraph_rs.InMemorySaver())
        snapshot = app.get_state({"configurable": {"thread_id": "nope"}})
        self.assertEqual(snapshot.values, {})
        self.assertEqual(snapshot.config, {"configurable": {"thread_id": "nope"}})
        self.assertIsNone(snapshot.metadata)
        self.assertEqual(snapshot.next, ())


class StreamEventsOrderTest(unittest.TestCase):
    """`astream_events` emits the graph stream once per superstep, like LangGraph."""

    def test_graph_events_per_superstep(self):
        import asyncio

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": state["value"] + 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        app = graph.compile()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        events = [(item["event"], item["name"]) for item in asyncio.run(collect())]
        self.assertEqual(
            events,
            [
                ("on_chain_start", "LangGraph"),
                ("on_chain_start", "a"),
                ("on_chain_stream", "a"),
                ("on_chain_end", "a"),
                ("on_chain_stream", "LangGraph"),
                ("on_chain_start", "b"),
                ("on_chain_stream", "b"),
                ("on_chain_end", "b"),
                ("on_chain_stream", "LangGraph"),
                ("on_chain_end", "LangGraph"),
            ],
        )


class SubgraphEventNestingTest(unittest.TestCase):
    """Subgraph `astream_events` nest like LangGraph's."""

    @staticmethod
    def _app():
        child = langgraph_rs.StateGraph({"value": "last_value"})
        child.add_node("c", lambda state: {"value": 1})
        child.add_edge(langgraph_rs.START, "c")
        child.add_edge("c", langgraph_rs.END)
        parent = langgraph_rs.StateGraph({"value": "last_value"})
        parent.add_node("sub", child.compile())
        parent.add_edge(langgraph_rs.START, "sub")
        parent.add_edge("sub", langgraph_rs.END)
        return parent.compile()

    def test_nested_sequence(self):
        import asyncio

        app = self._app()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        events = [(item["event"], item["name"]) for item in asyncio.run(collect())]
        self.assertEqual(
            events,
            [
                ("on_chain_start", "LangGraph"),
                ("on_chain_start", "sub"),
                ("on_chain_start", "LangGraph"),
                ("on_chain_start", "c"),
                ("on_chain_stream", "c"),
                ("on_chain_end", "c"),
                ("on_chain_end", "LangGraph"),
                ("on_chain_stream", "sub"),
                ("on_chain_end", "sub"),
                ("on_chain_stream", "LangGraph"),
                ("on_chain_end", "LangGraph"),
            ],
        )

    def test_parent_depth_increases_for_child_nodes(self):
        import asyncio

        app = self._app()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        events = asyncio.run(collect())
        depths = {item["name"]: len(item["parent_ids"]) for item in events if item["name"] == "c"}
        self.assertEqual(depths, {"c": 3})

    def test_graph_chunk_carries_the_step_updates(self):
        import asyncio

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        chunks = [
            item["data"]["chunk"]
            for item in asyncio.run(collect())
            if item["event"] == "on_chain_stream" and item["name"] == "LangGraph"
        ]
        self.assertEqual(chunks, [{"a": {"value": 1}}])


class EventMetadataTest(unittest.TestCase):
    """Node events carry LangGraph's `ls_integration` / `langgraph_step` / node name."""

    def test_step_and_node_metadata(self):
        import asyncio

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": state["value"] + 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", langgraph_rs.END)
        app = graph.compile()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        events = asyncio.run(collect())
        node_events = [item for item in events if item["metadata"].get("langgraph_node")]
        self.assertEqual(
            [(item["metadata"]["langgraph_node"], item["metadata"]["langgraph_step"]) for item in node_events],
            [("a", 1), ("a", 1), ("a", 1), ("b", 2), ("b", 2), ("b", 2)],
        )
        for item in node_events:
            self.assertEqual(item["metadata"]["ls_integration"], "langgraph")


class EventTaskRunsTest(unittest.TestCase):
    """Parallel `Send`s to one node get one run each; metadata mirrors LangGraph's keys."""

    def test_parallel_sends_have_separate_runs(self):
        import asyncio

        def add(left, right):
            return (left or 0) + (right or 0)

        def fan(state):
            return [langgraph_rs.Send("worker", {"v": 1}), langgraph_rs.Send("worker", {"v": 2})]

        # A callable channel spec keeps the reducer explicit (this module uses postponed
        # annotations, so `Annotated` metadata would arrive as text).
        graph = langgraph_rs.StateGraph({"total": add, "v": "last_value"})
        graph.add_node("fan", lambda state: {})
        graph.add_node("worker", lambda state: {"total": state.get("v", 0)})
        graph.add_edge(langgraph_rs.START, "fan")
        graph.add_conditional_edges("fan", fan, ["worker"])
        graph.add_edge("worker", langgraph_rs.END)
        app = graph.compile()
        self.assertEqual(app.invoke({"total": 0}), {"total": 3})

        async def collect():
            return [event async for event in app.astream_events({"total": 0}, version="v2")]

        events = asyncio.run(collect())
        workers = [item for item in events if item["metadata"].get("langgraph_node") == "worker"]
        self.assertEqual(
            [item["event"] for item in workers],
            [
                "on_chain_start",
                "on_chain_stream",
                "on_chain_end",
                "on_chain_start",
                "on_chain_stream",
                "on_chain_end",
            ],
        )
        self.assertEqual(len({item["run_id"] for item in workers}), 2)
        self.assertEqual(
            len({item["metadata"]["langgraph_checkpoint_ns"] for item in workers}), 2
        )

    def test_metadata_keys(self):
        import asyncio

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "a")
        graph.add_edge("a", langgraph_rs.END)
        app = graph.compile()

        async def collect():
            return [event async for event in app.astream_events({"value": 0}, version="v2")]

        events = asyncio.run(collect())
        node_event = next(
            item for item in events if item["metadata"].get("langgraph_node") == "a"
        )
        self.assertEqual(
            sorted(node_event["metadata"]),
            [
                "langgraph_checkpoint_ns",
                "langgraph_node",
                "langgraph_path",
                "langgraph_step",
                "ls_integration",
            ],
        )
        self.assertEqual(node_event["metadata"]["langgraph_path"], ["__pregel_pull", "a"])
        self.assertTrue(node_event["metadata"]["langgraph_checkpoint_ns"].startswith("a:"))


class StoreTtlTest(unittest.TestCase):
    """`InMemoryStore` rejects a TTL and unknown batch ops, like LangGraph."""

    def test_ttl_is_rejected(self):
        store = langgraph_rs.InMemoryStore()
        self.assertFalse(store.supports_ttl)
        with self.assertRaises(NotImplementedError):
            store.put(("a",), "k", {"v": 1}, ttl=1)
        store.put(("a",), "k", {"v": 1})
        self.assertEqual(store.get(("a",), "k").value, {"v": 1})

    def test_unknown_batch_op_is_a_value_error(self):
        store = langgraph_rs.InMemoryStore()
        with self.assertRaises(ValueError) as raised:
            store.batch([{"namespace": ("a",), "key": "k", "value": {}}])
        self.assertIn("Unknown operation type", str(raised.exception))


class BaseCheckpointSaverTest(unittest.TestCase):
    """`BaseCheckpointSaver` carries LangGraph's protocol surface."""

    def test_members_match_langgraph(self):
        from langgraph_rs.checkpoint.base import BaseCheckpointSaver

        self.assertEqual(
            sorted(name for name in dir(BaseCheckpointSaver) if not name.startswith("_")),
            sorted(
                [
                    "acopy_thread",
                    "adelete_for_runs",
                    "adelete_thread",
                    "aget",
                    "aget_delta_channel_history",
                    "aget_tuple",
                    "alist",
                    "aprune",
                    "aput",
                    "aput_writes",
                    "config_specs",
                    "copy_thread",
                    "delete_for_runs",
                    "delete_thread",
                    "get",
                    "get_delta_channel_history",
                    "get_next_version",
                    "get_tuple",
                    "list",
                    "prune",
                    "put",
                    "put_writes",
                    "serde",
                    "with_allowlist",
                ]
            ),
        )

    def test_get_uses_get_tuple(self):
        from langgraph_rs.checkpoint.base import BaseCheckpointSaver

        class Tuple:
            checkpoint = {"id": "c1"}

        class Saver(BaseCheckpointSaver):
            def get_tuple(self, config):
                return Tuple()

        self.assertEqual(Saver().get({"configurable": {"thread_id": "t"}}), {"id": "c1"})
        # Without an implementation the protocol method propagates the failure, as in LangGraph.
        with self.assertRaises(NotImplementedError):
            BaseCheckpointSaver().get({})

    def test_protocol_methods_raise(self):
        from langgraph_rs.checkpoint.base import BaseCheckpointSaver

        saver = BaseCheckpointSaver()
        for call in (
            lambda: saver.get_tuple({}),
            lambda: saver.list(None),
            lambda: saver.put({}, {}, {}, {}),
            lambda: saver.put_writes({}, [], "task"),
            lambda: saver.delete_thread("t"),
            lambda: saver.copy_thread("a", "b"),
            lambda: saver.prune(["t"]),
        ):
            with self.assertRaises(NotImplementedError):
                call()

    def test_next_version_counts_up(self):
        from langgraph_rs.checkpoint.base import BaseCheckpointSaver

        saver = BaseCheckpointSaver()
        self.assertEqual(saver.get_next_version(None), 1)
        self.assertEqual(saver.get_next_version(4), 5)
        with self.assertRaises(NotImplementedError):
            saver.get_next_version("v1")


class BaseStoreProtocolTest(unittest.TestCase):
    """`BaseStore` derives everything from `batch`, like LangGraph."""

    @staticmethod
    def _store():
        from langgraph_rs.store.base import BaseStore

        class Memory(BaseStore):
            def __init__(self):
                self.data = {}

            def batch(self, ops):
                out = []
                for op in ops:
                    name = type(op).__name__
                    if name == "PutOp":
                        key = (tuple(op.namespace), op.key)
                        if op.value is None:
                            self.data.pop(key, None)
                        else:
                            self.data[key] = op.value
                        out.append(None)
                    elif name == "GetOp":
                        value = self.data.get((tuple(op.namespace), op.key))
                        out.append(None if value is None else {"value": value, "key": op.key})
                    elif name == "SearchOp":
                        prefix = tuple(op.namespace_prefix)
                        out.append(
                            [
                                {"value": value, "key": key}
                                for (namespace, key), value in self.data.items()
                                if namespace[: len(prefix)] == prefix
                            ]
                        )
                    else:
                        out.append([namespace for namespace, _ in self.data])
                return out

        return Memory()

    def test_members_match_langgraph(self):
        from langgraph_rs.store.base import BaseStore

        self.assertEqual(
            sorted(name for name in dir(BaseStore) if not name.startswith("_")),
            [
                "abatch",
                "adelete",
                "aget",
                "alist_namespaces",
                "aput",
                "asearch",
                "batch",
                "delete",
                "get",
                "list_namespaces",
                "put",
                "search",
                "supports_ttl",
                "ttl_config",
            ],
        )

    def test_sync_surface_from_batch(self):
        store = self._store()
        store.put(("a",), "k", {"v": 1})
        self.assertEqual(store.get(("a",), "k")["value"], {"v": 1})
        self.assertEqual([hit["key"] for hit in store.search(("a",))], ["k"])
        store.delete(("a",), "k")
        self.assertIsNone(store.get(("a",), "k"))

    def test_async_surface_delegates(self):
        import asyncio

        store = self._store()

        async def scenario():
            await store.aput(("b",), "k", {"v": 2})
            return await store.aget(("b",), "k")

        self.assertEqual(asyncio.run(scenario())["value"], {"v": 2})

    def test_ttl_requires_support(self):
        store = self._store()
        with self.assertRaises(NotImplementedError):
            store.put(("a",), "k", {"v": 1}, ttl=1)


class StoreItemShapeTest(unittest.TestCase):
    """Store items are LangGraph's `Item`, with UTC timestamps and `dict()`."""

    def test_item_type_and_fields(self):
        from datetime import datetime, timezone

        from langgraph_rs.store.base import Item

        store = langgraph_rs.InMemoryStore()
        store.put(("n", "sub"), "k", {"v": 1})
        item = store.get(("n", "sub"), "k")
        self.assertIs(type(item), Item)
        self.assertEqual(item.namespace, ("n", "sub"))
        self.assertEqual(item.key, "k")
        self.assertEqual(item.created_at.tzinfo, timezone.utc)
        hit = store.search(("n",))[0]
        self.assertIs(type(hit).__name__, "SearchItem")
        # `Item.dict()` has no score; `SearchItem.dict()` adds it, like LangGraph.
        self.assertEqual(
            sorted(hit.dict()),
            ["created_at", "key", "namespace", "score", "updated_at", "value"],
        )
        self.assertEqual(
            sorted(item.dict()),
            ["created_at", "key", "namespace", "updated_at", "value"],
        )

    def test_item_dict_serialises_timestamps(self):
        store = langgraph_rs.InMemoryStore()
        store.put(("n",), "k", {"v": 1})
        payload = store.get(("n",), "k").dict()
        self.assertEqual(payload["namespace"], ["n"])
        self.assertTrue(payload["created_at"].endswith("+00:00"))

    def test_op_classes_are_named_tuples(self):
        from langgraph_rs.store.base import GetOp, ListNamespacesOp, MatchCondition, PutOp, SearchOp

        for cls in (GetOp, PutOp, SearchOp, ListNamespacesOp, MatchCondition):
            self.assertTrue(issubclass(cls, tuple), cls)
        op = PutOp(("a",), "k", {"v": 1})
        self.assertEqual((op.namespace, op.key, op.value), (("a",), "k", {"v": 1}))
        namespace, key, value, index, ttl = op
        self.assertEqual(key, "k")


class ToolNodeRunnableTest(unittest.TestCase):
    """`ToolNode` / `ValidationNode` are LangChain `Runnable`s, as in LangGraph."""

    def test_tool_node_is_a_runnable(self):
        from langchain_core.runnables import Runnable
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search."""
            return f"hit:{query}"

        node = langgraph_rs.prebuilt.ToolNode([search])
        self.assertIsInstance(node, Runnable)
        self.assertEqual(node.get_name(), "tools")
        self.assertEqual(node.get_name(" suffix"), "tools suffix")
        output = node.invoke(
            {
                "messages": [
                    {
                        "role": "ai",
                        "content": "",
                        "tool_calls": [{"name": "search", "args": {"query": "x"}, "id": "1"}],
                    }
                ]
            }
        )
        self.assertEqual(output["messages"][0]["content"], "hit:x")

    def test_validation_node_is_a_runnable(self):
        from langchain_core.runnables import Runnable
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search."""
            return f"hit:{query}"

        with self.assertWarns(langgraph_rs.warnings.LangGraphDeprecatedSinceV10):
            node = langgraph_rs.prebuilt.ValidationNode([search])
        self.assertIsInstance(node, Runnable)
        self.assertEqual(node.get_name(), "validation")


class RuntimeSurfaceTest(unittest.TestCase):
    """`Runtime`, `RunControl` and the small type exports match LangGraph."""

    def test_runtime_helpers(self):
        from langgraph_rs.runtime import ExecutionInfo, RunControl, Runtime

        info = ExecutionInfo(task_id="t1")
        self.assertEqual(info.patch(task_id="t2").task_id, "t2")
        runtime = Runtime(context={"a": 1})
        self.assertEqual(runtime.override(context={"b": 2}).context, {"b": 2})
        self.assertEqual(runtime.merge(Runtime(store="s")).store, "s")
        with self.assertRaises(RuntimeError):
            Runtime().patch_execution_info(task_id="x")

    def test_run_control_drain(self):
        control = langgraph_rs.RunControl()
        self.assertFalse(control.drain_requested)
        self.assertIsNone(control.drain_reason)
        control.request_drain("shutdown")
        self.assertTrue(control.drain_requested)
        self.assertEqual(control.drain_reason, "shutdown")

    def test_type_exports(self):
        from langgraph_rs.channels.base import EmptyChannelError
        from langgraph_rs.types import BaseCheckpointSaver, CacheKey, DeltaChannelHistory, StateUpdate

        self.assertEqual(StateUpdate._fields, ("values", "as_node", "task_id"))
        self.assertEqual(CacheKey._fields, ("ns", "key", "ttl"))
        self.assertEqual(list(DeltaChannelHistory.__annotations__), ["writes", "seed"])
        self.assertEqual(EmptyChannelError.__name__, "EmptyChannelError")
        self.assertEqual(
            len([name for name in dir(BaseCheckpointSaver) if not name.startswith("_")]), 24
        )


class SmallSurfaceTest(unittest.TestCase):
    """Small LangGraph members verified against the reference package."""

    def test_timeout_policy_coerce(self):
        self.assertIsNone(langgraph_rs.TimeoutPolicy.coerce(None))
        policy = langgraph_rs.TimeoutPolicy.coerce(0.5)
        self.assertEqual(policy.run_timeout, 0.5)
        given = langgraph_rs.TimeoutPolicy(run_timeout=2.0, refresh_on="heartbeat")
        self.assertEqual(langgraph_rs.TimeoutPolicy.coerce(given).refresh_on, "heartbeat")

    def test_serializer_allowlist(self):
        from langgraph_rs.checkpoint.serde import JsonPlusSerializer

        serializer = JsonPlusSerializer()
        extended = serializer.with_msgpack_allowlist([("mod", "Type")])
        self.assertIsNot(extended, serializer)
        self.assertIs(serializer.with_msgpack_allowlist([]), serializer)

    def test_tool_node_class_attributes(self):
        self.assertEqual(langgraph_rs.prebuilt.ToolNode.name, "tools")
        self.assertIn("tools_by_name", dir(langgraph_rs.prebuilt.ToolNode))

    def test_delta_channel_history_shape(self):
        from langgraph_rs.checkpoint.base import DeltaChannelHistory

        self.assertEqual(list(DeltaChannelHistory.__annotations__), ["writes", "seed"])


class PregelAndChannelSurfaceTest(unittest.TestCase):
    """`Pregel` is the compiled graph's base class; `BaseChannel` carries its protocol."""

    def test_compiled_graph_is_a_pregel(self):
        from langgraph_rs.pregel import Pregel

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", lambda state: {"value": 1})
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        app = graph.compile()
        self.assertIsInstance(app, Pregel)
        self.assertEqual(app.invoke({"value": 0}), {"value": 1})
        self.assertEqual(len([n for n in dir(Pregel) if not n.startswith("_")]), 69)

    def test_base_channel_protocol(self):
        from langgraph_rs.channels import BaseChannel, LastValue

        channel = LastValue(int)
        self.assertFalse(channel.is_available())
        with self.assertRaises(Exception):
            channel.get()
        self.assertFalse(channel.update([1]))
        self.assertFalse(channel.consume())
        self.assertIs(type(channel.copy()), LastValue)
        for name in ("checkpoint", "from_checkpoint", "consume", "finish"):
            self.assertTrue(hasattr(BaseChannel, name), name)


class ToolOutputDeltaTest(unittest.TestCase):
    """`ToolRuntime.emit_output_delta` streams on the `tools` channel."""

    def test_delta_reaches_the_tools_stream(self):
        from langchain_core.tools import tool

        from langgraph_rs.prebuilt import ToolRuntime as _Runtime

        # The annotation is resolved from module globals at call time.
        globals()["ToolRuntime"] = _Runtime

        @tool
        def streamer(query: str, runtime: ToolRuntime) -> str:
            """Stream partial output."""
            runtime.emit_output_delta({"partial": query})
            return "done"

        graph = langgraph_rs.StateGraph({"messages": "add_messages"})
        graph.add_node("tools", langgraph_rs.prebuilt.ToolNode([streamer]))
        graph.add_edge(langgraph_rs.START, "tools")
        graph.add_edge("tools", langgraph_rs.END)
        payload = {
            "messages": [
                {
                    "role": "ai",
                    "content": "",
                    "tool_calls": [
                        {"name": "streamer", "args": {"query": "q"}, "id": "c1"}
                    ],
                }
            ]
        }
        events = list(graph.compile().stream(payload, stream_mode="tools"))
        self.assertEqual(
            [event["event"] for event in events],
            ["tool-started", "tool-output-delta", "tool-finished"],
        )
        self.assertEqual(events[1]["delta"], {"partial": "q"})
