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
