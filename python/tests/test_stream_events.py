"""`stream_events(version="v3")`: projections, output, interrupts and parity with LangGraph.

A run stream is caller-driven: iterating one projection pulls the graph forward, and projections are
single-consumer, so each case drives exactly one projection (or asks for `output`).
"""

from __future__ import annotations

import asyncio
import importlib
import typing
import unittest

import langgraph_rs as rs
from langgraph_rs.stream import StreamChannel, StreamTransformer
from langgraph_rs.stream.transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    TasksTransformer,
    UpdatesTransformer,
)

try:
    import langgraph.graph as lg_graph

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


class _State(typing.TypedDict):
    x: int
    note: str


INPUT = {"x": 0, "note": ""}


def _writer_for(module):
    if module is rs:
        return rs.get_stream_writer
    from langgraph.config import get_stream_writer

    return get_stream_writer


def _graph(module, *, interrupt=False):
    """Build the same graph with either engine's `StateGraph`."""
    state_schema = {"x": "last_value", "note": "last_value"} if module is rs else _State
    get_writer = _writer_for(module)

    def step(state):
        writer = get_writer()
        writer({"progress": 1})
        if interrupt:
            answer = module.interrupt({"question": "ok?"})
            return {"x": state["x"] + 1, "note": str(answer)}
        return {"x": state["x"] + 1, "note": "done"}

    graph = module.StateGraph(state_schema)
    graph.add_node("step", step)
    graph.set_entry_point("step")
    graph.set_finish_point("step")
    return graph


EXTRA = {
    "updates": UpdatesTransformer,
    "custom": CustomTransformer,
    "checkpoints": CheckpointsTransformer,
    "debug": DebugTransformer,
    "tasks": TasksTransformer,
}


def _drive(
    app, projection, input_state=INPUT, *, thread_id=None, transformers=None, module=rs
):
    transformer_cls = EXTRA.get(projection)
    if transformer_cls is not None and transformers is None:
        # Each engine's mux validates its own transformer classes.
        stream_module = importlib.import_module("langgraph.stream" if module is lg_graph else "langgraph_rs.stream")
        transformer_cls = getattr(stream_module.transformers, transformer_cls.__name__)
        transformers = [transformer_cls]
    with app.stream_events(
        input_state, version="v3", thread_id=thread_id, transformers=transformers
    ) as run:
        return list(getattr(run, projection))


class StreamEventsTest(unittest.TestCase):
    def test_values_projection(self):
        app = _graph(rs).compile()
        self.assertEqual(
            _drive(app, "values"),
            [{"x": 0, "note": ""}, {"x": 1, "note": "done"}],
        )

    def test_updates_projection(self):
        app = _graph(rs).compile()
        self.assertEqual(_drive(app, "updates"), [{"step": {"x": 1, "note": "done"}}])

    def test_custom_projection(self):
        app = _graph(rs).compile()
        self.assertEqual(_drive(app, "custom"), [{"progress": 1}])

    def test_tasks_projection(self):
        app = _graph(rs).compile()
        items = _drive(app, "tasks")
        self.assertTrue(any(item.get("name") == "step" for item in items))
        self.assertTrue(any("result" in item for item in items))

    def test_checkpoints_projection(self):
        app = _graph(rs).compile(checkpointer=rs.InMemorySaver())
        # Checkpoints are per thread, so the run needs a `thread_id`.
        items = _drive(app, "checkpoints", thread_id="ckpt")
        self.assertTrue(items)
        self.assertIn("values", items[-1])

    def test_default_projections_match_langgraph(self):
        app = _graph(rs).compile()
        with app.stream_events(INPUT, version="v3") as run:
            # LangGraph's defaults: `values`, `messages` plus the lifecycle/subgraph discovery
            # transformers (measured on LangGraph 1.2.22: `run.messages` exists by default).
            self.assertEqual(
                sorted(run.extensions), ["lifecycle", "messages", "subgraphs", "values"]
            )
            with self.assertRaises(AttributeError):
                run.updates
            run.abort()

    def test_output_and_extensions(self):
        app = _graph(rs).compile()
        with app.stream_events(INPUT, version="v3") as run:
            self.assertIn("values", run.extensions)
            self.assertIsInstance(run.extensions["values"], StreamChannel)
            self.assertEqual(run.output, {"x": 1, "note": "done"})
            self.assertFalse(run.interrupted)
            self.assertEqual(run.interrupts, [])

    def test_interrupt_is_reported(self):
        app = _graph(rs, interrupt=True).compile(checkpointer=rs.InMemorySaver())
        with app.stream_events(INPUT, version="v3", thread_id="t1") as run:
            self.assertTrue(run.interrupted)
            self.assertEqual(run.interrupts[0].value["question"], "ok?")
            # The node paused before writing, so the snapshot still holds the input.
            self.assertEqual(run.output, {"x": 0, "note": ""})

    def test_raw_protocol_events(self):
        app = _graph(rs).compile(checkpointer=rs.InMemorySaver())
        with app.stream_events(
            INPUT, version="v3", thread_id="raw", transformers=list(EXTRA.values())
        ) as run:
            events = list(run)
        methods = {event["method"] for event in events}
        # `tasks` events are folded into the lifecycle/subgraph projections, not the raw log.
        self.assertEqual(
            methods, {"values", "updates", "custom", "checkpoints", "debug"}
        )
        self.assertEqual([event["seq"] for event in events], sorted(event["seq"] for event in events))

    def test_missing_projection_reports_registered_names(self):
        app = _graph(rs).compile()
        with app.stream_events(INPUT, version="v3") as run:
            with self.assertRaises(AttributeError) as caught:
                run.does_not_exist
            self.assertIn("registered projections", str(caught.exception))
            run.abort()

    def test_version_and_owned_kwargs_are_rejected(self):
        app = _graph(rs).compile()
        with self.assertRaises(NotImplementedError):
            app.stream_events(INPUT, version="v2")
        with self.assertRaises(TypeError):
            app.stream_events(INPUT, stream_mode="values")

    def test_custom_transformer_contributes_a_projection(self):
        class Lengths(StreamTransformer):
            def __init__(self, scope=()):
                super().__init__(scope)
                self.log = StreamChannel("lengths")

            def init(self):
                return {"lengths": self.log}

            def process(self, event):
                if event["method"] == "values":
                    self.log.push(len(event["params"]["data"]))
                return True

        app = _graph(rs).compile()
        with app.stream_events(INPUT, version="v3", transformers=[Lengths]) as run:
            # Non-native projections live in `extensions`, like LangGraph.
            self.assertEqual(list(run.extensions["lengths"]), [2, 2])
            run.abort()

    def test_astream_events(self):
        async def step(state):
            writer = rs.get_stream_writer()
            writer({"progress": 1})
            return {"x": state["x"] + 1, "note": "done"}

        graph = rs.AsyncStateGraph({"x": "last_value", "note": "last_value"})
        graph.add_node("step", step)
        graph.set_entry_point("step")
        graph.set_finish_point("step")
        app = graph.compile()

        async def collect():
            run = await app.astream_events(INPUT)
            values = [item async for item in run.values]
            return values, await run.output()

        values, output = asyncio.run(collect())
        self.assertEqual(values, [{"x": 0, "note": ""}, {"x": 1, "note": "done"}])
        self.assertEqual(output, {"x": 1, "note": "done"})


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class StreamEventsParityTest(unittest.TestCase):
    def test_projections_match_langgraph(self):
        ours = _graph(rs).compile()
        theirs = _graph(lg_graph).compile()
        # 	asks payloads carry engine-generated ids and extra fields, so only the\n        # payload-free projections are compared.
        for projection in ("values", "updates", "custom"):
            with self.subTest(projection=projection):
                self.assertEqual(
                    _drive(ours, projection),
                    _drive(theirs, projection, module=lg_graph),
                )

    def test_output_matches_langgraph(self):
        with _graph(rs).compile().stream_events(INPUT, version="v3") as run:
            ours = run.output
        with _graph(lg_graph).compile().stream_events(INPUT, version="v3") as run:
            theirs = run.output
        self.assertEqual(ours, theirs)


if __name__ == "__main__":
    unittest.main()