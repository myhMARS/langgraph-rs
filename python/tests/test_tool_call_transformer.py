"""`ToolCallTransformer` + `ToolCallStream`, and compile-time `transformers=` registration.

The transformer projects `tools` protocol events into handle objects; because the channel is
subscription-first, each case subscribes to `mux.extensions["tool_calls"]` before pushing events.
"""

from __future__ import annotations

import unittest

import langgraph_rs as rs
from langgraph_rs.prebuilt import ToolCallTransformer
from langgraph_rs.stream import StreamChannel, StreamTransformer
from langgraph_rs.stream._mux import StreamMux

try:
    import langgraph.prebuilt as lg_prebuilt
    from langgraph.stream._mux import StreamMux as LgStreamMux

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


def _tool_event(data, namespace=()):
    return {
        "type": "event",
        "method": "tools",
        "params": {"namespace": list(namespace), "timestamp": 0, "data": data},
    }


def _started(tool_call_id="c1", tool_name="search", tool_input=None):
    return _tool_event(
        {
            "event": "tool-started",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "input": tool_input,
        }
    )


def _delta(token, tool_call_id="c1"):
    return _tool_event(
        {"event": "tool-output-delta", "tool_call_id": tool_call_id, "delta": token}
    )


def _finished(output, tool_call_id="c1"):
    return _tool_event(
        {"event": "tool-finished", "tool_call_id": tool_call_id, "output": output}
    )


def _failed(message, tool_call_id="c1"):
    return _tool_event(
        {"event": "tool-error", "tool_call_id": tool_call_id, "message": message}
    )


def _run(transformer_cls, events, *, mux_cls=StreamMux, close=True):
    """Subscribe, push the events, then return the handles and their drained deltas."""
    transformer = transformer_cls()
    mux = mux_cls([transformer])
    cursor = iter(mux.extensions["tool_calls"])
    for event in events:
        mux.push(event)
    if close:
        mux.close()
    handles = list(cursor)
    deltas = [list(handle.output_deltas) for handle in handles]
    return handles, deltas


class ToolCallTransformerTest(unittest.TestCase):
    def _live_handle(self, transformer_cls=ToolCallTransformer, mux_cls=StreamMux):
        """Create a mux, subscribe, push `tool-started`, and return (mux, handle, deltas iterator).

        Deltas are only buffered for a subscribed channel, so the handle's `output_deltas` is
        subscribed before the delta events are pushed.
        """
        transformer = transformer_cls()
        mux = mux_cls([transformer])
        cursor = iter(mux.extensions["tool_calls"])
        mux.push(_started(tool_input={"q": "x"}))
        handle = next(cursor)
        deltas = iter(handle.output_deltas)
        return mux, handle, deltas

    def test_started_deltas_and_finish(self):
        mux, handle, deltas = self._live_handle()
        mux.push(_delta("he"))
        mux.push(_delta("llo"))
        mux.push(_finished("hello"))
        mux.close()
        self.assertEqual(list(deltas), ["he", "llo"])
        self.assertEqual(handle.tool_call_id, "c1")
        self.assertEqual(handle.tool_name, "search")
        self.assertEqual(handle.input, {"q": "x"})
        self.assertEqual(handle.output, "hello")
        self.assertIsNone(handle.error)
        self.assertTrue(handle.completed)

    def test_tool_error_marks_the_handle_failed(self):
        handles, _ = _run(ToolCallTransformer, [_started(), _failed("boom")])
        handle = handles[0]
        self.assertEqual(handle.error, "boom")
        self.assertIsNone(handle.output)
        self.assertTrue(handle.completed)
        self.assertIn("failed", repr(handle))

    def test_finalize_closes_in_flight_handles(self):
        mux, handle, _deltas = self._live_handle()
        mux.push(_delta("partial"))
        mux.close()
        self.assertTrue(handle.completed)
        self.assertIsNone(handle.output)

    def test_fail_propagates_the_error_to_active_handles(self):
        mux, handle, _deltas = self._live_handle()
        mux.fail(RuntimeError("graph failed"))
        self.assertTrue(handle.completed)
        self.assertEqual(handle.error, "graph failed")

    def test_events_are_passed_through(self):
        transformer = ToolCallTransformer()
        mux = StreamMux([transformer])
        log = iter(mux._events)
        cursor = iter(mux.extensions["tool_calls"])
        mux.push(_started())
        mux.close()
        list(cursor)
        self.assertEqual([event["method"] for event in log], ["tools"])

    def test_deeper_namespace_events_are_ignored(self):
        handles, _ = _run(
            ToolCallTransformer,
            [_tool_event({"event": "tool-started", "tool_call_id": "c9"}, namespace=["sub"])],
        )
        self.assertEqual(handles, [])

    def test_serialized_tool_message_output_is_unwrapped(self):
        serialized = {
            "type": "constructor",
            "id": ["langchain_core", "messages", "ToolMessage"],
            "kwargs": {"content": "raw result", "tool_call_id": "c1"},
        }
        handles, _ = _run(ToolCallTransformer, [_started(), _finished(serialized)])
        self.assertEqual(handles[0].output, "raw result")

    def test_unknown_events_are_ignored(self):
        handles, _ = _run(
            ToolCallTransformer,
            [_tool_event({"event": "tool-finished", "tool_call_id": "missing"})],
        )
        self.assertEqual(handles, [])


class CompileTimeTransformerTest(unittest.TestCase):
    def test_compile_time_transformers_reach_the_run(self):
        class Counter(StreamTransformer):
            def __init__(self, scope=()):
                super().__init__(scope)
                self.log = StreamChannel("counter")

            def init(self):
                return {"counter": self.log}

            def process(self, event):
                if event["method"] == "values":
                    self.log.push(event["params"]["data"].get("x"))
                return True

        graph = rs.StateGraph({"x": "last_value"})
        graph.add_node("node", lambda state: {"x": state["x"] + 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile(transformers=[Counter])
        with app.stream_events({"x": 0}, version="v3") as run:
            self.assertEqual(list(run.extensions["counter"]), [0, 1])

    def test_call_site_transformers_add_to_compile_time_ones(self):
        def counter(name):
            class Named(StreamTransformer):
                def __init__(self, scope=()):
                    super().__init__(scope)
                    self.log = StreamChannel(name)

                def init(self):
                    return {name: self.log}

                def process(self, event):
                    return True

            return Named

        graph = rs.StateGraph({"x": "last_value"})
        graph.add_node("node", lambda state: {"x": 1})
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile(transformers=[counter("from_compile")])
        with app.stream_events(
            {"x": 0}, version="v3", transformers=[counter("from_call")]
        ) as run:
            self.assertIn("from_compile", run.extensions)
            self.assertIn("from_call", run.extensions)
            run.abort()


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class ToolCallTransformerParityTest(unittest.TestCase):
    def test_projection_matches_langgraph(self):
        def run(transformer_cls, mux_cls):
            transformer = transformer_cls()
            mux = mux_cls([transformer])
            cursor = iter(mux.extensions["tool_calls"])
            mux.push(_started(tool_input={"q": "x"}))
            first = next(cursor)
            deltas = iter(first.output_deltas)
            mux.push(_delta("a"))
            mux.push(_delta("b"))
            mux.push(_finished("ab"))
            first_deltas = list(deltas)
            mux.push(_started(tool_call_id="c2", tool_name="lookup"))
            second = next(cursor)
            mux.push(_failed("nope", tool_call_id="c2"))
            mux.close()
            return [first, second], [first_deltas]

        ours, our_deltas = run(ToolCallTransformer, StreamMux)
        theirs, their_deltas = run(lg_prebuilt.ToolCallTransformer, LgStreamMux)
        self.assertEqual(
            [(handle.tool_call_id, handle.tool_name, handle.input, handle.output, handle.error, handle.completed) for handle in ours],
            [(handle.tool_call_id, handle.tool_name, handle.input, handle.output, handle.error, handle.completed) for handle in theirs],
        )
        self.assertEqual(our_deltas, their_deltas)


if __name__ == "__main__":
    unittest.main()