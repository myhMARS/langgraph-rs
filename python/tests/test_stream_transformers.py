"""Built-in stream transformers: scoped capture, values tracking, and LangGraph parity.

Channels are single-consumer and only buffer once subscribed, so each case subscribes (through the
mux's projection) before pushing events. A transformer must capture only its own protocol method, at
its own scope, in arrival order.
"""

from __future__ import annotations

import unittest

from langgraph_rs.stream import StreamChannel
from langgraph_rs.stream._mux import StreamMux
from langgraph_rs.stream.transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    TasksTransformer,
    UpdatesTransformer,
    ValuesTransformer,
)

try:
    import langgraph.stream as lg_stream
    from langgraph.stream._mux import StreamMux as LgStreamMux

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


def _event(method, data, namespace=()):
    return {
        "type": "event",
        "method": method,
        "params": {"namespace": list(namespace), "timestamp": 0, "data": data},
    }


CAPTURING = {
    "values": ValuesTransformer,
    "custom": CustomTransformer,
    "updates": UpdatesTransformer,
    "checkpoints": CheckpointsTransformer,
    "debug": DebugTransformer,
    "tasks": TasksTransformer,
}


def _collect(transformer, events, *, key=None, mux_cls=StreamMux, scope=()):
    """Build a mux, subscribe to the projection, push `events`, close, return the items."""
    mux = mux_cls([transformer])
    key = key or transformer.required_stream_modes[0]
    cursor = iter(mux.extensions[key])
    for event in events:
        mux.push(event)
    mux.close()
    return list(cursor)


class BuiltinTransformerTest(unittest.TestCase):
    def test_each_transformer_captures_only_its_method(self):
        for method, transformer_cls in CAPTURING.items():
            with self.subTest(method=method):
                items = _collect(
                    transformer_cls(),
                    [_event(method, {"payload": method}), _event("other", {"payload": "nope"})],
                )
                self.assertEqual(items, [{"payload": method}])

    def test_events_from_a_deeper_namespace_are_ignored(self):
        items = _collect(
            UpdatesTransformer(),
            [
                _event("updates", {"node": "inner"}, namespace=["sub"]),
                _event("updates", {"node": "root"}),
            ],
        )
        self.assertEqual(items, [{"node": "root"}])

    def test_scoped_transformer_captures_its_own_namespace(self):
        items = _collect(
            UpdatesTransformer(scope=("sub",)),
            [
                _event("updates", {"node": "inner"}, namespace=["sub"]),
                _event("updates", {"node": "root"}),
            ],
        )
        self.assertEqual(items, [{"node": "inner"}])

    def test_custom_payloads_keep_order(self):
        items = _collect(CustomTransformer(), [_event("custom", position) for position in range(3)])
        self.assertEqual(items, [0, 1, 2])

    def test_values_tracks_latest_and_interrupts(self):
        transformer = ValuesTransformer()
        interrupted = _event("values", {"step": 2})
        interrupted["params"]["interrupts"] = ({"question": "ok?"},)
        items = _collect(
            transformer, [_event("values", {"step": 1}), interrupted]
        )
        self.assertEqual(items, [{"step": 1}, {"step": 2}])
        self.assertEqual(transformer._latest, {"step": 2})
        self.assertTrue(transformer._interrupted)
        self.assertEqual(transformer._interrupts, [{"question": "ok?"}])

    def test_values_error_reflects_a_failed_run(self):
        transformer = ValuesTransformer()
        mux = StreamMux([transformer])
        error = RuntimeError("boom")
        mux.fail(error)
        self.assertIs(transformer.error, error)

    def test_suppressed_events_do_not_reach_the_main_log(self):
        mux = StreamMux([CustomTransformer()])
        cursor = iter(mux._events)
        channel_cursor = iter(mux.extensions["custom"])
        mux.push(_event("custom", "value"))
        mux.close()
        self.assertEqual(list(channel_cursor), ["value"])
        methods = [event["method"] for event in cursor]
        self.assertIn("custom", methods)

    def test_builtin_transformers_are_native_and_declare_modes(self):
        for method, transformer_cls in CAPTURING.items():
            with self.subTest(method=method):
                transformer = transformer_cls()
                self.assertTrue(transformer._native)
                self.assertEqual(transformer.required_stream_modes, (method,))

    def test_registration_merges_all_projections(self):
        mux = StreamMux([transformer_cls() for transformer_cls in CAPTURING.values()])
        self.assertEqual(sorted(mux.extensions), sorted(CAPTURING))
        self.assertEqual(mux.native_keys, set(CAPTURING))
        for key, transformer_cls in CAPTURING.items():
            self.assertIsInstance(mux.extensions[key], StreamChannel)
            self.assertIsInstance(mux.transformer_by_key(key), transformer_cls)


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class BuiltinTransformerParityTest(unittest.TestCase):
    def _run(self, module, mux_cls, transformer_cls, method, events):
        transformer = getattr(module.transformers, transformer_cls.__name__)()
        return _collect(transformer, events, key=method, mux_cls=mux_cls), transformer

    def test_capture_matches_langgraph(self):
        import langgraph_rs.stream as rs_stream

        for method, transformer_cls in CAPTURING.items():
            with self.subTest(method=method):
                events = [
                    _event(method, {"value": 1}),
                    _event(method, {"value": 2}, namespace=["deep"]),
                    _event("other", {"value": 3}),
                ]
                ours, _ = self._run(rs_stream, StreamMux, transformer_cls, method, events)
                theirs, _ = self._run(lg_stream, LgStreamMux, transformer_cls, method, events)
                self.assertEqual(ours, theirs)

    def test_values_tracking_matches_langgraph(self):
        import langgraph_rs.stream as rs_stream

        def run(module, mux_cls):
            transformer = module.transformers.ValuesTransformer()
            mux = mux_cls([transformer])
            cursor = iter(mux.extensions["values"])
            interrupted = _event("values", {"step": 1})
            interrupted["params"]["interrupts"] = ("x",)
            mux.push(interrupted)
            mux.close()
            return (
                list(cursor),
                transformer._latest,
                transformer._interrupted,
                list(transformer._interrupts),
            )

        self.assertEqual(run(rs_stream, StreamMux), run(lg_stream, LgStreamMux))


if __name__ == "__main__":
    unittest.main()