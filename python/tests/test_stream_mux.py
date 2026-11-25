"""Tests for `langgraph_rs.stream._mux.StreamMux` against LangGraph's implementation.

Both engines expose the mux as a private module, but its behaviour (registration order, projection
merging, `seq` assignment, channel auto-forwarding, close/fail cleanup) is what the public
transformers rely on, so it is compared directly.
"""

from __future__ import annotations

import unittest

from langgraph_rs.stream import ProtocolEvent, StreamChannel, StreamTransformer
from langgraph_rs.stream._mux import StreamMux

try:
    from langgraph.stream._mux import StreamMux as LgStreamMux

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


class NamedChannelTransformer(StreamTransformer):
    _native = True

    def __init__(self, scope=()):
        super().__init__(scope)
        self.channel = StreamChannel("named")

    def init(self):
        return {"named": self.channel}

    def process(self, event):
        self.channel.push(event["params"]["data"])
        return True


class UserChannelTransformer(StreamTransformer):
    def __init__(self, scope=()):
        super().__init__(scope)
        self.channel = StreamChannel("mine")

    def init(self):
        return {"mine": self.channel}

    def process(self, event):
        return True


class CountingTransformer(StreamTransformer):
    def __init__(self, scope=()):
        super().__init__(scope)
        self.seen = []

    def init(self):
        return {"counted": []}

    def process(self, event):
        self.seen.append(event)
        return True


def _event(method="values", data=None, namespace=()):
    return {
        "type": "event",
        "method": method,
        "params": {"namespace": list(namespace), "timestamp": 0, "data": data},
    }


class StreamMuxTest(unittest.TestCase):
    def test_extensions_merge_and_native_keys(self):
        mux = StreamMux([NamedChannelTransformer(), CountingTransformer()])
        self.assertEqual(sorted(mux.extensions), ["counted", "named"])
        self.assertEqual(mux.native_keys, {"named"})
        self.assertIsInstance(mux.transformer_by_key("named"), NamedChannelTransformer)

    def test_conflicting_projection_keys_raise(self):
        with self.assertRaises(ValueError):
            StreamMux([CountingTransformer(), CountingTransformer()])

    def test_init_must_return_a_dict(self):
        class Bad(StreamTransformer):
            def init(self):
                return []

            def process(self, event):
                return True

        with self.assertRaises(TypeError):
            StreamMux([Bad()])

    def test_async_transformer_rejected_on_the_sync_lane(self):
        class AsyncOnly(StreamTransformer):
            def init(self):
                return {}

            async def aprocess(self, event):
                return True

        with self.assertRaises(RuntimeError):
            StreamMux([AsyncOnly()], is_async=False)

    def test_events_get_monotonic_seq(self):
        mux = StreamMux([CountingTransformer()])
        cursor = iter(mux._events)
        for index in range(3):
            mux.push(_event(data=index))
        mux.close()
        self.assertEqual([event["seq"] for event in cursor], [1, 2, 3])

    def test_processing_can_suppress_events(self):
        class Suppressor(StreamTransformer):
            def init(self):
                return {}

            def process(self, event):
                return event["method"] != "updates"

        mux = StreamMux([Suppressor()])
        cursor = iter(mux._events)
        mux.push(_event(method="updates", data=1))
        mux.push(_event(method="values", data=2))
        mux.close()
        self.assertEqual([event["method"] for event in cursor], ["values"])

    def test_named_native_channel_forwards_as_its_own_method(self):
        mux = StreamMux([NamedChannelTransformer()])
        cursor = iter(mux._events)
        mux.push(_event(data="hello"))
        mux.close()
        forwarded = [event for event in cursor if event["method"] == "named"]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0]["params"]["data"], "hello")

    def test_user_channel_forwards_with_custom_prefix(self):
        mux = StreamMux([UserChannelTransformer()])
        cursor = iter(mux._events)
        mux.extensions["mine"].push("payload")
        mux.close()
        forwarded = [event for event in cursor if event["method"].startswith("custom:")]
        self.assertEqual(forwarded[0]["method"], "custom:mine")
        self.assertEqual(forwarded[0]["params"]["data"], "payload")

    def test_child_mux_requires_factories(self):
        mux = StreamMux([CountingTransformer()])
        with self.assertRaises(RuntimeError):
            mux._make_child(("sub",))

    def test_child_mux_shares_no_seq(self):
        mux = StreamMux(factories=[CountingTransformer])
        child = mux._make_child(("sub",))
        self.assertEqual(child.scope, ("sub",))
        event = _event()
        child.push(event)
        self.assertNotIn("seq", event)

    def test_close_closes_projections_and_calls_finalize(self):
        class Finalizer(StreamTransformer):
            def __init__(self, scope=()):
                super().__init__(scope)
                self.finalized = False
                self.channel = StreamChannel("fin")

            def init(self):
                return {"fin": self.channel}

            def process(self, event):
                return True

            def finalize(self):
                self.finalized = True

        transformer = Finalizer()
        mux = StreamMux([transformer])
        mux.close()
        self.assertTrue(transformer.finalized)
        self.assertTrue(transformer.channel._closed)

    def test_fail_propagates_errors_to_channels(self):
        transformer = CountingTransformer()
        mux = StreamMux([transformer])
        error = ValueError("boom")
        mux.fail(error)
        with self.assertRaises(ValueError):
            list(iter(mux._events))


def _counting_for(module):
    class Counting(module.StreamTransformer):
        def __init__(self, scope=()):
            super().__init__(scope)
            self.seen = []

        def init(self):
            return {"counted": []}

        def process(self, event):
            self.seen.append(event)
            return True

    return Counting


def _user_channel_for(module):
    class UserChannel(module.StreamTransformer):
        def __init__(self, scope=()):
            super().__init__(scope)
            self.channel = module.StreamChannel("mine")

        def init(self):
            return {"mine": self.channel}

        def process(self, event):
            return True

    return UserChannel


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class StreamMuxParityTest(unittest.TestCase):
    def test_dispatch_matches_langgraph(self):
        def run(mux_cls, module):
            mux = mux_cls([_counting_for(module)()])
            cursor = iter(mux._events)
            for index in range(3):
                mux.push(_event(data=index))
            mux.close()
            return [
                (event.get("seq"), event["method"], event["params"]["data"]) for event in cursor
            ]

        import langgraph.stream as lg_stream

        from langgraph_rs import stream as rs_stream

        self.assertEqual(run(StreamMux, rs_stream), run(LgStreamMux, lg_stream))

    def test_forwarding_matches_langgraph(self):
        def run(mux_cls, module):
            mux = mux_cls([_user_channel_for(module)()])
            cursor = iter(mux._events)
            mux.extensions["mine"].push("payload")
            mux.close()
            return [(event["method"], event["params"]["data"]) for event in cursor]

        import langgraph.stream as lg_stream

        from langgraph_rs import stream as rs_stream

        self.assertEqual(run(StreamMux, rs_stream), run(LgStreamMux, lg_stream))


if __name__ == "__main__":
    unittest.main()