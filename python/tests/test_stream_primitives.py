"""`langgraph.stream` primitives: `StreamChannel`, `ProtocolEvent`, `StreamTransformer`.

The channel is a single-consumer, caller-pumped queue, so every case subscribes *before* pushing
(lazy subscription is part of the contract). The LangGraph-backed cases run the same steps on both
implementations and compare the results and the raised errors.
"""

from __future__ import annotations

import asyncio
import unittest

import langgraph_rs.stream as rs_stream
from langgraph_rs.stream import ProtocolEvent, StreamChannel, StreamTransformer

try:
    import langgraph.stream as lg_stream

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


def _drain_sync(module, items):
    channel = module.StreamChannel()
    channel._bind(is_async=False)
    cursor = iter(channel)
    for item in items:
        channel.push(item)
    channel.close()
    return list(cursor)


def _drain_tee(module, items):
    channel = module.StreamChannel()
    channel._bind(is_async=False)
    first, second = channel.tee(2)
    for item in items:
        channel.push(item)
    channel.close()
    return list(first), list(second)


class StreamChannelTest(unittest.TestCase):
    def test_iteration_requires_binding(self):
        channel = StreamChannel()
        with self.assertRaises(TypeError):
            iter(channel)

    def test_sync_iteration_drains_pushed_items(self):
        self.assertEqual(_drain_sync(rs_stream, [1, 2, 3]), [1, 2, 3])

    def test_single_subscriber(self):
        channel = StreamChannel()
        channel._bind(is_async=False)
        iter(channel)
        with self.assertRaises(RuntimeError):
            iter(channel)

    def test_wrong_mode_raises(self):
        sync_channel = StreamChannel()
        sync_channel._bind(is_async=False)
        with self.assertRaises(TypeError):
            sync_channel.__aiter__()
        async_channel = StreamChannel()
        async_channel._bind(is_async=True)
        with self.assertRaises(TypeError):
            iter(async_channel)

    def test_tee_fans_out(self):
        self.assertEqual(_drain_tee(rs_stream, [1, 2]), ([1, 2], [1, 2]))

    def test_fail_surfaces_to_the_consumer(self):
        channel = StreamChannel()
        channel._bind(is_async=False)
        cursor = iter(channel)
        error = ValueError("boom")
        channel.fail(error)
        with self.assertRaises(ValueError):
            list(cursor)

    def test_async_iteration(self):
        async def drain():
            channel = StreamChannel()
            channel._bind(is_async=True)
            cursor = channel.__aiter__()
            channel.push("x")
            channel.push("y")
            channel.close()
            return [item async for item in cursor]

        self.assertEqual(asyncio.run(drain()), ["x", "y"])

    def test_maxlen_must_be_positive(self):
        with self.assertRaises(ValueError):
            StreamChannel(maxlen=0)


class StreamTransformerTest(unittest.TestCase):
    def test_default_process_raises(self):
        class Passive(StreamTransformer):
            def init(self):
                return {}

        with self.assertRaises(NotImplementedError):
            Passive().process({"type": "event", "method": "values", "params": {}})

    def test_async_detection(self):
        class SyncOnly(StreamTransformer):
            def init(self):
                return {}

            def process(self, event):
                return True

        class AsyncOnly(StreamTransformer):
            def init(self):
                return {}

            async def aprocess(self, event):
                return True

        self.assertFalse(rs_stream.transformer_requires_async(SyncOnly()))
        self.assertTrue(rs_stream.transformer_requires_async(AsyncOnly()))

    def test_schedule_requires_running_loop(self):
        class Scheduler(StreamTransformer):
            def init(self):
                return {}

            def process(self, event):
                return True

        async def nothing():
            return None

        coro = nothing()
        with self.assertRaises(RuntimeError):
            Scheduler().schedule(coro)
        coro.close()  # the call raised before scheduling, so close it explicitly

    def test_protocol_event_is_a_typed_dict(self):
        annotations = ProtocolEvent.__annotations__
        self.assertEqual(sorted(annotations), ["event_id", "method", "params", "seq", "type"])


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class StreamPrimitivesParityTest(unittest.TestCase):
    def test_sync_iteration_matches_langgraph(self):
        self.assertEqual(
            _drain_sync(rs_stream, [1, 2, 3]), _drain_sync(lg_stream, [1, 2, 3])
        )

    def test_tee_matches_langgraph(self):
        self.assertEqual(_drain_tee(rs_stream, [1, 2]), _drain_tee(lg_stream, [1, 2]))

    def test_async_iteration_matches_langgraph(self):
        async def drain(module):
            channel = module.StreamChannel()
            channel._bind(is_async=True)
            cursor = channel.__aiter__()
            channel.push("a")
            channel.close()
            return [item async for item in cursor]

        self.assertEqual(asyncio.run(drain(rs_stream)), asyncio.run(drain(lg_stream)))

    def test_surface_covers_langgraph(self):
        missing = [
            name
            for name in dir(lg_stream)
            if not name.startswith("_") and not hasattr(rs_stream, name)
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()