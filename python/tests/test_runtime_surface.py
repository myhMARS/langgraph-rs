"""`langgraph.runtime`, `checkpoint.serde` and `cache.base` surface parity."""

from __future__ import annotations

import dataclasses
import datetime
import unittest

import langgraph_rs as lg
from langgraph_rs.cache.base import BaseCache, FullKey
from langgraph_rs.checkpoint.serde import JsonPlusSerializer
from langgraph_rs.runtime import (
    DEFAULT_RUNTIME,
    ExecutionInfo,
    Runtime,
    ServerInfo,
    get_runtime,
)


class RuntimeTest(unittest.TestCase):
    def test_runtime_fields_match_langgraph(self):
        fields = [field.name for field in dataclasses.fields(Runtime)]
        expected = [
            "context",
            "store",
            "stream_writer",
            "heartbeat",
            "previous",
            "execution_info",
            "server_info",
            "control",
        ]
        self.assertEqual(fields[: len(expected)], expected)

    def test_get_runtime_outside_a_run_returns_the_default(self):
        self.assertIs(get_runtime(), DEFAULT_RUNTIME)
        self.assertIsNone(DEFAULT_RUNTIME.context)
        self.assertIsNone(DEFAULT_RUNTIME.store)

    def test_get_runtime_inside_a_node(self):
        seen = {}

        def node(state):
            runtime = get_runtime()
            seen["type"] = type(runtime).__name__
            seen["context"] = runtime.context
            seen["store"] = runtime.store
            seen["namespace"] = runtime.namespace
            return {"v": 1}

        graph = lg.StateGraph({"v": "last_value"})
        graph.add_node("n", node)
        graph.set_entry_point("n")
        graph.set_finish_point("n")
        graph.compile(store=lg.InMemoryStore()).invoke({"v": 0}, context={"user": "ada"})
        self.assertEqual(seen["type"], "Runtime")
        self.assertEqual(seen["context"], {"user": "ada"})
        self.assertIsNotNone(seen["store"])
        self.assertEqual(seen["namespace"], "")

    def test_execution_and_server_info_defaults(self):
        info = ExecutionInfo(thread_id="t1")
        self.assertEqual((info.thread_id, info.node_attempt, info.checkpoint_ns), ("t1", 1, ""))
        self.assertEqual(ServerInfo().assistant_id, "")


class SerdeTest(unittest.TestCase):
    def test_round_trip_json_extras(self):
        serializer = JsonPlusSerializer()
        payload = {
            "n": 1,
            "bytes": b"raw",
            "set": {3, 4},
            "date": datetime.date(2026, 11, 25),
            "ts": datetime.datetime(2026, 11, 25, 12, 0, 0),
        }
        kind, blob = serializer.dumps_typed(payload)
        self.assertEqual(kind, "json")
        self.assertEqual(serializer.loads_typed((kind, blob)), payload)

    def test_unsupported_object_raises(self):
        with self.assertRaises(TypeError):
            JsonPlusSerializer().dumps(object())

    def test_typed_payload_kind_is_validated(self):
        with self.assertRaises(ValueError):
            JsonPlusSerializer().loads_typed(("msgpack", b""))


class CacheBaseTest(unittest.TestCase):
    def test_full_key_and_subclassing(self):
        self.assertEqual(FullKey(("a", "b"), "k"), (("a", "b"), "k"))

        class _Cache(BaseCache):
            def __init__(self):
                self.data = {}

            def get(self, keys):
                return {key: self.data[key] for key in keys if key in self.data}

            def set(self, pairs):
                for key, value in pairs:
                    self.data[key] = value

            def clear(self, namespaces=None):
                self.data.clear()

        cache = _Cache()
        cache.set([(FullKey(("a",), "k"), 1)])
        self.assertEqual(cache.get([FullKey(("a",), "k")]), {FullKey(("a",), "k"): 1})
        cache.clear()
        self.assertEqual(cache.get([FullKey(("a",), "k")]), {})


if __name__ == "__main__":
    unittest.main()
