"""Custom Python long-term stores follow LangGraph's `BaseStore` protocol.

`compile(store=<object>)` accepts anything implementing `get` / `put` / `delete` / `search` /
`list_namespaces`, including LangGraph's own `InMemoryStore`.
"""

from __future__ import annotations

import unittest

import langgraph_rs as lg
from langgraph_rs.config import get_store


class _DictStore:
    """Minimal LangGraph-style store: namespaces are tuples, values are plain JSON."""

    def __init__(self) -> None:
        self.data: dict[tuple[tuple[str, ...], str], object] = {}

    def get(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        if value is None:
            return None
        return {"namespace": tuple(namespace), "key": key, "value": value}

    def put(self, namespace, key, value, index=None):
        self.data[(tuple(namespace), key)] = value

    def delete(self, namespace, key):
        self.data.pop((tuple(namespace), key), None)

    def search(self, namespace_prefix, query=None, filter=None, limit=10, offset=0):
        prefix = tuple(namespace_prefix)
        items = [
            {"namespace": namespace, "key": key, "value": value}
            for (namespace, key), value in self.data.items()
            if namespace[: len(prefix)] == prefix
        ]
        return items[:limit]

    def list_namespaces(self, **kwargs):
        return sorted({namespace for (namespace, _key) in self.data})


class CustomStoreTest(unittest.TestCase):
    def _graph(self, store):
        def node(state):
            current = get_store()
            current.put(("users",), "u1", {"name": "ada"})
            found = current.get(("users",), "u1")
            return {"v": found.value["name"]}

        graph = lg.StateGraph({"v": "last_value"})
        graph.add_node("n", node)
        graph.set_entry_point("n")
        graph.set_finish_point("n")
        return graph.compile(store=store)

    def test_custom_store_roundtrip(self):
        store = _DictStore()
        graph = self._graph(store)
        self.assertEqual(graph.invoke({"v": ""})["v"], "ada")
        self.assertEqual(store.data[(("users",), "u1")], {"name": "ada"})

    def test_langgraph_store_interoperates(self):
        try:
            from langgraph.store.memory import InMemoryStore as LgStore
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        graph = self._graph(LgStore())
        self.assertEqual(graph.invoke({"v": ""})["v"], "ada")


if __name__ == "__main__":
    unittest.main()