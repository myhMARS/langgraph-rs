"""Store search surface: `filter`, `offset` / `limit`, `list_namespaces` options and TTL.

The LangGraph-backed cases run the same operations on LangGraph's `InMemoryStore` and on
`langgraph_rs.InMemoryStore` and compare the results.
"""

from __future__ import annotations

import unittest

import langgraph_rs as rs

try:
    from langgraph.store.memory import InMemoryStore as LgInMemoryStore

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False

ITEMS = (
    ("a", {"text": "apple", "n": 1, "nested": {"kind": "fruit"}}),
    ("b", {"text": "banana", "n": 2, "nested": {"kind": "fruit"}}),
    ("c", {"text": "cherry", "n": 3, "nested": {"kind": "fruit"}}),
)


def _fill(store):
    for key, value in ITEMS:
        store.put(("u",), key, value)
    return store


class StoreSearchTest(unittest.TestCase):
    def test_filter_equality_and_operators(self):
        store = _fill(rs.InMemoryStore())
        self.assertEqual([h.key for h in store.search(("u",), filter={"n": 2})], ["b"])
        self.assertEqual(
            [h.key for h in store.search(("u",), filter={"n": {"$gt": 1}})], ["b", "c"]
        )
        self.assertEqual(
            [h.key for h in store.search(("u",), filter={"n": {"$lte": 2}})], ["a", "b"]
        )
        self.assertEqual(
            [h.key for h in store.search(("u",), filter={"n": {"$ne": 2}})], ["a", "c"]
        )
        self.assertEqual(
            [h.key for h in store.search(("u",), filter={"nested": {"kind": "fruit"}})],
            ["a", "b", "c"],
        )

    def test_pagination(self):
        store = _fill(rs.InMemoryStore())
        self.assertEqual([h.key for h in store.search(("u",), limit=1)], ["a"])
        self.assertEqual(
            [h.key for h in store.search(("u",), limit=2, offset=1)], ["b", "c"]
        )
        self.assertEqual([h.key for h in store.search(("u",), offset=10)], [])

    def test_list_namespaces_options(self):
        store = rs.InMemoryStore()
        for namespace in (("u", "a"), ("u", "b"), ("v", "c")):
            store.put(namespace, "k", {"n": 1})
        self.assertEqual(
            [tuple(ns) for ns in store.list_namespaces(prefix=("u",))],
            [("u", "a"), ("u", "b")],
        )
        self.assertEqual(
            [tuple(ns) for ns in store.list_namespaces(prefix=("u", "a"))], [("u", "a")]
        )
        self.assertEqual([tuple(ns) for ns in store.list_namespaces(max_depth=1)], [("u",), ("v",)])
        self.assertEqual(
            [tuple(ns) for ns in store.list_namespaces(prefix=("u",), limit=1)], [("u", "a")]
        )
        self.assertEqual(
            [tuple(ns) for ns in store.list_namespaces(prefix=("u",), limit=1, offset=1)],
            [("u", "b")],
        )

    def test_ttl_hides_expired_items(self):
        store = rs.InMemoryStore()
        store.put(("u",), "gone", {"text": "x"}, ttl=-0.01)
        store.put(("u",), "kept", {"text": "y"}, ttl=60)
        self.assertEqual([h.key for h in store.search(("u",))], ["kept"])
        self.assertIsNone(store.get(("u",), "gone"))
        self.assertEqual(store.get(("u",), "kept").value, {"text": "y"})

    def test_index_and_refresh_ttl_arguments_are_accepted(self):
        store = rs.InMemoryStore()
        store.put(("u",), "a", {"text": "apple"}, index=["text"])
        hits = store.search(("u",), query="apple", refresh_ttl=False)
        self.assertEqual([h.key for h in hits], ["a"])


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class StoreParityTest(unittest.TestCase):
    def _both(self):
        return _fill(rs.InMemoryStore()), _fill(LgInMemoryStore())

    def test_filter_and_pagination_match_langgraph(self):
        ours, theirs = self._both()
        cases = (
            {"filter": {"n": 2}},
            {"filter": {"n": {"$gt": 1}}},
            {"filter": {"n": {"$gte": 1, "$lt": 3}}},
            {"filter": {"nested": {"kind": "fruit"}}},
            {"limit": 2},
            {"limit": 1, "offset": 1},
            {"limit": 10, "offset": 10},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(
                    [h.key for h in ours.search(("u",), **case)],
                    [h.key for h in theirs.search(("u",), **case)],
                )

    def test_list_namespaces_match_langgraph(self):
        ours, theirs = self._both()
        for namespace in (("u", "a"), ("v", "b")):
            ours.put(namespace, "k", {"n": 1})
            theirs.put(namespace, "k", {"n": 1})
        for case in (
            {},
            {"prefix": ("u",)},
            {"max_depth": 1},
            {"limit": 2},
            {"limit": 1, "offset": 1},
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    [tuple(ns) for ns in ours.list_namespaces(**case)],
                    [tuple(ns) for ns in theirs.list_namespaces(**case)],
                )


if __name__ == "__main__":
    unittest.main()