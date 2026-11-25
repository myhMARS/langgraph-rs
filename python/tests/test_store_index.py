"""Vector-indexed store: `index=` configuration, semantic ranking and `score`.

The ranking cases run on both engines with the same deterministic embedder, so the scores and the
order can be compared directly.
"""

from __future__ import annotations

import unittest

import langgraph_rs as rs

try:
    from langgraph.store.memory import InMemoryStore as LgInMemoryStore

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


def fake_embed(texts):
    """Deterministic 3-dim embedding: length mod 7, count of `a`, count of `b`."""
    return [[len(text) % 7 + 1.0, text.count("a") + 1.0, text.count("b") + 1.0] for text in texts]


class ObjectEmbeddings:
    """LangChain-style embeddings object (separate document/query methods)."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return fake_embed(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return fake_embed([text])[0]


DOCS = (("d1", "apple pie"), ("d2", "banana bread"), ("d3", "apple sauce"))


def _fill(store):
    for key, text in DOCS:
        store.put(("docs",), key, {"text": text})
    return store


class StoreIndexTest(unittest.TestCase):
    def test_semantic_ranking_and_scores(self):
        store = _fill(rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]}))
        hits = store.search(("docs",), query="apple")
        self.assertEqual([hit.key for hit in hits], ["d3", "d1", "d2"])
        for hit in hits:
            self.assertIsNotNone(hit.score)
        self.assertGreater(hits[0].score, hits[1].score)

    def test_limit_and_offset_apply_after_ranking(self):
        store = _fill(rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]}))
        self.assertEqual(
            [hit.key for hit in store.search(("docs",), query="apple", limit=1)], ["d3"]
        )
        self.assertEqual(
            [hit.key for hit in store.search(("docs",), query="apple", limit=1, offset=1)], ["d1"]
        )

    def test_items_without_vectors_are_appended_unscored(self):
        store = _fill(rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]}))
        store.put(("docs",), "d4", {"text": "cherry"}, index=False)
        hits = store.search(("docs",), query="apple")
        self.assertEqual([hit.key for hit in hits], ["d3", "d1", "d2", "d4"])
        self.assertIsNone(hits[-1].score)

    def test_index_argument_selects_the_fields(self):
        store = rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]})
        store.put(("docs",), "n1", {"text": "apple", "note": "banana"}, index=["note"])
        hits = store.search(("docs",), query="banana")
        self.assertEqual([hit.key for hit in hits], ["n1"])

    def test_embeddings_object_is_used_for_documents_and_queries(self):
        embeddings = ObjectEmbeddings()
        store = rs.InMemoryStore(index={"dims": 3, "embed": embeddings, "fields": ["text"]})
        _fill(store)
        # The extension embeds per `put` (LangGraph batches several puts into one call);
        # what matters is that documents go through `embed_documents`.
        self.assertGreaterEqual(embeddings.document_calls, 1)
        self.assertEqual(embeddings.query_calls, 0)
        store.search(("docs",), query="apple")
        self.assertEqual(embeddings.query_calls, 1)

    def test_whole_value_is_indexed_without_fields(self):
        store = rs.InMemoryStore(index={"dims": 3, "embed": fake_embed})
        store.put(("docs",), "w1", {"text": "apple"})
        self.assertEqual([hit.key for hit in store.search(("docs",), query="apple")], ["w1"])

    def test_filter_still_applies_to_indexed_search(self):
        store = _fill(rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]}))
        store.put(("docs",), "d4", {"text": "apple juice", "tag": "juice"})
        hits = store.search(("docs",), query="apple", filter={"tag": "juice"})
        self.assertEqual([hit.key for hit in hits], ["d4"])


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class StoreIndexParityTest(unittest.TestCase):
    def test_ranking_matches_langgraph(self):
        ours = _fill(
            rs.InMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]})
        )
        theirs = _fill(
            LgInMemoryStore(index={"dims": 3, "embed": fake_embed, "fields": ["text"]})
        )
        for case in (
            {"query": "apple"},
            {"query": "apple", "limit": 1},
            {"query": "apple", "limit": 1, "offset": 1},
            {"query": "banana"},
            {},
        ):
            with self.subTest(case=case):
                ours_hits = ours.search(("docs",), **case)
                theirs_hits = theirs.search(("docs",), **case)
                self.assertEqual(
                    [hit.key for hit in ours_hits], [hit.key for hit in theirs_hits]
                )
                for ours_hit, theirs_hit in zip(ours_hits, theirs_hits, strict=True):
                    if theirs_hit.score is None:
                        self.assertIsNone(ours_hit.score)
                    else:
                        self.assertAlmostEqual(ours_hit.score, theirs_hit.score, places=6)


if __name__ == "__main__":
    unittest.main()