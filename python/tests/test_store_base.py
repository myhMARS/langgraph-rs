"""`langgraph.store.base` helpers are faithful ports of LangGraph's path logic.

`tokenize_path` / `get_text_at_path` decide which text an embedding-indexed store indexes, so the
tests pin the exact outputs (and a differential comparison against LangGraph when installed).
"""

from __future__ import annotations

import unittest

from langgraph_rs.store.base import (
    NOT_PROVIDED,
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    NotProvided,
    PutOp,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    override,
    tokenize_path,
    validate_op_namespace,
)


class TokenizePathTest(unittest.TestCase):
    def test_field_and_index_tokens(self):
        self.assertEqual(tokenize_path("a.b"), ["a", "b"])
        self.assertEqual(tokenize_path("a[0].b"), ["a", "[0]", "b"])
        self.assertEqual(tokenize_path("a[*]"), ["a", "[*]"])
        self.assertEqual(tokenize_path("{a,b}.c"), ["{a,b}", "c"])
        self.assertEqual(tokenize_path(""), [])
        self.assertEqual(tokenize_path("x{y}"), ["x", "{y}"])


class GetTextAtPathTest(unittest.TestCase):
    def test_whole_object_when_no_path(self):
        self.assertEqual(get_text_at_path({"b": 1, "a": 2}, "$"), ['{"a": 2, "b": 1}'])
        self.assertEqual(get_text_at_path("text", ""), ['"text"'])

    def test_simple_field_and_nested_field(self):
        self.assertEqual(get_text_at_path({"a": "hello"}, "a"), ["hello"])
        self.assertEqual(get_text_at_path({"a": {"b": "deep"}}, "a.b"), ["deep"])

    def test_arrays_and_wildcards(self):
        self.assertEqual(get_text_at_path({"a": [10, 20]}, "a[0]"), ["10"])
        self.assertEqual(get_text_at_path({"a": [10, 20]}, "a[-1]"), ["20"])
        self.assertEqual(get_text_at_path({"a": [{"c": 1}, {"c": 2}]}, "a[*].c"), ["1", "2"])
        self.assertEqual(get_text_at_path({"a": {"x": "1", "y": "2"}}, "a.*"), ["1", "2"])

    def test_multi_field_selection(self):
        obj = {"a": 1, "b": {"c": "two"}}
        self.assertEqual(get_text_at_path(obj, "{a,b.c}"), ["1", "two"])

    def test_missing_and_none_values_are_skipped(self):
        self.assertEqual(get_text_at_path({"a": None}, "a"), [])
        self.assertEqual(get_text_at_path({"a": 1}, "missing"), [])

    def test_matches_langgraph(self):
        try:
            from langgraph.store.base import get_text_at_path as lg_get_text_at_path
            from langgraph.store.base import tokenize_path as lg_tokenize_path
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        cases = [
            ({"a": [1, 2, {"x": "y"}]}, "a[*].x"),
            ({"a": {"b": [{"c": 1}, {"c": 2}]}}, "a.b[*].c"),
            ({"a": 1, "b": {"c": "two"}}, "{a,b.c}"),
            ({"a": {"x": "1", "y": "2"}}, "a.*"),
            ({"a": None, "b": True}, "a.b"),
            ([{"k": "v"}], "[0].k"),
        ]
        for obj, path in cases:
            self.assertEqual(get_text_at_path(obj, path), lg_get_text_at_path(obj, path), path)
            self.assertEqual(tokenize_path(path), lg_tokenize_path(path), path)


class OpTest(unittest.TestCase):
    def test_op_defaults(self):
        self.assertEqual(GetOp(("ns",), "k"), GetOp(("ns",), "k", False))
        self.assertEqual(SearchOp(("ns",)).limit, 10)
        self.assertEqual(ListNamespacesOp().limit, 100)
        self.assertIsNone(PutOp(("ns",), "k").value)

    def test_validate_op_namespace_rejects_dotted_labels(self):
        validate_op_namespace(GetOp(("ok",), "k"))
        for op in (
            GetOp(("bad.label",), "k"),
            PutOp(("",), "k"),
            SearchOp(("a", "b.c")),
            ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("x.y",)),)),
        ):
            with self.assertRaises(InvalidNamespaceError):
                validate_op_namespace(op)

    def test_not_provided_sentinel(self):
        self.assertIsInstance(NOT_PROVIDED, NotProvided)
        self.assertEqual(repr(NOT_PROVIDED), "<not provided>")

    def test_override_marks_the_method(self):
        class Base:
            def run(self):
                return 1

        class Child(Base):
            @override
            def run(self):
                return 2

        self.assertTrue(Child.run.__override__)
        self.assertEqual(Child().run(), 2)


class EmbeddingsTest(unittest.TestCase):
    def test_function_is_wrapped(self):
        embeddings = ensure_embeddings(lambda texts: [[0.1, 0.2] for _ in texts])
        self.assertEqual(embeddings.embed_query("hello"), [0.1, 0.2])
        self.assertEqual(embeddings.embed_documents(["a", "b"]), [[0.1, 0.2], [0.1, 0.2]])

    def test_none_and_string_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ensure_embeddings(None)
        with self.assertRaises(ValueError):
            ensure_embeddings("openai:text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()