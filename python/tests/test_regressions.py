"""Guards for regressions found by independent audits."""

import operator as op
import unittest
from typing import Annotated, TypedDict

import langgraph_rs


class RegressionGuardTest(unittest.TestCase):
    def test_reducer_over_a_non_json_value(self):
        """A `set` reducer folds from the first write, not from a fabricated empty mapping."""

        class TagState(TypedDict, total=False):
            tag_set: Annotated[set, op.or_]

        def build():
            graph = langgraph_rs.StateGraph(TagState)
            graph.add_node("n", lambda state: {"tag_set": {"c"}})
            graph.add_edge(langgraph_rs.START, "n")
            graph.add_edge("n", langgraph_rs.END)
            return graph.compile()

        self.assertEqual(sorted(build().invoke({})["tag_set"]), ["c"])
        self.assertEqual(
            sorted(build().invoke({"tag_set": {"a", "b"}})["tag_set"]), ["a", "b", "c"]
        )

    def test_get_store_inside_a_run_without_a_store_returns_none(self):
        from langgraph_rs.config import get_store

        seen = {}

        def node(state):
            seen["store"] = get_store()
            return {}

        graph = langgraph_rs.StateGraph({"value": "last_value"})
        graph.add_node("n", node)
        graph.add_edge(langgraph_rs.START, "n")
        graph.add_edge("n", langgraph_rs.END)
        graph.compile().invoke({"value": 0})
        self.assertIsNone(seen["store"])


if __name__ == "__main__":
    unittest.main()
