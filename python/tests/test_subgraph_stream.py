"""Subgraph discovery in `stream_events(version="v3")`: handles, nesting and lifecycle events.

`SubgraphTransformer` builds a `SubgraphRunStream` per direct child and `LifecycleTransformer`
reports `started`/`completed`/`failed`. Projections are single-consumer and lazy, so a handle's
projections must be iterated while the handle is live.
"""

from __future__ import annotations

import typing
import unittest

import langgraph_rs as rs

try:
    import langgraph.graph as lg_graph

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


class _State(typing.TypedDict):
    x: int


def _nested_graph(module, *, fail_inner=False):
    """parent -> mid -> grand, so nesting and lifecycle can both be observed."""
    state_schema = {"x": "last_value"} if module is rs else _State

    def leaf(inner_state):
        if fail_inner:
            raise RuntimeError("inner boom")
        return {"x": inner_state["x"] + 100}

    grand = module.StateGraph(state_schema)
    grand.add_node("g", leaf)
    grand.set_entry_point("g")
    grand.set_finish_point("g")

    mid = module.StateGraph(state_schema)
    mid.add_node("m", lambda state: {"x": state["x"] + 10})
    if module is rs:
        mid.add_subgraph("gsub", grand.compile())
    else:
        mid.add_node("gsub", grand.compile())
    mid.add_edge(module.START, "m")
    mid.add_edge("m", "gsub")
    mid.set_finish_point("gsub")

    parent = module.StateGraph(state_schema)
    parent.add_node("p", lambda state: {"x": state["x"] + 1})
    if module is rs:
        parent.add_subgraph("msub", mid.compile())
    else:
        parent.add_node("msub", mid.compile())
    parent.add_edge(module.START, "p")
    parent.add_edge("p", "msub")
    parent.set_finish_point("msub")
    return parent.compile()


class SubgraphStreamTest(unittest.TestCase):
    def test_direct_child_handle_reports_name_and_status(self):
        app = _nested_graph(rs)
        with app.stream_events({"x": 0}, version="v3") as run:
            for handle in run.subgraphs:
                self.assertEqual(handle.graph_name, "msub")
                self.assertEqual(handle.path, (handle.path[0],))
                self.assertTrue(handle.path[0].startswith("msub:"))
                self.assertEqual(list(handle.values), [{"x": 11}, {"x": 111}])
                self.assertEqual(handle.status, "completed")

    def test_grandchild_is_discovered_on_the_child_handle(self):
        app = _nested_graph(rs)
        seen: list[tuple[str | None, list[dict[str, int]]]] = []
        with app.stream_events({"x": 0}, version="v3") as run:
            for handle in run.subgraphs:
                for inner in handle.subgraphs:
                    seen.append((inner.graph_name, list(inner.values)))
                    self.assertEqual(inner.status, "completed")
        self.assertEqual(seen, [("gsub", [{"x": 111}])])

    def test_lifecycle_reports_started_and_completed(self):
        app = _nested_graph(rs)
        with app.stream_events({"x": 0}, version="v3") as run:
            events = list(run.lifecycle)
        names = [(event["event"], event.get("graph_name")) for event in events]
        self.assertEqual(
            names,
            [
                ("started", "msub"),
                ("started", "gsub"),
                ("completed", None),
                ("completed", None),
            ],
        )
        # The nested namespace carries one segment per level, like LangGraph.
        self.assertTrue(all(len(event["namespace"]) in (1, 2) for event in events))

    def test_failed_child_surfaces_the_error_through_the_stream(self):
        app = _nested_graph(rs, fail_inner=True)
        with app.stream_events({"x": 0}, version="v3") as run:
            seen: list[str] = []
            with self.assertRaises(RuntimeError) as caught:
                for handle in run.subgraphs:
                    seen.append(handle.graph_name or "")
                    list(handle.values)
            self.assertIn("inner boom", str(caught.exception))
            # The run failed on its first step, so no handle had been discovered yet.
            self.assertEqual(seen, [])

    def test_nested_namespace_has_one_segment_per_level(self):
        app = _nested_graph(rs)
        parts = [
            part
            for part in app.stream(
                {"x": 0}, stream_mode="values", version="v2", subgraphs=True
            )
            if isinstance(part, dict)
        ]
        depths = {len(part["ns"]) for part in parts}
        self.assertEqual(depths, {0, 1, 2})

    def test_projections_include_lifecycle_and_subgraphs_by_default(self):
        app = _nested_graph(rs)
        with app.stream_events({"x": 0}, version="v3") as run:
            self.assertEqual(
                sorted(run.extensions), ["lifecycle", "subgraphs", "values"]
            )
            run.abort()


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class SubgraphStreamParityTest(unittest.TestCase):
    def test_lifecycle_sequence_matches_langgraph(self):
        def collect(app):
            with app.stream_events({"x": 0}, version="v3") as run:
                return [(event["event"], event.get("graph_name")) for event in run.lifecycle]

        self.assertEqual(collect(_nested_graph(rs)), collect(_nested_graph(lg_graph)))

    def test_child_graph_names_match_langgraph(self):
        def collect(app):
            names = []
            with app.stream_events({"x": 0}, version="v3") as run:
                for handle in run.subgraphs:
                    names.append(handle.graph_name)
                    run.abort()
                    break
            return names

        self.assertEqual(collect(_nested_graph(rs)), collect(_nested_graph(lg_graph)))


if __name__ == "__main__":
    unittest.main()