"""Subgraphs inherit the parent's checkpointer when they are compiled without one.

LangGraph passes the saver through the run config, so `add_node("sub", subgraph_without_saver)`
still checkpoints the child and an `interrupt()` inside it is resumable. The extension used to
require an explicit saver on the child; this test pins the inherited behaviour for both engines,
including a nested (grandchild) graph.
"""

from __future__ import annotations

import typing
import unittest

import langgraph_rs as rs

try:
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph as LgGraph
    from langgraph.prebuilt import InjectedState as _unused  # noqa: F401  (import check only)
    from langgraph.checkpoint.memory import InMemorySaver as LgMemorySaver
    from langgraph.types import Command as LgCommand
    from langgraph.types import interrupt as lg_interrupt

    _LANGGRAPH = True
except Exception:  # pragma: no cover - optional dependency
    _LANGGRAPH = False


class State(typing.TypedDict):
    draft: str
    approved: bool


def _rs_child():
    def review(state):
        return {"approved": bool(rs.interrupt({"question": "approve?"}))}

    child = rs.StateGraph({"draft": "last_value", "approved": "last_value"})
    child.add_node("review", review)
    child.set_entry_point("review")
    child.set_finish_point("review")
    return child


def _rs_parent(child):
    parent = rs.StateGraph({"draft": "last_value", "approved": "last_value"})
    parent.add_node("work", lambda state: {"draft": "v1"})
    parent.add_subgraph("sub", child)
    parent.add_node(
        "done",
        lambda state: {
            "draft": state["draft"] + (" [done]" if state["approved"] else " [rejected]")
        },
    )
    parent.add_edge(rs.START, "work")
    parent.add_edge("work", "sub")
    parent.add_edge("sub", "done")
    parent.set_finish_point("done")
    return parent


def _lg_child():
    def review(state):
        return {"approved": bool(lg_interrupt({"question": "approve?"}))}

    child = LgGraph(State)
    child.add_node("review", review)
    child.add_edge(LG_START, "review")
    child.add_edge("review", LG_END)
    return child


def _lg_parent(child):
    parent = LgGraph(State)
    parent.add_node("work", lambda state: {"draft": "v1"})
    parent.add_node("sub", child)
    parent.add_node(
        "done",
        lambda state: {
            "draft": state["draft"] + (" [done]" if state["approved"] else " [rejected]")
        },
    )
    parent.add_edge(LG_START, "work")
    parent.add_edge("work", "sub")
    parent.add_edge("sub", "done")
    parent.add_edge("done", LG_END)
    return parent


class SubgraphInheritanceTest(unittest.TestCase):
    def test_child_without_saver_inherits_the_parent_saver(self):
        app = _rs_parent(_rs_child().compile()).compile(checkpointer=rs.InMemorySaver())
        paused = app.invoke({"draft": "", "approved": False}, thread_id="inherit")
        self.assertIn("__interrupt__", paused)
        final = app.resume("inherit", True)
        self.assertEqual(final, {"approved": True, "draft": "v1 [done]"})

    def test_grandchild_inherits_through_two_levels(self):
        inner = _rs_child().compile()

        middle = rs.StateGraph({"draft": "last_value", "approved": "last_value"})
        middle.add_subgraph("inner", inner)
        middle.add_node("noop", lambda state: {})
        middle.add_edge(rs.START, "inner")
        middle.add_edge("inner", "noop")
        middle.set_finish_point("noop")

        parent = rs.StateGraph({"draft": "last_value", "approved": "last_value"})
        parent.add_node("work", lambda state: {"draft": "v1"})
        parent.add_subgraph("middle", middle.compile())
        parent.add_node(
            "done",
            lambda state: {"draft": state["draft"] + (" [done]" if state["approved"] else "")},
        )
        parent.add_edge(rs.START, "work")
        parent.add_edge("work", "middle")
        parent.add_edge("middle", "done")
        parent.set_finish_point("done")

        app = parent.compile(checkpointer=rs.InMemorySaver())
        paused = app.invoke({"draft": "", "approved": False}, thread_id="deep")
        self.assertIn("__interrupt__", paused)
        self.assertEqual(app.resume("deep", True)["approved"], True)

    def test_child_with_its_own_saver_keeps_it(self):
        child_saver = rs.InMemorySaver()
        app = _rs_parent(_rs_child().compile(checkpointer=child_saver)).compile(
            checkpointer=rs.InMemorySaver()
        )
        paused = app.invoke({"draft": "", "approved": False}, thread_id="own")
        self.assertIn("__interrupt__", paused)
        self.assertEqual(app.resume("own", True)["draft"], "v1 [done]")


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class SubgraphInheritanceParityTest(unittest.TestCase):
    def test_langgraph_and_extension_agree(self):
        lg_app = _lg_parent(_lg_child().compile()).compile(checkpointer=LgMemorySaver())
        lg_cfg = {"configurable": {"thread_id": "lg"}}
        lg_paused = lg_app.invoke({"draft": "", "approved": False}, lg_cfg)
        lg_final = lg_app.invoke(LgCommand(resume=True), lg_cfg)

        rs_app = _rs_parent(_rs_child().compile()).compile(checkpointer=rs.InMemorySaver())
        rs_paused = rs_app.invoke({"draft": "", "approved": False}, thread_id="rs")
        rs_final = rs_app.resume("rs", True)

        self.assertIn("__interrupt__", lg_paused)
        self.assertIn("__interrupt__", rs_paused)
        self.assertEqual(lg_final["draft"], rs_final["draft"])
        self.assertEqual(lg_final["approved"], rs_final["approved"])


if __name__ == "__main__":
    unittest.main()