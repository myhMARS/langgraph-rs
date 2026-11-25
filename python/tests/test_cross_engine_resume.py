"""Cross-engine resume: LangGraph and langgraph_rs can continue each other's runs.

Two sharing models are covered:

* a LangGraph-schema SQLite saver (`rs.LangGraphSqliteSaver` + LangGraph's `SqliteSaver`)
* a single Python saver object used by both engines (LangGraph's `InMemorySaver`)

Both directions are exercised with a two-node graph that pauses on `interrupt()`.
"""

from __future__ import annotations

import typing
import unittest

import langgraph_rs as rs

try:
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph as LgGraph
    from langgraph.types import Command as LgCommand
    from langgraph.types import interrupt as lg_interrupt

    _LANGGRAPH = True
except Exception:  # pragma: no cover - langgraph is optional
    _LANGGRAPH = False


def _reducer(current, incoming):
    return (current or []) + (incoming or [])


if _LANGGRAPH:

    class LgState(typing.TypedDict):
        attempts: int
        log: typing.Annotated[list, _reducer]

    def lg_ask(state):
        answer = lg_interrupt({"question": "approve?"})
        return {"attempts": state.get("attempts", 0) + 1, "log": [f"lg-ask:{answer}"]}

    def lg_done(state):
        return {"log": ["lg-done"]}


def rs_ask(state):
    answer = rs.interrupt({"question": "approve?"})
    return {"attempts": state["attempts"] + 1, "log": [f"rs-ask:{answer}"]}


def rs_done(state):
    return {"log": ["rs-done"]}


def rs_graph(saver):
    graph = rs.StateGraph({"attempts": "last_value", "log": "append"})
    graph.add_node("ask", rs_ask)
    graph.add_node("done", rs_done)
    graph.set_entry_point("ask")
    graph.add_edge("ask", "done")
    graph.set_finish_point("done")
    return graph.compile(checkpointer=saver)


def lg_graph(saver):
    graph = LgGraph(LgState)
    graph.add_node("ask", lg_ask)
    graph.add_node("done", lg_done)
    graph.add_edge(LG_START, "ask")
    graph.add_edge("ask", "done")
    graph.add_edge("done", LG_END)
    return graph.compile(checkpointer=saver)


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class SharedSaverResumeTest(unittest.TestCase):
    """One saver object shared by both engines (LangGraph's in-memory saver)."""

    def setUp(self):
        from langgraph.checkpoint.memory import InMemorySaver

        self.saver = InMemorySaver()
        self.lg_app = lg_graph(self.saver)
        self.rs_app = rs_graph(self.saver)

    def test_langgraph_pause_is_resumed_by_langgraph_rs(self):
        self.lg_app.invoke({"attempts": 0, "log": []}, {"configurable": {"thread_id": "a"}})
        self.assertEqual(self.rs_app.get_state("a")["values"], {"attempts": 0, "log": []})
        out = self.rs_app.resume("a", "yes")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["log"], ["rs-ask:yes", "rs-done"])

    def test_langgraph_rs_pause_is_resumed_by_langgraph(self):
        self.rs_app.invoke({"attempts": 0, "log": []}, thread_id="b")
        cfg = {"configurable": {"thread_id": "b"}}
        snapshot = self.lg_app.get_state(cfg)
        self.assertEqual([task.name for task in snapshot.tasks], ["ask"])
        out = self.lg_app.invoke(LgCommand(resume="yes"), cfg)
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["log"], ["lg-ask:yes", "lg-done"])

    def test_langgraph_rs_round_trip_still_works(self):
        paused = self.rs_app.invoke({"attempts": 0, "log": []}, thread_id="c")
        self.assertIn("__interrupt__", paused)
        self.assertEqual(paused["__interrupt__"][0].value, {"question": "approve?"})
        out = self.rs_app.resume("c", "yes")
        self.assertEqual(out["log"], ["rs-ask:yes", "rs-done"])


@unittest.skipUnless(_LANGGRAPH, "langgraph is not installed")
class SqliteSchemaResumeTest(unittest.TestCase):
    """The LangGraph-schema SQLite saver, shared through a database file."""

    def test_both_directions(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver

        # LangGraph pauses, the extension resumes.
        path = Path(tempfile.mkdtemp()) / "lg-first.sqlite"
        connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        lg_saver = SqliteSaver(connection)
        lg_saver.setup()
        lg_app = lg_graph(lg_saver)
        rs_app = rs_graph(rs.LangGraphSqliteSaver(str(path)))
        lg_app.invoke({"attempts": 0, "log": []}, {"configurable": {"thread_id": "d1"}})
        out = rs_app.resume("d1", "yes")
        self.assertEqual(out["log"], ["rs-ask:yes", "rs-done"])

        # The extension pauses, LangGraph resumes.
        path = Path(tempfile.mkdtemp()) / "rs-first.sqlite"
        rs_app = rs_graph(rs.LangGraphSqliteSaver(str(path)))
        rs_app.invoke({"attempts": 0, "log": []}, thread_id="d2")
        connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        lg_saver = SqliteSaver(connection)
        lg_saver.setup()
        lg_app = lg_graph(lg_saver)
        out = lg_app.invoke(LgCommand(resume="yes"), {"configurable": {"thread_id": "d2"}})
        self.assertEqual(out["log"], ["lg-ask:yes", "lg-done"])


if __name__ == "__main__":
    unittest.main()