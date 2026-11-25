"""Custom Python checkpointers follow LangGraph's saver protocol.

`compile(checkpointer=<object>)` accepts anything implementing `put` / `put_writes` /
`get_tuple` / `list`, including LangGraph's own savers. The engine's scheduling fields travel in
`metadata["langgraph_rs"]`, so a saver that drops unknown metadata keys is rejected loudly.
"""

from __future__ import annotations

import unittest

import langgraph_rs as lg


class _DictSaver:
    """Minimal LangGraph-style saver backed by a list."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict, dict]] = []
        self.writes: list[tuple[str, object]] = []

    def put(self, config, checkpoint, metadata, new_versions):
        thread = config["configurable"]["thread_id"]
        self.rows.append((thread, checkpoint, dict(metadata)))
        return {
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(self, config, writes, task_id, task_path=""):
        self.writes.append((task_id, list(writes)))
        return None

    def get_tuple(self, config):
        cfg = config.get("configurable", {})
        thread = cfg.get("thread_id")
        wanted = cfg.get("checkpoint_id")
        for stored_thread, checkpoint, metadata in reversed(self.rows):
            if stored_thread != thread:
                continue
            if wanted and checkpoint["id"] != wanted:
                continue
            return (
                {
                    "configurable": {
                        "thread_id": thread,
                        "checkpoint_ns": "",
                        "checkpoint_id": checkpoint["id"],
                    }
                },
                checkpoint,
                metadata,
                None,
            )
        return None

    def list(self, config, **kwargs):
        cfg = config.get("configurable", {})
        thread = cfg.get("thread_id")
        for stored_thread, checkpoint, metadata in reversed(self.rows):
            if stored_thread == thread:
                yield (
                    {
                        "configurable": {
                            "thread_id": thread,
                            "checkpoint_ns": "",
                            "checkpoint_id": checkpoint["id"],
                        }
                    },
                    checkpoint,
                    metadata,
                    None,
                )


class _DroppingSaver(_DictSaver):
    """A saver that discards unknown metadata keys (should be rejected)."""

    def put(self, config, checkpoint, metadata, new_versions):
        thread = config["configurable"]["thread_id"]
        self.rows.append((thread, checkpoint, {"source": metadata.get("source")}))
        return config


def _graph(saver):
    graph = lg.StateGraph({"v": "last_value"})
    graph.add_node("inc", lambda state: {"v": state["v"] + 1})
    graph.set_entry_point("inc")
    graph.set_finish_point("inc")
    return graph.compile(checkpointer=saver)


class CustomSaverTest(unittest.TestCase):
    def test_custom_saver_persists_and_resumes(self):
        saver = _DictSaver()
        graph = _graph(saver)
        self.assertEqual(graph.invoke({"v": 0}, thread_id="t1")["v"], 1)
        self.assertEqual(graph.get_state("t1")["values"]["v"], 1)
        self.assertEqual(graph.invoke({"v": 5}, thread_id="t1")["v"], 6)
        self.assertGreaterEqual(len(graph.get_state_history("t1")), 1)
        self.assertGreaterEqual(len(saver.rows), 2)

    def test_saver_that_drops_metadata_falls_back_to_the_langgraph_payload(self):
        # Without the `langgraph_rs` sidecar the engine rebuilds the checkpoint from the
        # LangGraph-shaped payload (`__root__`, `branch:to:*`, `__interrupt__`), so reading and
        # resuming keep working instead of raising.
        graph = _graph(_DroppingSaver())
        graph.invoke({"v": 0}, thread_id="t2")
        self.assertEqual(graph.get_state("t2")["values"]["v"], 1)

    def test_langgraph_saver_interoperates(self):
        try:
            from langgraph.checkpoint.memory import InMemorySaver as LgSaver
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        graph = _graph(LgSaver())
        self.assertEqual(graph.invoke({"v": 0}, thread_id="lg1")["v"], 1)
        self.assertEqual(graph.get_state("lg1")["values"]["v"], 1)
        self.assertEqual(graph.invoke({"v": 10}, thread_id="lg1")["v"], 11)
        self.assertGreaterEqual(len(graph.get_state_history("lg1")), 1)


if __name__ == "__main__":
    unittest.main()