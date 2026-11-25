"""`langgraph.errors` surface matches LangGraph's classes, bases and messages."""

from __future__ import annotations

import unittest

import langgraph_rs as lg
import langgraph_rs.errors as errors


class ErrorsTest(unittest.TestCase):
    def test_graph_bubble_up_hierarchy(self):
        self.assertTrue(issubclass(errors.GraphInterrupt, errors.GraphBubbleUp))
        self.assertTrue(issubclass(errors.GraphDrained, errors.GraphBubbleUp))
        self.assertTrue(issubclass(errors.NodeInterrupt, errors.GraphInterrupt))
        self.assertEqual(str(errors.GraphDrained()), "Graph drained: shutdown")

    def test_node_timeout_error_messages(self):
        idle = errors.NodeTimeoutError("n", 1.5, kind="idle", idle_timeout=1.0)
        self.assertEqual(
            str(idle),
            "Node 'n' exceeded its idle timeout of 1.000s without making progress (elapsed: 1.500s).",
        )
        self.assertEqual((idle.node, idle.elapsed, idle.kind), ("n", 1.5, "idle"))
        run = errors.NodeTimeoutError("n", 2.0, kind="run", run_timeout=1.5)
        self.assertEqual(
            str(run),
            "Node 'n' exceeded its run timeout of 1.500s (elapsed: 2.000s).",
        )

    def test_node_error_wraps_the_cause(self):
        wrapped = errors.NodeError("n", ValueError("boom"))
        self.assertEqual(wrapped.node, "n")
        self.assertEqual(repr(wrapped), "NodeError(node='n', error=ValueError('boom'))")
        self.assertNotIsInstance(wrapped, BaseException)

    def test_unknown_interrupt_keeps_diagnostics(self):
        error = errors.UnknownInterruptError(
            "no such interrupt", reason="not_pending", interrupt_ids=["a"], pending_ids=["b"]
        )
        self.assertEqual(error.reason, "not_pending")
        self.assertEqual(error.interrupt_ids, ("a",))
        self.assertEqual(error.pending_ids, ("b",))

    def test_error_codes_and_simple_exceptions(self):
        self.assertEqual(
            [member.name for member in errors.ErrorCode],
            [
                "GRAPH_RECURSION_LIMIT",
                "INVALID_CONCURRENT_GRAPH_UPDATE",
                "INVALID_GRAPH_NODE_RETURN_VALUE",
                "MULTIPLE_SUBGRAPHS",
                "INVALID_CHAT_HISTORY",
            ],
        )
        for cls in (errors.EmptyChannelError, errors.InvalidUpdateError, errors.TaskNotFound):
            self.assertTrue(issubclass(cls, Exception))

    def test_deprecation_warning_chain(self):
        from langgraph_rs.warnings import LangGraphDeprecationWarning, LangGraphDeprecatedSinceV10

        self.assertTrue(issubclass(LangGraphDeprecatedSinceV10, LangGraphDeprecationWarning))
        self.assertTrue(issubclass(LangGraphDeprecationWarning, DeprecationWarning))
        self.assertIs(errors.LangGraphDeprecatedSinceV10, LangGraphDeprecatedSinceV10)


class RecursionErrorTest(unittest.TestCase):
    def test_recursion_limit_raises_langgraph_error(self):
        graph = lg.StateGraph({"x": "last_value"})
        graph.add_node("a", lambda state: state)
        graph.add_conditional_edges("a", lambda state: "a")
        graph.set_entry_point("a")
        compiled = graph.compile()
        with self.assertRaises(errors.GraphRecursionError) as raised:
            compiled.invoke({"x": 1}, config={"recursion_limit": 3})
        self.assertIsInstance(raised.exception, RecursionError)
        message = str(raised.exception)
        self.assertIn("Recursion limit of 3 reached without hitting a stop condition", message)
        self.assertIn("errors/GRAPH_RECURSION_LIMIT", message)


class HelperTest(unittest.TestCase):
    def test_create_error_message(self):
        message = errors.create_error_message(
            message="boom", error_code=errors.ErrorCode.GRAPH_RECURSION_LIMIT
        )
        self.assertEqual(
            message,
            "boom\nFor troubleshooting, visit: "
            "https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT",
        )

    def test_is_invalid_resume(self):
        plain = ValueError("bad tool args")
        self.assertFalse(errors.is_invalid_resume(plain))
        marked = ValueError("bad resume")
        marked._invalid_resume = True
        self.assertTrue(errors.is_invalid_resume(marked))

    def test_parent_command_and_empty_input(self):
        command = lg.Command(goto="next")
        error = errors.ParentCommand(command)
        self.assertIs(error.command, command)
        self.assertIsInstance(error, errors.GraphBubbleUp)
        self.assertIsInstance(errors.EmptyInputError(), Exception)


if __name__ == "__main__":
    unittest.main()