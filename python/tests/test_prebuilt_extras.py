"""`create_tool`, tool-call wrappers and the tool message templates match LangGraph."""

from __future__ import annotations

import unittest

from langgraph_rs.prebuilt.chat_agent_executor import create_tool_calling_executor
from langgraph_rs.prebuilt.tool_node import (
    TOOL_CALL_ERROR_TEMPLATE,
    TOOL_EXECUTION_ERROR_TEMPLATE,
    TOOL_INVOCATION_ERROR_TEMPLATE,
    TOOL_MESSAGE_BLOCK_TYPES,
    ToolCallRequest,
    ToolCallWithContext,
    ToolInvocationError,
    create_tool,
    get_all_basemodel_annotations,
    msg_content_output,
)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class CreateToolTest(unittest.TestCase):
    def test_callable_becomes_a_tool(self):
        created = create_tool(add)
        self.assertEqual(created.name, "add")
        self.assertEqual(created.invoke({"a": 1, "b": 2}), 3)

    def test_decorator_forms(self):
        @create_tool
        def double(value: int) -> int:
            """Double a value."""
            return value * 2

        self.assertEqual(double.invoke({"value": 4}), 8)

        @create_tool(description="Triple it")
        def triple(value: int) -> int:
            """Triple a value."""
            return value * 3

        self.assertEqual(triple.description, "Triple it")
        self.assertEqual(triple.invoke({"value": 3}), 9)

    def test_runnable_requires_a_name(self):
        class _Runner:
            name = "runner"

            def invoke(self, payload):
                return payload

        with self.assertRaises(TypeError):
            create_tool(_Runner())
        named = create_tool("runner", _Runner())
        self.assertEqual(named.name, "runner")

    def test_matches_langgraph(self):
        try:
            from langgraph.prebuilt.tool_node import create_tool as lg_create_tool
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        ours, theirs = create_tool(add), lg_create_tool(add)
        self.assertEqual(ours.name, theirs.name)
        self.assertEqual(ours.description, theirs.description)
        self.assertEqual(list(ours.args), list(theirs.args))
        self.assertEqual(ours.invoke({"a": 2, "b": 3}), theirs.invoke({"a": 2, "b": 3}))


class ToolCallHelpersTest(unittest.TestCase):
    def test_msg_content_output(self):
        self.assertEqual(msg_content_output("text"), "text")
        blocks = [{"type": "text", "text": "x"}]
        self.assertEqual(msg_content_output(blocks), blocks)
        self.assertEqual(msg_content_output({"a": 1}), '{"a": 1}')
        self.assertEqual(msg_content_output(object()).startswith("<object object"), True)

    def test_templates_are_langgraph_text(self):
        self.assertEqual(TOOL_CALL_ERROR_TEMPLATE, "Error: {error}\n Please fix your mistakes.")
        self.assertIn("Error executing tool", TOOL_EXECUTION_ERROR_TEMPLATE)
        self.assertIn("Error invoking tool", TOOL_INVOCATION_ERROR_TEMPLATE)
        self.assertIn("text", TOOL_MESSAGE_BLOCK_TYPES)

    def test_tool_call_objects(self):
        request = ToolCallRequest(tool_call={"name": "add"}, tool=add)
        self.assertEqual(request.tool_call["name"], "add")
        with_context = ToolCallWithContext(tool_call={"name": "add"}, state={"v": 1})
        self.assertEqual(with_context.state, {"v": 1})
        self.assertTrue(issubclass(ToolInvocationError, Exception))

    def test_executor_alias(self):
        import langgraph_rs as lg

        # `create_react_agent` is now a Python wrapper that applies the facade, so compare the
        # behaviour-producing wrapper rather than the native function identity.
        self.assertTrue(create_tool_calling_executor is lg.create_react_agent)

    def test_get_all_basemodel_annotations(self):
        try:
            from pydantic import BaseModel
        except ImportError:  # pragma: no cover - pydantic is optional
            self.skipTest("pydantic is not installed")

        class Model(BaseModel):
            run_id: str | None = None

        self.assertIn("run_id", get_all_basemodel_annotations(Model))


if __name__ == "__main__":
    unittest.main()