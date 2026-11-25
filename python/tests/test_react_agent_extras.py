"""`create_react_agent` extras: `response_format`, `pre_model_hook`, `post_model_hook`, `context_schema`.

Every case runs the same fake chat model through LangGraph and through `langgraph_rs`, and asserts
that both engines end up with the same state.
"""

from __future__ import annotations

import typing
import unittest

import langgraph_rs as rs
from langgraph_rs.prebuilt import create_react_agent

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent

    _LANGCHAIN = True
except Exception:  # pragma: no cover - optional dependency
    _LANGCHAIN = False


class Answer(typing.TypedDict):
    text: str


class ExtraState(typing.TypedDict, total=False):
    messages: typing.Annotated[list, lambda left, right: (left or []) + (right or [])]
    trail: str
    # LangGraph requires the managed `remaining_steps` channel in a custom state schema.
    remaining_steps: int


@unittest.skipUnless(_LANGCHAIN, "langchain_core / langgraph are not installed")
class ReactAgentExtrasTest(unittest.TestCase):
    def _model(self, responses):
        class FakeModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

            def with_structured_output(self, schema, **kwargs):
                return self

        return FakeModel(responses=responses)

    def _noop(self):
        @tool
        def noop(x: str) -> str:
            """Does nothing."""
            return x

        return noop

    def _contents(self, state):
        return [
            message.content if hasattr(message, "content") else message.get("content")
            for message in state["messages"]
        ]

    def test_response_format_sets_structured_response(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        lg_out = lg_create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], response_format=Answer
        ).invoke(payload)
        rs_out = create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], response_format=Answer
        ).invoke(payload)
        self.assertIn("structured_response", lg_out)
        self.assertIn("structured_response", rs_out)
        self.assertEqual(self._contents(lg_out), self._contents(rs_out))

    def test_hooks_run_in_order_and_can_write_state(self):
        seen = []

        def pre_hook(state):
            seen.append("pre")
            return {"messages": [{"role": "system", "content": "pre-hook"}]}

        def post_hook(state):
            seen.append("post")
            return {"trail": "post-hook"}

        payload = {"messages": [{"role": "user", "content": "hi"}]}
        kwargs = {
            "pre_model_hook": pre_hook,
            "post_model_hook": post_hook,
            "state_schema": ExtraState,
        }
        lg_out = lg_create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], **kwargs
        ).invoke(payload)
        lg_seen = list(seen)
        seen.clear()
        rs_out = create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], **kwargs
        ).invoke(payload)

        self.assertEqual(lg_seen, ["pre", "post"])
        self.assertEqual(seen, lg_seen)
        self.assertEqual(lg_out.get("trail"), "post-hook")
        self.assertEqual(rs_out.get("trail"), "post-hook")
        self.assertEqual(self._contents(lg_out), self._contents(rs_out))

    def test_hooks_keep_the_tool_loop_working(self):
        seen = []

        def pre_hook(state):
            seen.append("pre")
            return {}

        def post_hook(state):
            seen.append("post")
            return {}

        call = AIMessage(
            content="",
            tool_calls=[{"name": "noop", "args": {"x": "1"}, "id": "c1"}],
        )
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        kwargs = {"pre_model_hook": pre_hook, "post_model_hook": post_hook}

        lg_out = lg_create_react_agent(
            self._model([call, AIMessage(content="done")]), [self._noop()], **kwargs
        ).invoke(payload)
        lg_seen = list(seen)
        seen.clear()
        rs_out = create_react_agent(
            self._model([call, AIMessage(content="done")]), [self._noop()], **kwargs
        ).invoke(payload)

        # LangGraph routes `tools -> pre_model_hook`, so the pre hook runs once per model call.
        self.assertEqual(lg_seen, ["pre", "post", "pre", "post"])
        self.assertEqual(seen, lg_seen)
        self.assertEqual(self._contents(lg_out), self._contents(rs_out))

    def test_context_schema_is_accepted(self):
        class Ctx(typing.TypedDict):
            user: str

        payload = {"messages": [{"role": "user", "content": "hi"}]}
        # Both engines accept `context_schema`; LangGraph validates it at run time, the extension
        # passes it to the engine's context validation.
        lg_out = lg_create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], context_schema=Ctx
        ).invoke(payload)
        rs_out = create_react_agent(
            self._model([AIMessage(content="hello")]), [self._noop()], context_schema=Ctx
        ).invoke(payload)
        self.assertEqual(self._contents(lg_out), self._contents(rs_out))


if __name__ == "__main__":
    unittest.main()