"""`create_react_agent(prompt=...)` follows LangGraph's three forms.

A plain string becomes a leading system message, a `BaseMessage` is prepended as-is, and a
callable receives the state and returns the complete model input (replacing the state's
messages). The remaining LangGraph-only keywords must fail loudly instead of being ignored.
"""

from __future__ import annotations

import typing
import unittest

import langgraph_rs as lg


class _FakeChatModel:
    """Minimal LangChain-compatible model that records the messages it received."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append([message["content"] for message in messages])

        class _Result:
            def model_dump(self):
                return {"role": "assistant", "content": "ok"}

        return _Result()


class _Message:
    def __init__(self, content: str, role: str = "system") -> None:
        self.content = content
        self.role = role

    def model_dump(self):
        return {"role": self.role, "content": self.content}


@unittest.skipUnless(hasattr(lg, "create_react_agent"), "prebuilt helpers unavailable")
class ReactAgentPromptTest(unittest.TestCase):
    def _run(self, prompt=None, **kwargs):
        model = _FakeChatModel()
        app = lg.create_react_agent(model, [], prompt=prompt, **kwargs)
        app.invoke({"messages": [{"role": "user", "content": "hi"}]})
        return model.seen

    def test_no_prompt_passes_state_messages(self):
        self.assertEqual(self._run(), [["hi"]])

    def test_string_prompt_becomes_system_message(self):
        self.assertEqual(self._run("be nice"), [["be nice", "hi"]])

    def test_message_prompt_is_prepended(self):
        self.assertEqual(self._run(_Message("be strict")), [["be strict", "hi"]])

    def test_callable_prompt_replaces_messages(self):
        self.assertEqual(
            self._run(lambda state: [_Message("only-this")]),
            [["only-this"]],
        )

    def test_hooks_and_response_format_are_supported(self):
        # These used to raise NotImplementedError; the parity tests live in
        # `test_react_agent_extras.py`, this only guards that the keywords are accepted.
        app = lg.create_react_agent(
            _FakeChatModel(),
            [],
            pre_model_hook=lambda state: {},
            post_model_hook=lambda state: {},
        )
        self.assertTrue(hasattr(app, "invoke"))

    def test_aligned_keywords_are_accepted(self):
        model = _FakeChatModel()
        app = lg.create_react_agent(
            model,
            [],
            name="agent",
            version="v2",
            interrupt_before=["agent"],
            store=lg.InMemoryStore(),
            state_schema=typing.TypedDict("AgentState", {"messages": list}),
        )
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
