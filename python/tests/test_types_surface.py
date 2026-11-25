"""`langgraph.types` typing surface and message helpers match LangGraph.

These names are mostly annotations, but they are imported by migrating code (`StreamPart`,
`CheckpointPayload`, `AgentState`), and `xxh3_128_hexdigest` must return the same digest because it
feeds cache keys.
"""

from __future__ import annotations

import unittest

import langgraph_rs as lg
from langgraph_rs.graph.message import REMOVE_ALL_MESSAGES, push_message
from langgraph_rs.prebuilt.chat_agent_executor import (
    AgentState,
    AgentStatePydantic,
    AgentStateWithStructuredResponse,
    AgentStateWithStructuredResponsePydantic,
)
from langgraph_rs.types import (
    CheckpointPayload,
    DebugPayload,
    Durability,
    GraphOutput,
    NS_SEP,
    Overwrite,
    StreamPart,
    TaskPayload,
    ValuesStreamPart,
    default_cache_key,
    omit_payload,
    xxh3_128_hexdigest,
)


class StreamPartTest(unittest.TestCase):
    def test_stream_part_has_seven_members(self):
        self.assertEqual(len(StreamPart.__args__), 7)
        self.assertEqual(list(ValuesStreamPart.__annotations__), ["type", "ns", "data", "interrupts"])

    def test_ns_sep_and_durability(self):
        self.assertEqual(NS_SEP, "|")
        self.assertEqual(Durability, __import__("typing").Literal["sync", "async", "exit"])

    def test_payload_key_sets(self):
        self.assertEqual(
            list(CheckpointPayload.__annotations__),
            ["config", "metadata", "values", "next", "parent_config", "tasks"],
        )
        self.assertEqual(
            list(TaskPayload.__annotations__),
            ["id", "name", "input", "triggers", "metadata"],
        )
        self.assertEqual(
            list(DebugPayload.__annotations__), ["step", "timestamp", "type", "payload"]
        )


class HelperTest(unittest.TestCase):
    def test_xxh3_digest(self):
        self.assertEqual(xxh3_128_hexdigest(b"abc"), "06b05ab6733a618578af5f94892f3950")

    def test_omit_payload_records_nothing(self):
        # LangGraph's TracePolicy helper always returns an empty payload.
        self.assertEqual(omit_payload({"a": 1}), {})
        self.assertEqual(omit_payload(None), {})

    def test_default_cache_key_is_stable(self):
        first = default_cache_key({"a": 1}, b=2)
        second = default_cache_key({"a": 1}, b=2)
        self.assertEqual(first, second)
        self.assertNotEqual(first, default_cache_key({"a": 2}, b=2))

    def test_overwrite_and_graph_output(self):
        self.assertEqual(Overwrite(5).value, 5)
        output = GraphOutput(value={"v": 1}, interrupts=[1])
        self.assertEqual((output.value, output.interrupts), ({"v": 1}, (1,)))


class AgentStateTest(unittest.TestCase):
    def test_agent_state_schemas(self):
        self.assertEqual(
            list(AgentState.__annotations__), ["messages", "remaining_steps", "is_last_step"]
        )
        self.assertEqual(
            list(AgentStateWithStructuredResponse.__annotations__),
            ["messages", "remaining_steps", "is_last_step", "structured_response"],
        )
        state = AgentStatePydanticWithResponse()
        self.assertEqual(state.messages, [])
        self.assertIsNone(state.structured_response)


def AgentStatePydanticWithResponse():
    return AgentStateWithStructuredResponsePydantic(messages=[])


class MessageHelperTest(unittest.TestCase):
    def test_remove_all_sentinel(self):
        self.assertEqual(REMOVE_ALL_MESSAGES, "__remove_all__")
        message = {"role": "assistant", "content": "x"}
        self.assertIs(push_message(message), message)

    def test_matches_langgraph(self):
        try:
            import langgraph.types as lg_types
            from langgraph.graph.message import REMOVE_ALL_MESSAGES as lg_sentinel
        except ImportError:  # pragma: no cover - langgraph is optional here
            self.skipTest("langgraph is not installed")
        self.assertEqual(REMOVE_ALL_MESSAGES, lg_sentinel)
        self.assertEqual(
            xxh3_128_hexdigest(b"payload"),
            lg_types.xxh3_128_hexdigest(b"payload"),
        )
        self.assertEqual(omit_payload({"a": 1, "b": None}), lg_types.omit_payload({"a": 1, "b": None}))
        self.assertEqual(
            sorted(ValuesStreamPart.__annotations__),
            sorted(lg_types.ValuesStreamPart.__annotations__),
        )


if __name__ == "__main__":
    unittest.main()