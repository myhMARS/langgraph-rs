"""`langgraph.channels` channel objects are accepted as a state schema.

LangGraph lets state be declared as a mapping of channel instances; the extension maps those onto
its own channel kinds so names and reduction semantics stay identical to the equivalent TypedDict
form. Barrier channels are rejected loudly because the engine has no equivalent.
"""

from __future__ import annotations

import operator
import unittest

import langgraph_rs as lg
from langgraph_rs.channels import (
    AnyValue,
    BaseChannel,
    BinaryOperatorAggregate,
    DeltaChannel,
    EphemeralValue,
    LastValue,
    LastValueAfterFinish,
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
    Topic,
    UntrackedValue,
)


def _run(schema, node, value=None, writes=1):
    graph = lg.StateGraph(schema)
    names = []
    for index in range(writes):
        name = f"n{index}"
        names.append(name)
        graph.add_node(name, node)
    graph.set_entry_point(names[0])
    for left, right in zip(names, names[1:]):
        graph.add_edge(left, right)
    graph.set_finish_point(names[-1])
    payload = {} if value is None else dict(value)
    return graph.compile().invoke(payload)


class ChannelSchemaTest(unittest.TestCase):
    def test_last_value_keeps_the_last_write(self):
        self.assertEqual(_run({"v": LastValue(int)}, lambda s: {"v": 7}, {"v": 1}, writes=2), {"v": 7})

    def test_binary_operator_aggregate_accumulates(self):
        schema = {"total": BinaryOperatorAggregate(int, operator.add)}
        self.assertEqual(_run(schema, lambda s: {"total": 2}, {"total": 1}), {"total": 3})
        self.assertEqual(_run(schema, lambda s: {"total": 2}, None, writes=2), {"total": 4})

    def test_topic_accumulates_as_a_list(self):
        schema = {"log": Topic(str, accumulate=True)}
        self.assertEqual(_run(schema, lambda s: {"log": ["a"]}, {"log": []}), {"log": ["a"]})

    def test_plain_channels_are_accepted(self):
        for channel in (AnyValue(int), LastValueAfterFinish(int), EphemeralValue(int), UntrackedValue(int)):
            schema = {"v": channel}
            self.assertEqual(_run(schema, lambda s: {"v": 5}, {"v": 0}), {"v": 5}, type(channel).__name__)

    def test_delta_channel_reduces(self):
        schema = {"items": DeltaChannel(lambda current, delta: list(current or []) + list(delta))}
        self.assertEqual(_run(schema, lambda s: {"items": ["x"]}, {"items": []}), {"items": ["x"]})

    def test_barrier_channels_raise(self):
        for channel in (NamedBarrierValue(int, {"a"}), NamedBarrierValueAfterFinish(int, {"a"})):
            with self.assertRaises(NotImplementedError):
                lg.StateGraph({"v": channel})

    def test_channel_attributes_match_langgraph(self):
        self.assertEqual(LastValue(int).typ, int)
        self.assertEqual(Topic(str, accumulate=True).accumulate, True)
        self.assertEqual(EphemeralValue(int).guard, True)
        self.assertIsInstance(BinaryOperatorAggregate(int, operator.add), BaseChannel)
        self.assertEqual(DeltaChannel(lambda a, b: b).snapshot_frequency, 100)


if __name__ == "__main__":
    unittest.main()