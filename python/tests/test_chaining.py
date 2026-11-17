import unittest

import langgraph_rs as lg


class ChainingTests(unittest.TestCase):
    def test_builder_methods_chain(self):
        graph = (
            lg.StateGraph({"value": "last_value"})
            .add_node("inc", lambda state: {"value": state["value"] + 1})
            .set_entry_point("inc")
            .set_finish_point("inc")
        )
        self.assertEqual(graph.compile().invoke({"value": 0})["value"], 1)

    def test_compiled_state_graph_alias(self):
        self.assertTrue(hasattr(lg, "CompiledStateGraph"))

    def test_invoke_accepts_config_dict(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        output = compiled.invoke(
            {"value": 0}, {"configurable": {"thread_id": "chain-config"}}
        )
        self.assertEqual(output["value"], 1)

    def test_state_snapshot_supports_attributes_and_mapping(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        compiled.invoke({"value": 0}, {"configurable": {"thread_id": "snapshot"}})
        snapshot = compiled.get_state("snapshot")
        self.assertEqual(snapshot["values"]["value"], 1)
        self.assertEqual(snapshot.values["value"], 1)
        self.assertEqual(list(snapshot.next), [])
        self.assertEqual(
            snapshot.config["configurable"]["thread_id"], "snapshot"
        )
        self.assertIn("values", snapshot)


class MessagesStateTests(unittest.TestCase):
    def test_messages_state_can_be_subclassed_as_schema(self):
        class MyState(lg.MessagesState):
            pass

        graph = lg.StateGraph(MyState)
        graph.add_node(
            "a", lambda state: {"messages": [{"id": "1", "content": "x"}]}
        )
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        output = graph.compile().invoke({"messages": []})
        self.assertEqual(output["messages"][0]["content"], "x")


if __name__ == "__main__":
    unittest.main()
