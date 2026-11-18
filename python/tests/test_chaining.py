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
        self.assertEqual(snapshot.get("missing", "fallback"), "fallback")


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

    def test_messages_state_subclass_inherits_messages_channel(self):
        from typing import Annotated

        class MyState(lg.MessagesState):
            extra: Annotated[str, "last_value"]

        graph = lg.StateGraph(MyState)
        graph.add_node(
            "a",
            lambda state: {
                "messages": [{"id": "1", "content": "x"}],
                "extra": "e",
            },
        )
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        output = graph.compile().invoke({"messages": [], "extra": ""})
        self.assertEqual(output["messages"][0]["content"], "x")
        self.assertEqual(output["extra"], "e")

    def test_config_dict_covers_stream_history_and_update(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        config = {
            "configurable": {"thread_id": "config-everywhere"},
            "recursion_limit": 5,
        }
        self.assertTrue(compiled.stream({"value": 0}, config))
        history = compiled.get_state_history(config)
        self.assertEqual(history[0]["values"]["value"], 1)
        compiled.update_state(config, {"value": 7})
        self.assertEqual(compiled.get_state(config)["values"]["value"], 7)

    def test_command_resume_with_goto(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})

        def ask(state):
            value = lg.interrupt({"q": 1})
            return {"value": value, "log": ["ask"]}

        def other(state):
            return {"log": ["other"]}

        graph.add_node("ask", ask)
        graph.add_node("other", other)
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        graph.set_finish_point("other")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0, "log": []}, thread_id="resume-goto")
        output = compiled.invoke(
            lg.Command(resume=5, goto="other"), thread_id="resume-goto"
        )
        self.assertEqual(output["value"], 5)
        self.assertIn("other", output["log"])


    def test_threaded_invoke_merges_and_resumes(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node("a", lambda state: {"value": 1, "log": ["a"]})
        graph.add_node("b", lambda state: {"value": 2, "log": ["b"]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        compiled = graph.compile(
            checkpointer=lg.InMemorySaver(), interrupt_before=["b"]
        )
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke(
                {"value": 0, "log": []}, thread_id="threaded-invoke"
            )
        output = compiled.invoke(None, thread_id="threaded-invoke")
        self.assertEqual(output["log"], ["a", "b"])

        merged = lg.StateGraph({"value": "last_value", "log": "append"})
        merged.add_node("a", lambda state: {"value": state["value"] + 1, "log": ["a"]})
        merged.set_entry_point("a")
        merged.set_finish_point("a")
        compiled_merged = merged.compile(checkpointer=lg.InMemorySaver())
        compiled_merged.invoke({"value": 0, "log": []}, thread_id="merge")
        output = compiled_merged.invoke({"log": ["b"]}, thread_id="merge")
        self.assertEqual(output["log"], ["a", "b", "a"])


class ReactAgentModelTests(unittest.TestCase):
    def test_create_react_agent_accepts_bind_tools_model(self):
        class Message:
            def __init__(self, value):
                self.value = value

            def model_dump(self):
                return self.value

        class Model:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                if any(message.get("role") == "tool" for message in messages):
                    return Message({"role": "assistant", "content": "done"})
                return Message(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "1",
                            }
                        ],
                    }
                )

        graph = lg.create_react_agent(
            Model(), {"add": lambda args: args["a"] + args["b"]}
        )
        output = graph.invoke({"messages": []})
        self.assertEqual(len(output["messages"]), 3)


    def test_stream_can_resume_thread(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node("a", lambda state: {"value": 1, "log": ["a"]})
        graph.add_node("b", lambda state: {"value": 2, "log": ["b"]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        compiled = graph.compile(
            checkpointer=lg.InMemorySaver(), interrupt_before=["b"]
        )
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke(
                {"value": 0, "log": []}, thread_id="stream-resume"
            )
        chunks = compiled.stream(None, thread_id="stream-resume")
        self.assertTrue(chunks)
        self.assertEqual(
            compiled.get_state("stream-resume")["values"]["log"], ["a", "b"]
        )


    def test_stream_interrupt_does_not_raise(self):
        graph = lg.StateGraph({"value": "last_value"})

        def ask(state):
            lg.interrupt({"q": 1})
            return {"value": 1}

        graph.add_node("ask", ask)
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        chunks = compiled.stream(
            {"value": 0}, thread_id="stream-interrupt", stream_mode=["updates"]
        )
        self.assertTrue(chunks)
        self.assertTrue(
            any(
                isinstance(chunk, dict) and "__interrupt__" in chunk
                for chunk in chunks
            )
        )
        v2 = compiled.stream(
            {"value": 0},
            thread_id="stream-interrupt-v2",
            stream_mode=["updates"],
            version="v2",
        )
        updates = [chunk for chunk in v2 if chunk["type"] == "updates"]
        self.assertTrue(updates)
        self.assertIn("__interrupt__", updates[0]["data"])


class ReactAgentGuardTests(unittest.TestCase):
    def test_agent_stops_with_tool_calls_near_recursion_limit(self):
        def model(state):
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "noop", "args": {}, "id": "1"}],
                    }
                ]
            }

        graph = lg.create_react_agent(model, {"noop": lambda args: "ok"})
        output = graph.invoke({"messages": []}, config={"recursion_limit": 3})
        last = output["messages"][-1]
        self.assertIn("Sorry, need more steps", last.get("content", ""))


if __name__ == "__main__":
    unittest.main()
