import asyncio
import unittest

import langgraph_rs as lg


class AsyncStreamTests(unittest.TestCase):
    def test_astream_is_both_awaitable_and_async_iterable(self):
        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile()
            # LangGraph's `astream` is an async generator (not awaitable) and defaults to
            # `updates`, so one superstep yields one chunk.
            streamed = [chunk async for chunk in compiled.astream({"value": 0})]
            self.assertEqual(streamed, [{"inc": {"value": 1}}])

        asyncio.run(scenario())

    def test_async_builder_chains_and_astream_defaults_to_updates(self):
        async def scenario():
            graph = (
                lg.AsyncStateGraph({"value": "last_value"})
                .add_node("inc", lambda state: {"value": state["value"] + 1})
                .set_entry_point("inc")
                .set_finish_point("inc")
            )
            compiled = graph.compile()
            chunks = [chunk async for chunk in compiled.astream({"value": 0})]
            self.assertTrue(chunks)
            self.assertTrue(all(isinstance(chunk, dict) for chunk in chunks))

        asyncio.run(scenario())

    def test_async_state_accessors(self):
        async def scenario():
            graph = (
                lg.AsyncStateGraph({"value": "last_value"})
                .add_node("inc", lambda state: {"value": state["value"] + 1})
                .set_entry_point("inc")
                .set_finish_point("inc")
            )
            compiled = graph.compile(checkpointer=lg.InMemorySaver())
            config = {"configurable": {"thread_id": "async-state"}}
            await compiled.ainvoke({"value": 0}, thread_id="async-state")
            snapshot = await compiled.aget_state(config)
            self.assertEqual(snapshot["values"]["value"], 1)
            history = await compiled.aget_state_history(config)
            self.assertTrue(history)

        asyncio.run(scenario())

    def test_v2_debug_payload_shape(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        # LangGraph only reports `checkpoint` debug entries when a checkpointer persists them,
        # so the graph needs one for the checkpoint shape to exist at all.
        chunks = graph.compile(checkpointer=lg.InMemorySaver()).stream(
            {"value": 0},
            {"configurable": {"thread_id": "v2-debug"}},
            stream_mode=["debug"],
            version="v2",
        )
        debug = [chunk for chunk in chunks if chunk["type"] == "debug"]
        self.assertTrue(debug)
        kinds = {chunk["data"]["type"] for chunk in debug}
        self.assertIn("checkpoint", kinds)
        self.assertIn("task", kinds)
        self.assertIn("task_result", kinds)
        checkpoint = next(
            chunk for chunk in debug if chunk["data"]["type"] == "checkpoint"
        )
        self.assertIn("payload", checkpoint["data"])
        self.assertIn("timestamp", checkpoint["data"])


    def test_astream_can_resume_thread(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value", "log": "append"})
            graph.add_node("a", lambda state: {"value": 1, "log": ["a"]})
            graph.add_node("b", lambda state: {"value": 2, "log": ["b"]})
            graph.set_entry_point("a")
            graph.add_edge("a", "b")
            graph.set_finish_point("b")
            compiled = graph.compile(
                checkpointer=lg.InMemorySaver(), interrupt_before=["b"]
            )
            # Static breakpoints pause without raising.
            await compiled.ainvoke(
                {"value": 0, "log": []}, thread_id="async-stream-resume"
            )
            chunks = [
                chunk async for chunk in compiled.astream(None, thread_id="async-stream-resume")
            ]
            self.assertTrue(chunks)
            snapshot = await compiled.aget_state("async-stream-resume")
            self.assertEqual(snapshot["values"]["log"], ["a", "b"])

        asyncio.run(scenario())

    def test_ainvoke_command_send_on_existing_thread_keeps_state(self):
        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value", "log": "append"})
            graph.add_node("a", lambda state: {"value": 1, "log": ["a"]})
            graph.add_node(
                "b", lambda state: {"value": state["value"] + 10, "log": ["b"]}
            )
            graph.set_entry_point("a")
            graph.set_finish_point("a")
            graph.set_finish_point("b")
            compiled = graph.compile(checkpointer=lg.InMemorySaver())
            await compiled.ainvoke({"value": 0, "log": []}, thread_id="async-command")
            out = await compiled.ainvoke(
                lg.Command(goto=[lg.Send("b")]), thread_id="async-command"
            )
            self.assertEqual(out["log"], ["a", "b"])
            self.assertEqual(out["value"], 11)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
