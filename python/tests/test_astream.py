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
            awaited = await compiled.astream({"value": 0})
            self.assertIsInstance(awaited, list)
            self.assertTrue(awaited)
            streamed = []
            async for chunk in compiled.astream({"value": 0}):
                streamed.append(chunk)
            self.assertTrue(streamed)

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
            chunks = await compiled.astream({"value": 0})
            self.assertTrue(chunks)
            self.assertEqual({chunk.get("type") for chunk in chunks}, {"updates"})

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
        chunks = graph.compile().stream(
            {"value": 0}, stream_mode=["debug"], version="v2"
        )
        debug = [chunk for chunk in chunks if chunk["type"] == "debug"]
        self.assertTrue(debug)
        self.assertEqual(debug[0]["data"]["type"], "checkpoint")
        self.assertIn("payload", debug[0]["data"])
        self.assertIn("timestamp", debug[0]["data"])


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
            try:
                await compiled.ainvoke(
                    {"value": 0, "log": []}, thread_id="async-stream-resume"
                )
            except lg.GraphInterrupt:
                pass
            chunks = await compiled.astream(
                None, thread_id="async-stream-resume"
            )
            self.assertTrue(chunks)
            snapshot = await compiled.aget_state("async-stream-resume")
            self.assertEqual(snapshot["values"]["log"], ["a", "b"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
