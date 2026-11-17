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


if __name__ == "__main__":
    unittest.main()
