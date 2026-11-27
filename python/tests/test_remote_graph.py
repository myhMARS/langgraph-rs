"""`RemoteGraph`: the LangGraph Server client, exercised with injected fake SDK clients.

A live deployment is not available in tests, so the fake clients record the arguments they receive
and return SDK-shaped `StreamPart` chunks — which is exactly the seam `RemoteGraph` talks to.
"""

from __future__ import annotations

import asyncio
import unittest
from collections import namedtuple

from langgraph_rs.pregel.remote import RemoteException, RemoteGraph

StreamPart = namedtuple("StreamPart", "event data id")


class FakeRuns:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.chunks)


class FakeAsyncRuns(FakeRuns):
    async def stream(self, **kwargs):  # an async generator, like the SDK's
        self.calls.append(kwargs)
        for chunk in self.chunks:
            yield chunk


class FakeThreads:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_state(self, thread_id, **kwargs):
        self.calls.append(("get_state", {"thread_id": thread_id, **kwargs}))
        return {"values": {"x": 1}, "checkpoint_id": kwargs.get("checkpoint_id")}

    async def aget_state(self, thread_id, **kwargs):
        return self.get_state(thread_id, **kwargs)

    def get_history(self, thread_id, **kwargs):
        self.calls.append(("get_history", {"thread_id": thread_id, **kwargs}))
        return [{"values": {"x": 1}}]

    async def aget_history(self, thread_id, **kwargs):
        return self.get_history(thread_id, **kwargs)

    def update_state(self, thread_id, values, **kwargs):
        self.calls.append(("update_state", {"thread_id": thread_id, "values": values, **kwargs}))
        return {"checkpoint_id": "ck-2"}

    async def aupdate_state(self, thread_id, values, **kwargs):
        return self.update_state(thread_id, values, **kwargs)


class FakeAssistants:
    def get_graph(self, assistant_id, **kwargs):
        return {"nodes": [{"id": assistant_id}], "edges": []}

    async def aget_graph(self, assistant_id, **kwargs):
        return self.get_graph(assistant_id, **kwargs)


class FakeClient:
    def __init__(self, chunks=()):
        self.runs = FakeRuns(list(chunks))
        self.threads = FakeThreads()
        self.assistants = FakeAssistants()


def _graph(chunks=()):
    client = FakeClient(chunks)
    return RemoteGraph("agent", client=client, sync_client=client), client


CONFIG = {"configurable": {"thread_id": "t-1"}}


class RemoteGraphStreamTest(unittest.TestCase):
    def test_invoke_returns_the_last_values_chunk(self):
        graph, client = _graph(
            [
                StreamPart("values", {"x": 1}, None),
                StreamPart("values", {"x": 2}, None),
            ]
        )
        self.assertEqual(graph.invoke({"x": 0}, CONFIG), {"x": 2})
        call = client.runs.calls[-1]
        self.assertEqual(call["assistant_id"], "agent")
        self.assertEqual(call["thread_id"], "t-1")
        self.assertEqual(call["stream_mode"], ["values"])

    def test_v1_stream_yields_payloads_for_a_single_mode(self):
        graph, _ = _graph([StreamPart("updates", {"node": {"x": 1}}, None)])
        self.assertEqual(list(graph.stream({}, CONFIG, stream_mode="updates")), [{"node": {"x": 1}}])

    def test_v1_stream_yields_mode_and_payload_for_multiple_modes(self):
        graph, _ = _graph(
            [
                StreamPart("values", {"x": 1}, None),
                StreamPart("updates", {"node": {"x": 2}}, None),
            ]
        )
        self.assertEqual(
            list(graph.stream({}, CONFIG, stream_mode=["values", "updates"])),
            [("values", {"x": 1}), ("updates", {"node": {"x": 2}})],
        )

    def test_v2_stream_yields_stream_part_dicts_with_namespaces(self):
        graph, _ = _graph(
            [
                StreamPart("values", {"x": 1}, None),
                StreamPart("values|sub:1|inner:2", {"x": 3}, None),
            ]
        )
        parts = list(graph.stream({}, CONFIG, stream_mode="values", version="v2"))
        self.assertEqual(
            parts[0], {"type": "values", "ns": (), "data": {"x": 1}, "interrupts": ()}
        )
        self.assertEqual(parts[1]["ns"], ("sub:1", "inner:2"))

    def test_error_chunks_raise_remote_exception(self):
        graph, _ = _graph([StreamPart("error", {"message": "boom"}, None)])
        with self.assertRaises(RemoteException):
            list(graph.stream({}, CONFIG, stream_mode="values"))

    def test_astream_and_ainvoke(self):
        client = FakeClient()
        client.runs = FakeAsyncRuns([StreamPart("values", {"x": 5}, None)])
        graph = RemoteGraph("agent", client=client, sync_client=client)

        async def collect():
            values = [chunk async for chunk in graph.astream({}, CONFIG, stream_mode="values")]
            final = await graph.ainvoke({}, CONFIG)
            return values, final

        values, final = asyncio.run(collect())
        self.assertEqual(values, [{"x": 5}])
        self.assertEqual(final, {"x": 5})


class RemoteGraphStateTest(unittest.TestCase):
    def test_get_state_passes_thread_and_checkpoint(self):
        graph, client = _graph()
        config = {"configurable": {"thread_id": "t-2", "checkpoint_id": "ck-1"}}
        graph.get_state(config)
        _, kwargs = client.threads.calls[-1]
        self.assertEqual(kwargs["thread_id"], "t-2")
        self.assertEqual(kwargs["checkpoint_id"], "ck-1")
        self.assertFalse(kwargs["subgraphs"])

    def test_update_state_passes_values_and_as_node(self):
        graph, client = _graph()
        graph.update_state(CONFIG, {"x": 9}, as_node="writer")
        _, kwargs = client.threads.calls[-1]
        self.assertEqual(kwargs["values"], {"x": 9})
        self.assertEqual(kwargs["as_node"], "writer")

    def test_get_state_history_and_graph(self):
        graph, client = _graph()
        self.assertEqual(graph.get_state_history(CONFIG, limit=2), [{"values": {"x": 1}}])
        self.assertEqual(client.threads.calls[-1][1]["limit"], 2)
        self.assertEqual(graph.get_graph()["nodes"], [{"id": "agent"}])

    def test_missing_client_raises(self):
        graph = RemoteGraph("agent")
        with self.assertRaises(ValueError):
            graph.get_state(CONFIG)
        with self.assertRaises(ValueError):
            list(graph.stream({}, CONFIG))

    def test_with_config_merges_and_copy_isolates(self):
        graph, _ = _graph()
        configured = graph.with_config({"configurable": {"thread_id": "t-3"}})
        self.assertEqual(configured.config["configurable"]["thread_id"], "t-3")
        self.assertEqual(graph.config, {})
        self.assertEqual(configured.assistant_id, "agent")
        self.assertEqual(repr(configured), "RemoteGraph('agent', name='agent')")


if __name__ == "__main__":
    unittest.main()

class RemoteGraphSurfaceTest(unittest.TestCase):
    """`RemoteGraph` exposes the same public surface as LangGraph's."""

    def test_surface_matches_langgraph(self):
        from langgraph.pregel.remote import RemoteGraph as LgRemoteGraph
        from langgraph_rs.pregel.remote import RemoteGraph

        lg_names = {name for name in dir(LgRemoteGraph) if not name.startswith("_")}
        rs_names = {name for name in dir(RemoteGraph) if not name.startswith("_")}
        self.assertEqual(sorted(lg_names - rs_names), [])
        self.assertEqual(sorted(rs_names - lg_names), [])

    def test_bulk_update_state_uses_the_client(self):
        from langgraph_rs.pregel.remote import RemoteGraph

        calls = []

        class Threads:
            def bulk_update_state(self, thread_id, supersteps, headers=None):
                calls.append((thread_id, supersteps))
                return {"configurable": {"thread_id": thread_id}}

        class Client:
            threads = Threads()

        graph = RemoteGraph("agent", client=Client(), sync_client=Client())
        result = graph.bulk_update_state(
            {"configurable": {"thread_id": "t1"}}, [[({"value": 1}, "a", None)]]
        )
        self.assertEqual(result, {"configurable": {"thread_id": "t1"}})
        self.assertEqual(calls, [("t1", [[({"value": 1}, "a", None)]])])
