"""Unit tests for the Python bindings.

Run them after building the extension module, e.g. with maturin::

    maturin develop --features python
    python -m unittest discover -s python/tests -v

or with a plain cargo build::

    cargo build --features python
    # copy target/debug/langgraph_rs.dll (or .so) to a directory as langgraph_rs.pyd
    python -m unittest discover -s python/tests -v
"""

import unittest

import langgraph_rs as lg


class BuilderTests(unittest.TestCase):
    def test_conditional_loop_until_end(self):
        graph = lg.StateGraph({"count": "last_value", "log": "append"})
        graph.add_node(
            "increment",
            lambda state: {"count": state["count"] + 1, "log": ["inc"]},
        )
        graph.set_entry_point("increment")
        graph.add_conditional_edges(
            "increment",
            lambda state: "increment" if state["count"] < 3 else None,
        )
        compiled = graph.compile()
        result = compiled.invoke({"count": 0, "log": []})
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["log"], ["inc", "inc", "inc"])
        self.assertEqual(compiled.node_names(), ["increment"])

    def test_static_edges_and_path_map(self):
        graph = lg.StateGraph({"outcome": "last_value", "log": "append"})
        graph.add_node("decide", lambda state: {"outcome": "ok"})
        graph.add_node("finish", lambda state: {"log": ["finish"]})
        graph.set_entry_point("decide")
        graph.add_conditional_edges(
            "decide",
            lambda state: state["outcome"],
            {"ok": "finish"},
        )
        compiled = graph.compile()
        result = compiled.invoke({"log": []})
        self.assertEqual(result["outcome"], "ok")
        self.assertEqual(result["log"], ["finish"])

    def test_messages_reducer(self):
        graph = lg.StateGraph({"messages": "messages"})
        graph.add_node(
            "first", lambda state: {"messages": [{"id": "a", "content": "one"}]}
        )
        graph.add_node(
            "second",
            lambda state: {
                "messages": [
                    {"id": "a", "content": "one-updated"},
                    {"id": "b", "content": "two"},
                ]
            },
        )
        graph.add_edge("START", "first")
        graph.add_edge("first", "second")
        graph.set_finish_point("second")
        compiled = graph.compile()
        result = compiled.invoke({"messages": []})
        self.assertEqual(
            result["messages"],
            [
                {"id": "a", "content": "one-updated"},
                {"id": "b", "content": "two"},
            ],
        )

    def test_node_can_return_goto(self):
        graph = lg.StateGraph({"log": "append"})
        graph.add_node("a", lambda state: {"log": ["a"], "__goto__": "c"})
        graph.add_node("b", lambda state: {"log": ["b"]})
        graph.add_node("c", lambda state: {"log": ["c"]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("c")
        compiled = graph.compile()
        result = compiled.invoke({"log": []})
        self.assertEqual(result["log"], ["a", "c"])

    def test_mermaid_export(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        mermaid = graph.compile().to_mermaid()
        self.assertIn("graph TD", mermaid)
        self.assertIn("START --> n_a", mermaid)

    def test_validation_errors_are_python_errors(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {})
        with self.assertRaises(RuntimeError):
            graph.compile()  # no entry point


class HumanInTheLoopTests(unittest.TestCase):
    def test_interrupt_and_resume(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node(
            "ask",
            lambda state: {
                "value": state["value"]
                + lg.interrupt({"question": "add how much?"})["amount"],
                "log": ["ask"],
            },
        )
        graph.add_node("finish", lambda state: {"log": ["finish"]})
        graph.set_entry_point("ask")
        graph.add_edge("ask", "finish")
        graph.set_finish_point("finish")
        compiled = graph.compile(checkpointer=True)

        with self.assertRaises(lg.GraphInterrupt) as raised:
            compiled.invoke({"value": 0, "log": []}, thread_id="approval")
        payload = raised.exception.args[0]
        self.assertEqual(payload, {"question": "add how much?"})

        snapshot = compiled.get_state("approval")
        self.assertTrue(snapshot["interrupted"])
        self.assertEqual(snapshot["next"], ["ask"])

        result = compiled.resume("approval", {"amount": 41})
        self.assertEqual(result["value"], 41)
        self.assertEqual(result["log"], ["ask", "finish"])
        self.assertFalse(compiled.get_state("approval")["interrupted"])

    def test_resume_with_updates(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "ask",
            lambda state: {"value": state["value"] + lg.interrupt("delta")},
        )
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        compiled = graph.compile(checkpointer=True)
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0}, thread_id="edits")
        result = compiled.resume_with_updates("edits", 7, {"value": 100})
        self.assertEqual(result["value"], 107)

    def test_update_state_and_history_filters(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state["value"] + 1})
        graph.add_node("b", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        compiled = graph.compile(checkpointer=True)
        compiled.invoke({"value": 0}, thread_id="patch")

        snapshot = compiled.update_state("patch", {"value": 10}, as_node="a")
        self.assertEqual(snapshot["values"]["value"], 10)
        self.assertEqual(snapshot["next"], ["b"])
        self.assertIsNotNone(snapshot["parent_checkpoint_id"])
        self.assertEqual(snapshot["metadata"]["source"], "update")

        continued = compiled.continue_thread("patch")
        self.assertEqual(continued["value"], 11)

        limited = compiled.get_state_history("patch", limit=1)
        self.assertEqual(len(limited), 1)
        full = compiled.get_state_history("patch")
        before = full[1]["checkpoint_id"]
        filtered = compiled.get_state_history("patch", before=before)
        self.assertNotIn(before, [item["checkpoint_id"] for item in filtered])

    def test_thread_delete_and_copy(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        compiled = graph.compile(checkpointer=True)
        compiled.invoke({"value": 0}, thread_id="copy-source")

        compiled.copy_thread("copy-source", "copy-target")
        copied = compiled.get_state("copy-target")
        self.assertEqual(copied["values"]["value"], 1)
        self.assertEqual(copied["checkpoint_ns"], "")

        compiled.delete_thread("copy-source")
        self.assertIsNone(compiled.get_state("copy-source"))
        self.assertIsNotNone(compiled.get_state("copy-target"))

    def test_checkpoint_history(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": 1})
        graph.add_node("b", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        compiled = graph.compile(checkpointer=True)
        compiled.invoke({"value": 0}, thread_id="history")
        history = compiled.get_state_history("history")
        self.assertGreaterEqual(len(history), 3)
        self.assertEqual(history[0]["values"]["value"], 2)
        self.assertTrue(compiled.has_checkpointer())


class UsageTests(unittest.TestCase):
    def test_usage_report_per_node(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "planner",
            lambda state: {
                "value": 1,
                "__usage__": {
                    "model": "gpt-4o",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cost_usd": 0.01,
                },
            },
        )
        graph.add_node(
            "writer",
            lambda state: {"value": 2, "__usage__": {"input_tokens": 10, "output_tokens": 5}},
        )
        graph.add_edge("START", "planner")
        graph.add_edge("planner", "writer")
        graph.set_finish_point("writer")
        compiled = graph.compile()

        result = compiled.run({"value": 0})
        self.assertEqual(result["state"]["value"], 2)
        report = result["report"]
        self.assertEqual(len(report["executions"]), 2)
        planner = report["executions"][0]
        self.assertEqual(planner["node"], "planner")
        self.assertEqual(planner["step"], 0)
        self.assertEqual(planner["usage"]["model"], "gpt-4o")
        self.assertEqual(planner["usage"]["input_tokens"], 100)
        self.assertEqual(report["executions"][1]["usage"]["output_tokens"], 5)

    def test_record_tokens_helper(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "call",
            lambda state: (lg.record_tokens(3, 4), {"value": 1})[1],
        )
        graph.set_entry_point("call")
        graph.set_finish_point("call")
        compiled = graph.compile()
        result = compiled.run({"value": 0})
        usage = result["report"]["executions"][0]["usage"]
        self.assertEqual(usage["input_tokens"], 3)
        self.assertEqual(usage["output_tokens"], 4)


class StreamingTests(unittest.TestCase):
    def test_stream_emits_chunks(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "emit",
            lambda state: (
                lg.custom({"progress": 0.5}),
                {"value": 1, "__usage__": {"input_tokens": 2}},
            )[1],
        )
        graph.set_entry_point("emit")
        graph.set_finish_point("emit")
        compiled = graph.compile()
        chunks = compiled.stream({"value": 0})
        kinds = {chunk["type"] for chunk in chunks}
        self.assertIn("values", kinds)
        self.assertIn("updates", kinds)
        self.assertIn("custom", kinds)
        self.assertIn("usage", kinds)

    def test_stream_modes(self):
        graph = lg.StateGraph({"value": "last_value"})
        def emit(state):
            lg.custom({"progress": 1})
            lg.message({"token": "hi"})
            return {"value": state["value"] + 1}
        graph.add_node("emit", emit)
        graph.set_entry_point("emit")
        graph.set_finish_point("emit")
        compiled = graph.compile()

        values = compiled.stream({"value": 0}, stream_mode="values")
        self.assertTrue(all(chunk["type"] in ("values", "usage") for chunk in values))
        self.assertIn("values", {chunk["type"] for chunk in values})

        custom_modes = compiled.stream({"value": 0}, stream_mode=["values", "custom"])
        types = {chunk["type"] for chunk in custom_modes}
        self.assertIn("custom", types)
        self.assertIn("values", types)

        messages = compiled.stream({"value": 0}, stream_mode="messages")
        self.assertIn("messages", {chunk["type"] for chunk in messages})

    def test_module_metadata(self):
        self.assertEqual(lg.START, "__start__")
        self.assertEqual(lg.END, "__end__")
        self.assertIsInstance(lg.__version__, str)


class PythonApiParityTests(unittest.TestCase):
    def test_send_map_reduce(self):
        graph = lg.StateGraph({"log": "append", "value": "last_value"})
        graph.add_node(
            "dispatch",
            lambda state: [lg.Send("worker", {"value": item}) for item in (1, 2, 3)],
        )
        graph.add_node("worker", lambda state: {"log": [state["value"]]})
        graph.set_entry_point("dispatch")
        graph.set_finish_point("worker")
        output = graph.compile().invoke({"log": [], "value": 0})
        self.assertEqual(sorted(output["log"]), [1, 2, 3])

    def test_command_as_node_output(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node(
            "seed",
            lambda state: lg.Command(update={"value": 1}, goto="finish"),
        )
        graph.add_node("finish", lambda state: {"log": ["finish"]})
        graph.set_entry_point("seed")
        graph.set_finish_point("finish")
        output = graph.compile().invoke({"value": 0, "log": []})
        self.assertEqual(output["value"], 1)
        self.assertEqual(output["log"], ["finish"])

    def test_command_as_invoke_input(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "ask",
            lambda state: {"value": state["value"] + lg.interrupt("delta")},
        )
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        compiled = graph.compile(checkpointer=True)
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0}, thread_id="cmd-input")
        output = compiled.invoke(
            lg.Command(resume=5, update={"value": 10}), thread_id="cmd-input"
        )
        self.assertEqual(output["value"], 15)

    def test_retry_policy(self):
        calls = {"count": 0}

        def flaky(state):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError(f"transient {calls['count']}")
            return {"value": 7}

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "flaky",
            flaky,
            retry_policy=lg.RetryPolicy(
                max_attempts=3, initial_interval=0.001, backoff_factor=1.0
            ),
        )
        graph.set_entry_point("flaky")
        graph.set_finish_point("flaky")
        output = graph.compile().invoke({"value": 0})
        self.assertEqual(output["value"], 7)
        self.assertEqual(calls["count"], 3)

    def test_node_cache(self):
        calls = {"count": 0}

        def work(state):
            calls["count"] += 1
            return {"value": calls["count"]}

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("work", work, cache_policy=lg.CachePolicy(ttl=60))
        graph.set_entry_point("work")
        graph.set_finish_point("work")
        compiled = graph.compile(cache=True)
        first = compiled.invoke({"value": 0})
        second = compiled.invoke({"value": 0})
        self.assertEqual(first["value"], 1)
        self.assertEqual(second["value"], 1)
        self.assertEqual(calls["count"], 1)

    def test_error_handler(self):
        def broken(state):
            raise RuntimeError("boom")

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("work", broken, error_handler="fallback")
        graph.add_node("fallback", lambda state: {"value": -1})
        graph.set_entry_point("work")
        graph.set_finish_point("fallback")
        self.assertEqual(graph.compile().invoke({"value": 0})["value"], -1)

    def test_saver_object(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        saver = lg.InMemorySaver()
        compiled = graph.compile(checkpointer=saver)
        self.assertTrue(compiled.has_checkpointer())
        compiled.invoke({"value": 0}, thread_id="saver-object")
        self.assertEqual(compiled.get_state("saver-object")["values"]["value"], 1)

    def test_add_messages_alias(self):
        graph = lg.StateGraph({"messages": lg.add_messages})
        graph.add_node(
            "append",
            lambda state: {"messages": [{"id": "1", "content": "new"}]},
        )
        graph.set_entry_point("append")
        graph.set_finish_point("append")
        output = graph.compile().invoke(
            {"messages": [{"id": "1", "content": "old"}]}
        )
        self.assertEqual(len(output["messages"]), 1)
        self.assertEqual(output["messages"][0]["content"], "new")

    def test_sqlite_saver_when_available(self):
        if not hasattr(lg, "SqliteSaver"):
            self.skipTest("built without the sqlite feature")
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("a")
        graph.set_finish_point("a")
        compiled = graph.compile(checkpointer=lg.SqliteSaver.in_memory())
        compiled.invoke({"value": 0}, thread_id="sqlite-saver")
        self.assertEqual(compiled.get_state("sqlite-saver")["values"]["value"], 1)

    def test_batch_invokes_each_input(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        results = graph.compile().batch([{"value": 0}, {"value": 5}, {"value": 10}])
        self.assertEqual([item["value"] for item in results], [1, 6, 11])


    def test_channel_spec_aliases(self):
        graph = lg.StateGraph(
            {"pulse": "ephemeral_value", "value": "any_value", "seen": "last_value"}
        )
        graph.add_node(
            "write", lambda state: {"pulse": 42, "value": 1, "seen": -1}
        )
        graph.add_node("read", lambda state: {"seen": state.get("pulse", 0)})
        graph.set_entry_point("write")
        graph.add_edge("write", "read")
        graph.set_finish_point("read")
        output = graph.compile().invoke({"pulse": 0, "value": 0, "seen": 0})
        self.assertEqual(output["seen"], 0)

    def test_untracked_value_alias(self):
        graph = lg.StateGraph({"secret": "untracked", "seen": "last_value"})
        graph.add_node("write", lambda state: {"secret": 7})
        graph.add_node("read", lambda state: {"seen": state["secret"]})
        graph.set_entry_point("write")
        graph.add_edge("write", "read")
        graph.set_finish_point("read")
        output = graph.compile().invoke({"secret": 0, "seen": 0})
        self.assertEqual(output["seen"], 7)


class PythonParityRound2Tests(unittest.TestCase):
    def test_subgraph_nested_interrupt(self):
        child = lg.StateGraph({"value": "last_value"})
        child.add_node(
            "ask", lambda state: {"value": state["value"] + lg.interrupt("delta")}
        )
        child.set_entry_point("ask")
        child.set_finish_point("ask")
        saver = lg.InMemorySaver()
        child_compiled = child.compile(checkpointer=saver)

        parent = lg.StateGraph({"value": "last_value"})
        parent.add_node("seed", lambda state: {"value": 10})
        parent.add_subgraph("child", child_compiled)
        parent.set_entry_point("seed")
        parent.add_edge("seed", "child")
        parent.set_finish_point("child")
        compiled = parent.compile(checkpointer=saver)

        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0}, thread_id="py-sub")

        self.assertTrue(compiled.get_state("py-sub")["interrupted"])
        child_state = compiled.get_state("py-sub", checkpoint_ns="child")
        self.assertIsNotNone(child_state)
        self.assertTrue(child_state["interrupted"])

        output = compiled.resume("py-sub", 5)
        self.assertEqual(output["value"], 15)
        self.assertTrue(
            all(
                snapshot["checkpoint_ns"] == ""
                for snapshot in compiled.get_state_history("py-sub")
            )
        )
        child_history = compiled.get_state_history(
            "py-sub", checkpoint_ns="child"
        )
        self.assertGreater(len(child_history), 0)

    def test_store_long_term_memory(self):
        store = lg.InMemoryStore()
        graph = lg.StateGraph({"value": "last_value"})

        def remember(state):
            handle = lg.get_store()
            previous = handle.get(["mem"], "n") or 0
            value = previous + 1
            handle.put(["mem"], "n", value)
            return {"value": value}

        graph.add_node("remember", remember)
        graph.set_entry_point("remember")
        graph.set_finish_point("remember")
        compiled = graph.compile(store=store)

        self.assertEqual(compiled.invoke({"value": 0})["value"], 1)
        self.assertEqual(compiled.invoke({"value": 0})["value"], 2)
        self.assertEqual(store.get(["mem"], "n"), 2)
        items = store.search(["mem"])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["key"], "n")
        self.assertIn(["mem"], store.namespaces())
        store.delete(["mem"], "n")
        self.assertIsNone(store.get(["mem"], "n"))

    def test_runtime_context(self):
        graph = lg.StateGraph({"value": "last_value"})

        def inspect(state):
            runtime = lg.current_runtime()
            self.assertEqual(runtime["context"], {"tenant": "acme"})
            self.assertEqual(runtime["namespace"], "")
            self.assertIsNone(runtime["thread_id"])
            self.assertEqual(lg.current_context(), {"tenant": "acme"})
            self.assertIsNone(lg.get_store())
            return {"value": len(runtime["context"])}

        graph.add_node("inspect", inspect)
        graph.set_entry_point("inspect")
        graph.set_finish_point("inspect")
        output = graph.compile().invoke(
            {"value": 0}, context={"tenant": "acme"}
        )
        self.assertEqual(output["value"], 1)


    def test_router_can_return_send_tasks(self):
        graph = lg.StateGraph({"log": "append", "value": "last_value"})
        graph.add_node("start", lambda state: {})
        graph.add_node("worker", lambda state: {"log": [state["value"]]})
        graph.add_conditional_edges(
            "start",
            lambda state: [
                lg.Send("worker", {"value": 1}),
                lg.Send("worker", {"value": 2}),
            ],
        )
        graph.set_entry_point("start")
        graph.set_finish_point("worker")
        output = graph.compile().invoke({"log": [], "value": 0})
        self.assertEqual(sorted(output["log"]), [1, 2])


    def test_managed_values(self):
        graph = lg.StateGraph({"value": "last_value", "last": "last_value"})
        graph.add_node(
            "inspect",
            lambda state: {
                "value": lg.remaining_steps(),
                "last": int(lg.is_last_step()),
            },
        )
        graph.set_entry_point("inspect")
        graph.set_finish_point("inspect")
        output = graph.compile(recursion_limit=2).invoke(
            {"value": 0, "last": 0}
        )
        self.assertEqual(output["value"], 2)
        self.assertEqual(output["last"], 0)


    def test_deferred_node_runs_after_regular_nodes(self):
        graph = lg.StateGraph({"value": "last_value", "log": "append"})
        graph.add_node(
            "inc",
            lambda state: {"value": state["value"] + 1, "log": ["inc"]},
        )
        graph.add_node(
            "summary",
            lambda state: {
                "value": 100,
                "log": ["summary after {}".format(state["value"])],
            },
            deferred=True,
        )
        graph.set_entry_point("inc")
        graph.add_edge("inc", "summary")
        graph.set_finish_point("inc")
        output = graph.compile().invoke({"value": 0, "log": []})
        self.assertEqual(output["value"], 100)
        self.assertEqual(output["log"], ["inc", "summary after 1"])


    def test_destinations_reject_undeclared_target(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("a", lambda state: {}, destinations=["b"])
        graph.add_node("b", lambda state: {})
        graph.add_node("c", lambda state: {})
        graph.set_entry_point("a")
        graph.add_edge("a", "c")
        graph.set_finish_point("b")
        graph.set_finish_point("c")
        with self.assertRaises(RuntimeError):
            graph.compile().invoke({"value": 0})


    def test_delta_and_finish_channels(self):
        graph = lg.StateGraph(
            {
                "tokens": "delta",
                "answer": "last_value_after_finish",
                "value": "last_value",
            }
        )
        graph.add_node(
            "a", lambda state: {"tokens": ["a"], "answer": "draft"}
        )
        graph.add_node(
            "b",
            lambda state: {
                "tokens": ["b"],
                "answer": "done",
                "value": state["answer"],
            },
        )
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")
        output = graph.compile().invoke(
            {"tokens": [], "answer": "", "value": ""}
        )
        self.assertEqual(output["tokens"], ["a", "b"])
        self.assertEqual(output["answer"], "done")
        self.assertEqual(output["value"], "")


    def test_subgraph_streaming(self):
        child = lg.StateGraph({"value": "last_value"})
        child.add_node("work", lambda state: {"value": state["value"] + 1})
        child.set_entry_point("work")
        child.set_finish_point("work")
        parent = lg.StateGraph({"value": "last_value"})
        parent.add_node("seed", lambda state: {"value": 10})
        parent.add_subgraph("child", child.compile())
        parent.set_entry_point("seed")
        parent.add_edge("seed", "child")
        parent.set_finish_point("child")
        chunks = parent.compile().stream({"value": 0}, subgraphs=True)
        namespaces = [
            chunk.get("namespace")
            for chunk in chunks
            if isinstance(chunk, dict) and "namespace" in chunk
        ]
        self.assertIn("child", namespaces)


    def test_functional_api_task_memoization(self):
        store = lg.InMemoryStore()
        calls = []

        def add(value):
            calls.append(value)
            return value + 1

        task = lg.task(add, name="add")
        graph = lg.entrypoint(lambda state: {"value": task(state["value"])})
        compiled = graph.compile(store=store)
        first = compiled.invoke({"value": 1})
        second = compiled.invoke({"value": 1})
        self.assertEqual(first["value"], 2)
        self.assertEqual(second["value"], 2)
        self.assertEqual(calls, [1])

    def test_functional_api_interrupt_skips_completed_tasks(self):
        saver = lg.InMemorySaver()
        store = lg.InMemoryStore()
        first_calls = []
        second_calls = []

        def first(value):
            first_calls.append(value)
            return value * 10

        def second(value):
            second_calls.append(value)
            return value + 1

        first_task = lg.task(first, name="first")
        second_task = lg.task(second, name="second")

        def entry(state):
            head = first_task(state["value"])
            delta = lg.interrupt({"question": "delta"})
            tail = second_task(delta)
            return {"value": head + tail}

        compiled = lg.entrypoint(entry).compile(checkpointer=saver, store=store)
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 1}, thread_id="py-functional")
        self.assertEqual(first_calls, [1])
        self.assertEqual(second_calls, [])

        output = compiled.resume("py-functional", 4)
        self.assertEqual(output["value"], 15)
        self.assertEqual(first_calls, [1])
        self.assertEqual(second_calls, [4])

    def test_optional_saver_classes(self):
        for name in ("RedisSaver", "PostgresSaver"):
            if hasattr(lg, name):
                self.assertTrue(callable(getattr(lg, name)))


    def test_multiple_interrupts_resume_by_id_map(self):
        graph = lg.StateGraph({"a": "last_value", "b": "last_value"})
        graph.add_node("a", lambda state: {"a": lg.interrupt({"q": "a"})})
        graph.add_node("b", lambda state: {"b": lg.interrupt({"q": "b"})})
        graph.add_edge("START", "a")
        graph.add_edge("START", "b")
        graph.set_finish_point("a")
        graph.set_finish_point("b")
        compiled = graph.compile(checkpointer=True)

        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"a": 0, "b": 0}, thread_id="py-multi")
        snapshot = compiled.get_state("py-multi")
        interrupts = snapshot["interrupts"]
        self.assertEqual(len(interrupts), 2)
        self.assertTrue(all(item["id"] for item in interrupts))

        mapping = {
            item["id"]: (5 if item["node"] == "a" else 7)
            for item in interrupts
        }
        output = compiled.resume("py-multi", mapping)
        self.assertEqual(output["a"], 5)
        self.assertEqual(output["b"], 7)


    def test_messages_state_and_remove_message(self):
        schema = getattr(lg, "MessagesState", {"messages": "add_messages"})
        graph = lg.StateGraph(schema)
        graph.add_node(
            "append",
            lambda state: {
                "messages": [
                    {"id": "1", "content": "a"},
                    {"id": "2", "content": "b"},
                ]
            },
        )
        graph.add_node(
            "remove", lambda state: {"messages": [lg.RemoveMessage("1")]}
        )
        graph.set_entry_point("append")
        graph.add_edge("append", "remove")
        graph.set_finish_point("remove")
        output = graph.compile().invoke({"messages": []})
        self.assertEqual(len(output["messages"]), 1)
        self.assertEqual(output["messages"][0]["id"], "2")

    def test_node_input_schema_keyword(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node(
            "work",
            lambda state: {"value": 1},
            input_schema={"type": "object", "properties": {"value": {"type": "integer"}}},
        )
        graph.set_entry_point("work")
        graph.set_finish_point("work")
        self.assertEqual(
            graph.compile().invoke({"value": 0})["value"], 1
        )


class AsyncPythonTests(unittest.TestCase):
    def test_async_invoke(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile()
            return await compiled.ainvoke({"value": 0})

        self.assertEqual(asyncio.run(scenario())["value"], 1)

    def test_async_interrupt_resume(self):
        import asyncio

        saver = lg.InMemorySaver()

        def ask(state):
            delta = lg.interrupt({"q": "delta"})
            return {"value": state["value"] + delta}

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("ask", ask)
            graph.set_entry_point("ask")
            graph.set_finish_point("ask")
            compiled = graph.compile(checkpointer=saver)
            with self.assertRaises(lg.GraphInterrupt):
                await compiled.ainvoke({"value": 0}, thread_id="py-async")
            return await compiled.aresume("py-async", 5)

        self.assertEqual(asyncio.run(scenario())["value"], 5)


    def test_async_stream_and_subgraph(self):
        import asyncio

        async def scenario():
            child = lg.AsyncStateGraph({"value": "last_value"})
            child.add_node("work", lambda state: {"value": state["value"] + 1})
            child.set_entry_point("work")
            child.set_finish_point("work")
            child_compiled = child.compile()

            parent = lg.AsyncStateGraph({"value": "last_value"})
            parent.add_node("seed", lambda state: {"value": 10})
            parent.add_subgraph("child", child_compiled)
            parent.set_entry_point("seed")
            parent.add_edge("seed", "child")
            parent.set_finish_point("child")
            compiled = parent.compile()

            chunks = await compiled.astream(
                {"value": 0}, subgraphs=True
            )
            namespaces = [
                chunk.get("namespace")
                for chunk in chunks
                if isinstance(chunk, dict) and "namespace" in chunk
            ]
            updates = await compiled.astream(
                {"value": 0}, stream_mode=["updates"]
            )
            return namespaces, updates

        namespaces, updates = asyncio.run(scenario())
        self.assertIn("child", namespaces)
        self.assertTrue(any(chunk.get("type") == "updates" for chunk in updates))


    def test_async_batch_and_message_graph(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile()
            return await compiled.abatch([{"value": 0}, {"value": 10}])

        results = asyncio.run(scenario())
        self.assertEqual([item["value"] for item in results], [1, 11])

        graph = lg.message_graph()
        graph.add_node("add", lambda state: {"messages": [{"id": "1", "content": "hi"}]})
        graph.set_entry_point("add")
        graph.set_finish_point("add")
        output = graph.compile().invoke({"messages": []})
        self.assertEqual(len(output["messages"]), 1)


    def test_node_can_interrupt_repeatedly_across_resumes(self):
        saver = lg.InMemorySaver()

        def ask(state):
            first = lg.interrupt({"q": 1})
            second = lg.interrupt({"q": 2})
            return {"value": first + second}

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("ask", ask)
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        compiled = graph.compile(checkpointer=saver)

        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0}, thread_id="py-sequential")
        with self.assertRaises(lg.GraphInterrupt):
            compiled.resume("py-sequential", 3)
        output = compiled.resume("py-sequential", 4)
        self.assertEqual(output["value"], 7)


    def test_compile_name_and_graph_export(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("work", lambda state: {"value": 1})
        graph.set_entry_point("work")
        graph.set_finish_point("work")
        compiled = graph.compile(name="demo")
        self.assertEqual(compiled.name, "demo")
        description = compiled.get_graph()
        self.assertIn("work", description.draw_mermaid())
        spec = description.to_dict()
        self.assertIn("work", [node["id"] for node in spec["nodes"]])
        self.assertTrue(hasattr(lg, "AsyncInMemorySaver"))


    def test_subgraph_command_parent(self):
        child = lg.StateGraph({"value": "last_value"})
        child.add_node(
            "jump",
            lambda state: lg.Command(
                update={"value": 100},
                goto="parent_after",
                graph=lg.Command.PARENT,
            ),
        )
        child.set_entry_point("jump")

        parent = lg.StateGraph({"value": "last_value"})
        parent.add_subgraph("child", child.compile())
        parent.add_node("parent_after", lambda state: {"value": state["value"] + 1})
        parent.set_entry_point("child")
        parent.set_finish_point("parent_after")
        output = parent.compile().invoke({"value": 0})
        self.assertEqual(output["value"], 101)


    def test_v2_stream_parts(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        compiled = graph.compile()
        chunks = compiled.stream(
            {"value": 0}, stream_mode=["values", "updates"], version="v2"
        )
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIn("type", chunk)
            self.assertIn("ns", chunk)
            self.assertIn("data", chunk)
        types = {chunk["type"] for chunk in chunks}
        self.assertIn("values", types)
        self.assertIn("updates", types)

    def test_async_v2_stream_parts(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile()
            return await compiled.astream(
                {"value": 0}, stream_mode=["values"], version="v2"
            )

        chunks = asyncio.run(scenario())
        self.assertTrue(chunks)
        values = [chunk for chunk in chunks if chunk["type"] == "values"]
        self.assertTrue(values)
        self.assertIn("data", values[0])


    def test_async_subgraph_command_parent(self):
        import asyncio

        def jump(state):
            return lg.Command(
                update={"value": 100},
                goto="parent_after",
                graph=lg.Command.PARENT,
            )

        async def scenario():
            child = lg.AsyncStateGraph({"value": "last_value"})
            child.add_node("jump", jump)
            child.set_entry_point("jump")
            child_compiled = child.compile()

            parent = lg.AsyncStateGraph({"value": "last_value"})
            parent.add_subgraph("child", child_compiled)
            parent.add_node("parent_after", lambda state: {"value": state["value"] + 1})
            parent.set_entry_point("child")
            parent.set_finish_point("parent_after")
            compiled = parent.compile()
            return await compiled.ainvoke({"value": 0})

        self.assertEqual(asyncio.run(scenario())["value"], 101)


    def test_interrupt_response_schema(self):
        saver = lg.InMemorySaver()

        def ask(state):
            answer = lg.interrupt(
                {"question": "age?"}, response_schema={"type": "integer"}
            )
            return {"value": answer}

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("ask", ask)
        graph.set_entry_point("ask")
        graph.set_finish_point("ask")
        compiled = graph.compile(checkpointer=saver)

        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 0}, thread_id="py-schema")
        snapshot = compiled.get_state("py-schema")
        self.assertIn("response_schema", snapshot["interrupts"][0])
        with self.assertRaises(Exception):
            compiled.resume("py-schema", "nope")
        output = compiled.resume("py-schema", 9)
        self.assertEqual(output["value"], 9)


    def test_python_custom_reducer(self):
        def add(current, incoming):
            return current + incoming

        graph = lg.StateGraph({"total": add})
        graph.add_node("one", lambda state: {"total": 1})
        graph.add_node("two", lambda state: {"total": 2})
        graph.set_entry_point("one")
        graph.add_edge("one", "two")
        graph.set_finish_point("two")
        output = graph.compile().invoke({})
        self.assertEqual(output["total"], 3)

    def test_python_custom_reducer_merges_messages(self):
        def merge(current, incoming):
            return current + incoming

        graph = lg.StateGraph({"items": merge})
        graph.add_node("one", lambda state: {"items": ["a"]})
        graph.add_node("two", lambda state: {"items": ["b", "c"]})
        graph.set_entry_point("one")
        graph.add_edge("one", "two")
        graph.set_finish_point("two")
        output = graph.compile().invoke({})
        self.assertEqual(output["items"], ["a", "b", "c"])


    def test_async_python_coroutine_node(self):
        import asyncio

        async def work(state):
            await asyncio.sleep(0)
            return {"value": state["value"] + 1}

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("work", work)
            graph.set_entry_point("work")
            graph.set_finish_point("work")
            compiled = graph.compile()
            return await compiled.ainvoke({"value": 0})

        self.assertEqual(asyncio.run(scenario())["value"], 1)

    def test_async_python_coroutine_node_interrupts(self):
        import asyncio

        async def ask(state):
            delta = lg.interrupt({"q": "delta"})
            await asyncio.sleep(0)
            return {"value": state["value"] + delta}

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("ask", ask)
            graph.set_entry_point("ask")
            graph.set_finish_point("ask")
            compiled = graph.compile(checkpointer=lg.InMemorySaver())
            with self.assertRaises(lg.GraphInterrupt):
                await compiled.ainvoke({"value": 0}, thread_id="py-coroutine")
            return await compiled.aresume("py-coroutine", 5)

        self.assertEqual(asyncio.run(scenario())["value"], 5)


    def test_tool_node_executes_tool_calls(self):
        def add(args):
            return args["a"] + args["b"]

        graph = lg.StateGraph(getattr(lg, "MessagesState", {"messages": "add_messages"}))
        graph.add_node("tools", lg.ToolNode({"add": add}))
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        output = graph.compile().invoke(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call-1", "name": "add", "args": {"a": 1, "b": 2}}
                        ],
                    }
                ]
            }
        )
        messages = output["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-1]["tool_call_id"], "call-1")
        self.assertEqual(messages[-1]["content"], "3")

    def test_tools_condition_and_create_react_agent(self):
        def add(args):
            return args["a"] + args["b"]

        def model(state):
            messages = state.get("messages", [])
            last = messages[-1] if messages else None
            if last is not None and last.get("role") == "tool":
                return {"messages": [{"role": "assistant", "content": "done"}]}
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call-1", "name": "add", "args": {"a": 1, "b": 2}}
                        ],
                    }
                ]
            }

        self.assertEqual(
            lg.tools_condition(
                {
                    "messages": [
                        {"role": "assistant", "tool_calls": [{"id": "1", "name": "x"}]}
                    ]
                }
            ),
            "tools",
        )
        self.assertEqual(
            lg.tools_condition({"messages": [{"role": "assistant"}]}),
            lg.END,
        )

        graph = lg.create_react_agent(model, {"add": add})
        output = graph.compile().invoke({"messages": []})
        messages = output["messages"]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], "3")
        self.assertEqual(messages[2]["content"], "done")


    def test_async_graph_with_async_saver_alias(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile(checkpointer=lg.AsyncInMemorySaver())
            output = await compiled.ainvoke({"value": 0}, thread_id="async-saver")
            snapshot = compiled.get_state("async-saver")
            return output, snapshot

        output, snapshot = asyncio.run(scenario())
        self.assertEqual(output["value"], 1)
        self.assertEqual(snapshot["values"]["value"], 1)


    def test_async_thread_management(self):
        import asyncio

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("inc", lambda state: {"value": state["value"] + 1})
            graph.set_entry_point("inc")
            graph.set_finish_point("inc")
            compiled = graph.compile(checkpointer=lg.AsyncInMemorySaver())
            await compiled.ainvoke({"value": 0}, thread_id="async-thread")
            compiled.copy_thread("async-thread", "async-copy")
            copied = compiled.get_state("async-copy")
            history = compiled.get_state_history("async-thread")
            compiled.delete_thread("async-thread")
            deleted = compiled.get_state("async-thread")
            return copied, history, deleted

        copied, history, deleted = asyncio.run(scenario())
        self.assertIsNotNone(copied)
        self.assertTrue(history)
        self.assertIsNone(deleted)

    def test_v2_messages_metadata(self):
        import asyncio

        def emit(state):
            lg.message({"content": "token"})
            return {"value": state["value"] + 1}

        async def scenario():
            graph = lg.AsyncStateGraph({"value": "last_value"})
            graph.add_node("emit", emit)
            graph.set_entry_point("emit")
            graph.set_finish_point("emit")
            compiled = graph.compile()
            return await compiled.astream(
                {"value": 0}, stream_mode=["messages"], version="v2"
            )

        chunks = asyncio.run(scenario())
        messages = [chunk for chunk in chunks if chunk["type"] == "messages"]
        self.assertTrue(messages)
        metadata = messages[0]["data"][1]
        self.assertEqual(metadata["langgraph_node"], "emit")
        self.assertIn("tags", metadata)


    def test_checkpoint_langgraph_export_import(self):
        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("inc", lambda state: {"value": state["value"] + 1})
        graph.set_entry_point("inc")
        graph.set_finish_point("inc")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        compiled.invoke({"value": 0}, thread_id="export-src")
        document = compiled.export_checkpoint("export-src")
        self.assertEqual(document["config"]["configurable"]["thread_id"], "export-src")
        self.assertEqual(document["checkpoint"]["channel_values"]["value"], 1)
        compiled.import_checkpoint("export-dst", document)
        snapshot = compiled.get_state("export-dst")
        self.assertEqual(snapshot["values"]["value"], 1)


    def test_functional_tasks_persist_in_checkpoints_without_store(self):
        saver = lg.InMemorySaver()
        calls = []

        def first(value):
            calls.append(value)
            return value * 10

        first_task = lg.task(first, name="first")

        def entry(state):
            head = first_task(state["value"])
            delta = lg.interrupt({"question": "delta"})
            return {"value": head + delta}

        compiled = lg.entrypoint(entry).compile(checkpointer=saver)
        with self.assertRaises(lg.GraphInterrupt):
            compiled.invoke({"value": 1}, thread_id="py-functional-checkpoint")
        self.assertEqual(calls, [1])
        output = compiled.resume("py-functional-checkpoint", 4)
        self.assertEqual(output["value"], 14)
        self.assertEqual(calls, [1])


    def test_snapshot_config_get_config_threads_and_stream_writer(self):
        def cfg(state):
            config = lg.get_config()
            writer = lg.get_stream_writer()
            writer({"from": "cfg"})
            return {"value": config["configurable"]["thread_id"]}

        graph = lg.StateGraph({"value": "last_value"})
        graph.add_node("cfg", cfg)
        graph.set_entry_point("cfg")
        graph.set_finish_point("cfg")
        compiled = graph.compile(checkpointer=lg.InMemorySaver())
        chunks = compiled.stream({}, thread_id="closeout", stream_mode=["custom"])
        self.assertTrue(any(chunk.get("type") == "custom" for chunk in chunks))
        snapshot = compiled.get_state("closeout")
        self.assertEqual(snapshot["values"]["value"], "closeout")
        self.assertEqual(
            snapshot["config"]["configurable"]["thread_id"], "closeout"
        )
        self.assertIn("closeout", compiled.threads())

    def test_retry_policy_jitter_and_retry_on(self):
        policy = lg.RetryPolicy(
            max_attempts=2, jitter=0.25, retry_on=["connect"]
        )
        self.assertEqual(policy.jitter, 0.25)
        self.assertEqual(policy.retry_on, ["connect"])
        self.assertEqual(
            lg.RetryPolicy(max_attempts=2).retry_on, []
        )


if __name__ == "__main__":
    unittest.main()