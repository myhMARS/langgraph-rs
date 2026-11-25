"""Streaming liveness checks for langgraph_rs.

Every check asserts *ordering*, not wall-clock time: the first item of a live stream must arrive
before the producing node finishes. That makes the checks stable on slow machines while still
failing if chunks are batched per node.

Run from this directory:

    python compare.py            # print results
    python compare.py --check    # exit non-zero if any check fails
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import langgraph_rs as rs
from langgraph_rs.stream.transformers import CustomTransformer, MessagesTransformer


class Flag:
    """Cross-thread "the node finished" marker."""

    def __init__(self) -> None:
        self.done = False
        self._lock = threading.Lock()

    def set(self) -> None:
        with self._lock:
            self.done = True

    def __bool__(self) -> bool:
        with self._lock:
            return self.done


def check_node_messages_sync() -> tuple[bool, str]:
    graph = rs.StateGraph({"messages": "messages"})
    flag = Flag()

    def node(state):
        for piece in ("a", "b"):
            rs.message({"type": "AIMessageChunk", "id": "m1", "content": piece})
            time.sleep(0.05)
        flag.set()
        return {"messages": []}

    graph.add_node("node", node)
    graph.set_entry_point("node")
    graph.set_finish_point("node")
    app = graph.compile()

    seen: list[str] = []
    before = None
    with app.stream_events({"messages": []}, version="v3", transformers=[MessagesTransformer]) as run:
        handle = next(iter(run.messages))
        for index, chunk in enumerate(handle.text):
            if index == 0:
                before = not flag
            seen.append(chunk)
    ok = before is True and "".join(seen) == "ab"
    return ok, f"deltas={seen} first_before_node_end={before}"


def check_custom_sync() -> tuple[bool, str]:
    graph = rs.StateGraph({"messages": "messages"})
    flag = Flag()

    def node(state):
        writer = rs.get_stream_writer()
        for piece in ("x", "y"):
            writer({"progress": piece})
            time.sleep(0.05)
        flag.set()
        return {"messages": []}

    graph.add_node("node", node)
    graph.set_entry_point("node")
    graph.set_finish_point("node")
    app = graph.compile()

    seen: list[dict] = []
    before = None
    with app.stream_events({"messages": []}, version="v3", transformers=[CustomTransformer]) as run:
        for index, payload in enumerate(run.custom):
            if index == 0:
                before = not flag
            seen.append(payload)
    ok = before is True and [item["progress"] for item in seen] == ["x", "y"]
    return ok, f"payloads={seen} first_before_node_end={before}"


def check_node_messages_async() -> tuple[bool, str]:
    async def scenario():
        graph = rs.AsyncStateGraph({"messages": "messages"})
        flag = Flag()

        async def node(state):
            for piece in ("a", "b"):
                rs.message({"type": "AIMessageChunk", "id": "m1", "content": piece})
                await asyncio.sleep(0.05)
            flag.set()
            return {"messages": []}

        graph.add_node("node", node)
        graph.set_entry_point("node")
        graph.set_finish_point("node")
        app = graph.compile()

        run = await app.astream_events(
            {"messages": []}, version="v3", transformers=[MessagesTransformer]
        )
        seen: list[str] = []
        before = None
        async for handle in run.messages:
            index = 0
            async for chunk in handle.text:
                if index == 0:
                    before = not flag
                seen.append(chunk)
                index += 1
            break
        return before, seen

    before, seen = asyncio.run(scenario())
    ok = before is True and "".join(seen) == "ab"
    return ok, f"deltas={seen} first_before_node_end={before}"


def check_interrupt_live() -> tuple[bool, str]:
    graph = rs.StateGraph({"messages": "messages", "answer": "last"})

    def node(state):
        rs.message({"type": "AIMessageChunk", "id": "m1", "content": "x"})
        return {"answer": rs.interrupt({"question": "ok?"})}

    graph.add_node("node", node)
    graph.set_entry_point("node")
    graph.set_finish_point("node")
    app = graph.compile(checkpointer=rs.InMemorySaver())

    with app.stream_events(
        {"messages": []}, version="v3", thread_id="liveness", transformers=[MessagesTransformer]
    ) as run:
        list(run.values)
        ok = bool(run.interrupted) and len(run.interrupts) == 1
        return ok, f"interrupted={run.interrupted} interrupts={len(run.interrupts)}"


CHECKS = (
    ("node messages (sync)", check_node_messages_sync),
    ("custom writer (sync)", check_custom_sync),
    ("node messages (async)", check_node_messages_async),
    ("interrupt on live path", check_interrupt_live),
)


def main(argv: list[str]) -> int:
    failures = 0
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except BaseException as error:  # noqa: BLE001 - report and continue
            ok, detail = False, f"{type(error).__name__}: {error}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        failures += 0 if ok else 1
    if "--check" in argv:
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
