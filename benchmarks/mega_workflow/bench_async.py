"""Async twin of ``bench.py``: LangGraph vs Rust binding vs Rust native.

Same 15-node workflow, same phases, but every node is an ``async def`` and every call is
awaited (``ainvoke`` / ``aresume`` / ``astream`` / ``abatch`` / ``aget_state`` ...).

Usage::

    python benchmarks/mega_workflow/bench_async.py --iterations 10
    python benchmarks/mega_workflow/bench_async.py --iterations 10 --skip-native
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASES = ("invoke", "resume", "state", "update_state", "stream", "batch")

INPUT = {"count": 0, "log": [], "sink": 0}


def _median(samples: list[int]) -> int:
    return int(statistics.median(samples))


class LangGraphAsyncDriver:
    name = "langgraph-async"

    def __init__(self) -> None:
        import build_lg_async  # noqa: PLC0415

        bundle = build_lg_async.build()
        self.graph = bundle["graph"]
        self.Command = bundle["Command"]
        from langgraph.errors import GraphInterrupt  # noqa: PLC0415

        self.GraphInterrupt = GraphInterrupt

    @staticmethod
    def _cfg(thread: str) -> dict:
        return {"configurable": {"thread_id": thread}}

    async def suite(self, thread: str) -> dict:
        graph = self.graph
        cfg = self._cfg(thread)
        timings: dict[str, int] = {}

        started = time.perf_counter_ns()
        await graph.ainvoke(dict(INPUT), cfg)
        timings["invoke"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        self.final = await graph.ainvoke(self.Command(resume="go"), cfg)
        timings["resume"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        snapshot = await graph.aget_state(cfg)
        history = [entry async for entry in graph.aget_state_history(cfg, limit=5)]
        self.step, self.history = snapshot.metadata["step"], len(history)
        timings["state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        await graph.aupdate_state(cfg, {"count": 99}, as_node="done")
        timings["update_state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        chunks = 0
        try:
            async for _chunk in graph.astream(
                dict(INPUT), self._cfg(thread + "-stream"), stream_mode="values"
            ):
                chunks += 1
        except self.GraphInterrupt:
            pass
        self.chunks = chunks
        timings["stream"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        batched = await graph.abatch([dict(INPUT), dict(INPUT)], self._cfg(thread + "-batch"))
        self.batch = len(batched)
        timings["batch"] = time.perf_counter_ns() - started

        return timings


class BindingAsyncDriver:
    name = "binding-async"

    def __init__(self) -> None:
        import langgraph_rs as lg  # noqa: PLC0415

        bundle = __import__("build_rs_async").build()
        self.graph = bundle["graph"]
        self.GraphInterrupt = lg.GraphInterrupt

    async def suite(self, thread: str) -> dict:
        graph = self.graph
        timings: dict[str, int] = {}

        started = time.perf_counter_ns()
        try:
            await graph.ainvoke(dict(INPUT), thread_id=thread)
        except self.GraphInterrupt:
            pass
        timings["invoke"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        self.final = await graph.aresume(thread, "go")
        timings["resume"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        snapshot = await graph.aget_state(thread)
        history = await graph.aget_state_history(thread, limit=5)
        self.step, self.history = snapshot["step"], len(history)
        timings["state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        await graph.aupdate_state(thread, {"count": 99}, as_node="done")
        timings["update_state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        chunks = 0
        try:
            streamed = await graph.astream(
                dict(INPUT), thread_id=thread + "-stream", stream_mode="values"
            )
            chunks = len(streamed)
        except self.GraphInterrupt:
            chunks = 0
        self.chunks = chunks
        timings["stream"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        try:
            batched = await graph.abatch([dict(INPUT), dict(INPUT)])
            self.batch = len(batched)
        except self.GraphInterrupt:
            self.batch = 0
        timings["batch"] = time.perf_counter_ns() - started

        return timings


class NativeAsyncDriver:
    name = "native-async"

    def __init__(self, iterations: int, release: bool = True) -> None:
        self.iterations = iterations
        profile = ["--release"] if release else []
        subprocess.run(
            ["cargo", "build", "--example", "mega_workflow_async_native", *profile, "--features", "async"],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        binary = REPO / "target" / ("release" if release else "debug") / "examples"
        exe = "mega_workflow_async_native.exe" if os.name == "nt" else "mega_workflow_async_native"
        self.binary = binary / exe

    async def suite(self, thread: str) -> dict:
        output = await asyncio.to_thread(
            lambda: subprocess.run(
                [str(self.binary), str(self.iterations)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        timings: dict[str, int] = {}
        for line in output.splitlines():
            if line.startswith("FINAL "):
                fields = dict(item.split("=", 1) for item in line.split()[1:])
                self.final = {
                    "count": int(fields["count"]),
                    "sink": int(fields["sink"]),
                    "child": int(fields["child"]),
                    "log_len": int(fields["log"]),
                }
            elif line.startswith("PHASE="):
                phase, value = line[len("PHASE=") :].split(":", 1)
                timings[phase] = int(value)
        return timings


def _note(driver) -> list[str]:
    final = getattr(driver, "final", None)
    if not isinstance(final, dict):
        return []
    log_len = final.get("log_len") or len(final.get("log") or [])
    return [
        "final count={} sink={} child={} log={}".format(
            final.get("count"), final.get("sink"), final.get("child"), log_len
        )
    ]


async def main_async(args: argparse.Namespace) -> int:
    if str(REPO / "target" / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "target" / "python"))
    sys.path.insert(0, str(HERE))

    drivers = []
    if not args.skip_lg:
        drivers.append(LangGraphAsyncDriver())
    drivers.append(BindingAsyncDriver())
    if not args.skip_native:
        drivers.append(NativeAsyncDriver(args.iterations))

    # Warm up once per engine, then interleave the suites so machine drift is shared.
    for driver in drivers:
        await driver.suite(f"{driver.name}-warmup")
    samples: dict[str, dict[str, list[int]]] = {
        driver.name: {phase: [] for phase in PHASES} for driver in drivers
    }
    for iteration in range(args.iterations):
        for driver in drivers:
            timings = await driver.suite(f"{driver.name}-{iteration}")
            for phase, value in timings.items():
                samples[driver.name].setdefault(phase, []).append(value)

    results: dict[str, dict[str, int]] = {}
    notes: dict[str, list[str]] = {}
    for driver in drivers:
        results[driver.name] = {
            phase: _median(values) for phase, values in samples[driver.name].items() if values
        }
        notes[driver.name] = _note(driver)
        print(f"{driver.name}: {results[driver.name]}", flush=True)

    print("| Phase | " + " | ".join(results) + " |")
    print("| --- |" + " ---: |" * len(results))
    for phase in PHASES:
        cells = []
        for medians in results.values():
            value = medians.get(phase)
            cells.append("-" if value is None else f"{value / 1_000_000:.2f} ms")
        print(f"| {phase} | {' | '.join(cells)} |")

    base_name = "langgraph-async"
    if base_name in results:
        print("\nSpeedup vs LangGraph async (higher is faster):")
        print("| Phase | binding | native |")
        print("| --- | ---: | ---: |")
        for phase in PHASES:
            base = results[base_name].get(phase)
            row = [phase]
            for name in ("binding-async", "native-async"):
                value = results.get(name, {}).get(phase)
                row.append("-" if not (base and value) else f"{base / value:.2f}x")
            print(f"| {row[0]} | {row[1]} | {row[2]} |")

    report = HERE / "MEGA_BENCH_ASYNC.md"
    lines = [
        "# Mega workflow benchmark (async)",
        "",
        f"Generated by `python benchmarks/mega_workflow/bench_async.py --iterations {args.iterations}`.",
        "",
        "Same 15-node workflow as `MEGA_BENCH.md`, but every node is `async def` and every",
        "call is awaited (`ainvoke`/`aresume`/`astream`/`abatch`/`aget_state`/`aupdate_state`).",
        "",
        "| Phase | " + " | ".join(results) + " |",
        "| --- |" + " ---: |" * len(results),
    ]
    for phase in PHASES:
        cells = []
        for medians in results.values():
            value = medians.get(phase)
            cells.append("-" if value is None else f"{value / 1_000_000:.2f} ms")
        lines.append(f"| {phase} | {' | '.join(cells)} |")
    if base_name in results:
        lines += ["", "Speedup vs LangGraph async:", "", "| Phase | binding | native |", "| --- | ---: | ---: |"]
        for phase in PHASES:
            base = results[base_name].get(phase)
            row = [phase]
            for name in ("binding-async", "native-async"):
                value = results.get(name, {}).get(phase)
                row.append("-" if not (base and value) else f"{base / value:.2f}x")
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", "Final state per engine:", ""]
    for name, note in notes.items():
        lines.append(f"- {name}: {', '.join(note) if note else 'n/a'}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10, help="suites measured per engine")
    parser.add_argument("--skip-native", action="store_true", help="skip the cargo example")
    parser.add_argument("--skip-lg", action="store_true", help="skip the reference LangGraph run")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())