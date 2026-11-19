"""Compare one long workflow across LangGraph, the Rust binding and Rust native.

The workflow (see ``build_lg.py`` / ``build_rs.py`` / ``../mega_workflow_native.rs``)
exercises, in a single suite:

* a loop built from conditional edges (``grow`` runs three times),
* ``Send`` fan-out with three workers joined back into ``join``,
* node-level caching, a retried failure and an always-failing node with an error handler,
* a deferred node and an embedded child graph with explicit input/output mapping,
* an ``interrupt`` that pauses the run and a resume that completes it,
* ``get_state`` / ``get_state_history``, ``update_state``, ``stream`` and ``batch``.

Each phase is timed separately so a slow phase does not hide the others.

Usage::

    python examples/mega_workflow/bench.py --iterations 20
    python examples/mega_workflow/bench.py --iterations 20 --skip-native

The Rust binding must be importable (``PYTHONPATH=<repo>/target/python``); the native
side is built with ``cargo build --release --example mega_workflow_native``.
"""

from __future__ import annotations

import argparse
import json
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


class LangGraphDriver:
    name = "langgraph"
    color = "python-langgraph"

    def __init__(self) -> None:
        import build_lg  # noqa: PLC0415 - local example module

        self.build_lg = build_lg
        bundle = build_lg.build()
        self.graph = bundle["graph"]
        self.Command = bundle["Command"]
        self.GraphInterrupt = self._interrupt_type()

    @staticmethod
    def _interrupt_type():
        from langgraph.errors import GraphInterrupt  # noqa: PLC0415

        return GraphInterrupt

    def _cfg(self, thread: str) -> dict:
        return {"configurable": {"thread_id": thread}}

    def suite(self, thread: str) -> dict:
        graph = self.graph
        cfg = self._cfg(thread)
        timings: dict[str, int] = {}

        started = time.perf_counter_ns()
        graph.invoke(dict(INPUT), cfg)
        timings["invoke"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        self.final = graph.invoke(self.Command(resume="go"), cfg)
        timings["resume"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        snapshot = graph.get_state(cfg)
        history = list(graph.get_state_history(cfg, limit=5))
        self.step, self.history = snapshot.metadata["step"], len(history)
        timings["state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        graph.update_state(cfg, {"count": 99}, as_node="done")
        timings["update_state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        chunks = 0
        try:
            for _ in graph.stream(dict(INPUT), self._cfg(thread + "-stream"), stream_mode="values"):
                chunks += 1
        except self.GraphInterrupt:
            pass
        self.chunks = chunks
        timings["stream"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        batched = graph.batch([dict(INPUT), dict(INPUT)], self._cfg(thread + "-batch"))
        self.batch = len(batched)
        timings["batch"] = time.perf_counter_ns() - started

        return timings


class BindingDriver:
    name = "binding"
    color = "python-binding"

    def __init__(self) -> None:
        import langgraph_rs as lg  # noqa: PLC0415

        self.lg = lg
        bundle = __import__("build_rs").build()
        self.graph = bundle["graph"]
        self.GraphInterrupt = lg.GraphInterrupt

    def suite(self, thread: str) -> dict:
        graph = self.graph
        timings: dict[str, int] = {}

        started = time.perf_counter_ns()
        try:
            graph.invoke(dict(INPUT), thread_id=thread)
        except self.GraphInterrupt:
            pass
        timings["invoke"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        self.final = graph.resume(thread, "go")
        timings["resume"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        snapshot = graph.get_state(thread)
        history = graph.get_state_history(thread, limit=5)
        self.step, self.history = snapshot["step"], len(history)
        timings["state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        graph.update_state(thread, {"count": 99}, as_node="done")
        timings["update_state"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        chunks = 0
        try:
            chunks = len(graph.stream(dict(INPUT), thread_id=thread + "-stream", stream_mode="values"))
        except self.GraphInterrupt:
            chunks = 0
        self.chunks = chunks
        timings["stream"] = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        try:
            batched = graph.batch([dict(INPUT), dict(INPUT)])
            self.batch = len(batched)
        except self.GraphInterrupt:
            # Both inputs pause at `ask`; the batch surfaces the interrupt.
            self.batch = 0
        timings["batch"] = time.perf_counter_ns() - started

        return timings


class NativeDriver:
    name = "native"
    color = "rust-native"

    def __init__(self, iterations: int, release: bool = True) -> None:
        self.iterations = iterations
        profile = ["--release"] if release else []
        subprocess.run(
            ["cargo", "build", "--example", "mega_workflow_native", *profile, "--features", "async"],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        binary = REPO / "target" / ("release" if release else "debug") / "examples"
        exe = "mega_workflow_native.exe" if os.name == "nt" else "mega_workflow_native"
        self.binary = binary / exe

    def suite(self, thread: str) -> dict:
        output = subprocess.run(
            [str(self.binary), str(self.iterations)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10, help="suites measured per engine")
    parser.add_argument("--skip-native", action="store_true", help="skip the cargo example")
    parser.add_argument("--skip-lg", action="store_true", help="skip the reference LangGraph run")
    args = parser.parse_args()

    if str(REPO / "target" / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "target" / "python"))
    sys.path.insert(0, str(HERE))

    drivers = []
    if not args.skip_lg:
        drivers.append(LangGraphDriver())
    drivers.append(BindingDriver())
    if not args.skip_native:
        drivers.append(NativeDriver(args.iterations))

    # Warm up once per engine, then interleave the suites so machine drift hits every
    # engine the same way instead of inflating whichever engine ran first.
    for driver in drivers:
        driver.suite(f"{driver.name}-warmup")
    samples: dict[str, dict[str, list[int]]] = {
        driver.name: {phase: [] for phase in PHASES} for driver in drivers
    }
    for iteration in range(args.iterations):
        for driver in drivers:
            timings = driver.suite(f"{driver.name}-{iteration}")
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

    header = f"| Phase | {' | '.join(results)} |"
    print(header)
    print("| --- |" + " ---: |" * len(results))
    for phase in PHASES:
        cells = []
        for name, medians in results.items():
            value = medians.get(phase)
            cells.append("-" if value is None else f"{value / 1_000_000:.2f} ms")
        print(f"| {phase} | {' | '.join(cells)} |")

    if "langgraph" in results:
        print("\nSpeedup vs LangGraph (higher is faster):")
        print("| Phase | binding | native |")
        print("| --- | ---: | ---: |")
        for phase in PHASES:
            lg_value = results["langgraph"].get(phase)
            row = [phase]
            for name in ("binding", "native"):
                value = results.get(name, {}).get(phase)
                row.append("-" if not (lg_value and value) else f"{lg_value / value:.2f}x")
            print(f"| {row[0]} | {row[1]} | {row[2]} |")

    report = HERE / "MEGA_BENCH.md"
    lines = [
        "# Mega workflow benchmark",
        "",
        f"Generated by `python examples/mega_workflow/bench.py --iterations {args.iterations}`.",
        "",
        "All three engines run the same 15-node workflow per suite (loop, Send fan-out, cache,",
        "retry, error handler, deferred node, embedded child graph, interrupt+resume,",
        "state/history, update_state, stream, batch).",
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
    if "langgraph" in results:
        lines += ["", "Speedup vs LangGraph:", "", "| Phase | binding | native |", "| --- | ---: | ---: |"]
        for phase in PHASES:
            lg_value = results["langgraph"].get(phase)
            row = [phase]
            for name in ("binding", "native"):
                value = results.get(name, {}).get(phase)
                row.append("-" if not (lg_value and value) else f"{lg_value / value:.2f}x")
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", "Final state per engine:", ""]
    for name, note in notes.items():
        lines.append(f"- {name}: {', '.join(note) if note else 'n/a'}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())