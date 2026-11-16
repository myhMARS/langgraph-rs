#!/usr/bin/env python3
"""Run the langgraph-rs benchmarks and write a comparison report.

What it does:

1. builds the release profile (native bench) and the Python extension module;
2. runs ``cargo bench --bench bench_rust`` and parses its JSON lines;
3. runs ``python/benchmarks/bench.py`` for Python LangGraph and for the bindings;
4. writes ``benchmarks/results/latest.json`` and a markdown table to
   ``benchmarks/RESULTS.md``.

Usage::

    python benchmarks/run.py                 # full run
    python benchmarks/run.py --skip-build    # reuse existing artefacts
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"
PYDIR = ROOT / "benchmarks" / ".pydir"
ENGINES = ["rust", "binding", "langgraph"]
ENGINE_LABELS = {
    "rust": "Rust native",
    "binding": "Rust via Python binding",
    "langgraph": "Python LangGraph",
}


def run(command: list[str], cwd: Path, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def prepare_extension() -> None:
    run(["cargo", "build", "--release", "--features", "python"], ROOT)
    if sys.platform == "win32":
        source = ROOT / "target" / "release" / "langgraph_rs.dll"
        target = PYDIR / "langgraph_rs.pyd"
    elif sys.platform == "darwin":
        source = ROOT / "target" / "release" / "liblanggraph_rs.dylib"
        target = PYDIR / "langgraph_rs.so"
    else:
        source = ROOT / "target" / "release" / "liblanggraph_rs.so"
        target = PYDIR / "langgraph_rs.so"
    if not source.exists():
        raise SystemExit(f"extension module not found at {source}")
    PYDIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"copied {source.name} -> {target}", flush=True)


def parse_json_lines(output: str) -> list[dict]:
    records = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def run_rust_bench() -> list[dict]:
    result = run(["cargo", "bench", "--bench", "bench_rust"], ROOT)
    records = parse_json_lines(result.stdout + "\n" + result.stderr)
    return records


def run_python_bench() -> list[dict]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PYDIR) + (os.pathsep + existing if existing else "")
    json_path = RESULTS_DIR / "python.json"
    result = run(
        [
            sys.executable,
            str(ROOT / "python" / "benchmarks" / "bench.py"),
            "--engine",
            "both",
            "--json",
            str(json_path),
        ],
        ROOT,
        env=env,
    )
    records = parse_json_lines(result.stdout)
    if not records and json_path.exists():
        records = json.loads(json_path.read_text(encoding="utf-8"))
    return records


def version(command: list[str]) -> str:
    try:
        return run(command, ROOT, check=False).stdout.strip()
    except OSError:
        return "unknown"


def langgraph_version() -> str:
    code = (
        "import importlib.metadata as m\n"
        "try:\n"
        "    print(m.version('langgraph'))\n"
        "except Exception:\n"
        "    print('not installed')\n"
    )
    try:
        result = run([sys.executable, "-c", code], ROOT, check=False)
        return result.stdout.strip()
    except OSError:
        return "unknown"


def rust_version() -> str:
    try:
        result = run(["rustc", "--version"], ROOT, check=False)
        return result.stdout.strip()
    except OSError:
        return "unknown"


def build_table(records: list[dict]) -> str:
    scenarios: list[str] = []
    table: dict[str, dict[str, dict]] = {}
    for record in records:
        if "ns_per_op" not in record:
            continue
        scenario = record["scenario"]
        engine = record["engine"]
        if scenario not in table:
            table[scenario] = {}
            scenarios.append(scenario)
        table[scenario][engine] = record

    lines = [
        "| Scenario | Rust native | Rust via Python binding | Python LangGraph | Rust native vs LangGraph |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scenario in scenarios:
        row = table[scenario]
        cells = []
        for engine in ENGINES:
            record = row.get(engine)
            cells.append(f"{record['ns_per_op'] / 1000:,.2f} µs/op" if record else "—")
        rust = row.get("rust")
        langgraph = row.get("langgraph")
        if rust and langgraph and langgraph["ns_per_op"]:
            speedup = langgraph["ns_per_op"] / rust["ns_per_op"]
            cells.append(f"**{speedup:,.1f}× faster**")
        else:
            cells.append("—")
        lines.append("| " + " | ".join([scenario, *cells]) + " |")
    return "\n".join(lines)


def build_memory_table(records: list[dict]) -> str:
    scenarios: list[str] = []
    table: dict[str, dict[str, dict]] = {}
    for record in records:
        if "peak_bytes" not in record:
            continue
        scenario = record["scenario"]
        engine = record["engine"]
        if scenario not in table:
            table[scenario] = {}
            scenarios.append(scenario)
        table[scenario][engine] = record

    lines = [
        "| Scenario | Rust native | Rust via Python binding | Python LangGraph |",
        "| --- | --- | --- | --- |",
    ]
    for scenario in scenarios:
        row = table[scenario]
        cells = []
        for engine in ENGINES:
            record = row.get(engine)
            cells.append(
                f"{record['peak_bytes'] / 1024:,.1f} KiB peak" if record else "—"
            )
        lines.append("| " + " | ".join([scenario, *cells]) + " |")
    return "\n".join(lines)


def build_report(records: list[dict]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark results",
        "",
        f"- Generated: {generated}",
        f"- Machine: {platform.platform()} / {platform.processor() or 'unknown CPU'}",
        f"- Rust: {rust_version()}",
        f"- Python: {sys.version.split()[0]}",
        f"- LangGraph: {langgraph_version()}",
        "- Protocol: same node counts and iteration counts per scenario for all engines;",
        "  warm-up runs precede every measurement; times are wall-clock around the whole",
        "  measured loop (graph construction excluded except for `build_compile`).",
        "",
        "## Wall-clock cost per operation",
        "",
        build_table(records),
        "",
        "## Peak allocated memory",
        "",
        "Rust figures come from a counting global allocator (allocation high-water mark",
        "during the measured loop). Python figures use `tracemalloc`; allocations made",
        "inside the Rust binding are invisible to tracemalloc, so binding numbers are a",
        "lower bound on its real footprint.",
        "",
        build_memory_table(records),
        "",
        "## Raw data",
        "",
        "See [`results/latest.json`](results/latest.json) for every record and",
        "[`README.md`](README.md) for the scenario definitions and caveats.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="reuse existing artefacts")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        prepare_extension()

    records = run_rust_bench() + run_python_bench()
    if not records:
        raise SystemExit("no benchmark records collected")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "rust": rust_version(),
        "python": sys.version.split()[0],
        "langgraph": langgraph_version(),
        "records": records,
    }
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "benchmarks" / "RESULTS.md").write_text(build_report(records) + "\n", encoding="utf-8")
    print("\n" + build_report(records), flush=True)


if __name__ == "__main__":
    main()