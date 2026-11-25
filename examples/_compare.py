"""Shared harness behind every ``examples/<feature>/compare.py``.

Each feature keeps four implementations of the *same* workflow:

* ``langgraph_demo.py`` -- LangGraph reference
* ``binding_demo.py``   -- ``langgraph_rs`` Python binding
* ``native_demo.rs``    -- Rust native API (built as the ``<feature>_native`` example)

The harness runs all three, checks that they print the same JSON, then times the binding and the
native binary (and LangGraph when it makes sense). Timings for the native side are measured inside
the binary, so process startup is not part of the number.

A feature's ``compare.py`` is therefore just a few lines:

    from _compare import Feature, main

    FEATURE = Feature(name="quickstart", native="quickstart_native")
    if __name__ == "__main__":
        raise SystemExit(main(FEATURE, __file__))
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Feature:
    """One feature directory and the demos it contains."""

    name: str
    native: str
    langgraph: str = "langgraph_demo.py"
    binding: str = "binding_demo.py"
    #: Skip the LangGraph column (feature not expressible there, or it needs extra services).
    bench_langgraph: bool = True
    #: Extra arguments appended to every demo invocation (e.g. a stream mode).
    args: list[str] = field(default_factory=list)


def _script(feature_dir: Path, name: str) -> Path:
    path = feature_dir / name
    if not path.exists():
        raise SystemExit(f"missing demo: {path}")
    return path


def _run_python(script: Path, args: list[str], root: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{script.relative_to(root)} failed\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout.strip()


def _native_binary(name: str, root: Path) -> Path:
    for profile in ("release", "debug"):
        for suffix in (".exe", ""):
            candidate = root / "target" / profile / "examples" / f"{name}{suffix}"
            if candidate.exists():
                return candidate
    raise SystemExit(
        f"native example `{name}` is not built; run "
        f"`cargo build --release --example {name}` first"
    )


def _run_native(name: str, args: list[str], root: Path) -> str:
    result = subprocess.run(
        [str(_native_binary(name, root)), *args], capture_output=True, text=True, cwd=str(root)
    )
    if result.returncode != 0:
        raise SystemExit(
            f"native example `{name}` failed\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout.strip()


def _bench(fn, runs: int) -> float:
    return statistics.median(fn() for _ in range(runs))


def main(feature: Feature, compare_file: str) -> int:
    feature_dir = Path(compare_file).resolve().parent
    root = feature_dir.parents[1]

    parser = argparse.ArgumentParser(description=f"{feature.name}: LangGraph vs binding vs native")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-langgraph", action="store_true")
    args = parser.parse_args()

    langgraph_script = _script(feature_dir, feature.langgraph)
    binding_script = _script(feature_dir, feature.binding)

    with_langgraph = feature.bench_langgraph and not args.skip_langgraph

    results: dict[str, object] = {}
    outputs: dict[str, str] = {}
    if with_langgraph:
        outputs["langgraph"] = _run_python(langgraph_script, feature.args, root)
        results["langgraph"] = json.loads(outputs["langgraph"])
    outputs["binding"] = _run_python(binding_script, feature.args, root)
    results["binding"] = json.loads(outputs["binding"])
    outputs["native"] = _run_native(feature.native, feature.args, root)
    results["native"] = json.loads(outputs["native"])

    print(f"{feature.name}: result parity")
    for engine, payload in results.items():
        print(f"  {engine:<9}: {json.dumps(payload, sort_keys=True)}")
    reference = results["binding"]
    mismatched = [engine for engine, payload in results.items() if payload != reference]
    if mismatched:
        print(f"MISMATCH: {', '.join(mismatched)} disagree with the binding")
        return 1
    print("  -> identical\n")
    if args.check_only:
        return 0

    timings: dict[str, float] = {}
    if with_langgraph:
        timings["langgraph"] = _bench(
            lambda: float(
                json.loads(
                    _run_python(
                        langgraph_script, [*feature.args, "--bench", str(args.iterations)], root
                    )
                )["ns_per_op"]
            ),
            args.runs,
        )
    timings["binding"] = _bench(
        lambda: float(
            json.loads(
                _run_python(
                    binding_script, [*feature.args, "--bench", str(args.iterations)], root
                )
            )["ns_per_op"]
        ),
        args.runs,
    )
    timings["native"] = _bench(
        lambda: float(
            json.loads(
                _run_native(feature.native, ["--bench", str(args.iterations)], root)
            )["ns_per_op"]
        ),
        args.runs,
    )

    print(f"latency (median of {args.runs} runs, {args.iterations} iterations each)")
    for engine, ns in timings.items():
        print(f"  {engine:<9}: {ns / 1_000:8.2f} us/op")
    if "langgraph" in timings:
        print(
            f"  -> binding {timings['langgraph'] / timings['binding']:5.2f}x faster than LangGraph, "
            f"native {timings['binding'] / timings['native']:5.2f}x faster than the binding"
        )
    else:
        print(f"  -> native {timings['binding'] / timings['native']:5.2f}x faster than the binding")
    return 0