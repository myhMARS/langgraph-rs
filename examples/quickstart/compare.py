"""Compare the three implementations of the quickstart graph.

* `langgraph_demo.py`  -- LangGraph reference
* `binding_demo.py`    -- langgraph_rs Python binding
* `native_demo.rs`     -- Rust native API (built as the `quickstart_native` example)

The script first checks that all three produce the same JSON state, then measures the same graph in
all three (native timing happens inside the binary, so interpreter/process startup is excluded).

    python examples/quickstart/compare.py                 # check + default head-to-head
    python examples/quickstart/compare.py --iterations 5000
    python examples/quickstart/compare.py --check-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _python(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(ROOT),
    )
    return result.stdout.strip()


def _native(*args: str) -> str:
    for profile in ("release", "debug"):
        for suffix in (".exe", ""):
            binary = ROOT / "target" / profile / "examples" / f"quickstart_native{suffix}"
            if binary.exists():
                result = subprocess.run(
                    [str(binary), *args], capture_output=True, text=True, check=True
                )
                return result.stdout.strip()
    raise SystemExit(
        "native example not built; run `cargo build --release --example quickstart_native`"
    )


def _bench_python(script: str, iterations: int) -> float:
    payload = json.loads(_python(script, "--bench", str(iterations)))
    return float(payload["ns_per_op"])


def _bench_native(iterations: int) -> float:
    payload = json.loads(_native("--bench", str(iterations)))
    return float(payload["ns_per_op"])


def _median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    langgraph_result = json.loads(_python("langgraph_demo.py"))
    binding_result = json.loads(_python("binding_demo.py"))
    native_result = json.loads(_native())

    print("result parity")
    print(f"  langgraph        : {json.dumps(langgraph_result, sort_keys=True)}")
    print(f"  binding          : {json.dumps(binding_result, sort_keys=True)}")
    print(f"  native           : {json.dumps(native_result, sort_keys=True)}")
    if not (langgraph_result == binding_result == native_result):
        print("MISMATCH: the three implementations disagree")
        return 1
    print("  -> identical\n")
    if args.check_only:
        return 0

    langgraph = _median([_bench_python("langgraph_demo.py", args.iterations) for _ in range(args.runs)])
    binding = _median([_bench_python("binding_demo.py", args.iterations) for _ in range(args.runs)])
    native = _median([_bench_native(args.iterations) for _ in range(args.runs)])

    print(f"latency (median of {args.runs} runs, --iterations {args.iterations})")
    print(f"  langgraph        : {langgraph / 1_000:8.2f} us/op")
    print(f"  binding          : {binding / 1_000:8.2f} us/op  ({langgraph / binding:5.2f}x vs langgraph)")
    print(f"  native           : {native / 1_000:8.2f} us/op  ({binding / native:5.2f}x vs binding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())