"""Assemble the importable ``langgraph_rs`` package into ``target/python``.

The package layout lives in ``python/langgraph_rs`` (pure Python) and the compiled
extension is dropped into it as ``langgraph_rs/langgraph_rs.pyd`` (``.so`` on Linux/macOS),
so both the flat ``langgraph_rs.StateGraph`` API and the LangGraph module layout work::

    from langgraph_rs.graph import StateGraph, START, END
    from langgraph_rs.checkpoint.memory import InMemorySaver

Usage::

    cargo build --release --features "python python-async sqlite redis postgres msgpack"
    python python/build_package.py --release

Then point the tests at it::

    PYTHONPATH=<repo>/target/python python -m unittest discover -s python/tests
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEST = REPO / "target" / "python"
PACKAGE = DEST / "langgraph_rs"


def library_path(profile: str) -> Path:
    name = "langgraph_rs.dll" if os.name == "nt" else "liblanggraph_rs.so"
    path = REPO / "target" / profile / name
    if not path.exists():
        raise SystemExit(
            f"missing {path}; build it first, e.g.\n"
            f'  cargo build --{profile} --features "python python-async sqlite redis postgres msgpack"'
        )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        action="store_true",
        help="use target/release instead of target/debug (recommended: debug is 4-5x slower)",
    )
    args = parser.parse_args(argv)
    profile = "release" if args.release else "debug"
    lib = library_path(profile)

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    shutil.copytree(
        HERE / "langgraph_rs",
        PACKAGE,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    suffix = ".pyd" if os.name == "nt" else ".so"
    shutil.copy2(lib, PACKAGE / f"langgraph_rs{suffix}")

    # A bare `langgraph_rs.pyd` next to the package would be shadowed by it, but leaving
    # it around makes `import langgraph_rs` depend on filesystem ordering rules.
    for stale in DEST.glob(f"langgraph_rs{suffix}"):
        stale.unlink()

    print(f"assembled {PACKAGE} from {lib}")
    print(f"run with: PYTHONPATH={DEST}{os.pathsep}$PYTHONPATH")
    return 0


if __name__ == "__main__":
    sys.exit(main())