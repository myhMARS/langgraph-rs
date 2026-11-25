"""Compare LangGraph, the langgraph_rs binding and the native async API.

Each iteration awaits a three-node async chain (`ainvoke`).

    python examples/async/compare.py
    python examples/async/compare.py --iterations 500
    python examples/async/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="async", native="async_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))