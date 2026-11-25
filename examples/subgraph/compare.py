"""Compare LangGraph, the langgraph_rs binding and the native API on the subgraph demo.

Each iteration runs four `Send` worker tasks, the reduce step and one embedded subgraph.

    python examples/subgraph/compare.py
    python examples/subgraph/compare.py --iterations 200
    python examples/subgraph/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="subgraph", native="subgraph_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))