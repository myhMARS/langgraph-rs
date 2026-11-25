"""Compare LangGraph, the langgraph_rs binding and the native API on the reducer graph.

    python examples/state_reducers/compare.py
    python examples/state_reducers/compare.py --iterations 2000 --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="state_reducers", native="state_reducers_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))