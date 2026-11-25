"""Compare LangGraph, the langgraph_rs binding and the native API on the quickstart graph.

    python examples/quickstart/compare.py                     # parity + timings
    python examples/quickstart/compare.py --iterations 5000
    python examples/quickstart/compare.py --check-only        # parity only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="quickstart", native="quickstart_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))