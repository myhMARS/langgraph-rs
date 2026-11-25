"""Compare LangGraph, the langgraph_rs binding and the native API on the streaming graph.

Each iteration consumes the full update stream, so the numbers include chunk production, not just
the final state.

    python examples/streaming/compare.py
    python examples/streaming/compare.py --iterations 500
    python examples/streaming/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="streaming", native="streaming_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))