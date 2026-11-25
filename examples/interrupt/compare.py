"""Compare LangGraph, the langgraph_rs binding and the native API on the interrupt graph.

Each iteration is a full pause -> resume cycle, so the numbers include checkpointing.

    python examples/interrupt/compare.py
    python examples/interrupt/compare.py --iterations 200
    python examples/interrupt/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="interrupt", native="interrupt_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))