"""Compare LangGraph, the langgraph_rs binding and the native API on the checkpointing demo.

Each iteration builds a fresh in-memory savers, runs the thread four times and calls
`get_state_history`, so the number includes checkpoint writes.

    python examples/checkpointing/compare.py
    python examples/checkpointing/compare.py --iterations 200
    python examples/checkpointing/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="checkpointing", native="checkpointing_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))