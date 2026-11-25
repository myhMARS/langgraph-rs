"""Compare LangGraph, the langgraph_rs binding and the native API on the store demo.

    python examples/store/compare.py
    python examples/store/compare.py --iterations 2000
    python examples/store/compare.py --check-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _compare import Feature, main

FEATURE = Feature(name="store", native="store_native")

if __name__ == "__main__":
    raise SystemExit(main(FEATURE, __file__))