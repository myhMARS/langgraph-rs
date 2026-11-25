"""Long-term memory store — langgraph_rs Python binding.

Same operations as `langgraph_demo.py` on `langgraph_rs.InMemoryStore`: `put` / `get` / `search` /
`list_namespaces`, all keyed by a namespace tuple.

Run:  python examples/store/binding_demo.py
      python examples/store/binding_demo.py --bench 2000
"""

from __future__ import annotations

import json
import statistics
import sys
import time

from langgraph_rs import InMemoryStore


def run_once(store=None):
    store = store or InMemoryStore()
    store.put(("users", "1"), "name", {"first": "Ada", "last": "Lovelace"})
    store.put(("users", "1"), "email", {"address": "ada@example.com"})
    store.put(("users", "2"), "name", {"first": "Grace", "last": "Hopper"})
    store.put(("teams", "core"), "lead", {"name": "ada"})

    name = store.get(("users", "1"), "name")
    users = store.search(("users",))
    items = sorted(f"{'/'.join(item.namespace)}:{item.key}" for item in users)
    return {
        "namespaces": sorted("/".join(namespace) for namespace in store.list_namespaces()),
        "names": sorted(item.value["first"] for item in users if item.key == "name"),
        "read": name.value if name else None,
        "search": items,
    }


def bench(iterations: int) -> float:
    for _ in range(50):
        run_once(InMemoryStore())
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run_once(InMemoryStore())
        samples.append((time.perf_counter_ns() - start) / iterations)
    return statistics.median(samples)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        iterations = int(sys.argv[sys.argv.index("--bench") + 1])
        print(json.dumps({"engine": "langgraph_rs binding", "ns_per_op": bench(iterations)}))
    else:
        print(json.dumps(run_once(), sort_keys=True))