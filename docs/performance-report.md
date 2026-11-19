# langgraph-rs performance report

Consolidated results of the performance work tracked round by round in
[`compatibility-audit.md`](compatibility-audit.md). Every change was verified against LangGraph's
observable behaviour before it was kept; changes that looked fast but changed semantics were
reverted (they are listed at the end).

## Method

Three independent measurements are used, because each answers a different question:

| Tool | What it measures | Command |
| --- | --- | --- |
| `examples/perf_probe.rs` | engine internals: one run, fan-out, saves, reads, history | `cargo run --release --example perf_probe --features async` |
| `examples/mega_workflow/bench*.py` | the same large workflow (nodes, reducers, subgraph, interrupts, cache, retry, deferred) on LangGraph, the Python binding and the native API, interleaved, with warmup and medians | `python examples/mega_workflow/bench.py` |
| `benchmarks/parity.py` | behaviour equivalence **and** per-scenario timing for 38 scenarios | `python benchmarks/parity.py` |

The machine is a Windows developer workstation; absolute numbers move by ±15% between runs, so
ratios are compared within one run and only same-session deltas are claimed.

## Where the time went, and what changed

Engine (same process, no Python):

| Round | Change | Evidence |
| --- | --- | --- |
| 1 | per-node state borrowed instead of cloned; O(1) checkpoint append; paged history | 30-node run 1049 → 529 us |
| 5 | async writer thread skipped for in-memory savers | 1-node run with checkpointer 94.7 → 16.7 us |
| 6 | `limit` pushed into the history read | `history(5)` 33 → 23 us |
| 10 | checkpoints shared as `Arc` instead of deep-cloned before persisting | removes an O(state) copy per save |
| 11 | channel writes moved instead of cloned; single-write fast path in `apply_updates` | 30-node run 600 → 497 us, save 19 → 15.5 us |
| 12 | version bumps from the written-channel set, not a deep value compare | 256 KB state: checkpoint build 205.9 → 53.2 us |
| 13 | snapshots take ownership of the checkpoint's values | `get_state` 29 → 26 us, `history(5)` 142 → 128 us |
| 18 | `update_state` snapshots built from a borrowed checkpoint | `update_state` 35 → 33 us |
| 19 | reads share the stored checkpoint (`latest_shared`) | `get_state` 26 → 16 us |
| 20 | history reads share the stored checkpoints | `history(5)` 113 → 109 us, `aget_state_history(5)` 114 → 97 us |
| 22 | `versions_seen` entries shared behind `Arc` | `versions_for` 7.09 → 3.03 us/save, checkpoint build 7.50 → 3.46 us |
| 27 | `get_state_history` returns a lazy `StateHistory` sequence (LangGraph's generator semantics) | `len(history)` on 420 checkpoints ~27 ms → 0.037 ms; `history[0]` ~27 ms → 0.13 ms |
| 4 | process-wide worker pool for parallel branches | fan-out 8: 471 → ~100-220 us (load dependent) |

Python binding:

| Round | Change | Evidence |
| --- | --- | --- |
| 1 | callback config cached per thread, writer skipped for in-memory savers | — |
| 7 | one asyncio loop per OS thread instead of one per node | 10-node `ainvoke` 14.16 → 2.48 ms |
| 14 | in-memory savers keep the inline path instead of `BlockingSaver` (`spawn_blocking` hop) | `aupdate_state` 103-114 → 37 us, `aget_state` 51 → 18 us, `ainvoke` 579 → 350 us |
| 15 | `has_thread` instead of building a snapshot to test existence | `invoke` 116 → 79-94 us, `ainvoke` 407 → 357 us |
| 16 | `Command` detection without raising an exception per call | `invoke` (no checkpointer) 45 → 37 us |
| 17 | `operator.add` reducers concatenate in Rust instead of calling back into Python | 40-node list reducer 3.45 → 1.67 ms |

## Current results

Cross-engine medians, 10 interleaved suites (`examples/mega_workflow`, one run):

| Phase | sync binding | sync native | async binding | async native |
| --- | ---: | ---: | ---: | ---: |
| invoke | 4.45x | 30.2x | 2.17x | 35.4x |
| resume | 7.40x | 39.1x | 1.22x | 16.7x |
| state | 10.14x | 28.6x | 6.22x | 26.1x |
| update_state | 2.28x | 11.1x | 1.98x | 11.5x |
| stream | 5.11x | 17.9x | 2.00x | 12.9x |
| batch | 7.77x | 37.5x | 3.01x | 27.5x |

Engine probe (same run as above):

```
nodes= 1  invoke=   8.1 us   async= 19.4 us   history=3   per-save=6.5 us
nodes=10  invoke=  48.8 us   async=112.5 us   history=12  per-save=9.4 us
nodes=30  invoke= 183.4 us   async=409.4 us   history=32  per-save=12.8 us
fan-out 3: 141 us   fan-out 8: 146 us
get_state 2.1-2.5 us   history(5) 15.1 us   history(limit=50) 192 us
```

Parity: **36/38 scenarios equivalent**, the two known differences being `command_goto` (the
binding does not also run the entry point) and `messages_stream` (documented grouping delta).
Non-I/O scenarios are 3-30x faster than LangGraph; the timeout scenarios are bounded by the timer
and SQLite scenarios by the database.

## Known limits

* **Async per-call bridge: ~105 us.** `pyo3_async_runtimes::future_into_py` spawns a Tokio task
  and wakes the asyncio loop. Polling the future eagerly on the calling thread made `ainvoke`
  357 → 116 us, but it runs node code on the event loop's thread: async Python nodes fail with
  "Cannot run the event loop while another loop is running" and a slow synchronous node would
  block the loop. Reverted. Python's own cross-thread future completion costs 171 us, so the
  bridge is already below the language-level floor.
* **One value copy per checkpoint save** (`persisted_values`) and one per superstep
  (`values_to_state`): the engine keeps the live map and the checkpoint alive at the same time.
  Removing it requires per-channel `Arc` values in `Checkpoint::values`, which `serde_json::Map`
  cannot express (it only implements its traits for `Map<String, Value>`).
* **History is lazy** (`StateHistory`): like LangGraph's generator it resolves at iteration time,
  so deleting a thread before reading an already-obtained history yields nothing (verified
  against LangGraph). It additionally supports `len()`/indexing, which LangGraph's generator
  does not.
* Two documented parity differences above; everything else is byte-identical in the compared
  observable state.

## Reverted experiments

| Experiment | Why it looked good | Why it was reverted |
| --- | --- | --- |
| eager first poll of the graph future | `ainvoke` 357 → 116 us, `aresume` 24 us | runs nodes on the event loop thread (async nodes break, slow sync nodes block the loop) |
| single-task fast path in async `run_superstep` | 87 vs 105 us/node | placed outside the abort `select!` it broke cancellation; inside it the gain was noise |
| hand-written awaitable | ~85 us vs 114 us per call | 29 us saving did not justify losing cancellation/contextvar fidelity |
| `BlockingSaver` for in-memory savers | — | kept for blocking savers only, removed for in-memory (round 14) |

## How to reproduce

```powershell
$env:SQLITE3_LIB_DIR="$env:USERPROFILE\.cargo\sqlite3-lib"; $env:SQLITE3_STATIC='1'
cargo build --features "python python-async sqlite redis postgres msgpack" --offline
Copy-Item target\debug\langgraph_rs.dll target\python\langgraph_rs.pyd -Force
$env:PYTHONPATH = "$PWD\target\python"
python -m unittest discover -s python\tests      # 119 tests
python benchmarks\parity.py                      # 36/38 equivalent
cargo run --release --example perf_probe --features async
python examples\mega_workflow\bench.py
python examples\mega_workflow\bench_async.py
```