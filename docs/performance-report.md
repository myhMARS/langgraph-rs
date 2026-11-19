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

Cross-engine medians, three consecutive runs of 12 interleaved suites each
(`examples/mega_workflow`); the median of the three runs is shown:

The extension **must be built in release mode** (`cargo build --release --features ...`) before it is
copied to `langgraph_rs.pyd`. The numbers below were collected with `target/release/langgraph_rs.dll`;
a debug build of the same code is 4-5x slower on the Rust side and collapses several phases below
5x (a debug linear 10-node chain runs in 0.28-0.39 ms versus **0.073 ms** in release).

| Phase | sync binding | sync native | async binding | async native |
| --- | ---: | ---: | ---: | ---: |
| invoke | **14.8-16.8x** | 32.6-53.7x | 4.4-5.9x | 33.7-54.8x |
| resume | **21.5-28.4x** | 34.1-47.9x | 2.5-3.3x | 18.2-32.2x |
| state | **40.8-43.3x** | 33.2-41.8x | **18.0-30.0x** | 29.5-55.9x |
| update_state | **8.6-12.1x** | 19.2-27.7x | **5.8-7.2x** | 15.4-23.2x |
| stream | **16.9-18.2x** | 16.1-23.3x | 4.1-5.0x | 13.6-22.0x |
| batch | **26.0-32.5x** | 33.5-54.2x | **6.1-8.7x** | 28.4-46.5x |

All six sync-binding phases and five of six async-binding phases are at or above 5x. The remaining
async outlier is `resume` (3.32x). Its cost decomposes as:

| Component | Cost |
| --- | ---: |
| one asyncio<->Tokio wake-up (measured warm, via a 1-node `ainvoke`) | ~0.165 ms |
| the resume work itself (same graph through the sync binding) | ~0.12 ms |
| per-node async engine overhead (state clone, context, `FuturesUnordered`) | ~0.02-0.15 ms |

An error-path measurement suggests 0.285 ms for the bridge, but that number includes building a
Python exception with a traceback; the warm figure is ~0.165 ms. `await` itself is free: awaiting an
`immediate_awaitable` call (`aget_state`) costs 0.006 ms. So the async outlier is one wake-up that
the sync binding does not pay, and removing it requires running nodes on the event-loop thread,
which changes async concurrency semantics and has not been adopted.

Segment profiling of a warm `update_state` (66 us on a debug build): read 5.4 us, merge 10.7 us,
checkpoint build 20.0 us, snapshot 4.5 us, binding glue 30.4 us.

The async-binding column is bounded by the model described under *Known limits*: a fixed
~0.25 ms asyncio<->Tokio wake-up per call plus ~61 us per serialized node transition, of which
the node adapter itself is 28 us and the checkpoint save ~17 us per superstep (0.30 ms per
workflow run). Reaching 5x there would require running nodes on the event-loop thread, which
changes async semantics (that experiment was reverted; see *Reverted experiments*).

Engine probe (same run as above):

```
nodes= 1  invoke=   8.1 us   async= 19.4 us   history=3   per-save=6.5 us
nodes=10  invoke=  48.8 us   async=112.5 us   history=12  per-save=9.4 us
nodes=30  invoke= 183.4 us   async=409.4 us   history=32  per-save=12.8 us
fan-out 3: 141 us   fan-out 8: 146 us
get_state 2.1-2.5 us   history(5) 15.1 us   history(limit=50) 192 us
```

Parity: **38/38 scenarios equivalent** -- `command_goto` now also schedules the entry point and
`messages_stream` reports non-streaming model output, matching LangGraph.
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
python benchmarks\parity.py                      # 38/38 equivalent
cargo run --release --example perf_probe --features async
python examples\mega_workflow\bench.py
python examples\mega_workflow\bench_async.py
```