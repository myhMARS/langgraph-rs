# Performance audit and optimization results

Audit performed by Claude Code (static source review) plus the repo's own benchmark suite
(`python benchmarks/run.py`, release build, Windows 11, same machine per run).

## Optimizations applied

| Area | Change |
| --- | --- |
| Stream chunk construction | `invoke`/`run` now use `execute_quiet`; values/debug/task/updates/usage/custom/message chunks are only built when the API actually streams (`stream_with`/`stream_modes_with`). |
| Ephemeral channels | `clear_ephemeral` reports whether anything changed; the second `values_to_state` per superstep is skipped in the common case. |
| Channel merge | `apply_updates` borrows the current channel value instead of cloning it before every reducer. |
| Messages reducer | One `HashMap` id index per merge instead of a linear scan per incoming message (was O(n·m)); incoming array is borrowed, not cloned. |
| Checkpoint interop id | LangGraph SQLite sidecar stores the public hex id; no behavior cost. |
| Resume binding / deferred / finish persistence | Correctness fixes from the review (no benchmark regression expected). |

## Measured results (Rust native, ns/op)

| Scenario | Before | After | Change |
| --- | ---: | ---: | ---: |
| `linear_10` | 45.15 µs | 30.01 µs | **-33.5%** |
| `loop_50` | 225.64 µs | 150.13 µs | **-33.5%** |
| `fanout_4` | 319.01 µs | 251.03 µs | **-21.3%** |
| `checkpoint_5` | 64.63 µs | 54.08 µs | **-16.3%** |
| `build_compile` | 13.13 µs | 10.77 µs | -18.0% |
| `scale_100` | 502.90 µs | 282.40 µs | **-43.8%** |
| `mt_8x_linear_10` | 13.16 µs | 7.18 µs | **-45.4%** |

Peak allocated memory stayed flat or improved slightly (`mem_linear_10` 7.2 → 6.8 KiB,
`mem_loop_50` 15.9 → 15.5 KiB).

Rust native is now ~66–81× faster than Python LangGraph 1.2.22 on the same scenarios
(`mt_8x_linear_10` ~314× throughput). Raw data: `benchmarks/results/latest.json`,
report: `benchmarks/RESULTS.md`.

## Remaining high-value optimizations (not applied)

1. **Sync fan-out thread spawn**: one OS thread per task per superstep; `fanout_4` is still
   ~12× costlier per node than an inline node. A scoped worker pool/rayon would be the next
   big win.
2. **Checkpoint save copies**: `persisted_values` + `checkpoint.clone()` + pending-task clone
   per save; use `Arc<Checkpoint>` or persist by reference and pass the known changed-channel
   set instead of diffing every value.
3. **`Topic(accumulate=true)`** still clones the whole accumulated array per superstep.
4. **`values_to_state`** still serializes `S::default()` and deep-clones the map on each call;
   caching the serialized defaults / applying updates to typed state directly would help
   large states.
5. **Async runtime**: `join_all` + `Box::pin` per attempt; reusable checkpoint writer; async
   Redis put is 3 sequential RTTs.
6. **History reads**: `get_state_history_filtered_in` is O(n²) and clones the full history;
   Redis history/latest use N+1 round trips.
7. **Python binding**: callback installation per node, JSON state conversion, int/float
   extraction order.
8. **Benchmark harness**: move to criterion/percentiles and add message-heavy, async,
   streaming, subgraph and history scenarios to guard these areas.

## Re-benchmark after kernel caching (same machine)

| Scenario | Round 3 | Latest |
| --- | ---: | ---: |
| linear_10 | 30.01 µs | 28.80 µs |
| loop_50 | 150.13 µs | 137.30 µs |
| fanout_4 | 251.03 µs | 220.36 µs |
| checkpoint_5 | 54.08 µs | 45.39 µs |
| scale_100 | 282.40 µs | 239.82 µs |
| mt_8x_linear_10 | 7.18 µs | 7.35 µs |

Rust native is now ~78-108x faster than Python LangGraph on these scenarios.

## Latest benchmark (after adapter/refactor + Topic/caching)

| Scenario | Rust native | Python LangGraph | Speedup |
| --- | ---: | ---: | ---: |
| linear_10 | 27.15 µs | 2,054.22 µs | 75.7x |
| loop_50 | 129.82 µs | 12,377.14 µs | 95.3x |
| fanout_4 | 233.34 µs | 2,440.71 µs | 10.5x |
| checkpoint_5 | 49.97 µs | 3,926.57 µs | 78.6x |
| scale_100 | 255.21 µs | 18,210.23 µs | 71.4x |
| mt_8x_linear_10 | 6.17 µs | 2,015.58 µs | 326.8x |
