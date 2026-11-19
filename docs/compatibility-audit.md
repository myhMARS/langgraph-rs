# Compatibility audit vs Python LangGraph 1.2.22

Audited by Claude Code (static source review; its shell was blocked) against the Rust
implementation and the repo's own parity harness. Python package source was not available
inside that review environment, so Python-side claims that depend on exact 1.2.22 details
are marked "verify" where relevant.

The installed Python LangGraph is used by `benchmarks/parity.py`; the scenario groups there
are the executable comparison suite and currently report **34/36 equivalent, 2 known
differences** (`Command(goto=...)` does not also run the entry point here, and the `messages`
stream needs an explicit callback handler).

## Fixed in this pass

| Finding | Fix |
| --- | --- |
| `LangGraphSqliteSaver` returned an internal checkpoint id while storing LangGraph hex ids, so `resume_from(snapshot.checkpoint_id)` could not find the row | Sidecar now stores the hex id; the reader always takes the row's `checkpoint_id` column. Regression test: `tests/langgraph_sqlite.rs::saver_snapshot_ids_can_be_loaded_back` |
| `update_state` cleared `interrupted` / `paused_before` / `interrupts`, causing `interrupt_before` to pause again and multi-interrupt maps to stop working | Metadata is preserved; values are filtered through `schema.persisted_values` so private/untracked channels are not checkpointed |
| `messages` reducer was O(n²) (clone-array + linear id scan per message) | Single index map per merge; O(n+m). Covered by `tests/messages.rs` |
| `values_to_state` ran twice per superstep even when no ephemeral channel was cleared | `clear_ephemeral` reports whether it changed anything; second conversion is skipped |
| `apply_updates` cloned the current channel value before every merge | Reducers now borrow the current value |
| `invoke`/`run` built stream chunks (full values/debug/task/updates/usage payloads) even with a discarding sink | `execute_quiet` skips chunk construction; `stream*` paths keep full emission |
| Sync subgraph resume used the first resume history value rather than the latest | Uses `current_resume_last()` |

## Remaining compatibility gaps (not fixed in this pass)

### High impact

- **Python binding is not drop-in**: `invoke`/`get_state` take a thread id instead of a
  `RunnableConfig`; builder methods are not chainable; `get_state` returns a dict; no
  `CompiledStateGraph` alias; `MessagesState`/`add_messages` are Python-dict approximations;
  `create_react_agent` returns an uncompiled graph.
- **Python async graphs are effectively sequential** (each Python node fully runs inside one
  `Python::attach`; `astream` collects into a list rather than yielding an async iterator).
- **Streaming semantics**: Python `stream()` defaults to `updates`; the Rust binding with no
  modes returns every chunk; `usage`/`interrupt` have no mode and are always forwarded;
  `custom`/`message` are buffered per task rather than emitted live; `version` validation in
  the sync binding happens after the run.
- **Native async nodes** cannot use `interrupt()`/`custom()`/`get_stream_writer()` because
  those free functions still read the sync task thread-local; use `AsyncContext` methods.

### Medium impact

- `invoke(Command(goto=...))` also runs the normal entry point (Rust test locks this in;
  Python's exact behavior needs confirmation).
- Single vs multi interrupt use different pending-write policies, and a permanent failure
  without an error handler aborts before merging successful siblings.
- Checkpoint metadata/step numbering is synthesized (`source="loop"`, `step=total-2` for
  history) rather than stored faithfully; `update_state` inserts same-step checkpoints.
- Resume APIs accept no `RunConfig`, so recursion limit/durability/control are lost;
  managed values use the persisted step while the per-run counter resets.
- Interrupt ids are time-based/non-deterministic; static `interrupt_before` /
  `interrupt_after` hooks create synthetic `Interrupt` entries.
- `update_state` does not apply pending writes or infer `as_node`.
- `NamedBarrierValue` does not validate names, re-arm, or expose a set; it stores an array
  before opening.
- Checkpoint interop is sidecar-dependent and the msgpack decoder lacks bin8/16/32.
- Internal SQLite/Postgres history orders by `step`, not creation time.
- Async subgraph namespaces omit the task id; `AsyncStateGraph` has no mapped subgraph.
- v2 stream payload shapes differ from Python (`checkpoints`/`tasks`/`debug`/`messages`).
- `max_concurrency` bounds only `batch`, not intra-run parallel tasks.
- Python conversion is JSON-only; large ints lose precision.
- Python async callback context can leak on exception/interrupt; functional task results are
  thread-local.
- `RetryPolicy::default()` is `max_attempts=1` vs Python's 3; deterministic jitter and
  substring `retry_on`.
- `StateSchema::topic` defaults to accumulating, unlike Python's `Topic(accumulate=False)`.

### Low impact / polish

- `get_state_history` has no metadata filter and a missing `before` id returns full history.
- Thread prune/copy are partial; `InMemorySaver::put` does not dedupe by id.
- `add_messages` lacks `REMOVE_ALL_MESSAGES` and differs on remove-then-add ordering.
- `DeltaChannel` stores but never uses `snapshot_frequency`.
- Reserved node-output keys (`send`/`__goto__`/`__usage__`) shadow real channels.
- Missing APIs: `bulk_update_state`, `with_config`, JSON-schema getters, `output_keys`,
  `debug`, tags/metadata, `astream_events`, xray view.
- Sync `idle_timeout` is a fallback, not a real timer; `Durability::Async` writer errors are
  swallowed; default durability is `Sync` vs LangGraph's documented async.
- `LastValueAfterFinish` after-step checkpoint holds the pre-publish value under `Sync`.
- CI does not run `benchmarks/parity.py`; some sync persistence tests read different env
  names than CI sets.

## Follow-up fixes (rounds 5-6)

- Python `invoke`/`stream` accept a LangGraph `RunnableConfig`-shaped dict (`configurable.thread_id`) as well as a plain thread id.
- `CompiledStateGraph` alias exported.
- `create_react_agent` now returns a compiled graph (invoke works directly).
- Native async nodes can use the free helpers via a task-local `AsyncTaskOps`.
- Python `stream()` defaults to `updates`, strict mode filtering, version validated before running.
- History ordering is creation-time based for SQLite/Postgres; msgpack bin8/16/32 decoded; named barrier validates writer names; `before` unknown id returns empty history.


- Python builder methods (dd_node, dd_edge, dd_conditional_edges, entry/finish, dd_subgraph) now return the graph, so LangGraph-style chaining works; covered by python/tests/test_chaining.py.
- Python get_state/get_state_history return a StateSnapshot object that also supports mapping access (snapshot['values']) and attribute access (snapshot.values, snapshot.next, snapshot.config).
- Python stream returns an AsyncStream that supports both wait (returns the chunk list) and sync for; covered by python/tests/test_astream.py.
- MessagesState is now a real Python class; subclasses with __annotations__ are accepted as a schema (Annotated reducers map to channel specs), covering class MyState(MessagesState).

## Status after rounds 84-118 (sections above are historical)

The findings above were re-audited against Python LangGraph 1.2.22 source and live
behaviour; these are now implemented and covered by tests:

- **Checkpoint layout and step numbering.** A fresh run writes a raw-input checkpoint
  (`source="input"`, `next=["__start__"]`) followed by the post-input checkpoint and one
  checkpoint per superstep, and steps are thread-cumulative across turns. Exported/streamed
  `metadata.step` is the internal counter minus one, matching `get_state().metadata.step`.
- **Interrupts.** Interrupt checkpoints are saved at the post-increment step (verified:
  metadata step `0`, history `[0, -1]`, resume step `1`), and the interrupted superstep's
  nodes are recorded in `versions_seen`.
- **Static breakpoints.** `interrupt_before`/`interrupt_after` pause without raising in
  `invoke`/`ainvoke`/`resume*`; snapshots expose no interrupts for them; `updates` streams
  emit `{'__interrupt__': ()}` and `values` streams emit nothing.
- **Step-scoped topics.** `Topic(accumulate=False)` is cleared after a superstep's writes,
  and only for channels that superstep did not write (values remain readable in the next
  step, matching `_algo.apply_writes`).
- **`update_state`.** Applies the checkpoint's pending writes first, advances the step,
  recomputes channel versions without touching `versions_seen`, preserves pending `next` and
  `paused_before`, and behaves identically in the async runtime.
- **Interrupt ids.** Node-scoped namespace (`{ns}|{node}:{task_id}`) plus `|{answered}` for
  repeated interrupts; the SQLite saver uses the same separator.
- **Cross-engine import.** JSON/msgpack and SQLite readers decode `__interrupt__` control
  writes into interrupts, rebuild `next` only from interrupted/failed tasks, and keep plain
  state writes out of the pending set.
- **Streaming.** The starting state is emitted before the first superstep, v1 task payloads
  carry `triggers`, and `StateTask.state` is only populated for subgraph tasks.
- **Managed values and budget.** `remaining_steps` now follows LangGraph's `stop - step`
  rule (first superstep reports `limit - 1`), and the default recursion limit is 10007 like
  `LANGGRAPH_DEFAULT_RECURSION_LIMIT`.
- **Robustness.** An unkeyed resume with several pending interrupts is rejected; the entire
  feature matrix (none/async/sqlite/redis/postgres/msgpack/python/python-async) builds
  without warnings.

Still open, in rough priority order (refreshed in round 149; items 1-4 of the previous list are
implemented — see the round sections below):

1. **Router re-evaluation, not branch replay.** A pause replays the successful siblings by
   re-running `collect_next_nodes` against the resumed state, whereas LangGraph replays the
   stored branch write. A router that reads a channel the resumed task writes can pick a
   different target. Routes themselves (static/conditional edges, `goto`, `Send`) survive.
2. **Trigger-accurate bookkeeping.** `versions_seen` records the pre-step version of every
   channel for each executed node (reads are not tracked), which is enough for author
   inference but not identical to Python's per-trigger map. Named barriers are consumed on a
   normal superstep only (interrupt paths keep them armed) but not narrowed to read channels.
3. **Interop edge cases.** Tuple-form `pending_writes` carry no node, so a foreign document
   with only tuple writes cannot reconstruct failed/pending tasks (sidecar and `__error__`
   handling cover our own exports). `channel_values.__interrupt__` (`__root__` included) is
   decoded, `__root__` state is flattened, and control channels are stripped.
4. **Debug stream payload details.** Checkpoint records (input/loop) and `config.checkpoint_id`
   are emitted; per-step debug task entries still serialize the `Send` input as `state` and the
   shape of `result`/`metadata` is not byte-identical to Python.
5. **Python binding breadth** (unchanged from the historical section): drop-in
   `RunnableConfig` API surface, streaming iteration semantics, and v2 payload details.
## v4 task ids and the export version (rounds 130)

- Internally new checkpoints use LangGraph v4 semantics: ids are UUID-shaped hex and task ids
  are `xxh3_128(checkpoint_id_bytes + ns + str(step) + node + "__pregel_pull" + "branch:to:<node>")`
  (`"__pregel_push" + index` for `Send` tasks), rendered as an 8-4-4-4-12 UUID.
- The LangGraph-compatible SQLite writer/export still emits `"v": 1` and SHA-1 task ids. Reasons:
  a Python reader picks `_xxhash_str` for `v > 1` and `_uuid5_str` otherwise
  (`langgraph/pregel/_algo.py:550`); our writes table stores the ids written beside each row, so
  the export and the stored ids must use the same hash. Switching the export to `v: 4` while the
  saver still wrote SHA-1 ids made Python see unknown tasks ("multiple pending interrupts",
  `tuple index out of range` on the Send-fan-out case), and switching the saver to xxh3 did not
  resolve it because the row layout (`task_path`, `idx`) still assumes the v1 shape.
- Consequence: cross-language readers recompute SHA-1 ids for exported checkpoints, while the
  engine's own runtime uses v4/xxh3 ids. Both are self-consistent; only a reader that mixes our
  export with a Python writer on the same thread would see different ids.
- Multi-turn step numbering was re-verified against Python (single node, three turns):
  Python `[1,0,-1]` / `[4,3,2,1,0,-1]` / `[7,6,5,4,3,2,1,0,-1]`, Rust identical (no drift).
## Round 147 verifications (update_state on a paused thread)

Python and Rust were probed with the same graph (`a -> ask`, `ask` raises `interrupt`):

| call after the pause | Python `next` | Rust `next` |
| --- | --- | --- |
| `update_state({"value": 9}, as_node="a")` | `('ask',)` | `['ask']` |
| `update_state({"value": 10}, as_node="ask")` | `()` | `[]` |

So an `as_node` update does **not** unconditionally keep interrupted tasks: the paused node
stays pending only when the chosen author's outgoing route reaches it (`a -> ask`), and the
pause disappears when the author is the interrupted node itself. Rust matches, and the
earlier audit note claiming "Rust drops the pending task, Python keeps it" is incorrect.

Related verified behaviour: `update_state` on a finished thread infers the author with
Python's `_infer_as_node` rules (one-node shortcut, highest pre-step version wins), and
raises "Ambiguous update, specify as_node" when no author can be inferred.
## Closeout verification (round 156)

An independent static audit (Claude Code, reads-only) traced the pause/resume protocol and
concluded "nothing material remains" for the four interrupt flows (linear, fan-out sibling+join,
sequential interrupts, `Send` fan-out with one interrupt) under the default durability, with all
files under 700 lines (99 files, ~27.5k lines, largest `_internal/interop.rs` at 670).

Accepted P2 items, kept documented rather than fixed:

1. **Router re-evaluation.** A pause replays sibling routes by re-running `collect_next_nodes`
   against the resumed state instead of replaying the stored branch write. Routes survive; a
   router that reads a channel the resumed task writes may pick a different target.
2. **`Durability::Exit` route inheritance.** Route inheritance is gated to a run's first
   superstep (`PauseContext::first_tick`), which closes the double-run window; locked by
   `tests/interrupt_routing.rs::exit_durability_does_not_replay_consumed_routes`.
3. **Same-node executed+replayed routing.** `routing_executed` de-duplicates by node name, so a
   node that is both a replayed `Send` sibling and the re-executed task can appear twice when it
   has conditional `Send` edges. Not reachable through the standard static-continuation shapes.
## Performance work (round 1)

Functional behaviour is unchanged; every change below is a pure fast path, verified by the
Rust suite, the 119 Python binding tests and the mega-workflow examples.

| Change | Where | Effect (same process, 300 runs) |
| --- | --- | --- |
| Borrow the run state per node instead of cloning it | `src/pregel/superstep.rs` | 10-node run 64.4 -> 47.5 us |
| Move `channel_versions`/`versions_seen` out of the previous checkpoint instead of cloning both | `src/checkpoint.rs::versions_for`, `pregel/checkpointing.rs` | 30-node run 1049 -> 913 us |
| Skip the `Durability::Async` writer thread for in-memory checkpoints (`Checkpointer::is_in_memory`) | `_internal/ports.rs`, `adapters/memory.rs`, `pregel/mod.rs` | 1-node run 94.7 -> 16.7 us; 30-node 928 -> 590 us |
| Resolve `aget_state`/`aget_state_history` eagerly instead of scheduling a Tokio task | `adapters/python/async_support.rs`, `async_graph.rs` | `aget_state` 0.291 -> 0.067 ms; history(5) 0.347 -> 0.114 ms |
| One asyncio loop per OS thread for node coroutines (previous round) | `adapters/python/async_support.rs` | 10-node `ainvoke` 14.16 -> 2.48 ms |

Remaining known gaps: binding `state`/`update_state` are still ~1.5-2x slower than LangGraph
because snapshots are converted into fresh Python dicts; `aupdate_state` still goes through the
Tokio bridge because the underlying saver call is async.### Round 2 additions

| Change | Where | Effect |
| --- | --- | --- |
| `aupdate_state` runs inline for in-memory savers instead of scheduling a Tokio task | `adapters/python/async_graph.rs` | 0.58 -> 0.132 ms per call (4.4x), now faster than LangGraph async (0.33 ms) |
| Benchmarks warm up once and interleave engines per iteration | `examples/mega_workflow/bench.py`, `bench_async.py` | removes the ±20% drift between runs and whichever engine ran first |

Current per-phase medians (10 interleaved suites, single machine; LangGraph 1.2.22):

| Phase | sync binding | sync native | async binding | async native |
| --- | ---: | ---: | ---: | ---: |
| invoke | 4.40x | 19.5x | 1.57x | 22.9x |
| resume | 6.36x | 24.0x | 1.08x | 10.6x |
| state | 1.44x | 5.7x | 1.55x | 6.3x |
| update_state | 1.89x | 8.4x | 1.23x | 9.1x |
| stream | 4.28x | 11.7x | 1.62x | 11.2x |
| batch | 6.29x | 19.4x | 2.59x | 18.9x |### Round 3 additions

| Change | Where | Effect |
| --- | --- | --- |
| Parallel supersteps run on a process-wide worker pool instead of spawning scoped threads per task | `src/pregel/pool.rs`, `pregel/superstep.rs`, `pregel/backend.rs` | fan-out of 3: 200 -> 95 us; fan-out of 8: 471 -> 200 us |
| Nested runs started from a pool worker fall back to scoped threads (`pool::on_worker` guard) | `pregel/pool.rs`, `pregel/superstep.rs` | prevents self-deadlock when a node invokes a child graph with its own fan-out; covered by `nested_parallel_runs_do_not_deadlock_on_the_worker_pool` |

Mega-workflow native timings after the pool landed (mean of several runs): sync invoke
0.53 -> 0.28 ms, batch 0.98 -> 0.60 ms. The cross-engine ratio moves with machine load
(measured 29x-47x for sync invoke across runs), so the probe numbers above are the
authoritative effect size.### Round 4 additions

| Change | Where | Effect |
| --- | --- | --- |
| `history_page_in` trait method; `InMemorySaver` clones only the requested page instead of the whole thread history | `_internal/ports.rs`, `checkpoint.rs::slice_history`, `adapters/memory.rs`, `graph/state_api.rs` | probe `get_state_history(limit=5)` 33 -> 23 us |
| `InMemorySaver::put` checks the newest checkpoint first; a full scan only runs for pause re-writes | `adapters/memory.rs` | removes an O(history) scan from every save on long threads |
| LangChain callback context is built once per OS thread and reused, instead of rebuilding the handler and re-importing `langchain_core.runnables.config` per node | `adapters/python/convert.rs` | binding 10-node `invoke` 0.589 -> 0.513 ms (~7 us per node) |

Latest probe (30-node chain, checkpointer): 529 us per run, 17.2 us per save, fan-out of 3: 71 us,
fan-out of 8: 93 us, `get_state` 5.3 us, `history(5)` 22.9 us.### Round 5 additions

| Change | Where | Effect |
| --- | --- | --- |
| Async engine also skips its checkpoint-writer task for in-memory savers | `adapters/tokio_runtime/execute.rs` | removes a Tokio task and one channel clone per checkpoint for async graphs (no behaviour change: the saver cannot block) |

Measured bridge costs that bound further gains (Windows, CPython 3.12):

* `pyo3_async_runtimes::tokio::future_into_py` round trip: ~270-300 us per awaited call. This is
  why every async API that must return a Tokio future (`ainvoke`, `aresume`, `astream`, `abatch`)
  carries a fixed ~0.3 ms; `aget_state`/`aget_state_history`/`aupdate_state` avoid it through the
  eager path.
* Sync `invoke` on a one-node graph: 41 us without a checkpointer, 76-100 us with one, versus
  7 us / 17 us natively. Of that, ~15 us is fixed call overhead (config parsing, snapshot/output
  conversion, PyO3 dispatch) and the rest is the per-node Python round trip (state dict, call,
  output parse, GIL re-acquire).### Round 6 additions

| Change | Where | Effect |
| --- | --- | --- |
| `get_state_history_where*` pushes `limit` into the page read when no metadata filter is given (both sync and async graphs) | `graph/state_api.rs`, `adapters/tokio_runtime/state_api.rs` | binding `get_state_history(limit=1)` 229 -> 37 us, `limit=5` 279 -> 161 us on a 15-node thread |

The previous code materialised the whole history (every `StateSnapshot`, including value
deserialization and Python conversion) and truncated afterwards, so `limit` did not bound the
cost at all. The bench `state` phase (one `get_state` + one `history(5)`) moved from 1.17x to
2.05x of LangGraph sync.

Also measured this round: the async **engine** is as fast as the sync one natively (1 node 7.2 us,
10 nodes 45 us), so the async binding's fixed ~0.3 ms belongs to `pyo3_async_runtimes`'
`future_into_py` bridge (double spawn + contextvars restore), not to the Pregel loop; a
hand-written awaitable would be needed to remove it, which would change cancellation semantics.