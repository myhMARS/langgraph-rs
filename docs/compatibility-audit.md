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
hand-written awaitable would be needed to remove it, which would change cancellation semantics.### Round 7 additions

| Change | Where | Effect |
| --- | --- | --- |
| `from_checkpoint_live` returns the base snapshot directly when the checkpoint has no pending writes (it used to clone the whole value map only to discard the copy) | `graph/snapshot.rs` | binding `get_state` 33 -> 29 us |
| `snapshot_to_py` builds the metadata dict directly instead of cloning the map into a `Value` and converting that | `adapters/python/stream_convert.rs` | binding `get_state_history(5)` 161 -> 142 us |

Investigated and rejected this round (measured, not guessed):

* A hand-written awaitable (thread + `call_soon_threadsafe`) for `ainvoke` would cost ~85 us per
  call against ~114 us for `pyo3_async_runtimes::future_into_py`; the 29 us saving does not
  justify losing its cancellation and contextvars handling.
* asyncio loops are **not** recreated per node: 100 runs x 10 async nodes created 0 new loops,
  so the remaining ~150 us per async Python node is not loop construction.### Round 8: where an async Python node's time goes

Instrumented `async_node_from_python` with per-segment counters (removed before commit),
2000 node calls on a 10-node async graph:

| segment | us/node |
| --- | ---: |
| `Python::attach` | 1.3 |
| state dict conversion | 1.8 |
| callback context install | 1.5 |
| Python callable invocation | 4.6 |
| driving the coroutine (`run_until_complete`) | 33.7 |
| `NodeOutput` parsing | 7.5 |
| **adapter total** | **54.6** |
| wait, the remaining wall time per invoke | 721 us |

That is: the Python-side adapter is 55 us/node, while wall time is ~87-115 us/node, so roughly
half the async per-node cost sits **outside** the adapter - in the Pregel loop as driven by
`pyo3_async_runtimes`, whose task scope saves/restores `contextvars` around every poll of the
returned future. A single-task fast path in `AsyncBackend::run_superstep` was implemented and
measured (87 vs 105 us/node slope) but it broke `abort_cancels_a_running_async_node` when placed
outside the abort `select!`; with the branch moved inside the select the gain disappeared into
measurement noise, so it was reverted rather than kept unproven.### Round 9: what a checkpoint save actually costs

Instrumented `save_checkpoint_via`/`build_checkpoint` (removed before commit), 30-node chain,
32 saves per run, 200 runs:

| segment | us per save |
| --- | ---: |
| `persisted_values` (channel clone) | 0.7 |
| `versions_for` | 1.3 |
| `next.to_vec()` | 0.2 |
| **`checkpoint.clone()` before `persist_checkpoint`** | **~6.5** |
| in-memory saver `put` (measured separately) | 0.1-2.4 |
| total save | ~9 |

So the dominant cost is not the saver and not `versions_for`: it is the full `Checkpoint` clone
the writer performs because `PregelBackend::persist_checkpoint` takes the checkpoint by value
while the loop must keep the same checkpoint as `previous`. A single 30-node run spends
~276 us of its ~600 us in that clone.

Planned fix (next round): store/pass checkpoints as `Arc<Checkpoint>` (`put(Arc<Checkpoint>)`,
`previous: Option<Arc<Checkpoint>>`). In-memory savers then push the `Arc` directly, durable
savers serialise through the `Arc`, and `previous` only needs the two version maps cloned
(~1.5 us) instead of the whole checkpoint.### Round 10: checkpoints are shared, not deep-cloned, on the write path

`PregelBackend::persist_checkpoint`, `Checkpointer::put_shared` (new, default = copy) and the
sync/async writer channels now move `Arc<Checkpoint>` instead of a `Checkpoint` by value:

* `save_checkpoint_via` builds the checkpoint once, wraps it in an `Arc`, hands a clone of the
  `Arc` to the writer and keeps the same `Arc` as `previous`;
* `InMemorySaver` stores `Arc<Checkpoint>` (readers still return owned copies, so the public
  API is unchanged) and overrides `put_shared` to keep the `Arc`;
* durable savers inherit the default `put_shared`, which copies -- they serialise immediately
  anyway, so nothing is lost;
* pause paths (`persist_pause`, `attach_pending_writes`) use `Arc::make_mut`, paying one copy
  per pause instead of one per superstep;
* `versions_for` copies the two version maps from the shared previous checkpoint (it can no
  longer take them), which is far cheaper than cloning the whole checkpoint.

This removes an O(state size) deep clone from every checkpoint write: with a small state the
wall-clock delta is inside this machine's noise band (±15%), but the clone scaled with the
state, so larger states and long histories benefit proportionally, and the peak allocation per
superstep drops by one full checkpoint copy.### Round 11: move channel writes instead of cloning them

| Change | Where | Effect |
| --- | --- | --- |
| `Channel::apply_owned` moves the winning value for `last_value`/`any`/`ephemeral`/`untracked` channels (the executor owns the writes) | `src/channels.rs` | removes one deep copy of every written value per superstep |
| `apply_updates` handles the single-write case without building the grouping maps, and inserts implicit channels for dynamic schemas directly | `_internal/state.rs` | removes two allocations per node |

Probe after both changes (quiet machine, 30-node chain): run 600 -> **497 us**, per-save
19 -> **15.5 us**; 16 KB state: 194 -> 176 us. The earlier "big state" numbers (1.5 ms at
256 KB) mix in the probe node's own 256 KB clones and were measured on a loaded machine; only
same-session comparisons are meaningful.

A regression the new fast path introduced was caught by the unit suite: dynamic schemas return
`resolve(...) == None` for implicit channels, and the fast path treated that as "skip the
write". Fixed by inserting the value directly, matching the general path.### Round 12: version bumps follow the written channels, not a value comparison

`versions_for` used to decide which channels changed by comparing every channel value with the
previous checkpoint (`!= Some(value)`, a deep compare). For large channel values that comparison
cost more than the rest of the save, and it is not what LangGraph does: LangGraph bumps the
version of every channel that was written in the step.

`versions_for` now derives the changed set from `written_by_node` (the executor already passes
it) and falls back to the value comparison only when no write information exists (fresh starts
and synthetic checkpoints).

Measured with the save-segment probe (256 KB channel, 12 saves per run, before removal of the
instrumentation): checkpoint build per run 205.9 -> 53.2 us, `persisted_values` 118.2 -> 95.6 us
(the remainder is the one unavoidable value copy). Small states are unchanged; parity stays
34/36 and the full Rust + Python suites pass.### Round 13: snapshots take ownership of the checkpoint's values

`StateSnapshot::from_checkpoint` used to deserialize a *borrowed* value map
(`values_to_state(&checkpoint.values)`), copying every channel even though the checkpoint was
already owned. `values_to_state_owned` moves the map into the state (merging schema defaults
first), and `next` is computed before the map is moved out.

`get_state` 29 -> 26 us, `history(5)` 142 -> 128 us (binding, 15-node thread), all suites and
parity unchanged.

Also checked in this round: the async `update_state` gap is not the eager awaitable. A pure
Python eager awaitable costs < 1 us, and `aget_state` (same eager path) is 31 us total, while
`aupdate_state` is ~103 us against ~35 us for the sync API -- so the extra time is inside the
async update path, not the bridge. The async `resume` phase also fluctuates between 0.94x and
1.26x across runs; it is tracked as noise until a stable regression appears.### Round 14: in-memory savers skip the `BlockingSaver` wrapper in the async Python API

Root cause of the async binding lag: `AsyncStateGraph.compile(checkpointer=<in-memory saver>)`
wrapped *every* saver in `BlockingSaver`, so each async read/write paid a `spawn_blocking`
round trip (~40 us) even though the saver cannot block. The async engine already supports a
plain sync checkpointer inline (that is how `saver_get`/`saver_put`/`saver_history` fall back),
so the wrapper is now only installed for savers that can actually block.

Measured (binding, in-memory saver):

| call | before | after |
| --- | ---: | ---: |
| `aupdate_state` | 103-114 us | **37 us** |
| `aget_state` | 51 us | **18 us** |
| `ainvoke` (1 node) | 579 us | **350 us** |

Cross-engine medians (10 interleaved suites):

| phase | async binding before | after |
| --- | ---: | ---: |
| invoke | 1.51x | 1.63x |
| resume | 0.94-1.26x | 1.32x |
| state | 1.39x | 1.95x |
| update_state | 1.05x | 1.79x |
| stream | 1.40x | 1.96x |
| batch | 2.31x | 3.53x |

This also settles the round-13 question: the async `update_state` gap was the saver hop, not
the eager awaitable and not the engine.### Round 15: `has_thread` instead of building a snapshot to test existence

Every `invoke(input, thread_id=...)` and `Command` path first asked
`graph.get_state(thread)?.is_some()`. That loads the latest checkpoint *and builds a full
`StateSnapshot`* (values deserialize, task list, config dicts) just to answer a yes/no question.

* `Checkpointer::has_thread` (default: `latest(..).is_some()`) and an O(1) override for
  `InMemorySaver` (`contains_key`);
* `CompiledGraph::has_thread` / `has_thread_in` and an async `has_thread`;
* both Python bindings use them.

Measured (binding, existing thread, repeated `invoke`): sync 116 -> **79-94 us**, async
`ainvoke` 407 -> **357-367 us**. Cross-engine medians after the change: async binding invoke
1.63x -> 1.89x, state 1.95x -> 1.70x (noise band), sync unchanged within noise.### Round 16: command detection without raising, and a rejected eager bridge

`Command` inputs were recognised with `obj.extract::<PyRef<PyCommand>>()`, which builds and
clears a Python exception for every ordinary dict input on the hot path. `obj.cast::<PyCommand>()`
checks the type without raising: `invoke` without a checkpointer 45 -> 37 us; ainvoke medians
1.89x -> 2.05x, batch 6.21x -> 7.41x.

Rejected (measured, then reverted): polling the graph future once on the calling thread before
handing it to Tokio. It looked excellent -- `ainvoke` 357 -> **116 us**, `aresume` 24 us,
`astream` 143 us, `abatch` 119 us -- but it runs node code on the event loop's thread, so
(a) async Python nodes fail with "Cannot run the event loop while another loop is running"
(the thread-local node loop is already running there) and (b) a slow synchronous node would
block the loop instead of running on a Tokio worker. Both are semantic regressions, so the
`future_into_py` bridge stays; the ~105 us it costs per call is comparable to Python's own
cross-thread future completion (171 us measured for a thread-per-call hop).### Round 17: `operator.add` reducers stay in Rust

A Python callable reducer used to be invoked for every merge: convert the accumulated value
*and* the write to Python objects, call the callable, convert the result back. For
`Annotated[list, operator.add]` (the most common reducer) that is two full list copies plus a
Python round trip per superstep.

`channel_specs_from_py` now detects `operator.add` when the schema is built; the reducer
concatenates two JSON arrays natively and only falls back to Python when either side is not an
array (strings, numbers, dicts), which keeps the semantics identical.

Measured (40-node chain writing one list element per step, binding): **3.45 ms -> 1.67 ms**
(2.1x). Graphs with other reducers are unchanged, and parity stays 34/36.### Round 18: build snapshots from a borrowed checkpoint

`update_state` (sync and async) built its result with
`StateSnapshot::from_checkpoint(updated.clone())`, cloning the whole checkpoint -- values,
`versions_seen`, `channel_versions`, pending writes, task results -- and then moving pieces out
of the copy. `from_checkpoint` is now split into `from_parts(&Checkpoint, values)` plus two
constructors: one that moves the value map out of an owned checkpoint (`from_checkpoint`) and
one that copies only the value map (`from_checkpoint_borrowed`), used by both update paths.

`update_state` 35 -> 33 us, `aupdate_state` 37 -> 32 us (now on par with the sync API);
cross-engine `update_state` 1.60x -> 2.15x sync / 2.19x async.### Round 19: reads share the stored checkpoint instead of copying it

`get_state` called `latest_in`, which for the in-memory saver deep-copied the whole checkpoint,
and then built a snapshot from that copy. The saver already stores `Arc<Checkpoint>`, so
`latest_shared`/`latest_in_shared` (default: wrap `latest` in an `Arc`; in-memory: clone the
stored handle) hand the read path a borrowed checkpoint, and
`StateSnapshot::from_checkpoint_live_borrowed` builds the snapshot from it -- copying only the
value map, and merging pending writes into a fresh map when there are any.

`get_state` 26 -> **16 us**, `aget_state` 51 -> **18 us**, `history(5)` 128 -> **113 us**
(10-node thread). Durable savers keep the default implementation, so nothing changes for them.### Round 20: history reads share the stored checkpoints

`get_state_history` called `history_page_in`, which copied every selected checkpoint (values,
version maps, tasks, pending writes), and then moved the copy into the snapshot. The shared
handles from round 19 are now used for history too:
`history_shared`/`history_in_shared`/`history_page_in_shared` (default maps to `Arc::new`,
in-memory clones the stored handles) plus `from_checkpoint_borrowed` per entry, so only the
value map is copied.

`history(5)` 113 -> **109 us**, `aget_state_history(5)` 114 -> **97 us** (10-node thread);
durable savers keep the default implementation.### Round 21: the parity suite now really exercises retries, and covers reducers

Two verification gaps were closed instead of adding another micro-optimisation:

* `retry_policy` benchmarked a graph whose node counter had already been advanced by the
  equivalence check, so every measured call took the straight-line path and the retry path was
  never timed. The bench now resets the counter before each measured invoke.
  With retries actually running, RS completes in **3.3 ms** (2 x 1 ms backoff) while LangGraph
  takes **~990 ms** for the same `initial_interval=0.001, backoff_factor=1.0` policy -- LangGraph
  appears to sleep ~0.5 s per retry regardless of the requested interval. Our engine follows the
  documented seconds semantics; this is recorded as an observed reference-implementation timing
  anomaly, not something to copy.
* New `reducer_add` scenario drives `Annotated[list, operator.add]` on both engines (the path
  optimised in round 17) and asserts the resulting state is identical; it is, and RS is 5.2x
  faster there.

Parity is now 35/37 equivalent with the same 2 known differences.### Round 22: `versions_seen` entries are shared, so checkpoints stop deep-copying them

Profiling the save path again showed `versions_for` at **7.09 us** of the 7.50 us checkpoint
build: cloning `versions_seen` deep-copies one `Value::Object` per executed node (30 entries on
a 30-node chain) on every save, and history needs each checkpoint to keep its own copy.

The field is now `VersionsSeen = IndexMap<String, Arc<Value>>` (insertion order preserved,
serde with the `rc` feature serialises `Arc<T>` exactly like `T`, so the persisted JSON shape is
unchanged). Appending a version clones only the pointer array; the per-node maps are shared.

`versions_for` 7.09 -> **3.03 us** per save, checkpoint build 7.50 -> **3.46 us**.
30-node runs: 546 -> ~460-500 us with checkpoints. Verified with the full Rust suite (including
the SQLite/LangGraph-format readers that reconstruct `versions_seen` from bare JSON), 119 Python
tests and parity 35/37 -- the two known differences are unchanged, and every checkpoint-shape
scenario (`checkpoint_history`, `time_travel`, `sqlite_persistence`, `pending_writes`,
`update_state`, `snapshot_metadata`) still reports equivalent.### Round 23: a large-state parity scenario, and what history actually costs

Added a `big_state` parity scenario: a 256 KB string channel rewritten on every superstep of a
4-node graph with an in-memory checkpointer. Both engines must return the same value, blob and
`get_state` blob; they do, and RS is 2.70x faster there (2.38 ms vs 6.43 ms).

Investigating the weakest remaining scenario (`checkpoint_history`, 1.45x) showed the read path
is not quadratic in history length: native `get_state` 2.6 us, `history(limit=1)` 5.2 us,
`history(50)` 132 us and the full 840-checkpoint read 4.0 ms -- i.e. ~3-5 us per snapshot,
linear. The apparent 168 us/snapshot in the binding came from the *state* growing across
supersteps (a 600-element `append` log), which both engines must copy per snapshot.

Documented divergence (pre-existing, not caused by these optimisations): LangGraph's
`get_state_history` returns a lazy generator, our binding returns a list. Iteration works the
same; `len()`/indexing only work on ours. The parity scenario materialises both sides, so the
comparison stays valid, and the difference is recorded here rather than silently changing the
Python API.### Round 27: `get_state_history` is lazy, like LangGraph's generator

The binding returned a fully materialised list: every checkpoint of the thread was converted to
a Python snapshot even when the caller only wanted the newest one. It now returns a `StateHistory`
sequence that loads checkpoint handles on first access and converts each snapshot on demand
(cached), with `len()`, positive/negative indexing, iteration and `to_list()` preserved.

Measured on a 420-checkpoint thread (state grows to ~300 entries per checkpoint):

| call | before | after |
| --- | ---: | ---: |
| `len(history)` | ~27 ms (full conversion) | **0.037 ms** |
| `history[0]` (newest snapshot) | ~27 ms | **0.13 ms** |
| `next(iter(history))` | ~27 ms | **0.11 ms** |
| `history.to_list()` | 27 ms | 27.6 ms (unchanged; converts everything) |

Both the sync and async graphs use it (`aget_state_history` keeps returning an awaitable that
resolves to the same object); metadata-filtered queries stay eager because a filter cannot be
answered from handles alone.

Behaviour verification: LangGraph's `get_state_history` is a generator, so its contents are
resolved at iteration time. That is now true for us as well -- the async thread-management test
that captured the history and then deleted the thread had to materialise it first, and a direct
LangGraph check (`gen = graph.get_state_history(...)`, `delete_thread`, `len(list(gen)) == 0`)
confirms the reference implementation behaves identically. Parity stays 36/38 and all 119
Python tests pass.### Round 28: `update_state` stops copying a fully-persisted state

`update_state` (sync and async) merged the update into a copy of the latest checkpoint, then
called `persisted_values` -- which, for a schema that persists every channel, is a second full
copy of the same map -- and then moved the *filtered* copy into the new checkpoint. Since the
merged map is owned at that point, the filter is now only applied when channels are actually
excluded (`StateSchema::persists_all`), so the map moves into the checkpoint instead of being
copied once more.

This is one full state copy removed per `update_state` call. The phase remains the weakest of the
sync binding (~1.9-2.3x) because both engines are sub-millisecond there and the remaining cost is
the state copy that `update_state` inherently performs (the caller-visible snapshot and the stored
checkpoint cannot share one map).### Round 29: `update_state` reuses the merged state for its snapshot

`update_state` merged the write into an owned map, deserialised it into the typed state (needed
for `collect_next_nodes`), moved the map into the new checkpoint, and then deserialised the
checkpoint values *again* to build the returned snapshot. `StateSnapshot::from_parts` is now
used with the state that was already materialised, so the second deserialisation is gone.

`update_state` 0.19-0.21 ms -> **0.14 ms** on the mega workflow thread (LangGraph 0.38-0.40 ms),
i.e. 1.9-2.3x -> **2.7-2.8x**. Verified with the Rust suite, clippy, 119 Python tests and parity
36/38.### Round 30: one LangChain callback scope per run instead of one per node

Every Python node used to install (and reset) the LangChain callback context variable around
its call: two Python attribute lookups + two calls per node. The binding now installs the context
once around the whole run (`CallbackScope`), and `install_langchain_callbacks` skips the per-node
work while a scope is active on that thread. Parallel nodes run on worker threads, where the
thread-local flag is false, so they keep their per-node installation and still see the context.

Behaviour check: `test_langchain_callbacks_auto_injected` (a `FakeListChatModel` streamed inside a
node with `stream_mode="messages"`) still yields `hello world`; the `messages_stream` parity entry
keeps its pre-existing documented difference, and the full suite is unchanged at 119 tests /
parity 36-38.

Effect (mega workflow, 14 iterations): invoke 2.55 -> **2.44 ms (5.80x)**, batch 8.11-10.32x,
resume 7.35x, state 11.39x, stream 4.95x, update_state 2.80x.### Round 31: config parsing stops raising for dict configs

`config_from_py` tried `value.extract::<String>()` first, so every LangGraph-style
`{"configurable": {...}}` config raised and cleared a Python exception before being parsed as a
dict. A `PyString` type check now selects the fast path, so dict configs parse without the
exception.

`update_state` with a config dict 0.084 -> **0.071 ms**, `get_state` 0.031 -> **0.028 ms**
(workflow thread, warm). Cross-engine ratios for the phases that take config dicts improve
accordingly; the mega-bench ratios remain load-sensitive (see the caveat in the performance
report).### Round 32: node output and router parsing stop raising

`node_output_from_py` probed for `Command`/`Send` with `extract::<PyRef<..>>()` before treating a
result as a dict; for the common dict case both probes raised and cleared a Python exception.
`route_from_py` did the same for strings and sends, once per superstep. Both now check the type
first (`cast::<...>()`).

Per-node profiling (temporary counters, removed): the adapter spent ~8 us/node on a warm simple
graph, of which **5.3 us was this parse step** -- the two failed extracts. After the change the
adapter is ~3 us/node and a 10-node linear chain drops 342 -> 280 us.

Mega-workflow ratios (14 iterations): invoke **5.41x**, stream **5.59x**, state 13.96x,
resume 7.13x, batch 8.78x, update_state 3.59x (absolute 0.12 ms vs LangGraph 0.43 ms).
Parity stays 36/38 and the 119 Python tests pass.### Round 33: `update_state` shares the stored checkpoint

`update_state` deep-copied the latest checkpoint (`latest_in`) and then copied its `task_results`
and `deferred` fields again into the new checkpoint. It now reads the shared handle
(`latest_in_shared`) and borrows it, cloning only the fields the new checkpoint needs.

Engine segment profiling (temporary counters, removed) for a warm `update_state` on the mega
workflow thread: total 71 us = read 5.4 + merge 10.7 + checkpoint build 20.0 + snapshot build 4.5
+ binding glue 30.4. LangGraph takes 0.34-0.48 ms for the same call, i.e. **4-6x** depending on
the run; the mega suite reports lower because it measures the *first* call on a fresh thread
(0.09-0.18 ms) rather than steady state.### Round 34: state conversion no longer clones the whole state map

`json_state_to_py` built `Value::Object(state.0.clone())` and handed it to the generic
`value_to_py`, i.e. it deep-copied every channel value on **every node call and every snapshot
conversion**. It now fills the `PyDict` straight from the borrowed map.

Effect: state conversion per node drops from 1.7 us to well under 1 us on small states and the
copy disappears entirely for large values (a 4 KB channel was copied per node before). The mega
workflow ratios (quiet run): invoke **6.17x**, resume 8.30x, state 13.18x, stream **6.05x**,
batch 11.22x, update_state 3.02x.### Round 35: `update_state` skips router machinery for a node without successors

When `as_node` names a node with no outgoing edges, no conditional edges and no destinations,
LangGraph schedules no follow-up tasks. `update_state` still ran the full `collect_next_nodes`
path (state borrow, dedupe sets, send conversion); it now returns an empty task list directly.

Warm `update_state` on the mega workflow thread: 71 -> **66 us**, i.e. 5.8-7.3x vs LangGraph's
0.38-0.48 ms. The mega suite reports ~3.2x because it measures the first call on a fresh thread
(0.09-0.11 ms), where building the resumed checkpoint's task/interrupt snapshot dominates.

Two consecutive suite runs on a quiet machine: invoke 5.42/5.35x, resume 7.35/7.09x,
state 12.49/10.39x, stream 5.11/5.20x, batch 8.87/8.98x, update_state 3.28/3.16x.### Round 36: where the async binding's time actually goes

Temporary counters inside the async node adapter (removed) on a 10-node trivial async graph:

- `ainvoke` total: 1.01 ms, i.e. 101 us per node call in the caller's view
- inside the node future (attach + Python call + output parse): **28 us** per call
- per-node work that the adapter itself measures: 28 us; the remaining ~73 us/node happens
  outside the adapter (async scheduler + Tokio/asyncio plumbing)

A fan-out graph (12 nodes, 3 supersteps) takes 0.77 ms versus 0.94 ms for the 10-node linear
graph, so the cost is not per superstep: it tracks the serialized node transitions, and the
checkpointer's `spawn_blocking` is not involved (no checkpointer in that graph).

For comparison, the same nodes through the sync binding cost ~3-8 us each. Closing the async gap
therefore needs a scheduler-level change (running synchronous node futures inline instead of
handing each to Tokio), not another adapter micro-optimisation -- and that is exactly the
experiment that broke async-node semantics when tried earlier (eager polling runs nodes on the
event-loop thread).### Round 37: async cost split -- fixed bridge, node work, and checkpoint saves

Temporary counters inside `AsyncCompiledGraph::run_task` (removed) plus targeted measurements on an
8-node trivial async graph:

| Component | Cost |
| --- | ---: |
| fixed cost per `ainvoke` (bridge + future setup), from the n=1 intercept | ~0.25 ms |
| marginal per node | ~61 us |
| inside `run_task` (setup 1.2 + future build 1.7 + await 24.9) | 28 us |
| engine work outside `run_task` | ~33 us/superstep |
| InMemorySaver vs no saver (+0.243 ms over 8 supersteps) | ~30 us/save |

The same nodes through the sync binding cost ~30 us each end to end. The async gap is therefore
made of three roughly equal parts: the asyncio<->Tokio wake-up per call, the per-superstep engine
path, and the per-superstep checkpoint save. `BlockingSaver` is *not* used for in-memory savers
(already special-cased in the async builder), so the save cost is the sync save path plus async
bookkeeping, not a thread-pool round trip.### Round 38: checkpoint saves are not the async bottleneck

Temporary counters in `save_checkpoint_via` (removed) on an 8-node async graph with an
`InMemorySaver`:

| Save component | Cost |
| --- | ---: |
| `persisted_values` | 2.2 us |
| `build_checkpoint` (id, versions, task results) | 13.1 us |
| `put_shared` | 1.4 us |
| **total** | **16.7 us/save** |

A full async invoke of the mega workflow performs 11 saves totalling **0.30 ms** out of a 3.8 ms
call, i.e. the saver is ~8% of the async cost, not the dominating term the earlier "saver vs no
saver" delta suggested (that delta also carried the extra bookkeeping a checkpointer enables:
checkpoint loads, pending-write handling and commit notifications).

The async gap is therefore dominated by things a per-call bridge cannot avoid: the ~0.25 ms
asyncio<->Tokio wake-up per call plus the serialized node transitions.### Round 39: the benchmark was measuring a debug build

Every cross-engine number recorded before this round was collected from a **debug** build of the
extension (`cargo build --features ...`), while the native column used a release binary. A debug
Rust engine is 4-5x slower: a 10-node linear chain takes 0.28-0.39 ms in the debug binding versus
**0.073 ms** in release, and the per-superstep engine cost (26 us in debug) matches that of a
release native graph within a few microseconds.

Rebuilt with `cargo build --release --features "python python-async sqlite redis postgres msgpack"`:

| Phase | sync binding | async binding |
| --- | ---: | ---: |
| invoke | 14.8-16.1x | 6.09x |
| resume | 26.0-28.4x | 3.32x |
| state | 42.3-43.3x | 20.4x |
| update_state | 10.0-12.1x | 6.94x |
| stream | 17.6-18.2x | 5.97x |
| batch | 30.1-32.5x | 8.74x |

Behaviour is unchanged by the build profile: 119 Python tests and parity 36/38 pass with the
release extension as well.

The one remaining sub-5x cell is async `resume`. Its floor is the bridge, not the graph: awaiting
an async binding call that fails instantly still costs **0.285 ms**, so a resume whose actual work
is 0.13 ms (measured through the sync binding) cannot beat 5x against LangGraph's 2.30 ms without
removing the asyncio<->Tokio hop for that call.### Round 40: the async bridge is one 0.165 ms wake-up, not 0.285 ms

Measuring `await` costs directly on a release extension:

| Await | Cost |
| --- | ---: |
| an `immediate_awaitable` call (`aget_state`) | 0.006 ms |
| a 1-node `ainvoke` (future_into_py) | 0.197 ms |
| the same through the sync binding | ~0.03 ms |

So the warm bridge is ~0.165 ms and Python's `await` machinery is negligible (0.006 ms). The earlier
0.285 ms figure came from an error path, which also builds a Python exception with a traceback.
Async `resume` (0.40-0.55 ms) is therefore ~0.165 ms bridge + ~0.12 ms of work that the same resume
takes through the sync binding, plus per-node async bookkeeping; against LangGraph's 2.3 ms it sits
at about 5x, and dips below that only when the machine is loaded (the fixed wake-up does not shrink
while LangGraph's absolute time grows).### Round 41: single-task supersteps run without `FuturesUnordered` in the async engine

`AsyncBackend::run_superstep` always allocated a `FuturesUnordered`, boxed every task future and
ran the drain loop, even when a superstep has exactly one task -- the shape of every linear graph
and of the tail after a fan-in. It now awaits that single future directly (still honouring
`RunControl` via `tokio::select!`), matching what the sync engine already did.

An 8-node trivial async graph runs in **0.294 ms** on a release build, i.e. ~12 us per node on top
of the ~0.165 ms bridge. Android-style suites: invoke 5.93x, state 17.92x, update_state 5.77x,
batch 6.91x, stream 4.87x, resume 2.60x against LangGraph async.### Round 42: async and sync execute exactly the same work

Temporary counters in the run loop (removed) counted supersteps and task executions for the mega
workflow:

| Run | supersteps | task executions |
| --- | ---: | ---: |
| sync invoke | 11 | 13 |
| async invoke | 11 | 13 |
| sync resume | 5 | 5 |
| async resume | 5 | 5 |

So the async binding is not doing extra graph work -- the entire remaining gap is the per-call
asyncio<->Tokio wake-up (0.165 ms) plus ~12 us per node of async bookkeeping. Any further async
speed-up would have to remove that wake-up by running nodes on the event-loop thread, which changes
concurrency semantics and needs an explicit decision.