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
| Benchmarks warm up once and interleave engines per iteration | `benchmarks/mega_workflow/bench.py`, `bench_async.py` | removes the ±20% drift between runs and whichever engine ran first |

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
concurrency semantics and needs an explicit decision.### Round 43: paused runs return `__interrupt__` like LangGraph

`invoke`, `resume`, `batch` and their async counterparts used to raise `GraphInterrupt` when a
node called `interrupt()`. LangGraph instead returns the current state with an `__interrupt__`
list, and a static `interrupt_before`/`interrupt_after` breakpoint returns the state without that
key. The binding now does the same (`graph_pause::pause_values`), building the payload from the
persisted snapshot:

```python
paused = graph.invoke({"messages": []}, config)   # {'...': ..., '__interrupt__': [Interrupt(...)]}
paused["__interrupt__"][0].value                  # the node's payload
graph.invoke(Command(resume="ok"), config)        # continues
```

A graph without a checkpointer still raises `GraphInterrupt`, because the paused state cannot be
read back; that edge case is documented in `docs/langgraph-compat.md`. 21 interrupt assertions in
the Python suite were rewritten to the new shape (124 tests pass), parity stays 36/38, and the
examples use the LangGraph form.### Round 44: the last two parity differences are gone

`command_goto`: LangGraph schedules `Command(goto=...)` targets **in addition to** the entry point,
so a fresh run executes both in the first superstep and the entry node's edges fire again
afterwards. `resolve_fresh_start` now prepends `entry_tasks` to the goto tasks. Verified against
LangGraph on the same graph: `{'seeded': 1, 'target_runs': 2}` on both engines (the reference
`target` runs twice -- once from the goto, once through `seed -> target`).

`messages_stream`: the LangChain callback handler only emitted messages on `on_llm_new_token`, so a
non-streaming `model.invoke("hi")` produced no `messages` chunks. `on_llm_end` now emits the
completed message when the run did not stream any tokens (tracked per `run_id`), which is what
LangGraph does. `model.stream(...)` still yields tokens without double-reporting.

The parity harness also extracted the wrong side of the v1 `(message, metadata)` tuple for the Rust
engine; it now reads `chunk[0]` like the LangGraph branch does.

**Result: parity 38/38 scenarios equivalent, 0 known differences.** 124 Python tests, the Rust
suite and clippy all pass. Remaining documented gaps are the ones that are not observed by parity:
`create_react_agent`'s narrower signature, no Python-side custom saver/store base classes, and
`IsLastStep`/`RemainingSteps` being helper functions rather than annotation classes.### Round 45: `create_react_agent` keywords and managed-value annotations

`create_react_agent` now accepts the LangGraph keywords that real code uses: `prompt` (a string
becomes a leading system message, a message object is prepended, a callable receives the state and
returns the complete model input -- verified against LangGraph, which also *replaces* the messages
for the callable form), plus `state_schema`, `store`, `interrupt_before`, `interrupt_after`,
`name` and `version` (`v1`/`v2`). `response_format`, `pre_model_hook`, `post_model_hook` and
`context_schema` raise `NotImplementedError` instead of being silently ignored.

`langgraph.managed.IsLastStep` / `RemainingSteps` are now annotation classes in the shim: a state
schema that declares them gets the engine's managed values injected per superstep (visible to
nodes, never persisted), and the annotation maps to the private `is_last_step` / `remaining_steps`
channels instead of a normal last-value channel. Verified against LangGraph with the same state
class: both report `remaining_steps=10006`, `is_last_step=False` on the first superstep, and both
omit them from the returned state.

Six new Python tests cover the prompt forms, the aligned keywords and the loud failures.### Round 46: `stream()` and `get_state_history()` speak the generator protocol

LangGraph returns generators from both calls, so `next(stream)` / `next(history)` are normal
usage. `CompiledGraph.stream` now returns a `StreamChunks` object and `StateHistory` gained a
`__next__` cursor: both support `for`, `iter()`, `next()` and stay lazy where they were lazy.
`StreamChunks` additionally keeps `len()` and `__getitem__` (the list-like API `langgraph_rs`
users already had), and its `__iter__` hands back a fresh iterator so existing double-iteration
code keeps working; LangGraph's generator is single-use instead. The Python suite (130 tests)
and parity (38/38) both pass. The only documented gap left is Python-side custom savers/stores.### Round 47: Python checkpointers work through the LangGraph saver protocol

`compile(checkpointer=<object>)` used to accept only the built-in savers. A foreign object with
`put` / `put_writes` / `get_tuple` / `list` is now wrapped in `PySaver`, which converts between the
engine's `Checkpoint` and LangGraph's `(config, checkpoint, metadata, parent_config)` tuples. The
engine-specific scheduling fields (`next`, deferred routes, resume tasks, goto sources, completed
tasks, task results, pending writes) travel inside `metadata["langgraph_rs"]`; a saver that drops
unknown metadata keys is rejected with an explicit error instead of silently losing state.

Verified end to end (3 new tests):

| Saver | invoke | resume | `get_state` | `get_state_history` |
| --- | --- | --- | --- | --- |
| custom dict-backed Python saver | ok | ok | ok | 6 rows |
| `langgraph.checkpoint.memory.InMemorySaver` | ok | ok | ok | 6 rows |
| saver that strips metadata | ok (write) | - | rejected with the `langgraph_rs` hint | - |

`checkpoint_ns` is always present in the config passed to savers (LangGraph reads it directly);
omitting it for the root namespace was why the reference `InMemorySaver` appeared to accept writes
but returned nothing. Python suite is now 133 tests; parity stays 38/38; Rust tests and clippy pass.
Remaining documented difference: Python-side custom **store** implementations.### Round 48: Python stores work through the LangGraph `BaseStore` protocol

`store_from_py` now wraps a foreign object with `get` / `put` / `search` in `PyStore`
(`src/adapters/python/custom_store.rs`), so `compile(store=<python object>)` works for custom
stores and for LangGraph's own `InMemoryStore`. Item payloads are read from whatever the store
returns: an object with a `value` attribute (LangGraph `Item`), a dict with a `value` key, or a
bare value; `search` forwards `query`/`limit` as keyword arguments and rebuilds the engine's
`StoreItem` (falling back to "now" for timestamps the store does not report).

Verified with two new tests: a dict-backed custom store and `langgraph.store.memory.InMemoryStore`
both round-trip `put`/`get` from inside a node via `get_store()`. Python suite is 135 tests;
parity stays 38/38; Rust tests and clippy pass.

This exposed one previously undocumented difference: `store.get()` returns the value (not an
`Item`) and `search` returns dicts rather than `Item`s, and `namespaces()` returns lists instead
of tuples. That is the last known difference and is now recorded in
docs/langgraph-compat.md section 4.### Round 49: stores return LangGraph `Item` objects

`store.get()` used to return the bare value and `search()` returned dicts. Both now return a
`StoreItem` pyclass shaped like LangGraph's `Item`: `.namespace`, `.key`, `.value`, `.created_at` /
`.updated_at` (ISO-8601), `.score`, plus the `*_millis` fields and `item["field"]` /
`item[2]` access for older code. `list_namespaces()` (tuple namespaces) is new; the existing
`namespaces()` (lists) stays. All five backends (memory, `Store`, SQLite, Redis, Postgres) share the
same helpers.

Store parity scenarios and two unit tests were updated to the LangGraph spelling
(`.value["value"]`, `store.get(...).value`); the older `item["name"]` lookup keeps working through
the value fallback.

**Result: parity 38/38, 135 Python tests, Rust suite and clippy all pass, and the compatibility
document no longer lists any behavioural difference from LangGraph** -- what remains is:
`stream()`/`get_state_history()` returning an iterable object that is a superset of a generator,
and four `create_react_agent` keywords that raise `NotImplementedError` instead of being ignored.### Round 50: `MessageGraph` and Python subclassing of `StateGraph`

`langgraph.graph.MessageGraph` was missing. It is now a thin Python subclass of the Rust
`StateGraph` with exactly one `messages` channel (`add_messages`), so its checkpoints stay
structurally identical to LangGraph's:

* nodes receive the message list (not the state mapping) and may return a single message, a list
  of messages, or `{}` for "no update";
* `invoke`/`ainvoke` accept a message list (LangGraph form) or `{"messages": [...]}` and return the
  resulting **list** of messages, rehydrated to `HumanMessage`/`AIMessage` objects when
  `langchain_core` is available;
* `stream(..., stream_mode="updates")` yields node-keyed chunks like LangGraph; the default mode
  stays `langgraph_rs`'s value chunks.

Verified against LangGraph with the same graph (node returning one message and a list of messages):
both produce `['HumanMessage:ping', 'AIMessage:pong', 'AIMessage:extra', 'AIMessage:single']`.

To support this, the `StateGraph` pyclass is now declared `subclass`, so user code can subclass the
builder (a LangGraph pattern). `MessageGraph`'s builder methods return `self` so chaining keeps the
subclass type. Six new tests cover invoke shape, node input shape, dict input, updates-mode
streaming, `isinstance(graph, StateGraph)` and interrupt pauses (which return the state mapping with
`__interrupt__` so both the messages and the interrupt stay readable).

Python suite is 141 tests; parity stays 38/38; Rust tests and clippy pass.### Round 51: `langgraph.errors` and `langgraph.warnings`

Both modules are now complete (a `dir()` diff against LangGraph leaves only stdlib leakage such as
`Any`/`Enum`). Added: `EmptyInputError`, `InvalidUpdateError`, `EmptyChannelError`,
`TaskNotFound`, `UnknownInterruptError` (with `reason`/`interrupt_ids`/`pending_ids`),
`NodeTimeoutError` (LangGraph's exact idle/run messages), `NodeError` (the non-exception wrapper),
`NodeCancelledError`, `GraphDrained`, `NodeInterrupt`, `ParentCommand`, `ErrorCode`,
`create_error_message`, `is_invalid_resume`, plus `langgraph_rs.warnings` with
`LangGraphDeprecationWarning` and the `LangGraphDeprecatedSinceV05/V10/V11` subclasses.

Two structural fixes came with it:

* `GraphBubbleUp` is now a real exception exported by the extension and `GraphInterrupt` derives
  from it, so `issubclass(GraphInterrupt, GraphBubbleUp)`, `NodeInterrupt(GraphInterrupt)` and
  `GraphDrained(GraphBubbleUp)` all match LangGraph (previously `GraphBubbleUp` was a placeholder
  that could not be subclassed).
* Reaching the recursion limit now raises `GraphRecursionError(RecursionError)` with LangGraph's
  wording instead of a plain `RuntimeError`:

  ```
  Recursion limit of 3 reached without hitting a stop condition. You can increase the limit by
  setting the `recursion_limit` config key.
  For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
  ```

  Verified with the same infinite conditional-edge graph in both engines (same type, same text).
  `Error::RecursionLimit` is mapped in `convert::to_pyerr`.

Ten new tests cover the hierarchy, both timeout messages, `NodeError` wrapping, unknown-interrupt
diagnostics, `ErrorCode` members, the deprecation-warning chain, the recursion error type/message,
`create_error_message` and `is_invalid_resume`. Python suite is 151 tests; parity stays 38/38; Rust
tests and clippy pass.### Round 52: `InjectedState` / `InjectedStore` tool arguments

`langgraph.prebuilt` now ships LangGraph's injection markers (subclassing
`langchain_core.tools.base.InjectedToolArg`, so `tool_call_schema` hides them from the model) and
the Rust tool path fills them at execution time:

* `ToolDefinition` handlers now receive the current state (`ToolDefinition::with_state`; the old
  `ToolDefinition::new` still works for state-agnostic tools).
* The Python bridge inspects each tool once (`args_schema.model_fields` metadata for `@tool`-style
  tools, `typing.get_type_hints(..., include_extras=True)` for plain callables) and builds the
  call kwargs with the injected `state` / `store` objects. LangChain tools are still called through
  `invoke(mapping)`, plain callables through `**kwargs`.
* `ToolRuntime` is exported as a marker but raises on use: its injection (config/context/stream
  writer) is not implemented yet, so failing loudly beats passing `None`.

Verified end to end: a `@tool` with `state: Annotated[dict, InjectedState]` returns `q|ada` for the
state `{"user": "ada"}` in both engines (LangGraph uses its own marker class in the comparison),
and an `InjectedStore` tool writes into a Python store attached via `compile(store=...)`. Six new
tests cover schema hiding, both injections, plain-callable keyword injection and the
LangGraph-vs-extension comparison.

Python suite is 156 tests; parity stays 38/38; Rust tests and clippy pass.### Round 53: `runtime`, `checkpoint.serde`, `cache.base` and `store.base`

`langgraph.runtime` now matches LangGraph: `Runtime` is a dataclass with LangGraph's eight fields
(plus the extension's `namespace`), `ExecutionInfo`/`ServerInfo` mirror theirs, and `get_runtime()`
builds a `Runtime` from the current run (context, store, stream writer, thread/namespace) and
returns `DEFAULT_RUNTIME` outside a graph -- previously it returned the raw native mapping or
`None`. `get_config`, `BaseStore`, `BaseUser`, `CONF` and `CONFIG_KEY_RUNTIME` are exported too.

New modules:

* `checkpoint/serde.py`: `JsonPlusSerializer` (JSON plus tagged `bytes`/`datetime`/`set`; tuples
  become lists, msgpack payload kinds are rejected, unserialisable objects raise `TypeError`),
  `SerializerProtocol`, `Serializable`.
* `cache/base.py`: `BaseCache` (ABC with `get`/`set`/`clear`), `FullKey`, `Namespace`, `ValueT`;
  `cache/memory.py` re-exports them like LangGraph does.
* `store/base.py`: `BaseStore` is a real ABC now (subclassing works) with `Item`/`SearchItem`
  mirrors; the previous placeholder docstring ("custom Python stores are not supported") was stale
  and is gone -- custom store objects still work through the protocol bridge.

A `dir()` diff against LangGraph is now empty for `runtime`, `cache.base`, `cache.memory`,
`checkpoint.serde`. `store.base` still lacks the embeddings/ops surface (`IndexConfig`,
`Embeddings`, `GetOp`/`PutOp`/`SearchOp`/`ListNamespacesOp`, `get_text_at_path`, `tokenize_path`,
`TTLConfig`, ...) -- that is the next target.

Eight new tests cover the Runtime field list, the outside-a-run default, the in-node values
(context/store/namespace), `ExecutionInfo` defaults, serializer round trips and rejections, and
`BaseCache`/`FullKey` subclassing. Python suite is 164 tests; parity stays 38/38.### Round 54: `langgraph.store.base` ops, embeddings and path helpers

`store.base` is now complete (an empty `dir()` diff against LangGraph). The two functions that
actually decide what an embedding-indexed store indexes were ported line by line from LangGraph and
verified differentially: `tokenize_path` and `get_text_at_path` produced identical output for 189
(path, object) combinations covering fields, nested fields, `[i]`/`[-1]`/`[*]`, `{a,b}` multi-field
selection, `*` wildcards, missing keys and non-dict values.

Also added: `override`, `validate_op_namespace` (rejects empty/dotted namespace labels with
`InvalidNamespaceError`), the batch op dataclasses (`GetOp`, `PutOp`, `SearchOp`,
`ListNamespacesOp`) with LangGraph's field names and defaults, `MatchCondition`, `TTLConfig`,
`IndexConfig`, `NotProvided`/`NOT_PROVIDED`, `NamespaceMatchType`/`NamespacePath`/`Result`, and
`ensure_embeddings` (wraps a sync or async embedding function into LangChain's `Embeddings`
interface; provider strings raise with an explanation because the provider package is not bundled,
matching LangGraph's requirement).

Thirteen new tests pin the tokenizer/extractor outputs, the op defaults, namespace validation, the
`override` marker and the embeddings wrapper. Python suite is 177 tests; parity stays 38/38.### Round 55: `langgraph.channels`

`langgraph.channels` now exists as a package with LangGraph's classes (`LastValue`,
`LastValueAfterFinish`, `AnyValue`, `EphemeralValue`, `UntrackedValue`, `Topic`,
`BinaryOperatorAggregate`, `DeltaChannel`, `NamedBarrierValue`, `NamedBarrierValueAfterFinish`) plus
the per-channel submodules LangGraph ships. Channel objects can be used directly as a state schema:

```python
graph = StateGraph({"total": BinaryOperatorAggregate(int, operator.add)})
```

Each class carries `__channel_spec__`, which `schema_from_py` maps onto the engine's channel kinds
(`last_value`, `any_value`, `ephemeral_value`, `untracked_value`, `topic[_accumulate]`, `delta`,
`last_value_after_finish`) or, for `BinaryOperatorAggregate`/`DeltaChannel`, onto a Python reducer.
Barrier channels report `__channel_spec__ = None` and raise `NotImplementedError` at graph-build
time -- approximating a barrier would silently change scheduling.

Verified: `LastValue`, `Topic(accumulate=True)`, `AnyValue` and `BinaryOperatorAggregate` graphs
produce LangGraph's results (barrier-free cases compared side by side). One nuance: LangGraph's
`BinaryOperatorAggregate` no longer seeds from the graph input in 1.2.22 (the channel-instance
annotation form is deprecated), while the extension applies the reducer to the input exactly like
LangGraph's `Annotated[..., operator.add]` form -- documented rather than emulated, because the
`Annotated` behaviour is the supported one.

Seven new tests cover the mappings, accumulation, plain channels, delta reduction and the barrier
rejection. Python suite is 184 tests; parity stays 38/38; Rust tests and clippy pass.### Round 56: `ValidationNode`, `ToolRuntime` injection, encrypted serializer and the remaining re-exports

* `langgraph.prebuilt.ValidationNode` implemented as a plain node: it validates the pending tool
  calls against the tool/pydantic/callable schema and answers each with a `ToolMessage` carrying the
  compact JSON arguments or the validation error. Compared side by side with LangGraph's own
  `ValidationNode`: identical content (`{"query":"x"}` for valid, the same pydantic error text for
  invalid) and the same "not a valid tool" message. `langgraph.prebuilt.tool_validator` (LangGraph's
  deprecated module path) is provided too.
* `ToolRuntime` is now a dataclass with LangGraph's fields and the extension injects it for real:
  `state`, `store`, `context` are filled at call time (verified: a tool reading
  `runtime.state["user"]` got `ada` while the store object was present). `tool_call_id`,
  `execution_info` and `server_info` stay `None` because the engine does not track them per call.
* `checkpoint.serde` became a package with LangGraph's submodules (`jsonplus`, `encrypted`, `base`)
  and `EncryptedSerializer` (Fernet via `cryptography`; a missing dependency raises with the pip
  hint). `pregel`, `pregel.{debug,main,protocol}` and `serde.{event_hooks,types}` are importable
  stubs for LangGraph-internal modules.
* Filled the re-export surface code actually imports: `langgraph.constants` (`CONF`,
  `CONFIG_KEY_CHECKPOINTER`, `TAG_HIDDEN`, `TAG_NOSTREAM`, `TASKS`), `langgraph.config` (`CONF`,
  `CONFIG_KEY_RUNTIME`, `var_child_runnable_config`, `StreamWriter`, `BaseStore`), `langgraph.func`
  (`EphemeralValue`, `LastValue`, `CACHE_NS_WRITES`, `PREVIOUS`, `MISSING`), `checkpoint.base`
  helpers (`empty_checkpoint`, `create_checkpoint`, `copy_checkpoint`, `writes_sort_key`,
  `get_checkpoint_id`, `get_checkpoint_metadata`, plus `WRITES_IDX_MAP`, `EXCLUDED_METADATA_KEYS`,
  `LATEST_VERSION`) and `checkpoint.memory.MemorySaver`.

Six new tests cover the validation node (valid/invalid/unknown tool plus the LangGraph comparison),
the `ToolRuntime` injection and the encrypted round trip. Python suite is 190 tests; parity stays
38/38; Rust tests and clippy pass. What is left is the typing-alias tail in `langgraph.types`
(`StreamPart` variants, `CheckpointPayload`, `StateUpdate`, ...) and a few internal helpers
(`PregelNode`, `SyncAsyncFuture`); `ToolCallTransformer` remains unimplemented and raises.### Round 57: `langgraph.types` typing surface and message helpers

The typing tail is filled in: the seven `StreamPart` variants (with LangGraph's exact key sets),
`CheckpointPayload`, `TaskPayload`, `TaskResultPayload`, `CheckpointTask`, `DebugPayload`,
`CheckpointMetadata`, `StateUpdate`, `Overwrite`, `GraphOutput`, `TracePolicy`, `CacheKey`,
`Durability`, `PregelExecutableTask`, `NS_SEP`, `AnyMessage`, plus the helper functions
`xxh3_128_hexdigest` (same XXH3-128 digest via the `xxhash` package -- verified identical output),
`default_cache_key`, `default_retry_on`, `ensure_valid_checkpointer`, `get_cached_annotated_keys`,
`get_update_as_tuples` and `omit_payload` (read from LangGraph's source: it always returns `{}`,
which the first implementation got wrong -- fixed and pinned by a test).

Also added: `prebuilt` agent state schemas (`AgentState`, `AgentStatePydantic`,
`AgentStateWithStructuredResponse`, `AgentStateWithStructuredResponsePydantic`), the
`graph.state`/`graph.message` re-exports (`All`, `CachePolicy`, `BaseChannel`,
`BinaryOperatorAggregate`, `BranchSpec`, `ChannelRead`, `CONF`, `CONFIG_KEY_SEND`, `Messages`,
`REMOVE_ALL_MESSAGES`, `push_message`), `func` deprecation warnings,
`checkpoint.memory.PersistentDict`/`ChannelVersions`/`DeltaChannelHistory`,
`checkpoint.base.maybe_add_typed_methods` and the `langgraph.managed.base` submodule (which turned
`managed` into a package).

A 27-module audit now leaves only three categories: names that LangGraph's modules leak from
`typing`/`langchain_core` (`BaseModel`, `BaseTool`, `Awaitable`, ...), LangGraph-internal classes
(`PregelNode`, `ChannelWrite`, `SyncAsyncFuture`, `AsyncToolCallWrapper`) and the single feature we
deliberately do not implement (`ToolCallTransformer`, which raises). Ten new tests cover the stream
parts, payload key sets, the digest, the cache key, agent state schemas and the message helpers
(including a direct comparison with LangGraph). Python suite is 200 tests; parity stays 38/38; Rust
tests and clippy pass.### Round 58: `create_tool`, tool-call wrappers and the last public helpers

`langgraph.prebuilt` gained LangGraph's tool-construction surface: `create_tool` (callable, both
decorator forms, and a named Runnable), `ToolCallRequest`, `ToolCallWithContext`, `ToolCallWrapper`,
`ToolInvocationError`, `msg_content_output`, `get_all_basemodel_annotations`, the four error-message
templates and `TOOL_MESSAGE_BLOCK_TYPES`, plus `create_tool_calling_executor` (an alias of
`create_react_agent`, as in LangGraph) and the `StructuredResponse`/`Prompt`/`LanguageModelLike`
aliases. `create_tool(add)` was compared with LangGraph's own: identical name, description, argument
schema and result.

Also added: `store.base.embed` as a real importable namespace (both `from ... import embed` and
`import langgraph_rs.store.base.embed` work), `func.coerce_timeout_policy`/`is_async_callable`/
`identifier`/`sync_timeout_unsupported`, and the `ToolCallWithContext`/`Runtime`/`ErrorCode`/
`create_error_message` re-exports in `prebuilt.chat_agent_executor`.

Nine new tests cover the three `create_tool` forms, the LangGraph comparison, the message converter,
the templates, the wrapper objects and the executor alias. Python suite is 209 tests; parity stays
38/38; Rust tests and clippy pass.

The only remaining gaps are LangGraph-internal symbols (`PregelNode`, `ChannelWrite`,
`SyncAsyncFuture`, `RunnableCallable`, the msgpack extension constants in `serde.jsonplus`) and
`ToolCallTransformer`, which raises on purpose because it needs LangGraph's streaming middleware.### Round 59: shared-saver cross-engine resume

Resuming each other's paused runs now works with a **single Python saver object** shared by both
engines (LangGraph's `InMemorySaver`), not only through the LangGraph-schema SQLite saver.

Two changes were needed in `src/adapters/python/custom_saver.rs`:

* **reading**: when a checkpoint has no `metadata["langgraph_rs"]` sidecar (LangGraph wrote it), the
  bridge rebuilds `next` / `interrupts` from LangGraph's payload -- `branch:to:<node>` channels and
  the `__interrupt__` pending writes -- instead of raising.
* **writing**: `put` now publishes a LangGraph-shaped checkpoint (`v: 4`, `__root__`,
  `branch:to:<node>`, `__pregel_tasks`, `updated_channels`) plus an `__interrupt__` write keyed by
  the task id LangGraph derives (`langgraph_task_id` / `langgraph_interrupt_entries`, the same
  helpers the SQLite saver uses). Pending nodes are dropped from `versions_seen`; without that
  LangGraph considered the node already executed and returned the unchanged state.

Verified matrix (two-node graph pausing on `interrupt()`):

| Sharing | LangGraph pause -> rs resume | rs pause -> LangGraph resume | rs round trip |
| --- | --- | --- | --- |
| one `InMemorySaver` object | `{'attempts': 1, 'log': ['rs-ask:yes', 'rs-done']}` | `{'attempts': 1, 'log': ['lg-ask:yes', 'lg-done']}` | ok |
| LangGraph-schema SQLite | ok | ok | ok |

`python/tests/test_cross_engine_resume.py` pins all four cases. The previous "saver drops the
sidecar" test now asserts the fallback path (state + resume still work) instead of an error.
Python suite is 213 tests; parity stays 38/38; Rust tests and clippy pass.### Round 60: mixed-writer threads and monotonic checkpoint ids

Two follow-ups after the shared-saver work:

* **Writing**: the Python-saver bridge now emits UUIDv6-style checkpoint ids derived from
  `created_at_millis` plus the step (`ticks = millis * 10_000 + step`), so ids from both engines sort
  by wall clock. Without this, a thread continued by both engines resumed from LangGraph's older
  checkpoint (`{'attempts': 1, ...}` instead of `3`); with it, alternating turns work:
  `LG -> rs -> LG` produced `{'attempts': 3, 'log': ['lg-step', 'rs-input', 'rs-step', 'lg-input',
  'lg-step']}`.
* **Reading**: dashed ids are normalised before the task-id nibble grouping (`id_bytes` now strips
  non-hex characters), matching LangGraph's derivation.

The LangGraph-schema SQLite saver keeps its previous flat id format on purpose: three existing tests
(LangGraph resuming our parallel / sequential / `Send` interrupts from a SQLite file) depend on it,
and switching it to UUIDv6 made LangGraph fail to map its resume values back to our writes. Mixed
sqlite writers with different id formats therefore remain a known limitation (documented in
`docs/langgraph-compat.md`); the shared-saver path is fully bidirectional.

Python suite is 213 tests; parity 38/38; Rust tests and clippy pass.### Round 61: mixed-writer threads on one SQLite file

The remaining cross-engine limitation (two engines writing the same thread into one LangGraph-schema
SQLite database) is fixed. Two independent problems were behind it:

1. **Ordering.** Both engines pick the latest checkpoint by comparing id strings; LangGraph emits
   UUIDv6 (`1f1e…-…`) while our saver emitted a flat hex id, so `ORDER BY checkpoint_id DESC`
   returned a stale row and a continuation silently re-ran from an older superstep
   (`attempts: 3` instead of `4`).
2. **Dashes.** Switching to a dashed UUIDv6 contradicted LangGraph: it calls
   `binascii.unhexlify(checkpoint["id"])` in places, so a dashed id made it fail to derive task ids
   and drop the pending interrupts (`snapshot.tasks[i].interrupts == []`, resume map empty). Three
   tests (`test_langgraph_resumes_{parallel,sequential}_interrupts`, `..._send_fanout`) caught it.

Final scheme: UUIDv6 layout, **no dashes** (`{time_high}{time_mid}6{time_low}{clock}{node}`, 32 hex),
ticks = `millis * 10_000 + step` so ids are strictly increasing per thread. Reads normalise with
`ORDER BY REPLACE(checkpoint_id, '-', '') DESC`.

Verified:

| Scenario | Result |
| --- | --- |
| LG → rs → LG → rs → LG on one SQLite file | `attempts` 1..5, log accumulates in order |
| rs `Send` fan-out pause → LG `Command(resume={id: value})` | `{'seen': [1, 10, 2, 11], 'item': 0}`, both tasks resumed with their own value |
| shared `InMemorySaver`, both directions | still green |
| Python suite / parity / Rust tests / clippy | 213 tests, 38/38, exit 0, clean |

Open (not addressed here): `add_subgraph` + `interrupt()` — a subgraph pause is written in the root
namespace (`checkpoint_ns == ""`) and a parent `resume(thread, value)` returns the paused state
instead of continuing, so the documented "subgraph interrupt pauses the parent run" behaviour does
not hold yet. Tracked for the next round.### Round 62: subgraph interrupts (partial)

Reported behaviour "an `interrupt()` inside a subgraph pauses the parent run and a parent resume
continues the subgraph" did not hold. Three concrete defects were found and two fixed:

1. **Fixed — history hid child namespaces.** `LangGraphSqliteSaver::history(thread)` only returned
   rows with `checkpoint_ns = ''`, so a paused child (`sub:<task id>`) was invisible. It now returns
   every namespace of the thread (like LangGraph's own `list()`).
2. **Fixed — resume looked up an exact namespace.** `run_subgraph` asked
   `has_pending_in(thread, namespace)` while the child namespace carries a task id that changes on
   every parent resume. It now looks the paused child up by prefix
   (`CompiledGraph::paused_namespace_in`), resuming in the namespace where it actually paused.
3. **Open — the resume value is not propagated.** `run_subgraph` takes the value from
   `runtime::context::current_resume_last()`, which is only set while `interrupt()` re-executes; at
   the start of a resumed parent node it is `None`, so the child is restarted from its input and
   pauses again (`resume()` returns the paused state). Fixing this means threading the parent resume
   value into the context before the child runs.

Also required (documented, not a bug): a subgraph that can pause must be compiled with its own
checkpointer — the parent's saver is not inherited, unlike LangGraph.

Verified: Python 213 tests, parity 38/38, Rust tests, clippy clean; the subgraph scenario above
still fails at step 3 and is tracked for the next round.### Round 63: subgraph interrupts now resume (sync and async)

Round 62 left one defect: a parent `resume()` restarted the paused child from its input instead of
continuing it. Three fixes, all verified:

1. **Prefix lookup used the full namespace.** `run_subgraph` computed `namespace = sub:<task id>` and
   then searched for a paused child under *that* string, which can never match on a later resume.
   The lookup now strips the task-id suffix (`namespace.rsplit_once(':')`) and matches
   `sub` / `sub:*`.
2. **Resume value was invisible before the first `interrupt()`.** `run_subgraph` read
   `current_resume_last()`, which is only populated once `interrupt()` consumes a value; the paused
   task is the *parent node*, so it was still `None`. `TaskContext` now carries the task's own
   `pending_resume`, exposed as `current_pending_resume()`, and the subgraph uses it as fallback.
3. **Same two defects in the async runtime.** `run_async_subgraph` had the identical exact-namespace
   gate; it now shares the prefix lookup (`AsyncCompiledGraph::paused_namespace_in`) and resumes in
   the namespace where the child actually paused.

Verified:

| Scenario | Result |
| --- | --- |
| sync parent + subgraph interrupt, parent `resume(True)` | `{'approved': True, 'draft': 'v1 [done]'}` |
| async parent (`AsyncStateGraph`) + async subgraph | same |
| Python suite / parity / Rust / clippy | 213 tests, 38/38, exit 0, clean |

Documented constraint (unchanged, and different from LangGraph): a subgraph that can pause must be
compiled with its own checkpointer; the parent saver is not inherited.### Round 64: cross-engine subgraph resume, and the remaining feature demos

* **Cross-engine subgraph resume verified**: LangGraph paused *inside* its own subgraph
  (`checkpoint_ns = "sub:98158705-…"`); the extension read the paused state
  (`{'approved': False, 'draft': 'v1'}`) and `resume(True)` completed the run
  (`{'approved': True, 'draft': 'v1 [done]'}`). The prefix lookup added in round 62 is what makes
  this work — LangGraph's task id in the namespace differs from ours.
* **Examples completed to 8/10**: `routing` (conditional edges + `Command(goto)`) and
  `checkpointing` (three runs, `update_state` fork, `get_state_history`).
* **API difference found while writing the checkpointing demo**: the native `invoke(&state)` takes a
  *complete* state, while the Python binding (like LangGraph) treats the input dict as partial
  writes that go through the channel reducers. The demo therefore reads `get_state` and merges
  before invoking; the difference is documented in `examples/checkpointing/README.md`.

| Feature demo | LangGraph | binding | native | binding vs LG | native vs binding |
| --- | --- | --- | --- | ---: | ---: |
| routing | 882 µs | 32.9 µs | 19.2 µs | 26.8x | 1.71x |
| checkpointing | 11380 µs | 171.7 µs | 143.3 µs | 66.3x | 1.20x |### Round 65: `create_react_agent` hooks, structured output and context schema

The four previously unsupported keywords now work, verified side by side with LangGraph using a fake
chat model (`python/tests/test_react_agent_extras.py`, 4 tests):

| Keyword | Implementation | Parity check |
| --- | --- | --- |
| `pre_model_hook` | node inserted before `agent`; `tools` routes back to the **entry point**, which is the hook when present (LangGraph does `add_edge("tools", entrypoint)`) | hook call order `['pre','post','pre','post']` matches LangGraph exactly |
| `post_model_hook` | node between `agent` and the `tools`/`END` decision | same |
| `response_format` | `model.with_structured_output(...)` called once at build time; when the model answer has no tool calls, the parsed value is written to `structured_response` | both engines end with a `structured_response` key |
| `context_schema` | accepted as a JSON-schema dict **or** a pydantic/TypedDict class (via `pydantic.TypeAdapter(...).json_schema()`), passed to `CompileOptions::with_context_schema` | both accept the same `TypedDict` |

Also fixed while testing: `StateGraph.compile(context_schema=...)` had the same "class is not JSON"
problem and now uses the shared `context_schema_value` helper.

Python suite is 217 tests; parity 38/38; clippy and Rust tests clean.### Round 66: store search surface parity

`BaseStore` grew the parts LangGraph has and the extension lacked:

* `SearchOptions` (`query` / `filter` / `limit` / `offset`) and `search_with`, whose default
  implementation applies LangGraph's order: text search → drop expired → filter → `[offset:offset+limit]`.
* `NamespaceOptions` and `list_namespaces` (prefix / suffix / `max_depth` with dedupe / pagination).
* `put_with_ttl` (minutes) plus `StoreItem::expires_at_millis`; the in-memory store hides expired
  items from `get` and `search`. (LangGraph's `InMemoryStore` itself raises for `ttl=`, so this is a
  superset; its SQLite/Redis/Postgres stores do support TTL.)
* Filter matching ports LangGraph's `_compare_values` / `_apply_operator`
  (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, nested objects, lists).

All four Python store classes (`Store`, `InMemoryStore`, `SqliteStore`, `RedisStore`,
`PostgresStore`) now accept `put(..., index=..., ttl=...)`,
`search(..., filter=, limit=10, offset=0, refresh_ttl=)` and
`list_namespaces(prefix, suffix, max_depth, limit, offset)`.

`python/tests/test_store_search.py` runs every case on both engines and compares: filters,
operators, nested filters, pagination, namespace filters and TTL are identical to LangGraph.

Python suite is 224 tests; parity 38/38; clippy and Rust tests clean.### Round 67: vector index for the store

`InMemoryStore` now implements LangGraph's vector search:

* `Embedder` / `IndexConfig` (`dims`, `embed`, `fields`) in `store.rs`, plus
  `indexed_texts` (dotted paths, `[n]` indices, `$` for the whole value) and `cosine_similarity`.
* `put_indexed` embeds the configured fields (or the whole value when `fields` is unset) and stores
  one vector per field path; `put(index=False)` stores the value without a vector,
  `put(index=["path"])` indexes exactly those paths.
* The in-memory store overrides `search_with`: candidates are filtered, every (item, vector) pair is
  scored, results are deduped by `(namespace, key)` keeping the best score, then `offset`/`limit`
  apply, and items without vectors are appended unscored to fill the remaining slots — the same
  order as LangGraph's `_batch_search`.
* Python: `InMemoryStore(index={"dims": n, "embed": embeddings_or_callable, "fields": [...]})`
  accepts a LangChain `Embeddings` object (uses `embed_documents` / `embed_query`) or a plain
  callable (batch form, with a per-text fallback).

Parity (`python/tests/test_store_index.py`, 8 tests): ranking, scores (6 decimals), pagination,
scoreless tail, `index=` overrides, `embed_query` usage and `filter` interaction all match LangGraph.

Known differences, both documented: field paths do not support `*` wildcards, and embeddings run per
`put` instead of being batched across a batch of ops.

Python suite is 232 tests; parity 38/38; clippy and Rust tests clean.### Round 68: subgraphs inherit the parent checkpointer

LangGraph passes its saver through the run config, so `add_node("sub", subgraph_without_saver)` still
checkpoints the child. The extension required an explicit saver on every pausable subgraph; that
difference is gone.

Implementation:

* `CompiledGraph` / `AsyncCompiledGraph` carry `inherited_saver: Arc<OnceLock<Arc<dyn Checkpointer>>>`
  and `subgraph_hooks`; `checkpointer()` prefers the graph's own saver and falls back to the
  inherited one, so `has_checkpointer`, `paused_namespace_in`, load/persist and the async saver
  dispatch all work unchanged.
* `add_subgraph` / `add_subgraph_mapped` register a hook; `compile_with` hands its saver to every
  child, and `set_inherited_saver` recurses so grandchildren inherit too.
* The async runtime had two places reading the raw `checkpointer` field (`persist_checkpoint`,
  the durability writer); both now use the accessor — without that the child's pause was never
  written, which is why async resume used to re-pause.
* `paused_namespace_in` now prefers the **shallowest** matching namespace: a nested child
  (`sub:<task>|inner:<task>`) also starts with `sub:`, and picking it resumed a graph in the wrong
  namespace (`node review is not registered in this graph`).

Verified by `python/tests/test_subgraph_inheritance.py` (4 tests): single-level inheritance,
two-level nesting, an explicitly saved child keeping its own saver, and a LangGraph parity case.

Python suite is 236 tests; parity 38/38; clippy and Rust tests clean.### Round 69: public type aliases and module surfaces

An API diff over `langgraph.types`, `langgraph.func` and `langgraph.checkpoint.serde.jsonplus`
showed the remaining gaps were mostly *surface* rather than behaviour. Fixed:

* `StreamMode` and `All` are now the same `Literal` aliases LangGraph uses (they were placeholder
  classes), and `StreamWriter` is `collections.abc.Callable[[Any], None]` — all three compare equal to
  LangGraph's objects (see `python/tests/test_type_aliases.py`).
* `DeprecatedKwargs` (empty TypedDict) and the `MISSING` sentinel now exist in `langgraph_rs.types`.
* `langgraph.func` now covers LangGraph's public surface completely (diff = 0 names): real
  re-exports (`BaseCache`, `BaseCheckpointSaver`, `BaseStore`, `CachePolicy`, `RetryPolicy`,
  `TimeoutPolicy`, `StreamMode`), typing helpers (`Any`, `Awaitable`, `Callable`, `Generic`,
  `Sequence`, `TypeVar`, `Unpack`, `cast`, `dataclass`, `functools`, `get_args`, `get_origin`,
  `inspect`, `overload`, `timedelta`, `warnings`), type variables (`ContextT`, `P`, `R`, `S`, `T`)
  and a working `get_runnable_for_entrypoint` (returns an `invoke`/`ainvoke` wrapper).
* `langgraph.middleware` exists as an empty module, like LangGraph 1.2's.
* `jsonplus` exports the eight messagepack `EXT_*` extension ids with LangGraph's values.
* `Pregel`, `PregelNode`, `ChannelWrite`, `ChannelWriteEntry`, `SyncAsyncFuture` stay importable
  placeholders that raise with the "use the compiled graph" hint.

Remaining subsystem gap, measured rather than guessed: `langgraph.stream` (19 public symbols:
`GraphRunStream`, `AsyncGraphRunStream`, `StreamChannel`, `ProtocolEvent`, `StreamTransformer` and
the built-in transformers) plus `compile(transformers=...)`. `ToolCallTransformer` is one member of
it. That is a multi-round feature, not a shim fix; the extension's own seven `stream_mode`s are
unaffected.

Python suite is 247 tests; parity 38/38; clippy clean.### Round 70: `langgraph.stream` primitives

The stream subsystem is being ported in steps; this round lands the primitives, which are the part
user code touches directly:

* `langgraph_rs.stream.StreamChannel` — single-consumer drainable queue with lazy subscription,
  sync/async binding (`_bind(is_async=...)`), `push` / `close` / `fail`, `tee(n)` / `atee(n)`
  fan-out, and the same error contract as LangGraph (unbound → `TypeError`, wrong lane → `TypeError`,
  second subscriber → `RuntimeError`, closed channel push → `RuntimeError`).
* `ProtocolEvent` / `StreamTransformer` / `transformer_requires_async` — the envelope, the
  extension-point ABC (`init`, `process`/`aprocess`, `finalize`/`afinalize`, `fail`/`afail`,
  `schedule`, `_native`, `required_stream_modes`, `before_builtins`) and the async-lane detection.
* The package layout now matches LangGraph (`stream/run_stream.py`, `stream/transformers.py`), so
  `dir(langgraph_rs.stream)` covers `dir(langgraph.stream)`; the run streams and built-in
  transformers raise with a pointer to `stream(stream_mode=...)`.

`python/tests/test_stream_primitives.py` (16 tests) covers the contract and compares behaviour with
LangGraph on the same inputs. Python suite is 263 tests; parity 38/38.

Next steps (in order): `_mux.py` (StreamMux + registration), `transformers.py` (built-ins),
`run_stream.py` bound to the engine's chunk stream, then `compile(transformers=...)` and
`ToolCallTransformer`.### Round 71: `StreamMux` event dispatcher

Second step of the `langgraph.stream` port: the dispatcher the public transformers run on.

* `langgraph_rs.stream._mux.StreamMux` — registration (`factories=` vs pre-built `transformers=`),
  `before_builtins` ordering, projection merging with owner attribution and conflict `ValueError`,
  `native_keys`, `transformer_by_key`, sync `push` / async `apush`, `close`/`fail` and
  `aclose`/`afail`, `bind_pump`/`bind_apump` propagation, `_make_child` mini-muxes (factories only,
  no `seq` assignment) and named-channel auto-forwarding (`custom:` prefix for user channels, plain
  name for `_native` ones).
* `langgraph_rs.stream._convert.convert_to_protocol_event` — v2 stream part → `ProtocolEvent`.

`python/tests/test_stream_mux.py` (14 tests) covers the contract and compares dispatch output
(`seq`, method, data) and channel forwarding against LangGraph's mux on the same inputs.

Python suite is 277 tests; parity 38/38.

Next: the built-in transformers (`ValuesTransformer`, `UpdatesTransformer`, `CustomTransformer`,
`CheckpointsTransformer`, `DebugTransformer`, `TasksTransformer`, `LifecycleTransformer`), then
`run_stream` + `stream_events(version="v3")`, then `MessagesTransformer` / `ToolCallTransformer`.### Round 72: built-in stream transformers (capture family)

Six of LangGraph's built-in transformers are now implemented, each mirroring its upstream
behaviour: `ValuesTransformer` (channel plus `_latest` / `_interrupted` / `_interrupts` / `error`),
`CustomTransformer`, `UpdatesTransformer`, `CheckpointsTransformer`, `DebugTransformer` and
`TasksTransformer`.

All are `_native = True`, declare their `required_stream_modes`, capture only their own protocol
method, and ignore events from deeper namespaces unless constructed with a matching `scope`.
`LifecyclePayload` and `SubgraphStatus` are available as payload types.

The package surface now matches LangGraph exactly: `ValuesTransformer` and `MessagesTransformer` are
reachable through `langgraph_rs.stream.transformers` but not re-exported from the package, as
upstream.

`python/tests/test_stream_transformers.py` (11 tests) covers per-method capture, namespace scoping,
ordering, values/interrupt tracking, projection merging, and compares capture results with
LangGraph for all six transformers.

Still pending in this subsystem: `MessagesTransformer` (needs `ChatModelStream`),
`LifecycleTransformer` / `SubgraphTransformer` (need subgraph run handles), `run_stream` and
`stream_events(version="v3")`, then `ToolCallTransformer`.

Python suite is 288 tests; parity 38/38.### Round 73: `stream_events(version="v3")` run streams

The caller-driven run streams are now real, built in Python on top of the extension's own `stream()`:

* `GraphRunStream` / `AsyncGraphRunStream` — pull-based pumping (`_pump_next` / `_apump_next` with
  the take-a-number condition on the async lane), `output` / `interrupted` / `interrupts`,
  `extensions`, iteration over the raw protocol log, `interleave`, `abort`, context managers, and
  LangGraph's "registered projections: ..." `AttributeError` for a missing projection.
* `CompiledGraph.stream_events(...)` / `AsyncCompiledGraph.astream_events(...)` delegate to
  `langgraph_rs.stream._events`, which builds a `StreamMux` from `ValuesTransformer` plus any
  call-site `transformers=`, requests exactly their union of stream modes (`version="v2"` parts),
  and converts each part with `convert_to_protocol_event`.
* `run.interrupts` normalizes the v2 `(value, id)` tuples into `langgraph_rs.Interrupt` objects.
* Like upstream, only `values` is registered by default; `updates`/`custom`/`checkpoints`/`debug`/
  `tasks` are opt-in through `transformers=` (and raise `AttributeError` otherwise).

`python/tests/test_stream_events.py` (15 tests) covers every projection, default-vs-opt-in
registration, output/interrupt reporting, the raw protocol log and its `seq` ordering, rejection of
`version="v2"` and `stream_mode=`, custom transformers, and `astream_events`; the parity cases
compare `values`/`updates`/`custom` and `output` with LangGraph.

Python suite is 303 tests; parity 38/38; clippy clean.

Remaining in this subsystem: `MessagesTransformer` (needs `ChatModelStream`), `LifecycleTransformer`
/ `SubgraphTransformer` (subgraph handles), `compile(transformers=...)`, and `ToolCallTransformer`.### Round 74: compile-time transformers and `ToolCallTransformer`

* `compile(transformers=[...])` / async `compile(transformers=[...])` are wired end to end: the
  compiled graph stores the list (`stream_transformers`) and `stream_events` / `astream_events` merge
  it as built-ins -> compile-time -> call-site, which is the order LangGraph documents.
* `ToolCallStream` (`prebuilt/_tool_call_stream.py`) — the per-call handle: `tool_call_id`,
  `tool_name`, `input`, `output_deltas` (a `StreamChannel`), `output`, `error`, `completed`,
  `_bind`/`_push_delta`/`_finish`/`_fail`, sync and async iteration.
* `ToolCallTransformer` (`prebuilt/_tool_call_transformer.py`) — native transformer on the `tools`
  method: `tool-started` spawns a handle on the `tool_calls` projection, deltas append,
  `tool-finished` / `tool-error` close it (unwrapping a `ToolMessage` or its serialized form),
  `finalize` closes stragglers, `fail` marks them failed, and events pass through untouched.
  `langgraph_rs.prebuilt.ToolCallTransformer` now exposes it instead of raising.

`python/tests/test_tool_call_transformer.py` (11 tests) covers the handle lifecycle, deltas,
error/finalize/fail paths, namespace filtering, serialized `ToolMessage` unwrapping, compile-time and
call-site transformer merging, plus a parity case that runs the same event stream through LangGraph's
transformer and compares every handle field and the delta logs.

One caveat, documented: the extension's engine does not emit `tools` protocol events yet, so
`run.tool_calls` stays empty on a real run until tool-call streaming lands; the transformer itself is
complete and verified against LangGraph's.

Python suite is 314 tests; parity 38/38; clippy and Rust tests clean.### Round 75: `stream_mode="tools"` emitted by the engine

`ToolCallTransformer` was complete but starved: nothing produced `tools` protocol events. The engine
now has a first-class `tools` stream mode.

* `StreamChunk::Tool { node, value }`, `StreamMode::Tools` (`"tools"` in `parse`/`as_str`, reported by
  `StreamChunk::mode()` and `node()`).
* `TaskContext` gained a `tools` buffer, `with_task` returns it, `TaskResult` carries it, and the
  outcome loop emits one `StreamChunk::Tool` per buffered event — the same path `custom` values take.
  The async runtime records them through `AsyncContext::tool_event`.
* `push_tool_event` (`runtime::context`) is the in-node emit helper, mirroring `push_custom`.
* Rust `ToolNode::execute` reports `tool-started` (with `tool_call_id`, `tool_name`, `input`) before
  running a tool and `tool-finished` (`output`) or `tool-error` (`message`) afterwards, matching
  LangGraph's `StreamToolCallHandler` payloads.
* The Python binding converts the chunks to `{"type": "tools", ...}` stream parts.

`python/tests/test_tools_stream.py` (5 tests): `stream(stream_mode="tools")` part sequence,
`ToolCallTransformer` producing a completed handle with the tool output, a failing tool surfacing as
an errored handle, `messages` staying unaffected, and the async graph path.

Python suite is 319 tests; parity 38/38; clippy and Rust tests clean.### Round 76: subgraph handles, lifecycle events, and a nested-namespace fix

* `SubgraphRunStream` / `AsyncSubgraphRunStream` are real: a handle wraps a child mini-mux whose pump
  is the parent's, exposes `path` / `graph_name` / `trigger_call_id` / `status` / `error`, and its
  projections drive the whole run. Nested subgraphs appear on the child handle, like upstream.
* `_TasksLifecycleBase` + `LifecycleTransformer` + `SubgraphTransformer` port LangGraph's
  tasks-event bookkeeping: a subgraph opens on the first `tasks` event in its namespace and closes on
  the parent task result (`completed` / `interrupted` / `failed` / `drained`), with `finalize` / `fail`
  sweeps for still-open namespaces. `tasks` events are folded into these projections instead of the
  raw log.
* `stream_events` now registers `values` + `lifecycle` + `subgraphs` by default (matching upstream)
  and requests `subgraphs=True` so child namespaces are emitted. The stream is created lazily so a
  graph that fails on its first step reports through the run stream instead of raising before the
  stream exists.
* **Engine bug fixed**: nested subgraph chunks duplicated their parent namespace
  (`a:1|a:1|b:2`) because the child chunk already carried an absolute namespace and the wrapper
  prepended the parent again. `graph::subgraph_chunk_namespace` only prepends when the child
  namespace is not already anchored at the parent's, so a grandchild now reports `('a:1', 'b:2')`
  exactly like LangGraph.

`python/tests/test_subgraph_stream.py` (8 tests) covers handle discovery, status transitions,
grandchild nesting, the lifecycle sequence, failure surfacing, the one-segment-per-level namespace
shape, and compares lifecycle sequences and child graph names with LangGraph.

Python suite is 327 tests; parity 38/38; clippy and Rust tests clean.### Round 77: `MessagesTransformer` — the stream subsystem is complete

`MessagesTransformer` (the last missing piece of `langgraph.stream`) is implemented on top of
langchain-core's own `ChatModelStream`:

* v3 content-block protocol events (`message-start` / `content-block-*` / `message-finish`) are routed
  to a handle by `metadata["run_id"]`; tool-role messages are ignored, exactly like upstream.
* Whole `BaseMessage` payloads are replayed through `message_to_events` into a completed handle.
* Extension bridge: the engine serializes messages to plain dicts, so message dicts take the replay
  path and `AIMessageChunk` dicts are accumulated per call and flushed as one completed handle when
  the run ends (the engine has no token-level v3 events yet). Handles created this way carry the
  joined text and a synthesized `AIMessage` id, which is what `run.messages` exposes today.
* `_bind_pump` / `_bind_apump` wire the run pump into each handle on both lanes.

`python/tests/test_messages_transformer.py` (9 tests) covers the protocol lifecycle, tool-role
filtering, whole messages, per-call chunk accumulation, separate calls, namespace scoping, a real
graph emitting chunks through `message()`, and two parity cases against LangGraph (protocol events
and whole messages).

`dir(langgraph_rs.stream)` now covers `dir(langgraph.stream)` completely (0 missing names).

Python suite is 336 tests; parity 38/38; clippy and Rust tests clean.

Documented divergence: LangGraph's `run.messages` is token-live only when the model layer emits v3
content-block events; the extension's engine does not, so chunked messages surface as one completed
handle at run end instead of a live stream.### Round 78: missing modules from the full-tree API diff

A diff over every importable `langgraph.*` module (minus stdlib re-exports) listed what was still
absent. This round adds the genuinely missing modules and the cheap re-export surfaces:

* `langgraph_rs.version` (`__version__` from package metadata).
* `langgraph_rs.typing` (`StateT`, `StateT_co`, `StateT_contra`, `InputT`, `OutputT`, `ContextT`,
  `ContextT_contra`, `NodeInputT`, `NodeInputT_contra`, `StateLike`).
* `langgraph_rs.utils` + `utils.config` (`ensure_config`, `patch_configurable`, `get_config`,
  `get_store`) + `utils.runnable` (`RunnableCallable` with `invoke`/`ainvoke`, `RunnableLike`).
* `langgraph_rs._config` with the merging `ensure_config` / `patch_configurable` implementation.
* `langgraph_rs.pregel.types` (the deprecated re-export module, including
  `LangGraphDeprecatedSinceV10`).
* `langgraph_rs.managed` + `managed.base` (`ManagedValue`, `ManagedValueSpec`,
  `ManagedValueMapping`, `PregelScratchpad`, `is_managed_value`) and `managed.is_last_step`
  (`IsLastStep`, `RemainingSteps` and their managers).
* `langgraph_rs.graph.ui` (`UIMessage`, `RemoveUIMessage`, `push_ui_message`,
  `delete_ui_message`, `ui_message_reducer`). The helpers emit on the custom stream; writing the
  event into a state key needs `CONFIG_KEY_SEND`, which the engine does not expose yet, so that half
  is documented rather than faked.
* `langgraph_rs.prebuilt.interrupt` (`HumanInterruptConfig`, `ActionRequest`, `HumanInterrupt`,
  `HumanResponse`).
* Re-export surfaces LangGraph's modules carry: `types.Runnable`/`RunnableConfig`/`TypeAdapter`/
  `ValidationError`/`TypeAliasType`, `stream.run_stream.beta`, and the message/stream names in
  `stream.transformers`.

What remains is no longer feature surface: the leftover diff entries are LangGraph's *internal*
modules re-exporting their own imports (`pregel.main` with 163 names, `pregel.debug`,
`pregel.protocol`, `prebuilt.tool_node`, `chat_agent_executor`, `tool_validator`). Matching those
would mean mirroring LangGraph's private engine layout, which the extension deliberately does not
have.

One real feature gap is left: `pregel.remote.RemoteGraph`. `langgraph_sdk` is installed in this
environment, so it is implementable as a thin client; that is the next round's work.

Python suite is 336 tests; parity 38/38; clippy clean.### Round 79: `RemoteGraph` — the last feature gap

`langgraph_rs.pregel.remote` now exists with a working `RemoteGraph` / `RemoteException`.

* Construction mirrors upstream: `RemoteGraph(assistant_id, url=..., api_key=..., headers=...,
  client=..., sync_client=..., config=..., name=...)`. Passing `url` builds the SDK clients;
  injecting `client` / `sync_client` skips that (which is how the tests drive it).
* `stream` / `astream` translate the SDK's `StreamPart` chunks: `version="v2"` yields the same
  `{"type", "ns", "data", "interrupts"}` parts the extension's own `stream()` produces (the namespace
  is decoded from the SDK event name), `version="v1"` yields payloads, `(mode, payload)` pairs or
  `(ns, mode, payload)` triples depending on `stream_mode` / `subgraphs`. `error*` chunks raise
  `RemoteException`.
* `invoke` / `ainvoke` drive `stream(stream_mode="values")` and return the last state, like upstream.
* `get_state` / `get_state_history` / `update_state` (+ async) delegate to the SDK's `threads` API,
  forwarding `thread_id`, `checkpoint_id`, `subgraphs`, `as_node`, `limit` and `headers`;
  `get_graph` / `aget_graph` delegate to `assistants`.
* `with_config` / `copy` return configured copies; a missing client raises `ValueError` with what to
  pass.

`python/tests/test_remote_graph.py` (11 tests) covers invoke/stream shapes (v1, v2, single and
multiple modes, namespaces), error propagation, async streaming, state read/write/history, graph
introspection, the missing-client error and config copying, all through injected fake SDK clients
(there is no live deployment in this environment).

Also added the real `convert_to_messages` / `message_chunk_to_message` helpers on
`langgraph_rs.graph.message`.

After this round every importable `langgraph.*` module has a counterpart except LangGraph's internal
engine modules, whose leftover `dir()` entries are that module's own imports (`pregel.main`,
`pregel.debug`, `pregel.protocol`, `graph.state`, `prebuilt.tool_node`, `chat_agent_executor`,
`tool_validator`, `managed.base`, `pregel.remote`).

Python suite is 347 tests; parity 38/38; clippy and Rust tests clean.### Round 80: closing verification

Objective: "langgraph 支持的都需要支持". Final evidence gathered in this round:

* **Python suite**: 347 tests, all passing.
* **Parity benchmark**: `python benchmarks/parity.py` → 38/38 scenarios equivalent, 0 known differences.
* **Migration examples**: all ten `examples/*/compare.py --check-only` runs report `-> identical`.
* **Rust**: `cargo test --all-targets --features "async msgpack sqlite"` clean (64 result lines, no
  failures) and `cargo clippy --all-targets --all-features -- -D warnings` clean.
* **Module coverage**: every importable `langgraph.*` module resolves to a `langgraph_rs.*`
  counterpart. The remaining `dir()` differences live only in LangGraph's internal engine modules
  (`pregel.main`, `pregel.debug`, `pregel.protocol`, `graph.state`, `prebuilt.tool_node`,
  `chat_agent_executor`, `tool_validator`, `managed.base`, `pregel.remote`) and are that module's own
  imports of `BaseChannel` / `AIMessage` / `RunnableConfig` / ... — not API the extension needs to
  mirror, because it runs its own engine.

`docs/langgraph-compat.md` was rewritten to state the current status accurately: the obsolete
"not implemented" entries (`ToolCallTransformer`, `create_react_agent`'s `response_format` /
`pre_model_hook` / `post_model_hook` / `context_schema`, the store vector index) are gone, and the
intentional-divergence list is now the ten verified items (engine internals as placeholders, native
`invoke` taking a full state, `create_react_agent` accepting extra keys without `state_schema`,
positional args for plain callables, the store superset, `run.messages` not being token-live,
repeatable stream/history iterables, no-saver interrupts raising, `push_ui_message` not writing
state, and `RemoteGraph` not being exercised against a live deployment).

With that, the objective is complete: LangGraph's public surface — graph building, state and
channels, sync/async execution, interrupts and resume, prebuilt agents and tools, checkpoint
backends with cross-engine resume, long-term memory with vector search, serialization, the v3 stream
subsystem including transformers, and the remote-server client — is implemented, tested against
LangGraph on the same workloads, and documented, with every remaining difference deliberate and
recorded.### Round 81: `run.messages` at delta level

The last recorded divergence was that `run.messages` produced one completed handle per chat-model
call. Two of the three layers needed to close it landed this round.

* **Model node streams**: `runnable_model_node` now calls the model's `stream()` when it exists,
  aggregates the chunks with LangChain's `+` into the final message (falling back to `invoke` when
  streaming fails or is unsupported), so `create_react_agent` produces real chunks instead of a
  single message. Tokens reach the `messages` stream through the existing LangChain callback handler;
  pushing them again from Rust duplicated every chunk (`['Hello','Hello',' ',' ','world','world']`),
  so that extra push was removed.
* **Transformer dispatches deltas**: `MessagesTransformer` opens a `ChatModelStream` on the *first*
  chunk, emits `message-start` / `content-block-start`, dispatches every chunk as its own
  `content-block-delta`, and closes with `content-block-finish` / `message-finish` (at run end, on
  failure, or when a whole message for the same id arrives — filling in any content the deltas had
  not covered). Handle dicts emitted by the engine's callback (`{"content": token, ...}`, no `type`)
  are recognised as chunks.

Measured behaviour now: a model inside `create_react_agent` yields `run.messages` → one handle whose
`text` streams `["Hello", " ", "world"]`.

Still open (measured, not guessed): the engine buffers a task's message chunks and sinks them when
the **node finishes** — a node emitting three chunks 0.2 s apart produces the first handle at 0.6 s —
and `stream()` collects everything before returning, so a consumer cannot yet receive tokens while
the model is generating. Closing that needs the engine to flush chunks as they are produced plus a
pull-based binding iterator.

`python/tests/test_messages_transformer.py` grew to 15 tests (live-delta dispatch, handle-before-end,
whole-message close with the remaining content, and a `create_react_agent` case asserting
`["Hello", " ", "world"]` for both `run.messages` and `stream_mode="messages"`).

Python suite is 351 tests; parity 38/38; clippy and Rust tests clean.### Round 82: live token delivery into `run.messages`

The remaining half of the `run.messages` gap landed: tokens now reach `handle.text` while the model
is still generating, not when the node finishes.

How it works:

* `langgraph_rs._live.LiveBridge` — a thread-safe queue plus a module-level "active bridge".
  `langgraph_rs.stream._events` activates one for the duration of a `stream_events` run.
* LangChain's callback handler (`on_llm_new_token`) already runs on the engine thread as tokens
  arrive; it now also pushes each token to the active bridge (in addition to the normal
  `langgraph_rs.message(...)` write, so the engine's own `messages` stream keeps its batches).
* `stream_events` starts the graph on a worker thread and yields through a generator that drains the
  bridge while polling the engine's iterator, so `GraphRunStream`'s caller-driven pump receives
  tokens as they are produced. Engine `messages` parts are dropped when the bridge already carried
  them, which keeps `handle.text` free of duplicates.

Measured (model emitting three chunks 0.2 s apart):

```
before: handle at 0.64s, deltas at 0.64 / 0.64 / 0.64
after:  handle at 0.24s, deltas at 0.24 / 0.44 / 0.64   (i.e. during generation)
```

`python/tests/test_messages_transformer.py` gained `LiveTokenDeliveryTest`, which asserts the first
delta is observed *before* the model's `_stream` ends (no timing thresholds, so it is not flaky).

Boundaries left, documented in `docs/langgraph-compat.md`:
1. `astream_events` still uses the batched iterator — the bridge is sync-only so far.
2. Chunks written by nodes via `rs.message()` / `get_stream_writer()` are still flushed per node by
   the engine; only the LangChain token callback is truly live.
3. One bridge at a time: concurrent `stream_events` runs share the most recent one.

Python suite is 352 tests; parity 38/38; clippy and Rust tests clean.### Round 83: live tokens on the async lane, and per-run isolation

* `astream_events` now uses the live bridge too (`_live_aparts`): the graph runs as an asyncio task,
  the generator drains the bridge while polling the engine iterator, and engine `messages` parts are
  dropped when the bridge carried them. Measured with a model emitting three chunks 0.2 s apart:
  handle at 0.26 s, deltas at 0.26 / 0.44 / 0.64 — i.e. during generation, same as the sync lane.
* Bridges are now keyed by `thread_id` (`_live.activate(bridge, thread_id)`,
  `active_bridge(current_thread_id())`), so two concurrent runs no longer exchange tokens. Runs
  without a `thread_id` share the `None` bucket.

Tests: `AsyncLiveTokenDeliveryTest` asserts the first async delta arrives before the model's
`_stream` ends (no timing thresholds), and `ConcurrentLiveRunsTest` runs two thread-keyed runs in
parallel and asserts each `run.messages` only saw its own text.

Remaining, documented: chunks written by nodes via `rs.message()` / `get_stream_writer()` are still
flushed by the engine when the node finishes — only the LangChain token callback is generation-time.

Python suite is 354 tests; parity 38/38; clippy and Rust tests clean.### Round 84: node-written chunks are live too

The last recorded boundary — chunks a node writes itself arriving only when the node returns — is
closed on the sync lane.

Engine (`src/runtime/context_run.rs`, `context_task_helpers.rs`):

* `RunContext` gained `live_sink: Option<LiveSink>` (`Arc<Mutex<dyn FnMut(StreamChunk) + Send>>`),
  inherited by nested/subgraph runs and by task threads through the existing run-context capture.
* `install_run` consumes a thread-local *pending* sink, so a caller can install one for the next run
  on its thread without threading a parameter through the engine.
* `push_message` / `push_custom` / `push_tool_event` hand their chunk to the live sink when one is
  installed (wrapping it in `Subgraph` for the current namespace) and skip buffering, so nothing is
  delivered twice.

Binding (`src/adapters/python/graph_compiled.rs`):

* New `CompiledGraph.stream_live(...)` returns `_LiveStream`, a `poll()`-based pull iterator. The
  graph runs on a worker thread that installs the live sink and forwards every chunk through a
  channel. `__next__` was replaced by `poll()` because blocking in Rust holds the GIL and deadlocks
  the worker that needs it to run Python nodes — waiting happens in Python (`time.sleep`) instead.
* The worker installs the LangChain callback context (`install_langchain_callbacks`), and the react
  agent's Rust model node now installs it per node (`begin_callback_scope`) — without that, model
  tokens never reached `messages` on the worker thread.

Python (`stream/_events.py`): `_live_parts` prefers `stream_live`, polls the bridge with a zero
timeout, sleeps 1 ms when idle, and yields engine parts as they arrive (previously they were
deferred to the end, which is exactly what hid the liveness).

Measured (node writing three chunks 0.2 s apart): handle at 0.20 s, deltas at 0.20 / 0.26 / 0.45.
Model tokens through `create_react_agent`: 0.23 / 0.43 / 0.63.

Boundaries left: `astream_events` still uses the collected `astream` (model tokens live via the
bridge, node writes batched), and plain `stream()` keeps its eager semantics.

Python suite is 355 tests; parity 38/38; clippy and Rust tests clean.### Round 85: live sink reachable from a run config

The live sink no longer depends on a thread-local handshake: `RunConfig` carries it
(`with_live_sink`), `run_loop` hands it to `step::begin_run`, and `install_run` prefers it over the
pending thread-local and the enclosing context. That makes live output reachable from the **async**
lane, where the run starts on a runtime thread the caller never touches.

`AsyncCompiledGraph.astream(..., live_sink=callback)` wraps the Python callable into a `LiveSink`
that converts each chunk with `convert_stream_chunks` and calls it on the engine thread. Verified:

```
astream(stream_mode=["messages"], live_sink=cb)            -> [0.03, 0.23]
astream(stream_mode=["values","tasks","messages"], subgraphs=True, live_sink=cb) -> [0.0, 0.2]
```

Status of the remaining gap: `astream_events` still delivers node-written chunks at node end. The
async event lane now passes a `live_sink` into `astream`, but the chunks do not surface before the
node finishes, so the wiring is incomplete — the sync lane (`stream_events`) remains the one with
full liveness for both model tokens and node writes. Model tokens stay live on the async lane through
the callback bridge (measured 0.22 / 0.37 / 0.51 for a model emitting three chunks 0.15 s apart).

Also: `step::begin_run` grew a parameter and now carries `#[allow(clippy::too_many_arguments)]`.

Python suite is 355 tests; parity 38/38; clippy and Rust tests clean.### Round 86: async node writes and custom events are live

Two fixes closed the gap left after round 85.

1. **The async lane was deferring engine parts.** `_live_aparts` appended parts to `deferred` and only
   yielded them after the loop ended — the same bug that had been fixed in the sync twin. Yielding
   them immediately fixed node-written chunks: deltas at 0.02 / 0.23 / 0.43 with `node finished=False`.
   The graph also moved onto its own thread with its own event loop (`asyncio.run` in a worker
   thread) so a busy consumer loop cannot starve the driver.
2. **The live path was gated on the `messages` mode.** `stream_events` only took the live route when
   `"messages"` was among the requested modes, so `get_stream_writer()` output (`custom` mode) was
   still batched. The gate is now `"messages" in modes or "custom" in modes`.

Widening it to *every* mode was tried first and broke two existing expectations
(`run.interrupted` stayed false, and a failing child subgraph leaked a message before the error), so
the gate stops at the token/custom lanes; `values` / `tasks` stay on the collected path where they are
assembled and ordered per step.

Measured after both fixes:

```
sync  custom writer : 0.01 / 0.16   (node still running)
async custom writer : 0.01 / 0.17   (node still running)
async node messages : 0.02 / 0.23 / 0.43 with node finished=False
```

Tests: `AsyncNodeWriteLivenessTest` and `CustomLivenessTest` added (both assert "first item before the
node finished", no timing thresholds).

Python suite is 357 tests; parity 38/38; clippy and Rust tests clean.### Round 87: interrupts on the live path, and a liveness regression harness

Auditing the live lane turned up a real defect: a run that paused reported `interrupted=False` when
the live path was active, while the collected path reported `True`. The engine signals a pause as an
`updates` chunk carrying `__interrupt__`, and `GraphRunStream._observe_event` only inspected `values`
events, so live runs lost the flag (the collected path happened to receive a `values` event with
`params["interrupts"]`).

`_observe_event` now treats an `updates` event whose data carries `__interrupt__` as an interruption
(sync and async run streams). Verified both lanes: `interrupted=True, interrupts=1`, and the payload
matches (`{"question": "ok?"}`).

Added `examples/streaming-liveness/compare.py`, a regression harness that asserts *ordering* rather
than wall-clock timing — the first item of a live stream must arrive before the producing node
finishes, which stays stable on slow machines while still failing when chunks get batched:

```
PASS  node messages (sync):   deltas=['a', 'b'] first_before_node_end=True
PASS  custom writer (sync):   payloads=[{'progress': 'x'}, {'progress': 'y'}] ...
PASS  node messages (async):  deltas=['a', 'b'] first_before_node_end=True
PASS  interrupt on live path: interrupted=True interrupts=1
```

Python suite is 358 tests; parity 38/38; clippy and Rust tests clean.### Round 88: auditing the divergence list against real LangGraph

Each "intentional divergence" was re-checked by running the same workload on LangGraph 1.2.22. Three
of the ten turned out to be wrong — one stale, two genuinely fixable.

* **`interrupt` without a checkpointer (was #8) — stale.** LangGraph returns the input state plus
  `__interrupt__`; so do we, on both `invoke` and `stream`. The claim that we raised `GraphInterrupt`
  no longer held, so the entry is gone.
* **`create_react_agent` extra input keys (was #3) — fixed.** LangGraph's default agent state declares
  only `messages`, so `{"messages": ..., "extra_key": 42}` comes back as `["messages"]`. Ours kept
  `extra_key`. Root cause was broader than the agent: `StateSchema::filter_input` filtered only
  private/input-schema channels, so *any* declared schema kept unknown input keys. It now drops them
  when the schema is strict (`!allow_implicit_channels`), matching LangGraph for every declared
  schema, while dynamic schemas keep accepting anything.
* **`run.messages` not live (was #6) — no longer a divergence.** The streaming work in rounds 81-87
  made model tokens, node-written messages and custom events live on both lanes; the entry moved to
  a "differences that were closed" list so it cannot silently regress.

Remaining seven are deliberate and documented: engine internals as placeholders, native
`invoke(&state)` requiring a complete state, positional args for plain callables, the store superset,
repeatable `stream()`/history objects, `push_ui_message` not writing state (no `CONFIG_KEY_SEND`),
and `RemoteGraph` not exercised against a live deployment.

Tests added: `DeclaredSchemaInputTest` (no-saver interrupt payload, react agent dropping extra keys).
Python suite is 360 tests; parity 38/38; clippy and Rust tests clean.### Round 89: plain callables are called by keyword, like LangChain

Divergence #3 of the audit list is closed. LangChain wraps a plain callable into a `StructuredTool`
whose `_run` calls `func(**tool_input)`, so `def get_weather(city, unit="c")` is called with
`city=..., unit=...`. Ours passed the whole argument mapping as one positional value, which broke
keyword-only parameters and any tool whose parameter names differ from the argument order.

`tool_definitions` now inspects the callable once per tool (`inspect.signature`, kinds 0-4) and
records how it can be called:

* `**kwargs` present → always call by keyword;
* only positional-or-keyword / keyword-only parameters and every tool argument name is one of them
  → call by keyword (the LangChain behaviour);
* otherwise (`def tool(args: dict)`, positional-only parameters, uninspectable callables) → pass the
  mapping positionally, so the older style keeps working.

`BaseTool` entries (`is_tool == true`) always receive the mapping positionally: `BaseTool.invoke()`
takes it as its single positional `input`, and the first version of this change broke exactly that
(`BaseTool.invoke() missing 1 required positional argument: 'input'`) plus a `@tool`-wrapped test.

Verified against LangGraph: the same graph with `def get_weather(city, unit="c")` and
`args={"city": "Paris", "unit": "f"}` produces `Paris:f` on both. Locally: named params → `Paris:f`,
`def tool(args: dict)` → `echo:Rome`, `**kwargs` → `star:a,b`, keyword-only → `kwonly:Oslo`.

Tests: `PlainToolCallTest` covers all four conventions. Divergence count is down to six.

Python suite is 364 tests; parity 38/38; clippy and Rust tests clean.### Round 90: `push_ui_message` writes state (a node-side send primitive)

LangGraph's `push_ui_message` does two things: it emits the event on the custom stream **and** calls
`config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])` so the event also lands in the graph state. We
only did the first half, which is why the divergence list carried "writes no state".

Engine: `TaskContext` gained a `writes` buffer and a `push_write` setter; a new
`runtime::context::send(channel, value)` free function records a write from inside a node, and
`take_writes()` drains the buffer. Both Python node wrappers merge the drained writes into the node's
`NodeOutput` (`apply_node_writes`), grouping repeated writes to one channel into a single list so the
reducer sees them exactly like the node returning each value.

Binding: `langgraph_rs._send(channel, value)` exposes the primitive (LangGraph's `CONFIG_KEY_SEND`
equivalent). `python/langgraph_rs/graph/ui.py` now calls it for `push_ui_message` and
`delete_ui_message`.

Verified against LangGraph on the same workload: two `push_ui_message` calls produce
`[("1", {"text": "hi"}), ("2", {"text": "there"})]` on both sides, and `delete_ui_message("1")`
leaves `[]`. `_send` also works for arbitrary channels (`_send("note", "hello")`).

Mistake worth recording: the first version wrapped the node call in a **fresh** `TaskContext`, which
replaced the engine's own context, so `custom()` / `record_tokens()` / `message()` inside the node
went into a context that was then discarded — eight existing tests failed with `KeyError: 'custom'`.
The fix is to reuse the engine's context and only drain the writes.

Tests: `UiMessageStateTest` (push, delete, arbitrary channel). Divergence count is down to five.

Python suite is 367 tests; parity 38/38; clippy and Rust tests clean.### Round 91: the store `query` divergence was misdiagnosed — and closed

Two claims in the divergence list were checked against LangGraph's source instead of behaviour
guesses, and both were wrong in our favour:

* **`PregelNode` / `ChannelWrite` are not importable from `langgraph.pregel` in 1.2.22** (`ImportError`,
  and `langgraph.pregel.write` does not exist), so our importable-but-raising placeholders are a
  superset, not a gap.
* **`search(query=...)` without a vector index.** The original entry said "LangGraph ignores `query`,
  we do substring matching". The first measurement that produced `[]` from LangGraph was invalid: the
  `put` in that test used `ttl=60`, which LangGraph rejects with `NotImplementedError`, so the store
  was empty. Reading `langgraph/store/memory/__init__.py` shows `_filter_items` keeps items that have
  no embeddings and `_embed_search_queries` returns nothing without an index config, i.e. **the query
  is ignored entirely**. Measured side by side:

  ```
  plain put + query="apple"     LG ['a']      rs ['a']
  put(index=["text"]) + query   LG ['a','b']  rs ['a','b']
  filter + query                LG ['b']      rs ['b']
  no query                      LG ['a','b']  rs ['a','b']
  ```

  `InMemoryStore::search_with` now drops the query when no vector index is configured, so the query
  half of the divergence is gone; only the `ttl` superset remains.

`StoreQueryParityTest` locks the three cases in against LangGraph itself. The divergence list is now
four items plus the `RemoteGraph` caveat, and item 4 was reworded: `StreamChunks` supports `next()`
(the generator protocol) in addition to `len()`/indexing, so it is a superset rather than an
incompatibility — verified with `next(app.stream(...))`.

Python suite is 370 tests; parity 38/38; clippy and Rust tests clean.### Round 92: closing verification for the divergence layer

Final pass over the whole layer, with the evidence re-collected in this round rather than quoted from
earlier notes.

**Verification run**

| Check | Result |
| --- | --- |
| Python suite | **372 tests, all passing** |
| Parity benchmark | 38/38 scenarios equivalent, 0 known differences |
| Migration examples | 10/10 `compare.py --check-only` → `identical` (11th is the liveness harness) |
| Streaming liveness harness | 4/4 PASS (`examples/streaming-liveness/compare.py --check`) |
| Rust tests | `cargo test --all-targets --features "async msgpack sqlite"` clean |
| Clippy | `--all-targets --all-features -- -D warnings` clean |

**Each remaining item re-checked against LangGraph 1.2.22**

* Internal engine classes: `langgraph.pregel` publicly exports only `NodeBuilder` and `Pregel`;
  `ChannelWrite` / `ChannelWriteEntry` live in the private `langgraph.pregel._write`; `PregelNode` and
  `SyncAsyncFuture` are not importable from any public module. We now provide all four names (plus a
  `langgraph_rs.pregel._write` path) as importable placeholders that raise `NotImplementedError` with
  a `StateGraph.compile()` hint — a superset, and `PregelPlaceholderTest` locks it in.
* `ttl`: LangGraph's `InMemoryStore.put(..., ttl=…)` raises `NotImplementedError`; ours accepts it.
* `stream()`: LangGraph returns a generator (`next()` works, no `len()`); `StreamChunks` supports
  `next()` as well as `len()`/indexing.
* Native `invoke(&state)`: Rust API design, no LangGraph counterpart to compare.
* `RemoteGraph`: no live deployment available in this environment.

`docs/langgraph-compat.md` section 6 now carries a closing statement: five retained differences (three
supersets, one Rust API choice, one environment limit) and a list of the six differences that were
closed in this layer, each with the test that locks it in.### Round 93: module-level API gaps closed

A systematic sweep (import every `langgraph` module, compare the names *defined in* that module —
dropping re-exports of `Any`/`Callable`/`TypedDict`) found seven public gaps after the previous
round. All seven are closed:

* `langgraph.callbacks` — the whole module is new: `GraphCallbackHandler` with `on_interrupt` /
  `on_resume`, the frozen `GraphInterruptEvent` / `GraphResumeEvent` payloads (field lists verified
  identical to LangGraph), `GraphLifecycleStatus` / `GraphLifecycleEvent`, and working sync/async
  managers via `get_{sync,async}_graph_callback_manager_for_config` (they filter `config["callbacks"]`
  down to `GraphCallbackHandler` subclasses and bind `run_id`).
* `langgraph.pregel.protocol` — `PregelProtocol` with LangGraph's 15 abstract methods and
  `StreamProtocol` (a callable holder with `modes`). The compiled graphs are registered as virtual
  subclasses, so `isinstance(app, PregelProtocol)` matches LangGraph.
* `langgraph.pregel.debug` — nine helpers. Eight of them were compared against LangGraph on identical
  inputs and produce equal output; `map_debug_checkpoint` and `tasks_w_writes` work on the same data
  shapes but are necessarily best-effort for `PregelExecutableTask`.
* TypeVars: `Value` / `Update` / `Checkpoint` (channels.base), `U` / `V` (managed.base),
  `ContextT` / `StateT` (prebuilt.tool_node). `ToolRuntime` is now `Generic[ContextT, StateT]` like
  LangGraph, with the public names carrying their defaults.
* `langgraph.types.PregelTask` was a raise-on-use placeholder; it is now the real seven-field named
  tuple, which also let `tasks_w_writes` match LangGraph.
* `langgraph.pregel.main` re-exports `NodeBuilder` / `Pregel`; `langgraph.pregel._write` exports
  `ChannelWriteTupleEntry`.

Final sweep: **0 public module gaps**. Deliberately absent: the 36 `langgraph._internal.*` modules and
18 private `langgraph.pregel._*` modules (LangGraph's own engine layout).

Known behavioural gap, recorded in the compat doc: the engine does not yet dispatch `on_interrupt` /
`on_resume` to `GraphCallbackHandler` itself — the module and managers are implemented and tested, but
a caller must drive them.

Python suite is 381 tests; parity 38/38; clippy clean.### Round 94: the engine dispatches graph lifecycle callbacks

The gap recorded in the previous round ("`GraphCallbackHandler` exists but the engine never calls it")
is closed. LangGraph's semantics were measured first, with a handler attached to a real graph:

```
invoke  -> [('interrupt', 'pending', '1f1e93ff', (), 1)]
resume  -> [('resume', 'pending', '1f1e93ff', ())]
```

Implementation:

* `langgraph_rs.callbacks._dispatch_graph_lifecycle(config, kind, ...)` builds the event and hands it
  to the manager for `config["callbacks"]`; it is a no-op when no `GraphCallbackHandler` is configured
  and swallows handler errors so a broken callback cannot fail a run.
* `graph_pause::pause_values` — the single place every paused run flows through — now dispatches
  `on_interrupt` with the snapshot's `checkpoint_id`, `status="pending"` and the interrupt payload.
  `pause_values` gained the caller's config for that purpose.
* `graph_invoke` dispatches `on_resume` for both resume shapes (`Command(resume=...)` and
  `invoke(None)` on a paused thread), using the checkpoint being resumed.
* `CompiledGraph.stream` dispatches `on_interrupt` (when the collected chunks show a paused snapshot)
  and `on_resume` for `Command(resume=...)`.
* Async: `AsyncCompiledGraph.ainvoke` now accepts LangGraph's `config=` argument (it previously only
  took `thread_id`), and both pause (`static_pause_values`) and resume dispatch on that lane too.

Verified on all three lanes — `invoke`, `stream`, `ainvoke` — each producing
`("interrupt", "pending", checkpoint set, 1 interrupt)` followed by `("resume", "pending", checkpoint set)`.
`GraphLifecycleDispatchTest` covers the three lanes plus the "no handlers configured" case.

Python suite is 385 tests; parity 38/38; clippy and Rust tests clean.### Round 95: one compiled graph with both APIs

Comparing `dir()` and `inspect.signature` of the compiled graph against LangGraph's
`CompiledStateGraph` showed a structural difference: LangGraph returns **one object** carrying both
sync and async entry points, while the extension had `CompiledGraph` and `AsyncCompiledGraph`
separately. A migrating caller doing `await app.ainvoke(...)` on the sync object failed with
`AttributeError`.

`CompiledGraph` now exposes the async twins:

* `ainvoke(input, config=None, *, thread_id, context, checkpoint_ns)` — runs the synchronous engine in
  a worker thread via `asyncio`'s executor, so a coroutine can be created outside a running loop
  (LangGraph's shape) and the loop is not blocked.
* `astream(...)` — an async generator over the collected run, matching LangGraph's
  `inspect.isasyncgenfunction` shape.
* `astream_events(..., version="v3")` — delegates to the v3 shim so it returns the same run stream as
  the async class.

`AsyncCompiledGraph.ainvoke` and `.astream` also accept LangGraph's `config=` argument now (they only
took `thread_id` before), which is what makes `config={"configurable": {...}, "callbacks": [...]}`
work on the async lane.

Verified: `await app.ainvoke(...)`, `async for chunk in app.astream(...)`, and
`await app.astream_events(...)` all work on the sync object; `AsyncCompiledGraph.astream(config=...)`
runs with a checkpointer.

Still different, and documented: `batch`/`abatch` do not take a per-input config list, and the
`Runnable` helpers (`with_config`, `bind`, `with_retry`, `pipe`, `map`, `batch_as_completed`,
`astream_log`, `atransform`, `as_tool`, schema introspection helpers) are not implemented because the
graph does not subclass `RunnableSerializable`.

Python suite is 389 tests; parity 38/38; clippy and Rust tests clean.### Round 96: batch takes per-input configs

`batch`/`abatch` were the last signature gap on the compiled graph: LangGraph accepts one config or
**one config per input**, plus `return_exceptions`, while ours took only `(inputs)`.

* `batch(inputs, config=None, *, return_exceptions=False, context=None)` — a config list is parsed
  into one `RunConfig` per input (`per_input_configs`); when every parsed config is identical the
  engine's parallel `batch` path is used, otherwise the inputs run one by one with their own config.
  A list whose length differs from the number of inputs raises `TypeError`.
* `return_exceptions=True` returns the exception object in the result list instead of raising.
* `abatch` exists on both compiled graphs now: `AsyncCompiledGraph.abatch` runs natively (config list
  supported), and `CompiledGraph.abatch` offloads the sync `batch` to a worker thread.

Verified: shared config `[1, 1]`, per-input `[11, 21]`, length mismatch → `TypeError`, failing node →
`RuntimeError` object in the list, `abatch` matching `batch`.

Along the way a round of editing removed `AsyncCompiledGraph.astream_events`; it was restored (now
also accepting `config=`) and the suite caught the regression immediately.

Python suite is 394 tests; parity 38/38; clippy and Rust tests clean.### Round 97: async state twins, bulk updates and cache clearing

The state surface was the last cluster of missing methods on the compiled graph.

* `aget_state`, `aget_state_history`, `aupdate_state` — awaitables that run the synchronous
  implementation on a worker thread (`call_via_sync`), so they work outside a running loop and do not
  block the event loop.
* `bulk_update_state(config, supersteps)` — LangGraph's shape: a list of supersteps, each a list of
  `{values, as_node, task_id}` updates. Each update goes through the engine's existing
  `update_state_with_task`, and the last resulting snapshot is returned. `abulk_update_state` is the
  awaitable twin.
* `clear_cache(nodes=None)` / `aclear_cache()` — calls `NodeCache::clear()`. The backends only support
  a full clear, so `nodes=` is accepted for compatibility and clears everything (documented).
* `get_graph(config=None, *, xray=False)` — accepts LangGraph's arguments; `xray` does not change the
  rendering because subgraphs are already embedded as nodes.

Verified end to end: bulk update → 99, `clear_cache` → `None`, `aget_state` → 99,
`aget_state_history` → 4 snapshots, `aupdate_state` → 5, `abulk_update_state` → 7, `aclear_cache` → ok.

Python suite is 398 tests; parity 38/38; clippy clean.### Round 98: graph introspection and a full sweep of what is still missing

Subgraph tracking: `StateGraph::add_subgraph` used to consume the child graph into a node closure, so
the compiled parent could not report it. The builder now records `subgraphs` (name → `Arc<CompiledGraph>`)
and the compiled graph keeps the map, which makes `get_subgraphs(*, namespace=None, recurse=False)`
truthful: it returns `(name, CompiledGraph)` pairs, and `recurse=True` also walks nested subgraphs.

Also added: `copy(update=None)` (deep-clones the graph definition, applies `name` from `update`) and the
`channels` / `store` attributes (`channels` = declared channel names, `store` = the real store).

`checkpointer` was implemented and then removed: the extension stores only the Rust `Arc<dyn Checkpointer>`,
so returning a Python object would have been a half-truth. `get_name` hit a genuine pyo3 symbol collision
with an existing `__pymethod_get_name__` in this class and stays unimplemented; both are documented as
part of the `Runnable` / engine-internals groups.

A fresh `dir()` + `inspect.signature` sweep leaves 62 names, now classified in
`docs/langgraph-compat.md`: the `RunnableSerializable` family (not a `Runnable` subclass by design) and
LangGraph's own engine-internal attributes (`nodes`, `attach_node`, `trigger_to_nodes`, ...).

Python suite is 401 tests; parity 38/38; clippy and Rust tests clean.### Round 99: the compiled graph is now a real `Runnable`

LangGraph's compiled graph inherits `langchain_core.runnables.Runnable`
(`CompiledStateGraph -> Pregel -> PregelProtocol -> Runnable -> ABC`), which is why it ships
`bind`/`with_config`/`pipe`/`map`/`as_tool`/`batch_as_completed`/`astream_log` and the schema
introspection helpers. The extension returned the bare PyO3 object, so none of that existed.

`compile()` now returns a `CompiledStateGraph` facade (Python, in `langgraph_rs/graph/state.py`)
that holds the native graph and inherits `Runnable`:

* `invoke`/`ainvoke`/`stream`/`astream`/`batch`/`abatch`, plus `stream_events`/`astream_events` and
  `get_graph`, are forwarded explicitly. This matters: `Runnable` defines all of them, so without an
  override the base-class defaults shadow the engine (the v3 event stream raised
  "only supported on Runnable subclasses" until it was forwarded).
* Everything else (`resume`, `resume_from`, `stream_live`, `get_subgraphs`, `bulk_update_state`, ...)
  is answered by `__getattr__`; a metaclass `__getattr__` forwards class-level lookups so
  `hasattr(CompiledStateGraph, 'export_checkpoint_msgpack')` still works, and `__dir__` lists both.
* `get_name`/`name`/`input_schema`/`output_schema`/`config_schema` are provided from the native
  graph's channel names and compiled name (`_graph_name` added in Rust for that purpose).
* Unsupported run keywords (`interrupt_before`, `interrupt_after`, `output_keys`) raise
  `NotImplementedError` with the alternative; `durability` is folded into the config.

Engine-side changes: `compile()` returns the facade; `add_node(name, compiled)` and `add_subgraph`
accept either the native object or the facade (`compiled_graph_of`); `_graph_name` exposes the
compiled name.

Verified: `isinstance(app, Runnable)`, `app | fn` runs the graph, `app.bind()/with_config()` return
`RunnableBinding`, `get_graph(xray=True).draw_ascii()` is the native renderer, `stream_events(v3)`
still works, and `thread_id=` stays supported. Suite is 408 tests, parity 38/38, examples 10/10
identical plus 4/4 liveness checks; clippy and Rust tests clean.

Remaining names after this round are the engine-internal attributes (`nodes`, `attach_node`,
`trigger_to_nodes`, `config`, ...) and two `RunnableSerializable`-only helpers
(`validate`, `get_context_jsonschema`).### Round 100: the async graph shares the facade, plus truthful introspection

`AsyncStateGraph.compile()` returned the bare native object, so an async graph was not a `Runnable`
and could not be run synchronously. Both halves are fixed:

* The async engine gained blocking `invoke` / `stream` / `batch`:
  `pyo3_async_runtimes::tokio::get_runtime().block_on(...)` inside `py.detach`, i.e. the calling
  Python thread waits while the Tokio runtime drives the run. `invoke` handles plain state,
  `Command(...)` and `thread_id`-only continuation, and routes pauses through the same
  `static_pause_values` helper as `ainvoke`.
* `AsyncStateGraph.compile()` now wraps the native object in the same `CompiledStateGraph` facade, so
  `isinstance(app, Runnable)` holds for both graph kinds. The facade raises a clear
  `NotImplementedError` for `stream_events` on a graph without a synchronous event stream.
* `add_subgraph` on the async builder accepts the facade (and the native object).

Truthful engine attributes exposed on both graphs: `interrupt_before_nodes`, `interrupt_after_nodes`,
`retry_policy`, `step_timeout`, `cache_policy`, `stream_transformers`, `context_schema`, `debug`.
Supporting accessors were added to both engines (`CompiledGraph::{interrupt_before, interrupt_after,
retries, timeouts, cache_policies}`, `AsyncCompiledGraph::{...}`, `CachePolicy::has_custom_key`,
`AsyncCompiledGraph::{to_ascii, graph_spec}`).

Verified: async graph `invoke` -> `{'n': 1}`, `stream` -> 3 parts, `batch` -> `[2, 3]`, `ainvoke` -> 6,
`aget_graph().draw_ascii()` contains the node, and the facade guard message for `stream_events`.
Suite is 413 tests, parity 38/38, examples 10/10 identical plus 4/4 liveness, clippy and Rust tests clean.

Remaining names are now 20: LangGraph's mutable-attach API (`attach_node`/`attach_edge`/`attach_branch`),
its own object graph (`nodes`, `builder`, `config`, `checkpointer`, `cache`, `*channels*`,
`trigger_to_nodes`, `node_error_handler_map`, `schema_to_mapper`, `stream_eager`, `stream_mode`) and
two `RunnableSerializable` helpers (`validate`, `get_context_jsonschema`).### Round 101: custom `BaseChannel` subclasses and the remaining truthful attributes

LangGraph's `BaseChannel` is an open class hierarchy: inherit it, implement `update(values) -> bool`
and `get()`, and the channel owns its merge semantics. The extension silently treated such a subclass
as `LastValue` (or rejected it when `__channel_spec__` was `None`), which is the one architecture-level
capability gap that was actually closable.

`convert.rs` now detects a `BaseChannel` subclass that overrides `update`/`get` and bridges it to an
engine reducer: each write calls `channel.update([write])`, then reads `channel.get()`. Verified with a
`MaxValue` channel (max of 5, 3, 9 -> 9) and a `Union` channel (`['x','y','z']`) in a three-node graph.
Builtin channels (`LastValue(int)`, ...) still resolve through `__channel_spec__`, and unknown specs
still raise at build time instead of degrading.

Documented limitation: the channel's private state lives in the Python object, so resuming from a
checkpoint restores the value but not the channel's internal fields; such channels must be able to
derive their state from the JSON value.

Attributes added this round, all truthful:
* `checkpointer` — the exact object passed to `compile()` (stored alongside the Rust handle)
* `cache` — the cache backend object (sync graph)
* `trigger_to_nodes` — derived from the static edges (`CompiledGraph::static_triggers`,
  `AsyncCompiledGraph::static_triggers`)

Suite is 415 tests, parity 38/38, examples 10/10 identical, clippy and Rust tests clean.
Remaining name count is 15, all LangGraph internals (`nodes`, `builder`, `config`,
`attach_node`/`attach_edge`/`attach_branch`, `*channels*`, `stream_eager`, `stream_mode`,
`node_error_handler_map`, `schema_to_mapper`) plus `validate`/`get_context_jsonschema`.### Round 102: builder/stream attributes, plus a measured `durability` comparison

Attributes added (all truthful): `builder` (the `StateGraph` that produced the graph — `compile` now
takes `PyRef<Self>` and stores it), `stream_mode` (`"values"`), `stream_eager` (`false`), `config`
(empty), `validate()` (compilation already validates, so it returns the graph) and
`get_context_jsonschema()` (derived from the declared context schema through a new
`langgraph_rs._compat.jsonschema_of` helper).

Behaviour measured against LangGraph 1.2.22 rather than assumed — `durability` is read by LangGraph
from `config[CONF][CONFIG_KEY_DURABILITY]`, i.e. only the invoke/stream keyword reaches it:

```
placement                       LangGraph   extension
invoke(..., durability="exit")   1 checkpoint  1 checkpoint
config["configurable"][...]      ignored       ignored
config["durability"] (top)       ignored       honoured (superset)
no argument                      4 checkpoints 4 checkpoints
```

Correct LangGraph usage therefore matches exactly; the extra top-level acceptance is a superset and is
documented. `recursion_limit` was checked in the same pass: both raise `GraphRecursionError` with the
same message. `Command(parent=True)` is rejected by both (it is an engine-internal field, not a
LangGraph argument), so no divergence there either.

Suite is 421 tests, parity 38/38, clippy and Rust tests clean. 11 names remain, all LangGraph's own
object model (`nodes`, `attach_*`, `input_channels`/`output_channels`, `stream_channels*`,
`node_error_handler_map`, `schema_to_mapper`).### Round 103: every interrupt from a paused superstep is surfaced

Switching from attribute names to measured behaviour found a real gap. Two parallel nodes that both
call `interrupt()`:

```
LangGraph: __interrupt__ has 2 entries, resume with Command(resume={id: value, ...})
extension: __interrupt__ had 1 entry (the run did pause both tasks)
```

The engine was already correct — `pause_multi_interrupt` persists every interrupt on the pause
checkpoint (`snapshot.interrupts`, and `get_state().tasks[*].interrupts` showed one per task). The bug
was in the Python surfacing layer: `pause_values` built `__interrupt__` from
`Error::Interrupt`, which carries only the *first* interrupt. It now prefers the snapshot's interrupt
list (falling back to the error payload when there is no checkpointer), and the `stream` path does the
same before dispatching `GraphCallbackHandler.on_interrupt`.

Verified end to end:

```
interrupts: [{'ask': 'a'}, {'ask': 'b'}]
resume(Command(resume={id_a: 'va', id_b: 'vb'})) -> {'a': 'va', 'b': 'vb'}
get_state().tasks -> {'na': 1 interrupt, 'nb': 1 interrupt}
```

A new parity scenario (`parallel_interrupts`) runs the same graph on both engines and compares both
the payloads and the resumed state: 39/39 scenarios equivalent. Subgraph interrupt bubbling and
multi-mode streaming (`stream_mode=["updates", "values"]`) were compared in the same pass and matched.

Suite is 424 tests; clippy and Rust tests clean.### Round 104: subgraph nodes emit their own `updates` chunk

Continuing the behavioural comparison found a second real gap: streaming `updates` from a parent graph
that runs a subgraph never reported the subgraph node itself.

```
LangGraph: {'sub': {'value': 1}}, {'after': {'value': 11}}
extension:                              {'after': {'value': 11}}
```

Cause: `add_subgraph` writes with `NodeOutput::replace` (the child may overwrite channels regardless of
their reducer), and `process_outcomes` emitted `StreamChunk::Updates` only when the *update* list was
non-empty — replacements were silent. Both kinds now feed one chunk (`updates` first, then
`replaces`), which is exactly the `{node: {channel: value}}` shape LangGraph produces.

Verified in both modes:

```
updates:             {'first': ...,'sub': {'value': 6}, 'after': ...}
updates+subgraphs:   ((), {'first'}), (('sub',), {'inner'}), ((), {'sub'}), ((), {'after'})
```

New parity scenario `subgraph_updates` compares the flat and namespaced chunk sequences (namespace ids
are opaque and differ in form, so only the names are compared): 40/40 scenarios equivalent.

Two remaining nuances, both recorded in `docs/langgraph-compat.md`: when the parent's *entry* is a
subgraph LangGraph omits the parent input echo while the extension emits the child's input echo, and
namespace ids are deterministic here versus uuids in LangGraph.

Suite is 426 tests, examples 10/10 identical plus 4/4 liveness, clippy and Rust tests clean.### Round 105: LangChain objects in state (JsonPlus codec)

The most consequential gap found so far: graph state rejected LangChain objects.

```
node returns AIMessage  -> RuntimeError: value of type `AIMessage` cannot be converted to JSON
invoke({"messages": [HumanMessage("hi")]}) -> TypeError
```

LangGraph stores these with `JsonPlusSerializer`, which emits
`{"lc": 1, "type": "constructor", "id": [module, name], "kwargs": {...}}` and revives them by
importing `id`. The extension now uses the same shape:

* `langgraph_rs._compat.encode_value` turns anything with `model_dump`/`to_json` into that dict;
  `py_to_value` calls it only on the path that previously failed, so plain JSON values pay nothing.
* `value_to_py` revives a dict when `lc == 1` (`decode_value` imports `id` and calls it with
  `kwargs`), otherwise the dict is returned unchanged.
* Encoding in LangGraph's shape means checkpoints written here stay readable by LangGraph.

Verified: a node returning `AIMessage` yields an `AIMessage`; `HumanMessage` round-trips as input; a
message with `tool_calls` survives a checkpointer round trip with its calls intact; plain dict
messages behave exactly as before.

Two adjacent findings recorded for later: our v3 run stream registers only
`lifecycle`/`subgraphs`/`values` (LangGraph also exposes a `messages` projection), and
`update_state(as_node=...)` plus `Command(graph=PARENT)` were compared in the same pass and matched
LangGraph exactly (`next == ('b',)`, `{'log': ['from-sub', 'after']}`).

Suite is 430 tests, parity 40/40, examples 10/10 identical, clippy and Rust tests clean.### Round 106: message chunks and the `messages` projection

`stream_mode="messages"` produced `(dict, metadata)` — a plain dict where LangGraph yields an
`AIMessageChunk` — and the v3 run stream registered no `messages` projection.

* The callback handler now keeps `content`/`id`/`tool_call_chunks`/`additional_kwargs`/
  `response_metadata` when the token is a message chunk, and `_compat.message_chunk` rebuilds an
  `AIMessageChunk` (with `Human`/`System` variants) from the JSON payload.
* `MessagesTransformer` is part of `DEFAULT_TRANSFORMERS`, matching LangGraph, so `run.messages`
  exists without opt-in. Factories are de-duplicated by transformer type, so passing
  `transformers=[MessagesTransformer]` no longer collides with the default registration.
* The live path tolerates a graph without a live sink (the synchronous facade): it falls back to
  plain parts instead of raising.

Verified: first message part is `AIMessageChunk(content='Hello')`, metadata carries
`langgraph_node`/`langgraph_step`, `run.extensions == ['lifecycle','messages','subgraphs','values']`,
and consuming `values` then `messages` matches LangGraph's single-consumer behaviour exactly.

Three test expectations were updated because the new default is the LangGraph-accurate one: the two
projection-set assertions now include `messages`, and the nested-failure test no longer claims no
subgraph handle is discovered (the outer child did start, so its handle is reported before the error).

Known gap, recorded: metadata lacks LangGraph's `langgraph_triggers` and `langgraph_path`, and there
is no `langgraph_checkpoint_ns` at the root namespace; those need the task context in the callback
layer.

Suite is 433 tests, parity 40/40, examples 10/10, clippy and Rust tests clean.### Round 107: `langgraph_triggers` and `langgraph_path` in message metadata

LangGraph's message metadata carries the task's pull path and the nodes that triggered it; ours had
neither. The triggers now travel with the run:

* `TaskContext` gained a `triggers` field, filled from `Task.triggers` in both execution paths
  (inline and the timed helper thread).
* `current_triggers()` is exposed to Python next to `current_node()`/`current_step()`.
* The callback handler emits `langgraph_triggers` (only when non-empty) and `langgraph_path`, whose
  shape is LangGraph's `("__pregel_pull", node)` extended by the subgraph namespace segments.
* Both keys are restored as **tuples** on the Python side; JSON carries them as arrays, so the v1 and
  v2 chunk converters rebuild the tuple types (this was a real bug: an earlier attempt returned lists).

Verified: a node reached through `a -> b` reports `langgraph_path == ('__pregel_pull', 'b')` and
`langgraph_triggers == ('a',)`; `langgraph_rs.current_triggers()` returns `['a']` inside the node.

Known differences recorded: the *entry* task still has empty triggers (LangGraph reports
`('branch:to:<node>',)`), and there is no `langgraph_checkpoint_ns` at the root namespace because the
extension does not synthesise checkpoint ids.

Suite is 436 tests, parity 40/40, clippy and Rust tests clean.### Round 108: task triggers use LangGraph's channel convention

Measuring `stream_mode="tasks"` on LangGraph 1.2.22 showed the trigger is the scheduling **channel**,
not the node that ran:

```
a -> b        : a: ('branch:to:a',)   b: ('branch:to:b',)
Send fan-out  : work: ('__pregel_push',)  (twice)
```

The extension pushed the *source* node (`['a']`) and left entry tasks empty. `fill_triggers` now
emits `branch:to:<node>` (or `__pregel_push` for push tasks) and runs for every superstep, so entry
tasks and `Send` tasks are filled too. Message metadata follows automatically:
`langgraph_triggers == ('branch:to:b',)` for a node reached through `a -> b`.

Two test expectations were updated to the measured convention (they previously asserted the old
source-node spelling).

New parity scenario `task_triggers` compares the start-chunk `(name, triggers)` pairs for a linear
graph and a `Send` fan-out on both engines: 41/41 scenarios equivalent.

Suite is 436 tests, examples 10/10, clippy and Rust tests clean.### Round 109: bulk_update_state shape and history parameters

Measured against LangGraph 1.2.22:

```
bulk_update_state(config, [[({"value": 100}, "a", None)], [...]])   # 3-tuples, not dicts
get_state_history(config, before=snapshot.config)                    # a RunnableConfig
```

Both were wrong on our side, and one of them silently: `bulk_update_state` read `values`/`as_node`
keys off each update, so LangGraph-style tuples matched nothing and were **skipped without an error**
(the run returned an unchanged state). It now parses `(values, as_node, task_id)` tuples (and lists),
keeps the dict spelling as a superset, and raises `TypeError` for shapes it cannot read.
`get_state_history` accepted only a checkpoint-id string for `before`; it now also accepts a
`RunnableConfig`, which is what callers get from `snapshot.config`.

Verified equal to LangGraph: `limit=2` -> `[2, 1]`, `before=<step 1>` -> `[0, -1]`,
`filter={"next": ("b",)}` -> `[]`, and `bulk_update_state` produces `{'value': 200, 'note': 'n1'}`.

Recorded differences that need an engine change: LangGraph writes one checkpoint per *superstep*
while the extension writes one per *update* (two supersteps produced three checkpoints here), and
LangGraph derives `next` from the union of every update in the last superstep while the extension
uses only the last update.

Suite is 441 tests, parity 41/41, clippy and Rust tests clean.### Round 110: bulk_update_state writes one checkpoint per superstep

The two differences recorded last round needed an engine change rather than a binding fix, so
`CompiledGraph::bulk_update_superstep(thread_id, [StateUpdateGroup])` was added:

* every update in a superstep is merged into the same value map,
* `next` is the **union** of the tasks each update's `as_node` schedules (deduplicated by node, and
  by input for `Send` tasks),
* exactly one checkpoint is written for the superstep (`source="update"`, step + 1).

`bulk_update_state` now groups the Python supersteps and calls it once per superstep.

Measured against LangGraph 1.2.22 on the same two-superstep call:

```
before      [2, 1, 0, -1]
after       [4, 3, 2, 1, 0, -1]      # 2 checkpoints added, same as LangGraph
values      {'note': 'n1', 'value': 200}
next        ('b',)                    # union of a's and b's routes
```

New parity scenario `bulk_update_state` compares the added checkpoint count, the resulting values and
`next` on both engines: 42/42 scenarios equivalent.

Suite is 443 tests, clippy and Rust tests clean.### Round 111: message metadata step numbering, plus `goto=[...]` and `Send` after resume

Three behaviours were compared with measured LangGraph output:

1. `Command(goto=["a", "b"])` — both engines run both targets (`{'log': ['fan','a','b']}`), no change
   needed.
2. `Send` fan-out after resuming an `interrupt()` — both pause with one interrupt and log `[7, 8]`
   after `Command(resume=7)`, no change needed.
3. `langgraph_step` in message metadata — LangGraph counts supersteps **from 1** (first node is
   `langgraph_step == 1`, second is `2`), while the engine's `current_step()` is 0-based. The handler
   now emits `step + 1`.

With that, a message chunk and its metadata match LangGraph exactly for a two-node graph:

```
{'type': 'AIMessageChunk', 'content': 'second',
 'meta': {'langgraph_node': 'b', 'langgraph_step': 2,
          'langgraph_triggers': ['branch:to:b'], 'langgraph_path': ['__pregel_pull', 'b']}}
```

New parity scenario `message_stream` compares chunk type, content and the four metadata fields
directly: 43/43 scenarios equivalent. (The examples check was re-run by exit code after a text-grep
heuristic produced false failures: 10/10 identical.)

Suite is 443 tests, clippy and Rust tests clean.### Round 112: graph lifecycle events go through a run queue

The callback dispatch used to be inverted: Rust called into Python at every pause/resume site (five
of them), building a callback manager per event, with `status` hardcoded and `run_id` always `None`.
LangGraph does the opposite — its loop pushes events onto a queue and the streaming layer drains it
once per iteration.

The extension now follows that shape:

* `_internal::lifecycle` defines `LifecycleEvent` (`kind`, `status`, `checkpoint_id`, `namespace`,
  `interrupts`) and `LifecycleSink`. `RunConfig::with_lifecycle_sink` carries it.
* The pause and resume sites only **record** (`graph_pause::record_interrupt` / `record_resume`);
  there is no Python call from Rust any more, and `callbacks._dispatch_graph_lifecycle` is gone.
* Each entry point attaches a sink for its run and copies the drained events onto the object
  (`take_lifecycle_events()`), sync and async alike.
* The Python facade drains once per call in `invoke`/`ainvoke`/`stream`/`astream` and dispatches
  through **one** manager with a fresh `run_id`.

Verified on all three lanes: `invoke`, `stream` and `ainvoke` each report
`('interrupt', 'pending', run_id set)` on the pause and `('resume', 'pending', run_id set)` on the
resume, and the queue is cleared between runs so a resume does not replay the earlier interrupt.

Suite is 445 tests, parity 43/43, examples 10/10, clippy and Rust tests clean.### Round 113: the call direction is now Python -> Rust

New constraint from the goal text: no reverse calls from Rust into Python. Nested Python calls fall
into two groups, and only the first is acceptable:

* **forward** — the engine invokes a user-provided callable (node function, reducer, saver,
  transformer). These are the user's code, so the engine must call them.
* **reverse** — a native method imports a shim module to do work the Python side should own.

The reverse group was removed this round:

| Removed | Replaced by |
|---|---|
| native `ainvoke` / `astream` / `abatch` (called `*_via_sync`) | facade: `asyncio.to_thread` for the sync native, direct native async API otherwise |
| native `stream_events` / `astream_events` (imported `stream._events`) | facade imports the shim itself and passes the facade (not the native) |
| native `aget_state` / `aget_state_history` / `aupdate_state` / `abulk_update_state` / `aclear_cache` (`call_via_sync`) | facade wraps the synchronous twins in `asyncio.to_thread` |
| native `get_context_jsonschema` (`_compat.jsonschema_of`) | native returns the raw schema, the facade derives JSON Schema |

Shim imports from Rust dropped from 17 to 7; the seven left are construction-time helpers
(`compile()` wrapping, the `JsonPlus` codec, `BaseChannel` detection, `tool_node`), listed for the
next rounds.

`astream` also became an async generator, matching LangGraph (which is not awaitable); the tests that
awaited it were updated, and the facade degrades gracefully when a graph has no `live_sink`.

Verification: 445 Python tests, parity 43/43, examples 10/10, streaming-liveness PASS, clippy and
Rust tests clean.### Round 114: construction-time wrapping moves to Python

`compile()` no longer returns the facade from Rust; the native method returns the engine object and
a Python subclass applies the facade:

* `PyStateGraph` / `PyAsyncStateGraph` gained `#[pyclass(subclass)]`;
* `graph/state.py` defines `StateGraph` / `AsyncStateGraph` subclasses whose `compile()` wraps the
  native result in `CompiledStateGraph`, and both packages export those names;
* `create_react_agent` is the same pattern: a Python wrapper over the native builder, and
  `create_tool_calling_executor` is that same object (identity preserved, as in LangGraph);
* `wrap_compiled` is deleted, and `AsyncStream` moved out of a Rust `exec` of embedded Python source
  into `langgraph_rs/stream/_stream.py`.

Shim imports from Rust: 17 -> 4.

What is left, and why it is harder:

| Site | Why it needs Python |
|---|---|
| `_compat.encode_value` / `decode_value` | the JsonPlus codec recreates LangChain objects from checkpoints |
| `channels.BaseChannel` detection | the channel base class lives in the shim; candidates need it to invert |
| `prebuilt.tool_node.ToolRuntime` | the injected runtime handle is a Python dataclass the tool receives |

Verification: 445 Python tests, parity 43/43, examples 10/10, streaming-liveness PASS, clippy and
Rust tests clean.### Round 115: schema normalisation moves to Python

Two more reverse calls disappeared, along with a second block of Python source embedded in Rust:

* `langgraph_rs/graph/_schema.py` now owns `MessagesState`, `_schema_from_class` and
  `normalize_schema`. The builders override `__new__` (the native constructor is `tp_new`, so
  `__init__` would run after the engine already read the schema) and hand the native a plain
  `{channel: spec}` dict.
* A channel that implements its own merge is marked `{"__custom_channel__": channel}` by
  `normalize_schema`; `schema_from_py` looks for that key instead of importing
  `langgraph_rs.channels` and testing `isinstance`. `is_custom_base_channel` is gone.
* The `PyModule::from_code` block that defined `MessagesState` / `_schema_from_class` in Rust is
  deleted; `graph/message.py` and the package root re-export the Python ones.

Shim imports from Rust: 4 -> 3 (`_compat.encode_value`, `_compat.decode_value`, `ToolRuntime`).

One regression to note from the change: the first version of the marker check shadowed the
unwrapped channel with the marker dict, which broke custom-channel merges until the duplicate
binding was removed.

Verification: 445 Python tests, parity 43/43, examples 10/10, clippy and Rust tests clean.### Round 116: the last three reverse calls are gone

Shim imports from Rust are now **zero**.

* **JsonPlus codec, native.** `encode_object` reads the object's own `model_dump` / `to_json` and
  `type().__module__` / `__qualname__` to build `{"lc": 1, "type": "constructor", "id": [...],
  "kwargs": ...}`; `decode_object` imports the recorded module with `importlib`, resolves the class
  and rebuilds it from `kwargs`. Both stay behaviourally identical to the shim's versions (missing
  classes fall back to the raw dict, non-dict payloads are wrapped under `value`).
* **`ToolRuntime` is a `#[pyclass]`.** The engine constructs the handle it injects into tools
  natively; `langgraph_rs.prebuilt.tool_node` re-exports the native class so user `isinstance`
  checks and annotations keep working. `create_react_agent` dependency unchanged.

What remains, and why it is legitimate: the engine still *calls* Python — node functions, reducers,
savers, transformers and tools are user code, and returning values requires constructing Python
objects. Neither is a reverse call into the extension's own logic.

Verification: 445 Python tests, parity 43/43, examples 10/10, streaming-liveness PASS, clippy and
Rust tests clean.### Round 117: `durability="exit"` keeps one checkpoint at a pause

Two behaviours were compared; the first already matched, the second did not.

* `stream_mode="custom"` with `get_stream_writer()`: identical on both engines — plain payloads in
  `custom` mode (`[{'progress': 1}, 'half way']`), the same interleaving in multi-mode
  (`values`, `custom`, `custom`, `values`), **no** propagation from a subgraph unless
  `subgraphs=True`, and `((namespace,), payload)` tuples when it is requested.
* `durability="exit"` across a pause: LangGraph keeps **one** checkpoint (`source="loop"`,
  carrying `next=('gate',)`) and two after the resume. The extension flushed the deferred raw-input
  checkpoint as well, so its pause history had an extra `source="input"` entry.

`persist_pause` now flushes only the post checkpoint for a deferred turn. Verified:

```
rs  paused [('loop', 1, ['gate'])]   resumed [('loop', 2, []), ('loop', 1, ['gate'])]
lg  paused [('loop', 1, ('gate',))]  resumed [('loop', 2, ()), ('loop', 1, ('gate',))]
```

`sync` / `async` histories were already identical (`loop 1 / loop 0 / input -1`).

New parity scenario `durability_pause` compares pause history, resume history and final state:
44/44 scenarios equivalent.

Suite is 445 tests, examples 10/10, streaming-liveness PASS, clippy and Rust tests clean.### Round 118: `stream_mode="debug"` matches LangGraph'

Measured against LangGraph 1.2.22, three things differed:

1. LangGraph only emits `checkpoint` debug entries when a **checkpointer** persists them; without one
   the stream carries `task` / `task_result` only. The extension emitted them unconditionally.
2. The `checkpoint` payload keys are `config`, `metadata`, `next`, `parent_config`, `tasks`,
   `values`; the extension added an `executed` key.
3. `step` is the *displayed* checkpoint step — LangGraph's raw-input checkpoint reports `-1` and the
   following ones `0, 1, 2`, with `metadata.step` equal to the entry's `step`. The extension was off
   by one and could not express `-1`.

Fixes: debug chunks are gated on `backend.has_checkpointer()`, `executed` is gone from the payload,
`StreamChunk::Debug.step` is now `i64` and carries the displayed value. Both streams now match entry
for entry:

```
no checkpointer   task 1, task_result 1, task 2, task_result 2
with checkpointer checkpoint -1, checkpoint 0, task 1, task_result 1,
                  checkpoint 1, task 2, task_result 2, checkpoint 2
```

Two Rust stream tests and one Python test assumed checkpoint entries always exist, so they now give
the graph a checkpointer (which is what LangGraph requires as well).

New parity scenario `debug_stream` compares `(type, step, payload keys)` for both cases: 45/45
scenarios equivalent.

Suite is 445 tests, examples 10/10, clippy and Rust tests clean.### Round 119: callable error handlers and the introspection surface

Measuring `dir()` on both compiled graphs left 27 names LangGraph has and the extension lacks. Most
are PregelNode internals (`nodes`, `attach_*`, `*_channels*`, `schema_to_mapper`'s mapper machinery)
or private helpers; the ones users actually touch were added this round.

**Callable `error_handler`** was a behavioural gap, not just an attribute: LangGraph synthesises a
`__error_handler__<node>` node, routes the failure there, passes the handler the **state** (not the
exception), applies its update, and does **not** run the failed node's outgoing edges. The Python
builder now does the same (registering the synthetic node and handing its name to the engine), and
the observable result matches: `{'value': 0, 'note': "handled:{'value': 0}"}` on both.

**Introspection attributes:** `input_channels`, `output_channels`, `stream_channels`,
`stream_channels_list`, `stream_channels_asis`, `node_error_handler_map`, `schema_to_mapper` and
`_repr_mimebundle_` (PNG when graphviz is available, `text/plain` otherwise).

**Graph description:** `get_graph().to_dict()["nodes"]` now lists `__start__` / `__end__` as well,
matching LangGraph's description (its edges already referenced them).

Still intentionally absent: `nodes` as `PregelNode`s, `attach_node`/`attach_edge`/`attach_branch`,
`node_error_handler_map`'s internal dispatch, and the private `_*` helpers.

New parity scenario `error_handler` compares the result state and the node set: 46/46 scenarios
equivalent. Suite is 450 tests, examples 10/10, clippy and Rust tests clean.### Round 120: `get_graph()` returns a real `langchain_core` Graph

LangGraph's `get_graph()` hands back `langchain_core.runnables.graph.Graph` — `Node`/`Edge` named
tuples plus `draw_mermaid`/`draw_png`/`draw_ascii`. The extension returned its own `Graph` object with
a `to_dict()` and three drawing methods.

Now the native side only produces the description and the Python facade rebuilds it with
langchain-core's API, so the returned type, `graph.nodes`, `graph.edges` and the drawing helpers
match LangGraph. `to_dict()` stays on the returned object as a convenience.

The description itself needed three additions to be equal:

* `nodes` lists `__start__` / `__end__` (last round);
* the **`__start__ -> entry node` edges** were missing (a conditional entry expands through its path
  map);
* conditional edges now carry their concrete targets with `conditional: true` instead of only
  naming the source.

`draw_ascii()` needs `grandalf`; without it both engines raise the same `ImportError`, so the tests
that used it now check `draw_mermaid()` instead.

New parity scenario `graph_export` compares the graph type, node set and `(source, target,
conditional)` edge list: 47/47 scenarios equivalent. Suite is 450 tests, examples 10/10, clippy and
Rust tests clean.### Round 121: UUID-shaped task ids and subgraph namespaces

LangGraph's task ids are UUIDs and its subgraph namespace is `<node>:<task_id>`, so `stream_mode="tasks"`
and `subgraphs=True` payloads show values like `child:3820a0d1-dd47-...`. The extension generated
`<node>-<millis>-<sequence>` instead (`child:child-1795688892058-8`), which users see verbatim.

`new_task_id` now folds the clock, the process id and a counter through SplitMix64 into 128 bits and
formats them as a UUID v4 string — no new dependency. Both task ids and namespaces are UUID-shaped:

```
tasks id:   0e06feea-ef23-47a8-8fe7-edae83fac2a2
namespace:  ('child:92a7f918-320b-477c-9cf6-aeb5e03b66b3',)
```

LangGraph's ids differ on every run; the extension's are likewise generated per run and only need to
be stable within a task, which they are.

New parity scenario `task_id_shape` checks the shape on both engines (all ids and namespace suffixes
match the UUID pattern): 48/48 scenarios equivalent. Suite is 450 tests, examples 10/10, clippy and
Rust tests clean.### Round 122: multi-round resume verified (no engine change)

Investigated repeated resumes with the suspicion that a previous answer was replayed.

* **Multi-round resume** (a loop that asks three times): both engines produce
  `['one'] -> ['one','two'] -> ['one','two','three']`, the same `{'ask': n}` interrupt payloads and
  the same history steps `[3, 2, 1, 0, -1]`.
* **Two `interrupt()` calls in one node**: resuming `"A"` pauses on the second question and resuming
  `"B"` yields `['A', 'B']` on both engines — the resumed run replays the earlier answer, as
  LangGraph does.

The apparent divergence came from the probe itself: the LangGraph side used `log: list` (a
last-value channel) while the extension used `append`, so returning the full list doubled the
entries. With matching reducers the traces are identical. A short-lived change that cleared
`resume`/`resume_history` on paused tasks was reverted — it broke the two-interrupt replay above,
which is exactly the behaviour LangGraph relies on.

New parity scenario `multi_resume` locks both traces in: 49/49 scenarios equivalent. Suite is 450
tests, examples 10/10, clippy and Rust tests clean.### Round 123: list `path_map`, `Send` fan-out interrupts, subgraph interrupts

Three comparisons, one of which found a real API gap.

* **`path_map` as a list.** LangGraph accepts `add_conditional_edges(..., path_map=["a", "b"])`;
  the extension raised `TypeError: path_map must be a dict`. The converter now expands a list of
  names into `{name: name}` and keeps the dict form.
* **`Send` fan-out with interrupts.** Both engines pause with `[{ask:0},{ask:1}]`, and resuming each
  interrupt by id produces `['0:a'] -> ['0:a','1:b']` with the pending payload shrinking in step.
* **Interrupt inside a subgraph.** The parent's `invoke` surfaces the child's payload, a parent
  `Command(resume=...)` continues the child, and both end with no pending tasks and the same state.

New parity scenarios `send_interrupts` and `subgraph_interrupt` lock the traces in: 51/51 scenarios
equivalent. Suite is 452 tests (two new path-map tests), examples 10/10, clippy and Rust tests clean.### Round 124: the checkpointer read protocol

`dir()` on the extension's `InMemorySaver` was empty: checkpoint management lived on the graph
(`threads()`, `get_state_history()`), while LangGraph's savers expose `get_tuple`, `list`, `get`,
`delete_thread`, `get_next_version` and async twins.

The native saver now carries the read side (`_checkpoint_tuple`, `_checkpoint_tuples`,
`delete_thread`, `get_next_version`) and a Python subclass in `checkpoint/memory.py` shapes it into
LangGraph's API:

* `get_tuple(config)` returns a `CheckpointTuple` **NamedTuple** (attribute access, as in LangGraph;
  the shim previously stubbed it as unimplemented), with LangGraph's checkpoint body keys
  (`v`, `id`, `ts`, `channel_values`, `channel_versions`, `versions_seen`, `updated_channels`).
* `get(config)` returns the checkpoint, `list(...)` walks the history newest-first and honours
  `before` / `limit` / metadata `filter`, `delete_thread` removes the thread, and
  `get_next_version` returns LangGraph's `"<counter:032x>.0.<timestamp>"` string.
* `aget_tuple` / `alist` / `adelete_thread` mirror the async side.

`Checkpoint`, `CheckpointMetadata`, `PendingWrite` and `CheckpointTuple` are now real definitions in
`checkpoint/base.py` instead of "not implemented" stubs.

The write side (`put` / `put_writes`) still runs inside the engine rather than through the saver
object; that is documented in the compatibility notes.

Suite is 455 tests (three new saver tests), parity 51/51, examples 10/10, clippy and Rust tests
clean.### Round 125: `astream_events` v2 exists and both versions have LangGraph's shape

LangGraph's `astream_events` is a plain `def`: v3 returns an awaitable `AsyncGraphRunStream`, v2
returns an **async generator** of LangChain-style callback events. The extension implemented v3 only
and declared `astream_events` as `async def`, so `version="v2"` produced a coroutine and `async for`
failed with `'async for' requires an object with __aiter__ method`.

Now the facade is a plain `def` that hands back whichever object the version implies, and a v2
emitter synthesises `on_chain_*` events from the engine's `updates` / `values` stream. The sequence
matches LangGraph event for event:

```
on_chain_start(LangGraph) -> on_chain_start(a) -> on_chain_stream(a)
-> on_chain_end(a) -> on_chain_stream(LangGraph) -> on_chain_end(LangGraph)
```

with the same keys per event (`event`, `name`, `run_id`, `tags`, `metadata`, `data`, `parent_ids`) and
`langgraph_node` (plus `langgraph_checkpoint_ns` for subgraphs) in node metadata.

New parity scenario `events_v2` compares the `(event, name, data keys)` sequence: 52/52 scenarios
equivalent. Suite is 457 tests (two new), examples 10/10, clippy and Rust tests clean.### Round 126: `Command(graph=...)` only crosses graphs for `Command.PARENT`

LangGraph's `Command.PARENT` is the string `"__parent__"`; every other `graph=` value is a *graph
name* that only matters when a command crosses that graph's boundary, so from a parent node it has
no effect. The extension treated **any** `graph=` value as parent routing, which made
`Command(goto="child", graph="child")` raise
`Command(graph=PARENT) can only be returned from inside a subgraph`.

`to_output` now compares against the `PARENT_GRAPH` constant, so both
`Command(goto="child", graph="child")` and `Command(goto="child")` route normally and return
`{'value': 7}` on both engines.

Kept deliberately strict: `Command(goto="<unknown>")` fails with `unknown destination node`, where
LangGraph logs `wrote to unknown channel ... ignoring it` and continues. The strict form is an
existing, tested design choice of this project.

New parity scenario `command_graph` covers both forms: 53/53 scenarios equivalent. Suite is 457
tests, examples 10/10, clippy and Rust tests clean.### Round 127: LangGraph's timeout rules

Two divergences from LangGraph's timeout behaviour:

1. **A timeout on a synchronous node is rejected.** LangGraph raises at `compile()`:
   `Node timeouts are only supported for async nodes because sync Python execution cannot be safely
   cancelled in-process. Node 'x' is sync.` The extension used to "expire" the node, report a
   failure, and leave the thread running — exactly the hazard LangGraph avoids. The Python builders
   now reject it at compile with the same message.
2. **The error type is `NodeTimeoutError`.** LangGraph raises `langgraph.errors.NodeTimeoutError`
   with `Node 'x' exceeded its run timeout of 0.050s (elapsed: 0.063s).`; the extension surfaced
   `RuntimeError: node 'x' failed: python node raised TimeoutError: ...`. `to_pyerr` now maps both
   the raw `Error::Timeout` and the retry-wrapped node failure to `NodeTimeoutError` with that
   message shape (`elapsed` is approximated by the configured timeout; LangGraph reports the measured
   duration).

`add_node(timeout=...)` also accepts plain seconds and `timedelta` (LangGraph accepts seconds,
`timedelta` or a `TimeoutPolicy`) by wrapping them in a `TimeoutPolicy`.

Two per-Send timeout paths had to stay: the Python coroutine runs on a private loop, so the
executor's Tokio timeout cannot interrupt it — `asyncio.wait_for` enforces it and the resulting
`TimeoutError` is mapped back to `Error::Timeout`.

Suite is 459 tests (two new), parity 53/53, examples 10/10, clippy and Rust tests clean.### Round 128: `idle_timeout` works for Python nodes

LangGraph's `idle_timeout` fires when a node stops reporting progress, raising
`NodeTimeoutError: Node 'x' exceeded its idle timeout of 0.050s without making progress`. The
extension ignored it entirely: the node ran to completion and succeeded, because the Tokio-side
watchdog (`run_with_idle_timeout`) never got to poll while a Python coroutine held the worker inside
`Python::attach`.

* The engine now exposes `_progress_counter()` — the sum of the async context counter and the sync
  task context counter — and `get_stream_writer()` / `custom()` / `message` / usage writes bump it.
* Async nodes install a Python watchdog: it polls the counter every `idle` interval, cancels the
  coroutine when nothing moved, and raises `TimeoutError("langgraph-rs idle timeout")`.
* That marker maps back to `Error::IdleTimeout`, and `to_pyerr` turns both the raw error and the
  retry-wrapped node failure into `NodeTimeoutError(kind="idle")`.

Verified against LangGraph: a silent node raises the idle `NodeTimeoutError`, while a node writing
`custom()` every 20 ms with `idle_timeout=0.05` completes normally on both engines.

Suite is 461 tests (two new), parity 53/53, examples 10/10, clippy and Rust tests clean.### Round 129: `ToolNode(handle_tool_errors=...)`

LangGraph's `ToolNode` decides what a raising tool does through `handle_tool_errors`; the extension
did not accept the argument at all (`TypeError: ToolNode.__new__() got an unexpected keyword
argument`), and failed tool messages carried no `status`.

All four shapes now match LangGraph:

| `handle_tool_errors` | resulting `ToolMessage` |
|---|---|
| `True` (default) | `status="error"`, content `Error: ValueError('kaboom')\n Please fix your mistakes.` |
| string | `status="error"`, content is that string |
| callable | `status="error"`, content is the handler's return value |
| `False` | the exception propagates and the node fails |

Two details had to follow: a tool error is now formatted from the exception `repr()` (what LangGraph
writes into the message), and the exception handed to a callable has the original message as its
`str()` (`kaboom`), not the repr.

This also exposed a deeper problem: `ToolNode` only saw `tool_calls` when the state held plain dict
messages. LangChain message objects reach the engine as encoded constructor payloads, so
`AIMessage(...).tool_calls` was invisible and the node silently did nothing. The node now flattens
messages to `role` / `content` / `tool_calls` before executing.

Known gap: `handle_tool_errors=<exception type or tuple>` cannot filter by type, because the engine
only keeps the error text; those values fall back to the default message.

Suite is 465 tests (four new), parity 54/54 (new `tool_errors` scenario), examples 10/10, clippy and
Rust tests clean.### Round 130: `ValidationNode` semantics and `timedelta` timeouts

`ValidationNode` returned messages without `status`, never validated argument *types*, and turned an
unknown tool into an error message. LangGraph:

```
valid call      -> status="success", content = '{"query":"x"}'
bad arguments   -> status="success", content = '1 validation error for Validation\ncount\n  Input should be ...'
unknown tool    -> KeyError, so the node fails
```

All three now match. Argument validation follows LangGraph's approach: the callable's signature is
turned into a pydantic model named `Validation`, so annotations are enforced — previously only
`inspect.signature().bind()` ran and `"nope"` passed for an `int`.

Timeouts also accept `timedelta` now. LangGraph's `TimeoutPolicy`, `Send(timeout=...)` and
`add_node(timeout=...)` take "seconds / `timedelta` / `TimeoutPolicy`"; the extension only took a
`float` and raised `TypeError: must be real number, not datetime.timedelta`.

The `send_interrupts` parity scenario was flaky: parallel `Send` tasks have no defined order, so the
interrupt values are now compared sorted.

Suite is 470 tests (five new), parity 55/55 (new `validation_node` scenario), examples 10/10, clippy
and Rust tests clean.### Round 131: `ToolNode` constructor surface and tool-call middleware

LangGraph's `ToolNode` signature is
`(tools, *, name="tools", tags=None, handle_tool_errors=..., messages_key="messages",
wrap_tool_call=None, awrap_tool_call=None)` with `tools_by_name`, `get_name()` and `invoke()` on the
instance. The extension only had `(tools, messages_key=...)` plus `__call__`.

Added: `name` / `tags` / `tools_by_name` / `get_name()` / `invoke()` / `ainvoke()`, and the
middleware pair `wrap_tool_call` / `awrap_tool_call`. LangGraph passes the wrapper a
`ToolCallRequest(tool_call, tool, state, runtime)` and a `handler`; calling `handler(request)`
executes the tool and its return value becomes the `ToolMessage`. Both engines now agree:

```
wrap(request, handler) sees ("ToolCallRequest", "add")
handler(request) -> status="success", content="3", name="add", tool_call_id="c1"
```

The native node still runs unchecked when no wrapper is configured (no overhead); the Python path
only engages when a wrapper is present.

Also fixed a false parity failure: `send_interrupts` resumed interrupts by **list index**, but
parallel `Send` tasks have no defined order, so LangGraph sometimes resumed the other task. The
scenario now locates the interrupt by its own `ask` value; three consecutive runs are a stable 56/56.

Suite is 474 tests (four new), parity 56/56 (new `wrap_tool_call` scenario), examples 10/10, clippy
and Rust tests clean.### Round 132: `stream_mode="tasks"` payloads

LangGraph's task payload is the `PregelTask` shape `{id, name, input, triggers}`. The extension
added a `step` key, usually reported `input` as `None` (only `Send` tasks carried one), and emitted
`triggers` as a list instead of a tuple.

Now identical, and the `debug` mode `task` records (same data) are fixed with it:

```
{"id": "...", "name": "a", "input": {"value": 0}, "triggers": ("branch:to:a",)}
{"id": "...", "name": "b", "input": {"value": 1}, "triggers": ("branch:to:b",)}
```

`input` follows LangGraph: the state a task receives, which for an ordinary scheduled task is the
current channel values.

Also checked, and already matching: `values` (input echo plus one chunk per superstep), `updates`
(one chunk per node) and `debug` (`task` / `task_result` pairs).

Suite is 475 tests (one new), parity 57/57 (new `tasks_stream` scenario), examples 10/10, clippy and
Rust tests clean.### Round 133: `subgraphs=True` namespaces every chunk

With `subgraphs=True`, LangGraph wraps **every** chunk as `(namespace, payload)` and uses the empty
tuple for the top level:

```
[(('sub:<uuid>',), {'inner': {'value': 7}}), ((), {'sub': {'value': 7}}), ((), {'tail': {'value': 9}})]
```

The extension only wrapped chunks that came from a subgraph, so a caller doing
`for namespace, chunk in stream` hit `ValueError` on the top-level chunks. They are now wrapped with
`()`, consistently for `updates`, `values` and `astream` (the wrapping happens in one place, so v2
mode gets the `ns` key too).

Also verified as already matching: `stream_mode="checkpoints"` (four chunks with the same key set).

Suite is 477 tests (two new), parity 58/58 (new `subgraph_stream` scenario), examples 10/10, clippy
and Rust tests clean.### Round 134: `stream_mode="messages"` streams tokens

LangGraph reports model output token by token in `messages` mode, with a full `langgraph_*` / `ls_*`
metadata set. The extension reported one chunk with the node's final message (`'hello world'`) and
left out `checkpoint_ns`, `lc_versions` and the `ls_*` keys.

Two root causes:

1. The LangChain callback handler was not a `_StreamingCallbackHandler`, so
   `BaseChatModel._should_stream()` returned False and a node's `model.invoke(...)` never took the
   streaming path — no tokens were produced. It now inherits both `BaseCallbackHandler` and
   `langchain_core.tracers._streaming._StreamingCallbackHandler`, so `invoke` streams exactly as it
   does under LangGraph.
2. `checkpoint_ns` was missing. LangGraph uses the **task-level** namespace `<node>:<task id>` while
   `langgraph_path` only appends *subgraph* segments — they are different values and are now read
   separately.

Verified identical:

```
tokens: ['hello', ' ', 'world']
meta:   checkpoint_ns, langgraph_checkpoint_ns, langgraph_node, langgraph_path,
        langgraph_step, langgraph_triggers, lc_versions, ls_integration,
        ls_model_type, ls_provider
```

`tags` was also dropped from the message metadata, where LangGraph does not have it.

Suite is 478 tests (one new), parity 59/59 (new `message_tokens` scenario), examples 10/10, streaming
liveness check PASS, clippy and Rust tests clean.### Round 135: `create_react_agent` runs tools again

Comparing against LangGraph with a fake model that returns a tool call and then an answer exposed a
real bug: the tool call was swallowed.

```
LG: [human, ai(tool_calls), tool('42'), ai('the answer is 42')]
rs: [user, ai('the answer is 42')]        # ai(tool_calls) and tool missing
```

`runnable_model_node` called `model.stream()` explicitly — a workaround from before the callback
handler could stream. It consumed the first message of the fake model, the chunk accumulation fell
back to `model.invoke()`, and that consumed the second one, so the node only saw the final answer.

LangGraph calls `invoke` **once**: its `_StreamingCallbackHandler` makes LangChain switch `invoke` to
the streaming path by itself. Streaming is now a run-time switch:

* `stream_mode="messages"` and `stream_live` (the events API) use
  `StreamingLangChainCallbackHandler`;
* a plain `invoke` uses the handler without the streaming marker, as LangGraph does.

`stream_live` runs the graph on a worker thread, so the flag is set there as well (it is
thread-local).

After the fix the output matches LangGraph message for message.

Documented difference: LangGraph message objects have `type` = `human` / `ai` / `tool`, while the
extension keeps dict messages with `role` = `user` / `ai` / `tool`; the parity scenario folds the two
spellings.

Suite is 479 tests (one new), parity 60/60 (new `react_agent_loop` scenario), examples 10/10,
streaming liveness PASS, clippy and Rust tests clean.### Round 136: `StateSnapshot` field types

LangGraph's `StateSnapshot` exposes `next`, `interrupts` and `tasks` as **tuples** and always puts
`parents` in `metadata` (an empty dict at the top level). The extension returned lists and had no
`parents`:

```
LG: tuple tuple tuple | metadata keys: parents, source, step
rs: list  list  list  | metadata keys: source, step
```

Both now match. The change surfaced a latent bug: `embed_subgraph_states` (the
`get_state(..., subgraphs=True)` path) grabbed `tasks` with `cast::<PyList>()`, so once the field
became a tuple the child snapshots were silently not embedded and `child_task["state"]` turned into
`None`. It iterates instead, accepting either container.

Also verified as already matching: `response_format` (`structured_response`), `prompt=` (system
message prepended), `store=` (`runtime.store` inside tools) and `context_schema` + `context=`
(`runtime.context` inside tools).

Suite is 481 tests (two new), parity 61/61 (new `snapshot_shape` scenario), examples 10/10, clippy
and Rust tests clean.### Round 137: `PregelTask` and `Interrupt` objects in snapshots

LangGraph's `snapshot.tasks[i]` is a `PregelTask` named tuple (`id, name, path, error, interrupts,
state, result`) and `snapshot.interrupts[i]` is an `Interrupt` object. The extension returned plain
dicts, so the documented attribute access (`task.name`, `interrupt.value`) raised
`TypeError: tuple indices must be integers` / `'Interrupt' is not subscriptable`.

Both are now built as objects:

```
task:      PregelTask(name='ask', path=('__pregel_pull', 'ask'), error=None, ...)
interrupt: Interrupt(id='...', value='q?')
```

`snapshot_to_py` was split into a raw-dict stage and a final-object stage, because
`get_state(..., subgraphs=True)` has to write child states into the task dicts first and named tuples
are immutable. That split also fixed the child-state embedding, which had silently stopped working
once tasks stopped being dicts.

Suite is 483 tests (two new), parity 62/62 (new `snapshot_objects` scenario), examples 10/10, clippy
and Rust tests clean.### Round 138: policy objects take LangGraph's argument types

LangGraph's `RetryPolicy`, `CachePolicy` and `TimeoutPolicy` are dataclasses, so their constructor
arguments and reprs are user-visible. The extension was missing pieces of both:

* `RetryPolicy` had no `__repr__` (`<builtins.RetryPolicy object at 0x...>`), took `jitter` only as a
  float where LangGraph uses a **bool** defaulting to `True`, and accepted `retry_on` only as strings
  where LangGraph takes **exception types or tuples of them**;
* `CachePolicy` had no `key_func`, one of its main fields;
* `TimeoutPolicy` had no `refresh_on` (`"auto"` / `"heartbeat"`).

All three now match, with reprs in LangGraph's field order:

```
RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=())
CachePolicy(key_func=None, ttl=60)
TimeoutPolicy(run_timeout=0.5, idle_timeout=None, refresh_on="auto")
```

`retry_on=(ValueError, KeyError)` participates by name and was verified end to end (a flaky node
succeeds on the second attempt). `TimeoutPolicy.refresh_on` is stored and exposed; the engine's idle
watchdog already refreshes on progress, which is what `"auto"` means.

Suite is 487 tests (four new), parity 63/63 (new `policy_surface` scenario), examples 10/10, clippy
and Rust tests clean.### Round 139: `CachePolicy.key_func` works, `InMemoryStore` gains batch/async

`CachePolicy.key_func` was stored but never called; LangGraph calls `key_func(val)` (one argument, the
node input state) to compute a node's cache key. It is now wired into the engine's cache-key
computation: two identical runs call the function twice and hit the cache on the second (verified
against LangGraph).

`InMemoryStore` was missing most of its surface:

* async twins `aget` / `aput` / `adelete` / `asearch` / `alist_namespaces` / `abatch`;
* `batch(ops)` dispatching `GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp`;
* `supports_ttl` / `ttl_config`;
* `Item.namespace` is now a **tuple** and `created_at` / `updated_at` are **datetimes** (they were a
  list and ISO strings).

`batch` also follows a subtle LangGraph rule: reads see the state the batch started from, so a
`PutOp` in the same batch is invisible to a later `GetOp` / `ListNamespacesOp`. The implementation
runs reads first and applies writes afterwards.

Suite is 491 tests (four new), parity 64/64 (new `store_surface` scenario), examples 10/10, clippy and
Rust tests clean.### Round 140: top-level saver/store are the compatibility classes

`langgraph_rs.InMemorySaver` pointed at the **native** class (`delete_thread` and
`get_next_version` only), while the class with LangGraph's surface (`get_tuple`, `list`,
`aget_tuple`, `alist`, `adelete_thread`) lived in `langgraph_rs.checkpoint.memory`. So the same
import spelled at top level was near-empty. The top-level names now point at the compatibility
classes (`InMemorySaver`, `MemorySaver`, `AsyncInMemorySaver`, `InMemoryStore`, `Store`).

The saver also gained the rest of its public surface: the async twins `aget_tuple`, `alist`, `aput`,
`aput_writes`, `adelete_thread`, `acopy_thread`, and `copy_thread(source, target, *,
checkpoint_id=None)` backed by a new native implementation (verified by copying a thread's state).

Still missing, documented as such: `put` / `put_writes` (engine-internal writes never pass through
the saver), `prune` / `with_allowlist` / `get_delta_channel_history` / `delete_for_runs`, and the
internal `serde` / `storage` / `writes` / `blobs` attributes.

A Python subclass of a pyclass must also build in `__new__`, not `__init__` — `InMemoryStore.__init__`
calling `super().__init__(index)` produced `object.__init__() takes exactly one argument`, caught by
the store-index tests.

Suite is 495 tests (four new), parity 65/65 (new `saver_surface` scenario), examples 10/10, clippy and
Rust tests clean.### Round 141: error types and `Command` on a fresh thread

A sweep of common failure paths against LangGraph found two divergences:

* **Conflicting writes to a plain channel** — LangGraph raises `InvalidUpdateError` with
  `At key 'value': Can receive only one value per step. Use an Annotated key to handle multiple
  values.` plus the troubleshooting link; the extension raised a bare `RuntimeError: conflicting
  concurrent writes to channel ...`. `Error::ConcurrentUpdate` now maps to `InvalidUpdateError` with
  LangGraph's exact message.
* **A `Command` for a thread without checkpoints** — LangGraph simply starts the run: `resume` is
  ignored and `update` becomes the initial input (`invoke(Command(resume=5))` gives the node `{}`,
  `invoke(Command(update={"value": 9}))` gives it `{"value": 9}`). The extension raised
  `RuntimeError: checkpoint failure: no checkpoint found for thread ...`; it now starts the run the
  same way.

Also checked and already matching: the recursion limit (`GraphRecursionError` with the documented
message), unknown channel writes (silently ignored in both), and `get_state` for an unknown thread.

Suite is 497 tests (two new), parity 66/66 (new `error_types` scenario), examples 10/10, clippy and
Rust tests clean.### Round 142: the `source="input"` checkpoint holds no values

`get_state_history` yielded four entries, but the oldest one (`metadata.source == "input"`, `step ==
-1`) carried the raw input where LangGraph leaves it empty:

```
step: 2 | source: loop  | values: {'value': 2} | next: ()
step: 1 | source: loop  | values: {'value': 1} | next: ('b',)
step: 0 | source: loop  | values: {'value': 0} | next: ('a',)
step: -1| source: input | values: {}            | next: ('__start__',)   # was {'value': 0}
```

The input lands in the post-input checkpoint that immediately follows; the `input` checkpoint only
carries the `__start__` task.

Also verified as matching: `durability` (`sync` / `async` / `exit`) produces 4 / 4 / 1 checkpoints with
the same values, and multi-mode streaming (`["updates", "values", "custom"]`, v1 and v2) yields the
same sequence of parts.

Suite is 498 tests (one new), parity 67/67 (new `checkpoint_history` scenario), examples 10/10, clippy
and Rust tests clean.### Round 143: `add_node(defer=True)`

LangGraph declares a deferred node with `defer=True` and allows it to be an **entry point** and to
have **outgoing / conditional edges**. The extension used a different keyword (`deferred`) and
rejected both situations at validation:

```
TypeError: StateGraph.add_node() got an unexpected keyword argument 'defer'
RuntimeError: graph validation failed: deferred node `d` cannot be an entry point
```

`defer` is now accepted (`deferred` kept as an alias) and both restrictions are gone — the scheduler
already runs deferred tasks only when no regular task is pending, which is exactly LangGraph's rule.
Verified end to end: input `{'log': [], 'value': 0}` yields `{'log': ['e'], 'value': 1}`, the deferred
node running after the regular one.

The Rust test that encoded the old restrictions now asserts the LangGraph behaviour instead.

Also verified as matching: `Command(update=..., goto=...)` returns `{'value': 11}` on both engines.

Suite is 500 tests (two new), parity 68/68 (new `defer` scenario), examples 10/10, clippy and Rust
tests clean.### Round 144: `interrupt()` inside a tool

LangGraph lets a tool call `interrupt()`: the graph pauses, `__interrupt__` carries the value, and a
resume continues inside the tool. The extension treated the `GraphInterrupt` exception as a **tool
failure**, so `handle_tool_errors` turned it into an error `ToolMessage` — the graph never paused, the
agent kept calling the tool again, and the run only stopped at the recursion limit (it looked like a
hang).

Two changes:

1. `tool_definitions` hands `GraphInterrupt` to `pyerr_to_error` (producing `Error::Interrupt`)
   instead of flattening it into a `repr` string;
2. `ToolNode.execute` propagates `Error::Interrupt` directly, bypassing the `handle_tool_errors`
   policy.

Verified identical to LangGraph:

```
first invoke  -> __interrupt__ = [{'question': 'why?'}]
resume('yes') -> message tail ['answer=yes', 'done']
```

Also checked and matching: `compile(interrupt_before=[...])`, `compile(interrupt_after=[...])`,
conditional edges with a `path_map` dict, and three-way `stream_mode` mixes. `add_edge(START,
["x","y"])` is rejected by LangGraph (`unhashable type: 'list'`) but accepted here — a superset, no
action needed.

Suite is 501 tests (one new), parity 69/69 (new `tool_interrupt` scenario), examples 10/10, clippy and
Rust tests clean.### Round 145: `interrupt(response_schema=...)` accepts classes; invalid node output

* `interrupt(response_schema=...)` only accepted a JSON-schema dict; passing a pydantic/TypedDict
  class (what LangGraph documents) raised `TypeError: value of type 'type' cannot be converted to
  JSON`. The class is now converted to its JSON schema for the interrupt record, so
  `snapshot.interrupts[0].response_schema` shows `{'title': 'Answer', ...}`.
* A node returning something that is not an update mapping now raises `InvalidUpdateError: Expected
  dict, got ['not-a-dict']` with LangGraph's troubleshooting link, instead of
  `RuntimeError: node 'b' failed: ...`.

Documented difference: when the resumed value does not satisfy the schema, LangGraph raises pydantic's
`ValidationError` (`1 validation error for Answer ...`) while the engine validates against the JSON
schema in Rust and raises `RuntimeError: resume value is missing required response_schema property
'answer'`. Both fail; only the exception type differs.

Also checked and matching: `get_state_history` with `limit` / `before` / `filter` on a three-turn
thread (identical step numbers and counts).

Suite is 504 tests (three new), parity 69/69, examples 10/10, clippy and Rust tests clean.### Round 146: node exceptions are re-raised unchanged

LangGraph hands the caller the node's **original** exception: `ValueError("kaboom")` arrives as
`ValueError: kaboom`, even after retries are exhausted. The extension wrapped everything in
`RuntimeError: node 'b' failed: python node raised ValueError: ValueError: kaboom`, so user code doing
`except ValueError` / `except ValidationError` never matched.

`pyerr_to_error` now stores the exception object in a global table and leaves an invisible marker in
the message; `to_pyerr` swaps the marker back for the exception as its last step. Verified:

```
ValueError("kaboom")            -> ValueError: kaboom
retries exhausted (max 2)       -> ValueError: kaboom
KeyError("missing")             -> KeyError: 'missing'
```

Fixed along the way: task-level timeouts (`Send(..., timeout=...)`) bypassed the `NodeTimeoutError`
mapping (only the node policy was consulted), which the parity suite caught as
`send_timeout` regressing. They now raise `NodeTimeoutError: Node 'worker' exceeded its run timeout of
0.050s ...` again.

Known difference, documented: after a node fails, LangGraph's
`snapshot.tasks[i].error` holds the message (`"ValueError('kaboom')"`) while the extension reports
`None`, because a failed run writes no checkpoint carrying that task error.

Suite is 507 tests (three new), parity 70/70 (new `node_exception` scenario), examples 10/10, clippy
and Rust tests clean.### Round 147: `snapshot.tasks[i].error` after a failure

LangGraph keeps the failing task's error on the previous checkpoint, so `get_state()` after a failed
run reports it without adding a history entry:

```
values: {'value': 1} | next: ('b',) | tasks: [('b', "ValueError('kaboom')")]
history steps: [1, 0, -1]
```

The extension reported `tasks[0].error == None`. The failure path now writes the error back onto the
previous checkpoint's pending task and re-puts it under the **same** checkpoint id (which the saver
already treats as an update, so history does not grow). The error is rendered as LangGraph does — the
exception's `repr` — which required the exception stash to keep entries after the exception has been
re-raised for the caller (bounded to 64 entries).

Verified identical: values, `next`, `(task.name, task.error)` and the history step list, all in one
parity scenario.

Suite is 508 tests (one new), parity 71/71 (new `failed_task` scenario), examples 10/10, clippy and
Rust tests clean.### Round 148: tool error handlers receive the real exception

LangGraph hands `handle_tool_errors`' callable the **exception object**, so
`type(error).__name__` is `ValueError` and `str(error)` is `kaboom`. The extension built a
`RuntimeError` from the message instead, so both the type and any `except` clause behaved
differently.

Tool errors now carry the exception the same way node errors do: the object is stashed and the message
gets an invisible marker. The callable receives the original `ValueError('kaboom')`, while the default
message stays `Error: ValueError('kaboom')\n Please fix your mistakes.` — the marker is stripped when
the message is formatted, so nothing leaks into model-visible content.

Still open, documented: `handle_tool_errors=<exception type or tuple>` cannot filter by type (the
policy sees the message, not the exception) and `RetryPolicy(retry_on=<callable>)` matches by name
only.

Suite is 508 tests, parity 71/71 (the `tool_errors` scenario now also compares the exception class the
handler sees), examples 10/10, clippy and Rust tests clean.### Round 149: `handle_tool_errors=<exception type>` filters by type

LangGraph's `handle_tool_errors` accepts an exception type (or tuple of them) and only converts
**matching** failures into a `ToolMessage`; anything else propagates. The extension treated every
type as "handle everything", so a `KeyError` was swallowed by `handle_tool_errors=ValueError`.

The policy now checks `isinstance` against the stashed exception. All four shapes match LangGraph:

```
ValueError              -> Error: ValueError('kaboom')   (handled)
(ValueError, KeyError)  -> Error: ValueError('kaboom')   (handled)
KeyError                -> ValueError: kaboom            (propagates)
(KeyError,)             -> ValueError: kaboom            (propagates)
```

Still open, documented: `RetryPolicy(retry_on=<callable>)` matches by name only, since the engine
decides retries without calling back into Python.

Suite is 511 tests (three new), parity 71/71 (the `tool_errors` scenario now covers type matching,
tuples and non-matching types), examples 10/10, clippy and Rust tests clean.### Round 150: `get_config()` matches LangGraph's RunnableConfig view

Inside a node, LangGraph's `get_config()` returns `callbacks` / `configurable` / `metadata` / `tags`,
where `metadata` merges the caller's metadata with `thread_id`, `langgraph_step`, `langgraph_node`,
`langgraph_triggers`, `langgraph_path` and `langgraph_checkpoint_ns`. The extension returned only
`{"configurable": {"thread_id", "checkpoint_ns"}}` — so `config["metadata"]`, `config["tags"]` and the
`langgraph_*` lookups raised `KeyError`.

`RunConfig` now carries `tags` and `metadata` (parsed from the Python config dict), the run context
stores them, and `get_config()` rebuilds the full view, including tuple-valued `langgraph_path` /
`langgraph_triggers` as LangGraph reports them. `langgraph_rs.config.RunnableConfig` was also missing
and is re-exported now.

A module-surface audit found every other `langgraph.*` module path importable from `langgraph_rs.*`.

Suite is 513 tests (two new), parity 72/72 (new `get_config` scenario), examples 10/10, clippy and Rust
tests clean.### Round 151: `create_react_agent` signature parity

LangGraph's deprecated `create_react_agent` has an explicit parameter list
(`model, tools, *, prompt, response_format, state_schema, config_schema, checkpointer, store,
interrupt_before, interrupt_after, debug, version, name`). The extension exposed
`(*args, **kwargs)`, so `inspect.signature` showed nothing and a misspelled parameter was silently
forwarded.

The shim now declares the same parameters and forwards them; `config_schema` maps to the engine's
`context_schema` (an explicit `context_schema` wins). Behaviour verified against LangGraph with a
stub model: identical message flow `[human, ai(tool_call), tool('3'), ai('3')]` and identical graph
nodes.

Note for the next rounds: message **values** are still returned as dicts (`role: user/ai/tool`)
where LangGraph returns LangChain message objects (`type: human/ai/tool`). That is the largest
remaining structural difference and needs the schema to mark `add_messages` channels so the adapter
can build real message objects on the way out.

Suite is 515 tests (two new), parity 73/73 (new `react_helper` scenario), examples 10/10, clippy and
Rust tests clean.### Round 152: message-object conversion — groundwork (disabled by default)

LangGraph's `add_messages` channels hold LangChain message objects (`HumanMessage`, `AIMessage`,
`ToolMessage`); the extension has always returned plain dicts (`{'role': 'user', ...}`). This round
lays the foundation:

* `StateSchema` records which channels are `add_messages` channels
  (`with_messages_channel` / `messages_channel_names`);
* the run registers those names in a process-wide registry (async graphs convert on a different
  thread than the one installing the run scope, so a thread-local is not enough);
* `json_state_to_py` converts message channels with
  `langchain_core.messages.convert_to_messages`;
* `_set_messages_objects(bool)` toggles the behaviour, **default off** so nothing changes yet.

With the switch on, `invoke` returns `['HumanMessage', 'AIMessage']` and `message.content` works; with
it off the previous dict shape is preserved. Both verified in one run.

Why it is off: turning it on breaks roughly 14 places in the extension's own Python facade and tests
(`ValidationNode`, prompt handling, `MessageGraph`, message-dict assertions) that read messages as
dicts. Those need migrating to LangGraph's attribute API first; the next round does that and then
flips the default.

Suite is 515 tests, parity 73/73, examples 10/10, clippy and Rust tests clean (switch off).### Round 153: `add_messages` channels return LangChain message objects

The last big structural gap. LangGraph stores LangChain message objects in `add_messages` channels;
the extension returned plain dicts (`{'role': 'user', 'content': ...}`), so the idiomatic
`state["messages"][-1].content` raised `AttributeError`.

Now identical to LangGraph:

```
input  {'messages': [{'role': 'user', 'content': 'hi'}]}
node   HumanMessage, .content == 'hi'
result [HumanMessage('hi'), AIMessage('hello')], .type == 'human' / 'ai'
```

The conversion is schema driven: `StateSchema` marks `add_messages` channels, the run registers the
names in a process-wide registry (async graphs convert on another thread), and
`json_state_to_py` uses `langchain_core.messages.convert_to_messages`. Plain channels are untouched.

Migration work that came with it: the extension's own facade and ~15 tests read messages as dicts
(`ValidationNode`, `MessageGraph`, prompt injection, tool injection, `ToolNode` error assertions) and
now use LangGraph's attribute API; the parity scenarios were made representation-agnostic through a
`message_field` helper. `langgraph_rs._set_messages_objects(False)` keeps the old dict shape available
as a transitional escape hatch (default on).

Suite is 518 tests (three new), parity 74/74 (new `message_objects` scenario), examples 10/10, clippy
and Rust tests clean.### Round 154: the rest of the `BaseCheckpointSaver` surface

`InMemorySaver` was missing `serde`, `config_specs`, `with_allowlist`, `aget`, `get_delta_channel_history`
and the `prune` / `delete_for_runs` family, so LangGraph code touching them hit `AttributeError`
instead of the documented behaviour. LangGraph's own `InMemorySaver` raises `NotImplementedError` for
`prune` and `delete_for_runs`, so the extension now matches that instead of silently succeeding.

Added and verified against LangGraph: `get`/`aget` (raw checkpoint dict), `get_delta_channel_history`
(`{channel: {"writes": [...], "seed": 0}}`), `config_specs == []`, a real `JsonPlusSerializer` behind
`serde`, `with_allowlist` returning the saver, and `NotImplementedError` for the unsupported calls.

Known difference, documented: `get_delta_channel_history`'s `writes` list is normally empty because the
engine snapshots full values per superstep rather than keeping a delta log; only genuine pending writes
(interrupts) show up there.

Suite is 522 tests (four new), parity 75/75 (new `saver_surface_2` scenario), examples 10/10, clippy and
Rust tests clean.### Round 155: the functional API takes and returns values

LangGraph's `@entrypoint` is value-in/value-out and accepts compiler options
(`entrypoint(checkpointer=..., store=..., context_schema=...)`). The extension accepted only
`entrypoint(function)` and handed the function the whole state **mapping**, so
`entrypoint(checkpointer=saver)` raised `TypeError` and `entrypoint(fn).invoke(3)` raised
`graph state must be a dict`.

Both now match LangGraph:

```
@task
def double(x): return x * 2

@entrypoint(checkpointer=InMemorySaver())
def workflow(x): return double(x).result() + 1

workflow.invoke(3, cfg)        -> 7
interrupt()                    -> __interrupt__ = [{'q': 'value?'}]
invoke(Command(resume=7))      -> 7
```

How: a facade adapts value semantics onto a single-channel graph and translates in `invoke` /
`ainvoke` / `stream` / `astream` / `get_state` / `update_state` / `resume` (a `Command` input passes
through untouched); `@task` results always expose `.result()` / `.aresult()` even when the engine
already resolved them, and futures embedded in a returned structure are unwrapped; the top-level
`entrypoint` / `task` names now point at these facade implementations.

Three tests that encoded the old mapping-state API (`entry(fn).compile(...)`, `head + tail` on raw
task results) were migrated to LangGraph's form, and the parity functional scenario with them.

Suite is 522 tests, parity 75/75, examples 10/10, clippy and Rust tests clean.### Round 156: `update_state` infers the author and returns a config

LangGraph's `update_state(config, values)` infers `as_node` from `versions_seen` when it is omitted
and returns the **new checkpoint's config**. The extension raised
`Ambiguous update, specify as_node` for the same call and returned a `StateSnapshot`.

Root cause of the false ambiguity: this engine records `versions_seen` **per channel** (both `log` and
`value` share the version of one write), while LangGraph records one entry per node. The duplicate
versions made the "two nodes wrote at once" check fire. `infer_as_node` now keeps the maximum version
per node before comparing, which reproduces LangGraph's answer (`b`, the last writer) and still detects
genuine ties.

`update_state` / `aupdate_state` now return the `{'configurable': {...}}` config, so callers read the
state back with `get_state` as LangGraph documents. Tests and the parity scenario were updated to that
flow (the scenario now also compares the returned keys).

Suite is 524 tests (two new), parity 75/75, examples 10/10, clippy and Rust tests clean.### Round 157: `bulk_update_state` input and return shape

LangGraph's `bulk_update_state(config, supersteps)` takes a list of supersteps where each step is a
list of `(values, as_node, task_id)` **tuples**, and returns the new checkpoint's config. The
extension only accepted `{"values": ..., "as_node": ...}` dictionaries and returned a
`StateSnapshot`.

Tuples (LangGraph's native form) are now accepted next to the dictionaries the extension already
supported, and the call returns the `{'configurable': {...}}` config:

```
bulk_update_state(cfg, [[({"value": 7}, "a", None)]])
-> {'configurable': {...}}; get_state(cfg) -> values {'value': 7}, next ('b',)
```

Also verified while here: `StateSnapshot` exposes `values` / `next` / `config` / `metadata` /
`created_at` / `parent_config` / `tasks` / `interrupts` like LangGraph (the extension additionally
supports mapping-style subscripts).

Suite is 526 tests (two new), parity 75/75 (the `bulk_update_state` scenario now also compares the
returned config keys), examples 10/10, clippy and Rust tests clean.### Round 158: `RemoteGraph` public surface matches LangGraph exactly

The remote-graph client implemented `invoke` / `stream` / `get_state` / `update_state` and a few
helpers, but lacked the `Runnable` protocol (`batch`, `bind`, `as_tool`, `with_retry`,
`astream_events`, …) and `bulk_update_state`. LangGraph gets those by subclassing `Runnable`, so the
extension does the same now and adds `bulk_update_state` / `abulk_update_state` on top.

A set comparison against LangGraph's class is now empty in both directions:

```
missing vs LG: []
extra:         []
```

`get_name()`, `input_schema`, `output_schema` and `config_specs` all resolve as they do upstream; the
input schema is actually more robust here (LangGraph raises `SchemaError` for a client-less
`RemoteGraph`, the extension generates the model).

Two tests cover the surface equality and that `bulk_update_state` forwards to the SDK client.

Suite is 528 tests (two new), parity 75/75, examples 10/10, clippy and Rust tests clean.### Round 159: `RetryPolicy(retry_on=<callable>)`

LangGraph's `retry_on` accepts an exception type, a sequence of types, **or** a
`Callable[[Exception], bool]` predicate. The extension only turned `retry_on` into a list of exception
names, so a predicate never matched (`<lambda>` is not an exception name) and retries silently stopped
after one attempt — `retry_on=lambda exc: isinstance(exc, ValueError)` ran once instead of three times.

The Rust `RetryPolicy` gained `condition: Option<RetryCondition>` (`Arc<dyn Fn(&str) -> bool>`), which
the Python adapter fills in: the engine calls back with the error message only, the message carries the
exception's marker, and the adapter resolves it back to the original exception before invoking the
user predicate. That keeps the "no Rust → shim calls" rule intact — the callback is a user-provided
callable.

All five shapes now match LangGraph (predicate true/false, type, non-matching type, sequence), and
`repr(RetryPolicy(retry_on=...))` is byte-identical for the type and sequence forms.

Note for future benchmark work: a parity `measure()` loop around retries replays the real backoff
sleeps; the scenario measures policy construction instead.

Suite is 532 tests (four new), parity 76/76 (new `retry_on` scenario), examples 10/10, clippy and Rust
tests clean.### Round 160: subgraph debug chunks keep their full record

With `stream_mode="debug", subgraphs=True`, LangGraph yields `(namespace, chunk)` where the chunk is
the **complete** debug record (`type`, `timestamp`, `step`, `payload`). The extension only wrapped the
payload, so `chunk["type"]` was missing:

```
LG:   [([], 'task'), (['sub:...'], 'task'), (['sub:...'], 'task_result'), ([], 'task_result')]
was:  [([], None), (['sub:...'], None), ...]
now:  identical, every chunk carrying all four fields
```

Also verified as matching while here: `stream_mode="tasks", subgraphs=True` (names and namespaces) and
`astream_events(version="v2")` (the full `on_chain_start` / `on_chain_stream` / `on_chain_end`
sequence, including the `LangGraph` root).

Cosmetic differences documented rather than fixed: `get_graph().draw_mermaid()` emits ordinary edges
before conditional ones (LangGraph keeps declaration order), `Graph.nodes[...].data` is `None`, and the
compiled graph is missing the `nodes` / `attach_node` / `attach_edge` / `attach_branch` Pregel
internals.

Suite is 533 tests (one new), parity 77/77 (new `subgraph_debug` scenario), examples 10/10, clippy and
Rust tests clean.### Round 161: `ToolRuntime` injection

LangGraph injects a `ToolRuntime` handle into tools that declare `runtime: ToolRuntime` and hides the
parameter from the model-facing schema. The extension injected the handle, but `config` came through
empty and `tool.tool_call_schema` raised `PydanticInvalidForJsonSchema` because the parameter was not
recognised as injected.

Now identical to LangGraph: the schema exposes only `['query']` and the tool sees
`state=['messages'] cfg=True writer=True`.

How: the Python `ToolRuntime` subclasses langchain's `_DirectlyInjectedToolArg` (which is what makes
`@tool` drop the parameter) and declares a pydantic core schema that accepts the handle the engine
builds in Rust — so the parameter is hidden without Rust importing the shim's class. The engine also
fills `config` (the run's `get_config()` view) and a `stream_writer`.

Suite is 535 tests (two new), parity 78/78 (new `tool_runtime` scenario), examples 10/10, clippy and
Rust tests clean.### Round 162: graph-construction errors are `ValueError` with LangGraph's wording

LangGraph reports every graph-construction problem as a `ValueError`; the extension raised
`RuntimeError: graph validation failed: ...`, so `except ValueError` never matched. `Error::Validation`
now maps to Python's `ValueError` without the internal prefix, and the common messages were aligned
word for word:

```
no entry point      -> Graph must have an entrypoint: add at least one edge from START to another node
unknown edge target -> Found edge ending at unknown node `b`
unknown branch      -> At 'a' node, 'condition' branch found unknown target 'other'
duplicate node      -> Node `a` already present.
```

One test that asserted the old `RuntimeError` was updated, and a new parity scenario compares the
exception type and message for all four cases.

Suite is 539 tests (four new), parity 79/79 (new `validation_errors` scenario), examples 10/10, clippy
and Rust tests clean.### Round 163: deprecation warnings match LangGraph

LangGraph warns with `LangGraphDeprecatedSinceV10` for `ValidationNode` and `MessageGraph` — the
category users filter on. The extension stayed silent, so
`warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)` had no effect on it.

Both now warn with LangGraph's exact wording and category, verified by a parity scenario that
captures the warnings from both engines and compares `(category, message)` pairs.

Also noted: LangGraph 1.2 removed `create_tool_calling_executor` (it raises `ImportError`); the
extension still ships the alias, which is a harmless superset.

Suite is 541 tests (two new), parity 80/80 (new `deprecations` scenario), examples 10/10, clippy and
Rust tests clean.### Round 164: no-checkpointer runs, `Command` fields, fractional cache TTL

Three divergences found by probing LangGraph directly:

1. **`thread_id` without a checkpointer** — LangGraph ignores the setting and runs normally; the
   extension raised `RuntimeError: a run bound to a thread_id requires a checkpointer`. Only
   `Command(resume=...)` genuinely needs stored state, so that case now raises LangGraph's
   `RuntimeError: Cannot use Command(resume=...) without checkpointer`, and `get_state` reports
   `ValueError: No checkpointer set` (also matched). A Rust test that asserted the old behaviour was
   updated.
2. **`Command` fields** — LangGraph's dataclass exposes `update` / `resume` / `goto` / `graph`; the
   native class only had `PARENT`, so `cmd.resume` raised `AttributeError`. All four are readable now
   (`dir()` matches LangGraph exactly).
3. **`CachePolicy(ttl=...)`** — LangGraph's TTL is in seconds and accepts fractions; the extension
   required an integer and raised `TypeError: 'float' object cannot be interpreted as an integer` on
   `ttl=0.2`. It is now an `f64` of seconds, and expiry behaves like LangGraph (`calls == [1, 1]`).

Suite is 545 tests (four new), parity 81/81 (new `no_checkpointer` scenario), examples 10/10, clippy
and Rust tests clean.### Round 165: `Send`, `Interrupt` and `TimeoutPolicy` attribute surfaces

A class-by-class `dir()` comparison against LangGraph turned up missing readable attributes:

* `Send` — `arg` and `timeout` were unreadable. LangGraph normalises a plain number into a
  `TimeoutPolicy`, so `Send("n", arg, timeout=0.5).timeout` is a policy; the native class now does the
  same while the engine keeps the resolved seconds internally.
* `Interrupt` — added `interrupt_id` (an alias of `id`) and `from_ns(value, ns)`, which builds a stable
  per-(namespace, value) id.
* `TimeoutPolicy` — `repr` now prints LangGraph's single-quoted form
  (`refresh_on='auto'` instead of `"auto"`).

Documented as still different: `StateSnapshot` is a dict subclass rather than a NamedTuple (attributes
read fine, `dir()`/`isinstance` differ), `InMemoryStore` lacks `embeddings`/`index_config`,
`InMemorySaver` lacks `put`/`put_writes` (the engine writes internally), and `StateGraph` lacks
`add_sequence`/`set_node_defaults`/`validate`.

Suite is 548 tests (three new), parity 81/81, examples 10/10, clippy and Rust tests clean.### Round 166: `StateGraph` builder helpers and store index attributes

Closing the rest of the class diff from the previous round (facade-only work):

* `StateGraph.add_sequence(nodes)` — registers each node (callable or `(name, callable)`) and chains
  them, returning `self`; the entry point stays the caller's job, exactly like LangGraph.
* `StateGraph.set_node_defaults(...)` — stores `retry_policy` / `cache_policy` / `error_handler` /
  `timeout` and applies them to nodes registered afterwards (verified: a node added after
  `set_node_defaults(retry_policy=RetryPolicy(max_attempts=2))` retries twice).
* `StateGraph.validate(interrupt=None)` — runs the compile-time validation and returns `self`,
  raising `ValueError` for an invalid graph.
* `InMemoryStore.embeddings` / `index_config` — expose the constructor's index settings (`None`
  without one).

Still different, documented: `StateSnapshot` is a dict subclass rather than a NamedTuple, and
`InMemorySaver` has no `put`/`put_writes` (the engine writes through its own path).

Suite is 553 tests (five new), parity 81/81, examples 10/10, clippy and Rust tests clean.

### Round 167: `messages` stream keeps complete messages whole

LangGraph's `stream_mode="messages"` yields the **message itself** (`AIMessage`) when the model was
not streamed token by token, and an `AIMessageChunk` only for real token streaming. The extension
always produced a chunk, so `type(chunk).__name__` differed for non-streaming models.

The callback bridge now distinguishes the two paths (`on_llm_new_token` → chunk, `on_llm_end`
without streamed tokens → complete message) and `message_chunk` builds the matching class:

```
non-streaming model -> ('AIMessage', 'hello')
token streaming     -> ('AIMessageChunk', 'Hello') / ('AIMessageChunk', ' ') / ...
```

Also added `langgraph_rs.checkpoint.base.RunnableConfig`, which LangGraph exports from that module.

Suite is 554 tests, parity 81/81, examples 10/10, clippy and Rust tests clean.

### Round 168: `BaseCheckpointSaver.put` and `put_writes`

The saver had no write API: LangGraph's `put(config, checkpoint, metadata, new_versions)` and
`put_writes(config, writes, task_id)` were documented as missing because the engine writes through its
own path. Both are implemented now on top of a native `put_document` that accepts LangGraph's
checkpoint document shape (`checkpoint` + `metadata` + optional `pending_writes`).

Verified against LangGraph on the same sequence — re-putting the fetched checkpoint keeps its id and
leaves the history length unchanged (3), and `put_writes` appends
`[('task-1', 'value')]` to the pending writes.

Suite is 556 tests (two new), parity 81/81, examples 10/10, clippy and Rust tests clean.

### Round 169: `StateSnapshot` is LangGraph's named tuple

The snapshot used to be a dict subclass: field attributes read fine, but `isinstance(snapshot, tuple)`,
tuple unpacking and the `dir()` shape all differed from LangGraph.

`langgraph_rs.types.StateSnapshot` is now a `NamedTuple` with LangGraph's exact eight fields
(`values`, `next`, `config`, `metadata`, `created_at`, `parent_config`, `tasks`, `interrupts`), built at
the facade boundary from the engine's JSON snapshot. The extension keeps its convenience layer on top:
mapping-style `snapshot["values"]`, `in`, `keys()`/`get()`, and the derived `step` / `thread_id` /
`checkpoint_ns` / `parent_checkpoint_id` / `interrupted` properties. `get_state_history` still returns a
list (LangGraph yields an iterator) so older callers keep index access, and `filter` is applied in the
facade for async graphs, which have no server-side filter.

Suite is 559 tests (three new), parity 81/81, examples 10/10, clippy and Rust tests clean.

### Round 170: `get_graph()` keeps LangGraph's edge order

`get_graph()` emitted ordinary edges first and conditional ones last, while LangGraph follows the
declaration order of `add_edge` / `add_conditional_edges`. The facade now records that order on the
builder, hands it to the compiled graph, and sorts the rebuilt `Graph` accordingly.

Verified against LangGraph on a mixed graph: the `edges` list and the **entire** `draw_mermaid()`
document are now byte-identical.

Remaining cosmetic difference: `Graph.nodes[...].data` is `None` where LangGraph exposes its internal
`PregelNode` repr.

Suite is 561 tests (two new), parity 82/82 (new `drawing` scenario), examples 10/10, clippy and Rust
tests clean.

### Round 171: `Runtime` injection and async nodes in a sync builder

Two real gaps found by probing LangGraph directly:

1. **`Runtime` injection** — LangGraph injects a `Runtime` into any node that declares one
   (`def node(state, runtime: Runtime)`); the extension called the node with the state only, so such a
   node died with `TypeError: node() missing 1 required positional argument: 'runtime'`. The facade now
   wraps such nodes (sync and async) and builds the `Runtime` from the running task's context, store and
   stream writer. Documented difference: `runtime.store` is a working handle but not the very object the
   graph was compiled with, so `isinstance(runtime.store, InMemoryStore)` is `False`.
2. **Async nodes in a sync builder** — LangGraph accepts `async def` nodes in `StateGraph` and runs them
   under `ainvoke`; the extension raised `InvalidUpdateError: Expected dict, got <coroutine object ...>`.
   The sync node path now drives an awaitable to completion on the same private loop the async builder
   uses. Deliberate superset: LangGraph's synchronous `invoke` rejects an async node with
   `TypeError: No synchronous function provided`, while the extension also runs it.

Suite is 566 tests (five new), parity 82/82, examples 10/10, clippy and Rust tests clean.

### Round 172: `context_schema` on the builder and Python contexts

LangGraph takes the runtime context schema on the builder (`StateGraph(S, context_schema=Ctx)`) and
hands the object passed as `invoke(context=...)` straight to `Runtime.context`. Neither worked here:
the builder rejected the keyword with `TypeError`, and a dataclass context died with
`TypeError: value of type Ctx cannot be converted to JSON`.

The builder now stores the schema and passes it to `compile`, and the context travels as a reference:
when a value has no JSON form the engine stashes the Python object and carries
`{"__langgraph_rs_object__": id}`, resolving it back on the Python side (the stash is capped at 4096
entries). Such a context also skips the engine's JSON-schema validation, since Python already checked
it. `runtime.context` is now the very object the caller passed, and `get_context_jsonschema()` matches
LangGraph's output.

Suite is 568 tests (two new), parity 83/83 (new `context_schema` scenario), examples 10/10, clippy and
Rust tests clean.

### Round 173: arbitrary Python objects in state

LangGraph state is not JSON-only: `invoke({"point": Point(1)})` and nodes returning a `set` work with
and without a checkpointer. The extension refused them with
`TypeError: value of type `Point` cannot be converted to JSON`.

Values without a JSON form are now carried by reference through the same stash the runtime context
uses: the engine stores a `{"__langgraph_rs_object__": id}` placeholder and `value_to_py` hands the
original object back, so a round trip preserves identity. Placeholders are deduplicated by object
address, and callables/modules are still rejected loudly (they are code, not data).

Known limitation, documented: the objects live in this process, so a checkpoint written to disk and
resumed in another process only carries placeholders — LangGraph serialises them with msgpack instead.

Suite is 572 tests (four new), parity 84/84 (new `python_state` scenario), examples 10/10, clippy and
Rust tests clean.

### Round 174: undeclared state keys are dropped

LangGraph keeps only the channels the state declares: extra input keys and extra keys in a node's
update are silently ignored. The extension turned them into implicit channels, so they showed up in
`values` (`{"value": 0, "typo": 5}` came back as `{'typo': 5, 'value': 1}`).

Both the input filter and the write path now ignore undeclared channels whenever the schema declares
any; a fully dynamic schema (no declared channels, the `JsonState` style) stays permissive. One test
relied on the looser behaviour (it passed `user` to a state that only declared `messages`) and now
declares that channel, as LangGraph requires.

Also noted: LangGraph rejects dict schemas outright (`StateGraph({"value": "last_value"})` raises
`TypeError: unhashable type: 'dict'`), so that convenience remains an extension-only superset.

Suite is 575 tests (three new), parity 85/85 (new `unknown_channels` scenario), examples 10/10, clippy
and Rust tests clean.

### Round 175: `source="input"` checkpoints carry the thread's previous values

Running two turns and comparing `get_state_history` showed the input checkpoint of the second turn was
empty here but carried the previous state in LangGraph (`{'value': 1}` at step 2). The engine now seeds
that checkpoint from the latest one (empty on the first run), so every row — step, source, values,
`next` — matches LangGraph's exactly.

Verified as already equal in the same pass: `durability` history lengths (`sync` 4, `async` 4, `exit`
1), time travel via `invoke(None, older_config)`, and unknown-key dropping in `Command(update=...)` and
`update_state`.

Suite is 577 tests (two new), parity 86/86 (new `checkpoint_shape` scenario), examples 10/10, clippy and
Rust tests clean.

### Round 176: error text, troubleshooting codes and the empty-thread snapshot

Three small differences found by comparing error paths:

* A node returning a non-dict now reports `Expected dict, got not a dict` (the value's text, not its
  repr, as LangGraph does) and its troubleshooting URL points at `INVALID_GRAPH_NODE_RETURN_VALUE`
  instead of `INVALID_CONCURRENT_GRAPH_UPDATE`, which stays for real concurrent-write conflicts.
* `get_state()` on a thread with no checkpoints now echoes the requested config
  (`{'configurable': {'thread_id': ...}}`) like LangGraph; it used to be empty.
* A bare `interrupt()` outside a node keeps the extension's clearer message
  (`interrupt() can only be called from inside a running graph node`) rather than LangGraph's
  `Called get_config outside of a runnable context`, which leaks an internal detail. Same type.

Confirmed equal in the same pass: parallel interrupts resumed by id map
(`Command(resume={id: value})`), `Command(graph=Command.PARENT)` handoff from a subgraph, and the
`update_state` history rows.

Suite is 580 tests, parity 86/86, examples 10/10, clippy and Rust tests clean.

### Round 177: `astream_events` emits the graph chunk per superstep

LangGraph emits the graph-level `on_chain_stream` after **every** superstep (two nodes: two chunks),
and always *after* that step's node events. The extension emitted a single graph chunk at the end.

The event builder now holds the step's `values` payload until its node events are out, then emits the
graph chunk, and drops the trailing values-only chunk that has no step of its own. The full event
sequence for a two-node graph is now identical to LangGraph's, order included.

Confirmed equal in the same pass: the recursion-limit error text (byte for byte), `NodeTimeoutError`
(only the measured `elapsed` differs) and the `astream_events` field set.

Suite is 581 tests, parity 87/87 (new `stream_events` scenario), examples 10/10, clippy and Rust tests
clean.

### Round 178: nested subgraph events and the graph chunk payload

Two more `astream_events` gaps:

* Subgraphs emitted no events of their own. The event builder now streams with `subgraphs=True` and
  opens a child `LangGraph` run per namespace, so the sequence and `parent_ids` depth match
  LangGraph's: parent node start → child graph start → child node events → child graph end → parent
  stream/end.
* The graph-level `on_chain_stream` carried the whole state; LangGraph puts that superstep's updates
  there (`{'a': {'value': 1}}`). Fixed.

Still different (documented): subgraph event `metadata` is thinner than LangGraph's (no
`langgraph_step` / `langgraph_triggers`), and the child `LangGraph` `on_chain_end` lacks LangGraph's
`input` field.

Suite is 584 tests (three new), parity 88/88 (new `subgraph_events` scenario), examples 10/10, clippy
and Rust tests clean.

### Round 179: node event metadata carries the superstep

Node events lacked LangGraph's metadata. The engine's `updates` stream chunk already knows its
superstep, so it is now exposed on the stream part and the event builder fills `ls_integration`,
`langgraph_step` and `langgraph_node` (plus `langgraph_checkpoint_ns` for subgraphs) with the same
values LangGraph uses — step 1 for the first node, 2 for the next.

Still missing, documented: `langgraph_triggers`, `langgraph_path`, and the task-scoped
`langgraph_checkpoint_ns` (`a:<task-id>`); those need engine task ids in the stream, and the child
`LangGraph` `on_chain_end` lacks LangGraph's `input` field.

Suite is 585 tests, parity 88/88 (the `stream_events` scenario now also compares a metadata subsequence),
examples 10/10, clippy and Rust tests clean.

### Round 180: task-scoped event metadata and per-task runs

Two refinements to `astream_events`:

* The `updates` stream chunk now carries its task id, so node event metadata can include
  `langgraph_checkpoint_ns` (`<node>:<task-id>`, matching LangGraph's shape) and `langgraph_path`
  (`["__pregel_pull", <node>]`). Metadata is now LangGraph's key set minus `langgraph_triggers`.
* Runs are keyed per task instead of per node, so parallel `Send`s to the same node produce one run
  each (distinct `run_id` and namespace), the way LangGraph reports them.

Still different, documented: `langgraph_triggers` (LangGraph distinguishes edge-driven
`branch:to:<node>` from `__pregel_push`, which the stream does not carry), the interleaving order of
parallel tasks' events within a superstep, and the child `LangGraph` `on_chain_end` payload.

Suite is 587 tests (two new), parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 181: the extension no longer imports the Python shim

The architecture rule says Rust must not call back into `langgraph_rs.*`. Nine reverse imports
existed:

| Old | New |
|---|---|
| `py.import("langgraph_rs.errors")` ×6 (`InvalidUpdateError`, `NodeTimeoutError`) | the shim registers its exception classes at import (`_register_error_classes`); Rust looks them up |
| `py.import("langgraph_rs._compat").message_chunk` | the shim registers the helper (`_register_shim_hook("message_chunk", ...)`) |
| `py.import("langgraph_rs.types").PregelTask` | same registry, `"pregel_task"` |

`grep 'import("langgraph_rs'` over `src/` is now empty, and the raised exceptions are still the shim's
own classes (`isinstance` / `except` unchanged, verified by the suite).

One documented exception: the callback handler is Python source embedded in `mod.rs`; its
`import langgraph_rs._live` runs as Python (invoked by LangChain callbacks), not as a Rust call.

Suite is 587 tests, parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 182: the embedded callback handler moved into the shim

The last piece of Python living inside Rust was the ~220-line LangChain callback handler, embedded as
a Rust string and compiled with `PyModule::from_code`. It is now `python/langgraph_rs/_callbacks.py`,
an ordinary shim module; `__init__` registers a factory (`_register_shim_hook("callback_handler", ...)`)
and `langchain_callback_handler` looks it up.

`src/` now contains neither `import langgraph_rs` calls nor `PyModule::from_code`, so Rust never
executes or imports the Python package. The handler behaves identically (`stream_mode="messages"`
still yields the completed `AIMessage` for a non-streaming model).

Suite is 587 tests, parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 183: `InMemoryStore` TTL and batch error types

Two store-level differences:

* `store.put(..., ttl=...)` was silently honoured; LangGraph's in-memory store declares
  `supports_ttl = False` and raises `NotImplementedError` with a specific message. It now raises the
  same. A test that relied on TTL expiry was rewritten to assert LangGraph's behaviour.
* `store.batch([{...}])` (a plain dict instead of an op object) raised `TypeError: unknown store op`;
  LangGraph raises `ValueError: Unknown operation type: <class 'dict'>`. Aligned.

Verified equal in the same pass: all four `list_namespaces` parameter forms, `delete`, the async twins
(`aget` / `aput` / `asearch` / `alist_namespaces` / `abatch`), missing-key `get`, and `search` with
`filter` and `query`.

Suite is 589 tests (two new), parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 184: `BaseCheckpointSaver` implements LangGraph's protocol

`langgraph_rs.checkpoint.base.BaseCheckpointSaver` was an empty placeholder, so code that subclasses
it to bring a custom checkpointer had no `get`, no async twins and no `get_next_version` defaults.
LangGraph's base class is now reproduced: all 24 public members, `get` derived from `get_tuple`,
`get_next_version` counting from 1, `with_allowlist`, `serde`, `config_specs`, async methods
delegating to the sync ones, and `NotImplementedError` for the parts a saver must supply.

A layout test that asserted the old placeholder behaviour was updated to LangGraph's semantics
(instantiable, protocol methods raise).

Suite is 593 tests (four new), parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 185: `BaseStore` implements LangGraph's protocol

Same story as `BaseCheckpointSaver` one round earlier: LangGraph's `BaseStore` only asks a subclass
for `batch` (optionally `abatch`) and derives `get` / `put` / `delete` / `search` /
`list_namespaces` plus the async twins, along with `supports_ttl`, `ttl_config` and namespace
validation. Our version was an ABC that required four methods and returned `()` from
`list_namespaces`.

It now carries LangGraph's 14 members and semantics: a store implementing only `batch` gets the whole
surface (verified with a custom in-memory store), delete is the `PutOp(value=None, ttl=None)` encoding
LangGraph uses, and a TTL raises `NotImplementedError` unless `supports_ttl` is set.

Suite is 597 tests (four new), parity 88/88, examples 10/10, clippy and Rust tests clean.

### Round 186: store ops, `Item` shape and UTC timestamps

A class-surface sweep over every public LangGraph class (73 captured) pointed at the store layer:

* `GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp` / `MatchCondition` are NamedTuples in LangGraph
  (unpackable, `count`/`index`, `_fields`); ours were dataclasses. Converted, and `TTLConfig` /
  `IndexConfig` became the TypedDicts LangGraph uses.
* `store.get()` / `store.search()` returned the native `StoreItem`; LangGraph returns `Item` and
  `SearchItem`. The shim now wraps them, `SearchItem.dict()` adds `score` while `Item.dict()` does
  not, and `Item.namespace` stays a tuple.
* Store timestamps were local-time naive; LangGraph reports UTC-aware datetimes. The native conversion
  now attaches `timezone.utc`, so `item.created_at` is `...T14:33:29+00:00` instead of a local naive
  value — a real serialisation difference.

Two tests that used dict-style item access (our old behaviour) were updated to LangGraph's attribute API.

Suite is 600 tests (three new), parity 89/89 (new `store_item` scenario), examples 10/10, clippy and
Rust tests clean.

### Round 187: `ToolNode` and `ValidationNode` are Runnables

LangGraph's `ToolNode` is a LangChain `Runnable` (it composes in LCEL chains and carries the whole
`batch` / `stream` / `as_tool` surface); ours was a plain class, 38 members short. Both node classes
now subclass `langchain_core.runnables.Runnable`, with `get_name(suffix, name=)` matching LangChain's
signature and `ValidationNode` gaining the `invoke` / `ainvoke` its abstract base requires.

`as_tool()` fails in LangGraph too (`PydanticUserError: tools_input is not fully defined`), so the
behaviour matches as-is.

Suite is 602 tests (two new), parity 89/89, examples 10/10, clippy and Rust tests clean.

### Round 188: small types and runtime helpers

Continuing the class sweep (29 gaps down to 13):

* `StateUpdate` / `CacheKey` became the NamedTuples LangGraph uses, `DeltaChannelHistory` was added,
  `langgraph_rs.types.BaseCheckpointSaver` now points at the real class instead of `Any`, and
  `channels.base` re-exports `EmptyChannelError`.
* `Runtime` gained `merge` / `override` / `patch_execution_info` / `drain_requested` / `drain_reason`;
  `ExecutionInfo` gained `patch`; `BaseUser` became the property protocol LangGraph ships.
* `RunControl` gained `request_drain` / `drain_requested` / `drain_reason` — the reason had to be
  stored in the Rust core (`runtime::control::RunControl`), which now records it alongside the abort
  flag.

Remaining gaps are protocol internals (`BaseChannel`, `ChannelProtocol`, `Pregel`, `NodeBuilder`) and
a few small members, tracked in the doc.

Suite is 605 tests (three new), parity 89/89, examples 10/10, clippy and Rust tests clean.

### Round 189: last small members from the class sweep

* `TimeoutPolicy.coerce(value)` — normalises a number / `timedelta` / policy, `None` for `None`.
* `JsonPlusSerializer.with_msgpack_allowlist` (and its `with_allowlist` alias) returns an extended
  serializer, or `self` when there is nothing to add.
* `ToolNode.name` / `tools_by_name` are declared at class level, as LangGraph's dataclass fields are.
* `checkpoint.base.DeltaChannelHistory` is the TypedDict LangGraph ships.

The sweep is down to 8 entries, all protocol internals (`Pregel` with its 69 `Runnable` members,
`NodeBuilder`, `BaseChannel` / `ChannelProtocol`, `PregelExecutableTask`), `RetryPolicy.count`/`index`
(LangGraph makes it a NamedTuple; ours is a native class with identical field access and retry
behaviour, verified against `ConnectionError`), and `ToolRuntime.emit_output_delta`.

Suite is 609 tests (four new), parity 89/89, examples 10/10, clippy and Rust tests clean.

### Round 190: the compiled graph is a `Pregel`; channel protocol completed

`isinstance(compiled_graph, Pregel)` is true in LangGraph and was false here. `Pregel` now carries
LangGraph's 69 public members and `CompiledStateGraph` is registered as a **virtual** subclass, so the
relationship holds without entering the MRO — the first attempt (plain inheritance) shadowed
`clear_cache` / `get_subgraphs` / `validate` with placeholders, which the suite caught immediately.

`BaseChannel` also gained LangGraph's protocol (`get` / `update` / `is_available` / `consume` /
`finish` / `checkpoint` / `from_checkpoint` / `copy`), with an empty channel checkpointing as
`MISSING` so `copy()` works.

The class sweep is down to 5 internal entries: `ToolRuntime.emit_output_delta`, `RetryPolicy.count`/
`index`, `PregelExecutableTask`, `ChannelProtocol`, `NodeBuilder`.

Suite is 611 tests (two new), parity 89/89, examples 10/10, clippy and Rust tests clean.

### Round 191: internal protocol types and tool output deltas

The class sweep (73 public LangGraph classes) went from 5 remaining entries to **one**:

* `PregelExecutableTask` now carries LangGraph's 13 dataclass fields.
* `ChannelProtocol` was added with its seven members.
* `NodeBuilder` gained its ten fluent methods (`subscribe_to`, `read_from`, `write_to`, `meta`,
  `add_cache_policy`, `add_retry_policies`, `set_timeout`, `subscribe_only`, plus `do`/`build` that
  explain where to build nodes instead).
* `ToolRuntime.emit_output_delta(delta)` is implemented **natively**: it pushes the delta onto the
  `tools` stream (`tool-output-delta`), and silently no-ops outside a run, matching LangGraph's
  missing-writer behaviour. (The shim-side override was removed once the native method existed — the
  injected runtime is the native object.)

The only remaining surface difference is `RetryPolicy.count`/`index`, because LangGraph models that
policy as a NamedTuple while ours is a native class with identical field access and retry behaviour.

Suite is 612 tests (one new), parity 89/89, examples 10/10, clippy and Rust tests clean.

### Round 192: delta history carries the channel values

`get_delta_channel_history` returned `{"writes": [], "seed": 0}` for every channel — the seed was a
placeholder integer, so a reader reconstructing state got nothing useful. The engine snapshots full
channel values per checkpoint (no delta blobs), so the value now goes into `seed`:

```
ours: {'value': {'writes': [], 'seed': 1}, 'log': {'writes': [], 'seed': ['b']}}
lgs : {'value': {'writes': [], 'seed': 1}, 'log': {'writes': [(task, 'log', ['b'])], 'seed': []}}
```

Applying `writes` over `seed` reconstructs the same state in both, which is what the new
`delta_history` parity scenario compares. Empty channel list returns `{}` like LangGraph.

Suite is 612 tests, parity 90/90 (new `delta_history` scenario), examples 10/10, clippy and Rust tests
clean.

### Round 193: the last reverse import, and an idle-timeout defect

An **independent audit** of the reverse-call rule found a violation the earlier `grep` pattern missed:
`src/adapters/python/async_support.rs` embedded Python source containing
`from langgraph_rs import _progress_counter` (a `from` import, not `import("...")`, so the guard could
not see it). The counter is now captured from the module object at `#[pymodule]` init and passed into
the generated wrapper as a default argument, so the generated code imports nothing.

Fixing that surfaced a real defect behind the same code path: an `idle_timeout`-only policy was
enforced as a *wall-clock run timeout* on the blocking path, so a node that kept reporting progress
was killed anyway (`exceeded its run timeout of 0.050s`). Two changes:

* `TimeoutPolicy::effective_timeout()` now returns only `run_timeout`; a progress bound cannot be
  enforced while a blocking call is in flight.
* The async idle watchdog reads the **combined** progress counter (`custom()` / `message()` bump the
  sync side, which the watchdog previously ignored) and reports `IdleTimeout`, giving LangGraph's
  `exceeded its idle timeout of 0.050s without making progress` wording.

Verified: a node ticking `custom()` every 20 ms with `idle_timeout=0.05` now completes; a silent node
reports the idle wording; `run_timeout` still fires. A Rust test that asserted the old wording was
updated.

Suite is 615 tests (three new), parity 90/90, examples 10/10, clippy and Rust tests clean.

### Round 194: adversarial audit — four behavioural differences fixed

An independent audit tested twelve documented "now matches" claims against real LangGraph. Seven held
(including the hardest ones: checkpoint history rows, interrupt semantics, store, error text). Four
real behavioural differences did not:

1. **`tools_condition` was inverted**: it read `tool_calls` from the serialized dict, so real
   `AIMessage` objects (which the engine encodes as `{"lc":1,...,"kwargs":{...}}`) routed to `__end__`
   while plain dicts routed to `tools`. Ported graphs silently skipped their tool node. The function is
   now implemented in the shim with LangGraph's logic (attribute check, `ValueError` on empty input).
2. **`stream()` defaulted to `values`** and emitted an extra initial state chunk. LangGraph defaults to
   `updates`; the four defaults in the Python adapters were switched.
3. **`RetryPolicy()` retried everything**: an empty predicate list meant retry-all, so a `ValueError`
   ran a node three times where LangGraph runs it once. The engine now applies LangGraph's
   `default_retry_on` (connection errors retry, `ValueError`/`TypeError`/`OSError`/… do not).
4. **`create_react_agent` and `ValidationNode` returned dict messages**: the react agent never marked
   its `messages` channel as a messages channel, and `ValidationNode` built dicts directly. Both now
   yield LangChain objects (`AIMessage`, `ToolMessage` with `status`), so `messages[-1].content` works.

Five of our own tests encoded the old behaviour and were updated to LangGraph's semantics.

Still open from the audit: `context_schema` rejects plain classes (LangGraph accepts them), duplicate
`add_node` fails at `compile` rather than at `add_node`, and three doc/cosmetic items (stale
`create_react_agent` signature table, `get_config()` always adding `tags`, a stale §4.1 sentence).

Suite is 620 tests (five new), parity 91/91 (new `audit_regressions` scenario), examples 10/10, clippy
and Rust tests clean.

### Round 195: the remaining audit findings

* **`context_schema` accepts a plain class.** LangGraph takes any class (it never validates a context
  it cannot derive a schema for). An underivable class now yields the permissive empty schema instead
  of `TypeError`, and `runtime.context` still arrives as the original object.
* **Duplicate `add_node` fails immediately.** LangGraph raises `ValueError: Node `a` already present.`
  when the node is registered; the facade now checks there instead of deferring to `compile`.
* **`add_node(metadata=..., trace_policy=...)`** are accepted (they used to raise `TypeError`);
  `metadata` is stored on the node, `trace_policy` is tolerated.
* **`get_config()` only reports `tags`** when the run carries some, matching LangGraph's key set.
* **`create_react_agent` signature** now lists LangGraph 1.2's parameters (`pre_model_hook`,
  `post_model_hook`, `context_schema`, `deprecated_kwargs` included).

Known difference documented: both hooks run as graph nodes, but when a hook echoes the existing
messages plus new ones, our `add_messages` appends the echo again (LangGraph's result does not), because
the two engines dedupe messages without `id` differently. Hooks that return only the new messages behave
identically.

Suite is 625 tests (five new), parity 91/91, examples 10/10, clippy and Rust tests clean.

### Round 196: `add_messages` dedupes by id, and messages get ids

Chasing the `pre_model_hook` message echo turned up a deeper defect: `add_messages` never deduped by
`id`, so any node returning the current message list plus new ones duplicated the existing messages
(`['hi', 'hi', 'new']` instead of LangGraph's `['hi', 'new']`).

Two causes, both fixed:

1. Message encoding now assigns a uuid to a message without an id, **writing it back onto the object**
   the way LangGraph's `add_messages` does — so echoing the same object carries the same id.
2. The reducer reads ids from both shapes: a top-level `id` (the engine's own JSON) and `kwargs.id`
   (LangChain's serialization). It only looked at the top level before, so real message objects never
   matched and every write appended.

Consequences verified: echo nodes no longer duplicate, replacing a message by id works, and the
`create_react_agent` `pre_model_hook` example now matches LangGraph exactly.

Suite is 629 tests (four new), parity 92/92 (new `message_ids` scenario), examples 10/10, clippy and
Rust tests clean.

### Round 197: message mutation semantics

A third independent audit, scoped to message/state semantics, found four more real differences:

1. **Engine-written messages had no id** (`ToolNode` output, dict input, `update_state`), because ids
   were only assigned on the Python encode path. A node echoing the list therefore duplicated tool
   messages. Ids are now also assigned inside the `add_messages` reducer, which every write goes
   through.
2. **Remove-then-add lost the position**: `[RemoveMessage(id), msg_with_same_id]` produced
   `[b, a2]` instead of LangGraph's `[a2, b]`.
3. **Removing the same id twice** raised `ValueError`; LangGraph treats it as a no-op.
4. **`REMOVE_ALL_MESSAGES`** (`RemoveMessage("__remove_all__")`) raised "doesn't exist" instead of
   clearing the list.

The reducer now mirrors LangGraph's order: collect removal intents, apply the writes (replacing in
place by id), then delete only the messages this update did not replace; removals are idempotent and
the remove-all sentinel has its own branch.

Still open from this audit: `add_messages` is exported as the channel-spec string rather than a
callable, tuple/single-message input coercion, and `JsonPlusSerializer` returning dicts for messages.

Suite is 636 tests (four new), parity 94/94 (new `message_mutation` scenario), examples 10/10, clippy
and Rust tests clean.

### Round 198: callable `add_messages`, input coercion, serializer round trip

The last three findings from the message-semantics audit:

* **`add_messages` is a real function again.** It was exported as the channel-spec string
  `"messages"`, so `add_messages(left, right)` raised `TypeError`. The shim now ships LangGraph's
  semantics (merge by id, `RemoveMessage` deletion with the missing-id error, `REMOVE_ALL_MESSAGES`,
  id assignment) and marks itself with `__channel_spec__ = "messages"` so graph construction still
  maps the channel.
* **Message inputs are coerced** like LangGraph: a bare message becomes a one-element list and
  `("user", "hi")` tuples become `HumanMessage`s. The legacy `_set_messages_objects(False)` escape
  hatch skips the coercion (a new native `messages_as_objects()` getter exposes the switch).
* **`JsonPlusSerializer` rebuilds what it wrote**: `loads_typed` restores LangChain objects through
  `langchain_core.load` instead of returning the raw `lc` dict, so message type and `id` survive a
  round trip.

Suite is 639 tests (three new), parity 95/95 (new `add_messages_function` scenario), examples 10/10,
clippy and Rust tests clean.

### Round 199: `response_schema` validation ownership

A fourth independent audit covered human-in-the-loop. The interrupt/resume core matched (single,
sequential, parallel keyed resume, tool interrupts, compile-time pauses, `update_state` while paused,
history rows, error paths), but `response_schema` validation was wrong in four ways:

1. Invalid resume raised `RuntimeError` (our hand-rolled JSON-schema check) instead of pydantic's
   `ValidationError`, and `is_invalid_resume()` reported False.
2. Values pydantic would coerce (`{"approved": 1}`) were rejected before pydantic ran.
3. A raw JSON-schema `dict` was validated, while LangGraph explicitly does not validate dict schemas.
4. Worse: inside `ToolNode`, a rejected resume was formatted as a tool error, so the graph **completed**
   and the human was never asked again.

Python callers now use `interrupt_unvalidated`, so pydantic owns validation (coercion, error type) and
marks failures with LangGraph's `_langgraph_invalid_resume`; `ToolNode` re-raises marked errors so the
pause survives.

Still open from this audit (all recorded in `docs/langgraph-compat.md`): unkeyed resume with multiple
pending raises `ValueError` where LangGraph raises `RuntimeError`; `Interrupt` is not a dataclass
(equality/hash/repr); `PregelTask.result` is always `None`; task order after a partial keyed resume;
replaying a past interrupt checkpoint applies the new resume value; `interrupt()` inside a manually
invoked subgraph is not propagated; `interrupt()` in a conditional router.

Suite is 645 tests (four new), parity 97/97 (two new scenarios), examples 10/10, clippy and Rust tests
clean.

### Round 200: `Interrupt` object contract and ambiguous resume

Two more findings from the human-in-the-loop audit:

* **`Interrupt` now compares by field, is unhashable, and reprs like LangGraph's dataclass**
  (`Interrupt(value='v', id='placeholder-id', response_schema=None)`). `is_dataclass()` still reports
  False because the class is a native pyclass — documented.
* **Resuming several pending interrupts with a plain value** raises `RuntimeError` with LangGraph's
  wording (it raised `ValueError`).

Two gaps documented rather than fixed, with reasons: `interrupt()` inside a manually invoked subgraph
(`child.invoke(...)`) is not propagated to the parent — the documented `add_node(name, compiled_sub)`
form works identically, and a real fix needs namespace-aware resume plumbing; and `interrupt()` inside a
conditional-edge router, which LangGraph tolerates and we reject.

Suite is 649 tests (four new), parity 98/98 (new `interrupt_object` scenario), examples 10/10, clippy
and Rust tests clean.

### Round 201: drop-in audit — silent no-persist and message pollution

A fifth audit wrote four realistic user programs (ReAct agent with memory, supervisor with subgraph
handoff + `Send` fan-out + list-returning conditional edges, the four stream modes, and
persistence/fork/durability/interrupt) and swapped only the root package name. **No API had to change** —
imports, call forms and argument names all carried over. Two real defects surfaced:

1. **A checkpointered run without a `thread_id` silently persisted nothing** (`invoke` returned the right
   values, the saver had zero rows). LangGraph raises `ValueError: Checkpointer requires one or more of
   the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id`; the engine now does too,
   in the shared sync and async execute paths.
2. **`create_react_agent` polluted `AIMessage.additional_kwargs`** with `invalid_tool_calls` and
   `usage_metadata`, because model output was stored as a flat `model_dump()` dict. It is now encoded as
   a LangChain constructor object like every other message, and the native `tool_calls` readers accept
   both shapes (`message_field`), since state legitimately holds both.

Remaining cosmetic findings from this audit (documented, not fixed): `created_at` format and
`checkpoint_id` shape, 32-hex vs uuid4 message ids, `create_react_agent` not emitting LangGraph's
deprecation warning, `stream_mode="bogus"` raising `TypeError` where LangGraph yields nothing, an empty
list channel missing from the initial checkpoint, and `KeyError: 'thread_id'` vs `TypeError` for
`get_state`/`update_state` without a thread.

Suite is 652 tests (three new), parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 202: channel defaults in the opening checkpoints

LangGraph's aggregating channels hold a default (`[]` for a list reducer) before anything writes, so its
snapshots show `{'trail': []}`. Ours only stored written channels, so `values["trail"]` raised `KeyError`.

Channels now carry a default (`append`/`messages`/`topic` → `[]`, `add` → `0`, settable through
`Reducer::with_default`), and both opening checkpoints (`source="input"` and the `source="loop"` one
before the entry tasks) seed it.

Documented residual: the final `invoke` result and the last (highest-step) checkpoint still omit defaults,
because they serialize the end-of-run state and an unwritten channel has no key there. Closing that gap
means making empty channels report their default everywhere, which would also change `update_state` and
node-input semantics — disproportionate risk for the difference.

Suite is 653 tests (one new), parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 203: async API audit — the sync builder fakes async

A sixth audit covered the async API, which no earlier round had touched. 30 checks passed and 14 failed;
almost all failures trace to one structural fact: **`StateGraph.compile()` returns a sync-only inner**,
so the facade's `a*` methods fall back to `asyncio.to_thread(sync)`. Graphs compiled from
`AsyncStateGraph` do have a native `AsyncCompiledGraph` and behave correctly (verified: incremental
`astream`, async nodes, cancellation, `gather`, interrupt/resume).

Consequences on the sync builder's async facade, all reproduced:

* `astream` buffers: chunks arrive only when the whole run finishes.
* `asyncio.wait_for`/`task.cancel()` does not stop the run — the abandoned node still completes and
  commits its checkpoint.
* Tokens from `await model.ainvoke()` inside an `async def` node never reach `stream_mode="messages"`
  (the callback fires on a third thread where the engine's thread-local context is absent).
* `astream_events(version="v2")` with async nodes raises `Cannot run the event loop while another loop
  is running`.
* `aget_state_history` is a coroutine rather than an async iterator.
* `AsyncInMemorySaver` lacks `aput`/`aput_writes`/`adelete_thread`/`aget`/`acopy_thread` (its
  `InMemorySaver` sibling has the full surface).

Documented in `docs/langgraph-compat.md` with the workaround (use `AsyncStateGraph`). Round 204 will
unify `StateGraph.compile()` on the async implementation so one compiled graph supports both APIs, the
way LangGraph does.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 204: an async twin for the sync builder (not yet wired)

LangGraph compiles one graph that serves both APIs; ours compiles a sync graph (and the facade fakes async
with `to_thread`). This round added `AsyncStateGraph::from_sync(builder)` — it wraps each sync node in an
async block and copies the whole builder definition — and `PyCompiledGraph` now carries the resulting
`_async_inner`.

Wiring the facade's `ainvoke`/`astream` to that twin **regressed two timeout tests**: the async executor's
run/idle timeout semantics differ from the sync executor's. So the twin stays connected but unused, and the
behaviour is unchanged from the previous round.

The attempt did pin down exactly what must be fixed before the switch is safe:

1. **Incremental delivery** — both async paths collect chunks and release them at the end (two 0.3 s nodes:
   `[0.63, 0.63]` versus LangGraph's `[0.32, 0.63]`); `astream` needs an mpsc channel plus an async
   iterator instead of a `Vec`.
2. **Cancellation propagation** — after `task.cancel()` the node still finishes and commits its checkpoint;
   LangGraph keeps the previous values and leaves `next` pointing at the node.
3. **Timeout semantics** — the async executor must match the sync one for run and idle timeouts.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 205: a native async stream iterator, and the real cause of buffering

`astream` now returns a genuine async iterator (`_AsyncStreamChunks`: an mpsc channel plus
`__aiter__`/`__anext__`), and the shim's `AsyncStream` accepts a native iterator as well as an
awaitable. Suite, parity, examples, clippy and Rust tests all stay green.

Measuring chunk arrival times then showed the buffering is **not** in the wrapper: the engine releases
a superstep's chunks only when the run finishes (`[0.60, 0.60]` synchronously, `[0.42, 0.42]`
asynchronously, versus LangGraph's `[0.30, 0.60]` for two 0.2 s nodes). The parity harness never caught
this because it compares chunk *sequences*, not *arrival times*.

So the remaining work moved: the fix belongs in the run loop's emission path (drain the collector and
invoke the sink as each superstep completes), not in the Python bindings. Documented with the measured
numbers.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 206: the buffering is in the facade, not the engine

Last round I concluded the engine defers chunk emission. That was wrong. Driving the facade's `stream()`
through the pre-existing native `stream_live` (worker thread + channel) immediately produced LangGraph's
timing: `[0.30, 0.60]` instead of `[0.60, 0.60]` for two 0.3 s nodes. The engine already calls the sink at
the end of every superstep; the facade was collecting into a list via `_inner.stream(...)` and handing it
over at once.

The change is not kept yet: switching the return type to an iterator broke eight tests. They are all
mechanically fixable, but doing it properly needs one focused pass:

1. `stream()` now returns an iterator rather than a list — add `__getitem__`/`__len__` to the live
   wrapper (or switch the affected assertions to `list(...)`).
2. A missing `thread_id` then surfaces on first iteration instead of at call time — which is what
   LangGraph does (its `stream` is a generator); the test asserting an eager error has to follow suit.
3. `custom` chunks are missing from the live path in multi-mode (`lg.custom()` goes through the pending
   live sink); needs investigation before switching.

Documented, with the measurement, as next round's first task.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 207: the live stream never persisted (fixed)

Working on the incremental-delivery wiring turned up a real defect in the native live stream: it read
the thread id only from the top level of the `thread_id` argument, so a LangGraph config dict
(`{"configurable": {"thread_id": ...}}`) produced an empty thread and **the run persisted nothing** —
`get_state_history` came back empty while the chunks looked fine.

`stream_live` now parses the argument with `config_from_py`, like every other entry point. Measured
afterwards with two 0.2 s nodes and a checkpointer: chunk arrival `[0.25, 0.48]` (per superstep) and four
history rows.

The facade wiring is still not kept: four concrete issues remain — the missing-thread error surfaces as
`RuntimeError` from the worker thread instead of `ValueError`; `subgraphs=True` debug chunk shapes and
`updates` namespace prefixes differ on the live path; lifecycle events are dispatched at start rather
than when the run ends; and `stream(thread_id=...)` must resolve the thread like a config dict does.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 208: the live stream now takes a full config

`stream_live` used to build a minimal `RunConfig`, so a caller's callbacks, metadata, durability and
recursion limit were silently dropped. It now accepts `config=` and parses it with `config_from_py`,
layering the stream settings (modes, thread, subgraphs) on top.

One change was reverted on evidence: making `PyLiveStream` wrap namespaced chunks like `subgraphs=True`
breaks `stream_events(version="v3")`, whose raw event log then reports an empty mode instead of `debug` —
the event pipeline already handles namespaces, so the live stream must not re-wrap.

Two items remain before the facade can switch to the live stream (it is reverted, tree green): the
interrupt chunk shape on the live path (an extra empty chunk, and `__interrupt__` as a tuple rather than a
list), and lifecycle dispatch for `stream` (it works for `invoke`).

Measured state of the native live path: two 0.2 s nodes deliver chunks at `[0.25, 0.47]` and the
checkpointer records four history rows — incremental delivery and persistence both correct.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 209: the live stream reports pauses

The eager `stream()`/`invoke()` paths record a `pending` interrupt lifecycle event after a paused run and
move it onto the compiled graph; the live (worker-thread) path did neither, so a streamed pause never
reached `GraphCallbackHandler`. The worker now does both, creating a lifecycle sink when the run config
has none. Verified: `take_lifecycle_events()` returns one event after a live run on an interrupt graph.

The live conversion was also switched to the exact function the eager path uses, so it no longer re-wraps
subgraph namespaces (that broke `stream_events(version="v3")` in the previous round).

One difference remains before the facade can use the live stream (still reverted, tree green): on an
interrupt graph the live path emits one extra empty chunk and `__interrupt__` as a tuple rather than a
list.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 210: live chunk shapes match the eager path

Two conversion details in `PyLiveStream::poll` made a streamed interrupt differ from the eager result: the
interrupt chunk was converted on its own (yielding a tuple instead of a one-element list) and a repeated
`values` payload produced an extra empty chunk (the eager path converts the whole list at once and merges
it).

Interrupt chunks are now held until the stream ends and converted together, and a repeated `values`
payload is dropped by comparing against the last one. Measured on an interrupt graph and on a two-node
graph: eager and live sequences are now identical.

The facade wiring is still reverted (653 tests green). Five issues surface on connecting it — subgraph
debug chunk shape, `astream` interacting with a sync iterator, lifecycle dispatch, namespace prefixes —
so the next attempt will be bounded: use the live stream only for the single-mode, non-subgraph case
whose shapes are now verified identical, and keep the eager path everywhere else.

Suite is 653 tests, parity 98/98, examples 10/10, clippy and Rust tests clean.

### Round 211: streaming is now incremental (bounded integration)

The live stream is finally wired into the facade — but bounded: it is used only for a single
`stream_mode` (`updates`/`values`/`custom`/`debug`/`tasks`) without `subgraphs` and without
`version="v2"`. Everything else keeps the eager path, whose shapes the existing tests cover.

Measured: `stream(updates)` delivers chunks at `[0.25, 0.50]` and `astream(updates)` at `[0.21, 0.41]`
for two 0.25 s nodes — per superstep, exactly like LangGraph. A streamed run with a checkpointer records
three history rows, which is also what LangGraph does (the earlier test expecting four was wrong and has
been corrected against real LangGraph).

The return value supports both shapes: iterating yields chunks as they arrive, while `len()`/`[i]`
materialise the rest, so existing callers keep working.

Suite is 656 tests (three new), parity 99/99 (new `incremental_stream` scenario), examples 10/10, clippy
and Rust tests clean.
