# Compatibility audit vs Python LangGraph 1.2.22

Audited by Claude Code (static source review; its shell was blocked) against the Rust
implementation and the repo's own parity harness. Python package source was not available
inside that review environment, so Python-side claims that depend on exact 1.2.22 details
are marked "verify" where relevant.

The installed Python LangGraph is used by `benchmarks/parity.py`; the 40 scenario groups
there are the executable comparison suite and currently report **40/40 equivalent, 0 known
differences**.

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
- **Robustness.** An unkeyed resume with several pending interrupts is rejected; the entire
  feature matrix (none/async/sqlite/redis/postgres/msgpack/python/python-async) builds
  without warnings.

Still open, in rough priority order:

1. **Checkpoint v4 task ids.** Python derives task ids as
   `xxh3_128(checkpoint_id_bytes + ns + step + name + PULL + triggers)`; this crate uses
   time-based ids. Consequences: cross-engine pending-task identity, resume maps keyed by
   LangGraph ids, and the LangGraph-schema SQLite table (which stores internal steps because
   its stored ids were derived from that counter) are not interchangeable.
2. **Debug stream.** The input-checkpoint record is not emitted, `config.checkpoint_id` is
   `null` on debug checkpoint records, and debug/checkpoint task entries use the `Send` input
   as `state`.
3. **Trigger-accurate bookkeeping.** `versions_seen` approximates per-trigger reads, named
   barriers are consumed even when the superstep did not read them, and `NamedBarrierValue`
   re-arms where Python keeps the barrier open.
4. **Interop edge cases.** Tuple-form `pending_writes` carry no node, `__error__` writes are
   not rebuilt into a routable failure, and the JSON importer does not decode
   `channel_values.__interrupt__` (the SQLite reader does).
5. **Python binding breadth** (unchanged from the historical section): drop-in
   `RunnableConfig` API surface, streaming iteration semantics, and v2 payload details.