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
