# Functional comparison with LangGraph

Audited against **Python LangGraph 1.2.22** (with `langgraph-checkpoint` 4.3.2) on
2026-11-16. The LangGraph side was checked by introspecting the installed package
(`langgraph.graph.StateGraph`, `langgraph.types`, `langgraph.channels`,
`langgraph.pregel.Pregel`, `langgraph.func`, `langgraph.managed`, `langgraph.config`,
`langgraph.store`, `langgraph.cache`); the Rust side by reading the public API and the
integration tests in `tests/` and `python/tests/`.

Status definitions:

- ✅ **parity** — the same workflow can be expressed and behaves equivalently.
- ⚠️ **partial** — the capability exists but is narrower than LangGraph (documented in Notes).
- ❌ **missing** — not implemented in langgraph-rs.

## Headline verdict

For **single-graph orchestration** (typed state, channels/reducers, static and conditional
edges, parallel supersteps, `Send`, `goto`, recursion limit, checkpoint/resume, interrupts,
streaming of values/updates/custom events, retries, Mermaid export) langgraph-rs is
functionally equivalent in behaviour, with a different execution model (OS threads instead
of asyncio) and a different API spelling in places.

It is **not a drop-in replacement** for the whole of LangGraph. The large gaps are the
functional API (`@entrypoint` / `@task`), long-term memory (`BaseStore`), runtime/context
injection, timeout policies, deferred nodes, checkpoint namespaces (`checkpoint_ns`),
LangGraph's full streaming modes (`messages` / `debug` / `tasks`), async checkpointers,
subgraph state mapping/streaming, and a large part of the Python surface (no `Send`,
`Command`, cache/retry configuration, persistent savers or async from Python). The matrix
below is the complete audit.

## Matrix

### Graph model and construction

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Typed state + channels | ⚠️ `State::schema` / `StateSchema`, `JsonState` | ✅ `StateGraph(StateT)` with annotated channels | one state schema per graph; no separate input/output/private schemas |
| `LastValue` channel | ✅ | ✅ `LastValue` | concurrent writes fail in both |
| `Topic` channel | ⚠️ always appends | ✅ `Topic(accumulate=...)` | no `accumulate=False` mode |
| Custom binary reducers | ✅ `Reducer`, `reducers::{append,add}` | ✅ `BinaryOperatorAggregate`, `Annotated[..., reducer]` | |
| Extra channel types | ✅ `AnyValue`, `EphemeralValue`, `UntrackedValue`, `NamedBarrierValue`, `DeltaChannel` (`delta`), `LastValueAfterFinish`, `Overwrite`; Python specs `any_value` / `ephemeral_value` / `untracked_value` / `delta` / `last_value_after_finish` | ✅ all of these | barrier value is the seen-name array until open, then `true`; ephemeral values cleared between steps; untracked values excluded from checkpoints; finish-value writes published only at graph end; delta `snapshot_frequency` stored but this engine always snapshots full values |
| `add_messages` semantics | ✅ id-based replace/append (`reducers::messages`), `RemoveMessage` removals, `MessagesState` convenience state (Rust) and `lg.MessagesState` schema (Python); `MessageGraph` type alias (Rust) and `lg.message_graph()` factory | ✅ `add_messages` incl. `RemoveMessage` | no LangGraph `pending_writes` configuration |
| Static edges | ✅ | ✅ | `add_edge` takes one source; `add_edges_from` fans several sources into one target |
| Conditional edges + path map | ✅ | ✅ | |
| Conditional entry point | ✅ | ✅ | |
| `add_sequence` | ✅ (takes `(name, NodeFn)` pairs) | ✅ | heterogeneous closures need `Arc::new(...)` |
| Node metadata / `input_schema` / `destinations` / custom names | ⚠️ metadata via `add_node_with_metadata` / `node_metadata()`; destination allow-lists via `set_destinations` / `add_node_with_destinations` (Python `add_node(..., destinations=...)`); JSON `input_schema` stored via `set_node_input_schema` / `add_node(..., input_schema=...)` (not enforced); arbitrary node names supported | ✅ | `input_schema` is metadata only, no runtime validation |
| `Send` map/reduce from nodes | ✅ `Send::to` / `Send::with_arg` / `Send::with_timeout` (per-task timeout; Python `lg.Send(..., timeout=seconds)`) | ✅ `Send` (node, arg, timeout) | |
| `Send` from routers / conditional edges | ✅ `Route::send` / `Route::sends` (also conditional entry points; Python routers may return `Send` objects) | ✅ | |
| `goto` | ✅ `NodeOutput::goto` | ✅ `Command(goto=...)` / node return | |
| `Command` | ⚠️ `Command::resume` + `with_update` only | ✅ `Command(graph=, update=, resume=, goto=, PARENT)` | ours cannot `invoke(Command)`, no `graph=PARENT`, no goto via Command |
| Node metadata / input schema / destinations / name | ❌ | ✅ `add_node(..., metadata=, input_schema=, destinations=, name=)` | |
| Deferred nodes | ❌ | ✅ `add_node(defer=True)` | |
| Graph name / graph object export | ❌ | ✅ `compile(name=...)`, `get_graph()` | we export Mermaid text only |

### Execution

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Supersteps + recursion limit | ✅ default 25, per compile/run | ✅ default 25 | |
| Parallel superstep execution | ✅ scoped OS threads | ✅ asyncio tasks | CPU parallelism vs cooperative concurrency |
| Deterministic channel merge | ✅ task-order merge | ✅ | |
| Async engine | ⚠️ separate `AsyncStateGraph` adapter: checkpoints/HITL/retry/error routing, stream modes, store/context, per-node timeouts, subgraphs (namespaces + nested interrupts), node cache, batch, subgraph streaming; Python async bindings (`AsyncStateGraph`/`ainvoke`/`aresume`) behind `python-async` | ✅ native `ainvoke`/`astream` | still missing: async checkpointer adapters (sync savers work from async graphs) |
| Batch (`batch` / `abatch`) | ✅ sync `batch` / `batch_run`, async `batch` / `batch_run` (bounded by `max_concurrency`), Python `batch` and async `abatch` | ✅ | async `batch_run` runs sequentially (only the run report is collected) |
| `max_concurrency` / run control | ⚠️ `RunConfig::max_concurrency` bounds `batch` workers; no `RunControl` | ✅ `max_concurrency`, `RunControl` | |
| Durability modes | ⚠️ `Durability::{Sync, Async, Exit}` (`Async` behaves like `Sync`) | ✅ `durability=` | `Exit` persists only initial/interrupt/final checkpoints |
| Managed values (`IsLastStep` / `RemainingSteps`) | ⚠️ `remaining_steps()` / `is_last_step()` accessors (Rust + Python) | ✅ managed values in state annotations | no `RemainingSteps` state annotation / `deferred` nodes |
| TimeoutPolicy | ⚠️ `TimeoutPolicy` with `run_timeout` enforced (sync helper thread, async `tokio::time::timeout`); `idle_timeout` stored but not enforced | ✅ `TimeoutPolicy` (`run_timeout`, `idle_timeout`) | sync timeout cannot cancel a blocking closure |
| Deferred nodes (`deferred=True`) | ✅ sync + async (`add_node_deferred` / `set_deferred`, Python `add_node(..., deferred=True)`); run after all regular work, may interrupt/resume | ✅ | deferred nodes may not be entry points or have outgoing/conditional edges |
| Context / runtime injection (`Runtime`, `context_schema`) | ❌ | ✅ | |
| `get_config` / `get_stream_writer` | ⚠️ thread-local `custom()` only | ✅ context helpers | no config object |
| Managed values (`IsLastStep`, `RemainingSteps`) | ❌ | ✅ `langgraph.managed` | |
| Retry policy | ⚠️ `RetryPolicy` (attempts, exponential backoff, cap) | ✅ `RetryPolicy(initial_interval, backoff_factor, max_interval, max_attempts, jitter, retry_on)` | ours has no jitter and no `retry_on` exception selector |
| Error handling of nodes | ✅ `add_error_edge` + `last_error()` | ✅ per-node `add_node(error_handler=...)` | equivalent capability, different API; our handler runs in the next superstep |
| Timeout policy (`TimeoutPolicy`, node/run/idle timeouts) | ❌ | ✅ | |
| Functional API (`@entrypoint` / `@task`) | ⚠️ `functional::entrypoint` compiles to a one-node graph; `functional::task` memoizes results in `BaseStore` so completed tasks are not re-run after interrupt/resume; Python `lg.entrypoint` / `lg.task` mirror this | ✅ `langgraph.func` | tasks execute inline (no parallel task scheduling), no per-task checkpoint records |
| Prebuilt agents (`ToolNode`, `create_react_agent`, `tools_condition`) | ❌ | ✅ `langgraph.prebuilt` | outside the current scope |

### Checkpointing and state management

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Memory checkpointer | ✅ `InMemorySaver` | ✅ `InMemorySaver` | |
| Persistent checkpointers (sync) | ✅ `SqliteSaver`, `RedisSaver`, `PostgresSaver` (features) | ✅ SQLite/Postgres/Redis saver packages | ours store an internal JSON format; **not interoperable** with LangGraph's checkpoint format |
| Async checkpointers | ❌ | ✅ `AsyncSqliteSaver`, `AsyncPostgresSaver`, ... | our savers are synchronous `Mutex` adapters |
| Checkpoint namespaces (`checkpoint_ns`) | ✅ `RunConfig::with_checkpoint_ns`, `get_state_in` / `resume_in` / `continue_thread_in` / `update_state_in`; namespace-aware queries in all adapters (sync + async) | ✅ | subgraph-internal namespace trees are still unsupported |
| Pending writes / `put_writes` | ⚠️ pending `Task`s embedded in the checkpoint | ✅ separate writes + checkpoint tuple | resume replays pending tasks in both |
| `get_state` / history | ⚠️ `get_state(thread)` / `get_state_in(thread, ns)`, `get_state_history(before=, limit=)`; snapshots expose values/next/tasks/interrupts/metadata/parent id/checkpoint_ns | ✅ + `subgraphs=`, filter, metadata/config, tasks | no LangGraph `config` object and no interrupt/task ids |
| `update_state` (`as_node`, `task_id`) | ⚠️ `update_state(thread, values, as_node=)` / Python; no `task_id` | ✅ | recomputes next tasks from `as_node`'s edges, or keeps pending tasks |
| Thread management | ⚠️ `delete_thread` + `copy_thread` (all adapters, exposed from Python); no `prune` / filtered `list` | ✅ `delete_thread`, `prune`, `copy_thread`, `list` | |
| Time travel | ⚠️ `resume_from(thread, checkpoint_id, value)` | ✅ `get_state_history` + replay from any config | history now supports `before`/`limit`; still no `StateSnapshot.config` |
| Checkpoint metadata / created_at semantics | ⚠️ basic (`parent_id`, `created_at_millis`, `source`) | ✅ rich metadata, versions/versions_seen | our checkpoint model is simpler |

### Interrupts / HITL

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| `interrupt(value)` pause | ✅ `Error::Interrupt` | ✅ `GraphInterrupt` / `Interrupt` | node re-executes on resume in both |
| Resume with a value | ✅ `resume` / `Command::resume` | ✅ `Command(resume=...)` | |
| Multiple interrupts in one node | ❌ | ✅ resume dict/list matched per interrupt | our context consumes a single resume value |
| `interrupt(response_schema=...)` | ❌ | ✅ | |
| `interrupt_before` / `interrupt_after` | ✅ | ✅ | our pause payload is `null` |
| Resume with state edits | ✅ `resume_with_updates` / `Command::with_update` | ✅ `Command(update=...)`, `update_state` | |

### Streaming

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| `values` / `updates` / `custom` chunks | ✅ `StreamChunk::{Values,Updates,Custom}` + `stream_modes` filter | ✅ `stream_mode` | ours is a single eager chunk list / sync sink |
| `interrupt` / `checkpoint` chunks | ✅ | ✅ (via stream modes) | |
| `messages` (LLM token streaming) | ⚠️ `message()` / `ctx.message()` emitted by nodes; `stream_mode="messages"` | ✅ `stream_mode="messages"` (LangChain callback tokens) | payloads are node-supplied, no automatic LLM callback capture |
| `debug` / `tasks` / `checkpoints` stream modes | ✅ `StreamChunk::{Debug,Task}` + checkpoint chunks, filterable via `stream_modes` / Python `stream_mode` | ✅ | debug payloads carry step/executed/next rather than LangGraph's full payload schema |
| Subgraph streaming (`subgraphs=True`) | ✅ `RunConfig::with_stream_subgraphs(true)` wraps child chunks in `StreamChunk::Subgraph` with a checkpoint namespace (nested namespaces joined with `|`); Python `stream(..., subgraphs=True)`; sync + async | ✅ | enable per run; default hides subgraph chunks |
| Async streaming (`astream`) | ✅ `AsyncCompiledGraph::stream` / `stream_modes` | ✅ | same chunk shapes, no v2 events |
| v2 stream events | ❌ | ✅ `version="v2"` | |
| Stream writer from arbitrary helpers | ⚠️ thread-local `custom()` | ✅ `get_stream_writer()` | |

### Subgraphs, caching, async, extras

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Subgraphs | ✅ `add_subgraph` (shared state) + `add_subgraph_mapped` (different state types); child checkpoint namespaces, nested interrupts/resume and subgraph streaming supported | ✅ first-class | the interrupt surfaces as the parent subgraph node |
| Node-level cache | ⚠️ `CachePolicy` + `NodeCache` port, `InMemoryCache`, sync only | ✅ `CachePolicy(key_func, ttl)` + `BaseCache` (`InMemoryCache`, `RedisCache`) passed to `compile(cache=...)` | ours: in-memory only, no async cache, no size limits, no cache adapters |
| Long-term memory (`BaseStore`) | ⚠️ `BaseStore` port + `InMemoryStore`; `StateGraph::with_store` / `CompileOptions::with_store` / `RunConfig::with_store`; nodes read it with `get_store()`; not exposed from Python yet | ✅ `compile(store=...)`, `Runtime.store`, `PostgresStore`, ... | no persistent store adapters; `search` supports prefix + substring only |
| Runtime / context | ⚠️ `Runtime`, `current_runtime()`, `current_context()`, `current_thread_id()`, `current_namespace()`, `get_store()`; `RunConfig::with_context` carries a JSON context | ✅ `Runtime`, `context_schema`, `get_config`, `get_store` | no context schema validation |
| Mermaid export | ✅ sync + async `to_mermaid()` (+ Python) | ✅ `draw_mermaid()`, PNG, ASCII, xray subgraph view | static topology export only |
| Per-node usage/report (`RunReport`, `TraceSink`) | ✅ first-class (extension) | ⚠️ callbacks / metadata / LangSmith | not a LangGraph feature; superset here |
| Python bindings | ⚠️ PyO3 bindings (`python` feature): `StateGraph`/`CompiledGraph`, `interrupt`/`resume` (including multiple pending interrupts via an `{id: value}` map), `Send`, `Command` (node output / invoke input), `RetryPolicy`/`CachePolicy`/`error_handler`, `compile(cache=True, store=...)`, saver objects (`InMemorySaver`, `SqliteSaver` with `sqlite`, `RedisSaver`/`PostgresSaver` with the matching features), `InMemoryStore` + `get_store`, subgraphs (namespaces, nested interrupts, streaming), runtime context, functional API (`lg.entrypoint` / `lg.task`), `add_messages`, `update_state`, thread delete/copy, `stream_mode`, `batch`, run reports | ✅ full Python API | still missing from Python: `Command(graph=PARENT)`; async Python covers `AsyncStateGraph` (incl. `add_subgraph`), `ainvoke`, `aresume`, `abatch`, `astream` (stream modes + subgraph streaming), `get_state` (`python-async` feature) |

## How the matrix was produced

Statuses are based on the public Rust/Python APIs of this crate, its integration tests
(`tests/graph.rs`, `tests/channels.rs`, `tests/delta_channels.rs`, `tests/messages.rs`, `tests/subgraph_streaming.rs`, `tests/multi_interrupt.rs`, `tests/advanced.rs`, `tests/deferred.rs`, `tests/destinations.rs`, `tests/functional.rs`, `tests/managed.rs`, `tests/store.rs`,
`tests/checkpoint.rs`, `tests/state_model.rs`, `tests/usage.rs`,
`tests/resilience.rs`,
`tests/subgraph.rs`, `tests/cache.rs`, `tests/visualize.rs`, `tests/async_graph.rs`,
`tests/async_resilience.rs`, `tests/async_subgraph.rs`, `tests/persistence.rs`, `python/tests/test_bindings.py`) and
introspection of the installed LangGraph 1.2.22 API. Performance is measured separately in
[`RESULTS.md`](RESULTS.md).