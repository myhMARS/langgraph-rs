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
| Extra channel types | ❌ | ✅ `EphemeralValue`, `AnyValue`, `NamedBarrierValue`, `UntrackedValue`, `LastValueAfterFinish`, `DeltaChannel` | |
| `add_messages` semantics | ⚠️ id-based replace/append (`reducers::messages`) | ✅ `add_messages` incl. `RemoveMessage` | no remove/delete messages; no `MessagesState` / `MessageGraph` |
| Static edges | ✅ | ✅ | `add_edge` takes one source; `add_edges_from` fans several sources into one target |
| Conditional edges + path map | ✅ | ✅ | |
| Conditional entry point | ✅ | ✅ | |
| `add_sequence` | ✅ (takes `(name, NodeFn)` pairs) | ✅ | heterogeneous closures need `Arc::new(...)` |
| Node metadata / `input_schema` / `destinations` / custom names | ❌ | ✅ | node options are not modelled |
| `Send` map/reduce from nodes | ✅ `Send::to` / `Send::with_arg` | ✅ `Send` (node, arg, timeout) | ours has no per-`Send` timeout |
| `Send` from routers / conditional edges | ❌ | ✅ | our routers return destinations only |
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
| Async engine | ⚠️ separate `AsyncStateGraph` adapter | ✅ native `ainvoke`/`astream` | see Async row |
| Batch (`batch` / `abatch`) | ⚠️ sync `batch` / `batch_run` (+ Python `batch`); no async batch | ✅ | results stay in input order, failures are per item |
| `max_concurrency` / run control | ⚠️ `RunConfig::max_concurrency` bounds `batch` workers; no `RunControl` | ✅ `max_concurrency`, `RunControl` | |
| Durability modes | ⚠️ `Durability::{Sync, Async, Exit}` (`Async` behaves like `Sync`) | ✅ `durability=` | `Exit` persists only initial/interrupt/final checkpoints |
| Context / runtime injection (`Runtime`, `context_schema`) | ❌ | ✅ | |
| `get_config` / `get_stream_writer` | ⚠️ thread-local `custom()` only | ✅ context helpers | no config object |
| Managed values (`IsLastStep`, `RemainingSteps`) | ❌ | ✅ `langgraph.managed` | |
| Retry policy | ⚠️ `RetryPolicy` (attempts, exponential backoff, cap) | ✅ `RetryPolicy(initial_interval, backoff_factor, max_interval, max_attempts, jitter, retry_on)` | ours has no jitter and no `retry_on` exception selector |
| Error handling of nodes | ✅ `add_error_edge` + `last_error()` | ✅ per-node `add_node(error_handler=...)` | equivalent capability, different API; our handler runs in the next superstep |
| Timeout policy (`TimeoutPolicy`, node/run/idle timeouts) | ❌ | ✅ | |
| Functional API (`@entrypoint` / `@task`) | ❌ | ✅ `langgraph.func` | separate programming model; not implemented |
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
| Subgraph streaming (`subgraphs=True`) | ❌ | ✅ | |
| Async streaming (`astream`) | ✅ `AsyncCompiledGraph::stream` / `stream_modes` | ✅ | same chunk shapes, no v2 events |
| v2 stream events | ❌ | ✅ `version="v2"` | |
| Stream writer from arbitrary helpers | ⚠️ thread-local `custom()` | ✅ `get_stream_writer()` | |

### Subgraphs, caching, async, extras

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Subgraphs | ⚠️ `add_subgraph` (shared state) + `add_subgraph_mapped` (different state types); child checkpoint namespaces and nested interrupts/resume supported | ✅ first-class | subgraph **streaming** (`subgraphs=True`) is still missing; the interrupt surfaces as the parent subgraph node |
| Node-level cache | ⚠️ `CachePolicy` + `NodeCache` port, `InMemoryCache`, sync only | ✅ `CachePolicy(key_func, ttl)` + `BaseCache` (`InMemoryCache`, `RedisCache`) passed to `compile(cache=...)` | ours: in-memory only, no async cache, no size limits, no cache adapters |
| Long-term memory (`BaseStore`) | ⚠️ `BaseStore` port + `InMemoryStore`; `StateGraph::with_store` / `CompileOptions::with_store` / `RunConfig::with_store`; nodes read it with `get_store()`; not exposed from Python yet | ✅ `compile(store=...)`, `Runtime.store`, `PostgresStore`, ... | no persistent store adapters; `search` supports prefix + substring only |
| Runtime / context | ⚠️ `Runtime`, `current_runtime()`, `current_context()`, `current_thread_id()`, `current_namespace()`, `get_store()`; `RunConfig::with_context` carries a JSON context | ✅ `Runtime`, `context_schema`, `get_config`, `get_store` | no context schema validation |
| Mermaid export | ✅ sync + async `to_mermaid()` (+ Python) | ✅ `draw_mermaid()`, PNG, ASCII, xray subgraph view | static topology export only |
| Per-node usage/report (`RunReport`, `TraceSink`) | ✅ first-class (extension) | ⚠️ callbacks / metadata / LangSmith | not a LangGraph feature; superset here |
| Python bindings | ⚠️ PyO3 bindings (`python` feature): `StateGraph`/`CompiledGraph`, `interrupt`/`resume`, `Send`, `Command` (node output / invoke input), `RetryPolicy`/`CachePolicy`/`error_handler`, `compile(cache=True)`, saver objects (`InMemorySaver`, `SqliteSaver` with `sqlite`), `add_messages`, `update_state`, thread delete/copy, `stream_mode`, run reports | ✅ full Python API | still missing from Python: async API, subgraphs, Redis/Postgres saver classes, `Command(graph=PARENT)`, multi-resume interrupts |

## How the matrix was produced

Statuses are based on the public Rust/Python APIs of this crate, its integration tests
(`tests/graph.rs`, `tests/checkpoint.rs`, `tests/state_model.rs`, `tests/usage.rs`,
`tests/resilience.rs`,
`tests/subgraph.rs`, `tests/cache.rs`, `tests/visualize.rs`, `tests/async_graph.rs`,
`tests/async_resilience.rs`, `tests/persistence.rs`, `python/tests/test_bindings.py`) and
introspection of the installed LangGraph 1.2.22 API. Performance is measured separately in
[`RESULTS.md`](RESULTS.md).