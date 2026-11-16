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

It is **not a drop-in replacement**: the execution model differs (OS threads / Tokio
instead of asyncio), checkpoint payloads are JSON documents (LangGraph-tuple JSON export/import is provided, not its msgpack database format), and a
few APIs are spelled differently or narrower. The remaining gaps are documented in the
matrix below. The most notable ones are: the functional API is approximated with inline
tasks plus store memoization, v2 stream parts are Python-only (Rust exposes the typed enum),
`Command(graph=PARENT)` works in sync and async subgraphs,
async checkpointers are aliases over the synchronous savers, and `input_schema`
is stored as metadata without runtime validation. Everything else — graph API, channels,
checkpointing/interrupts, streaming modes (including subgraph streaming), subgraphs
(mapping, checkpoint namespaces, nested interrupts), stores/runtime context, retries and
error routing, caching, batch/concurrency/durability, timeouts, deferred nodes, functional
API and the Python / Python-async bindings — is implemented and covered by the test suite.

## Matrix

### Graph model and construction

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Typed state + channels | ✅ `State::schema` / `StateSchema` (also `with_input_channels` / `with_output_channels` / `with_private_channels`; Python `StateGraph(..., input_schema=, output_schema=, private_schema=)`), `JsonState` | ✅ `StateGraph(StateT)` with annotated channels | input/output/private are channel allow-lists over one value schema |
| `LastValue` channel | ✅ | ✅ `LastValue` | concurrent writes fail in both |
| `Topic` channel | ✅ `Topic { accumulate }`: `topic`/`Topic(accumulate=true)` keeps values across supersteps, `topic_current` / `Topic(accumulate=false)` only the current superstep (lists are flattened, like LangGraph) | ✅ `Topic(accumulate=...)` | default plugin `topic` accumulates (LangGraph's constructor default is false) |
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
| `Command` | ⚠️ `Command::{resume, update, goto, goto_parent}`; nodes may return one via `add_command_node` or `From<Command> for NodeOutput`; invoke input supports resume+update; Python `Command(update=, goto=, graph=PARENT)` | ✅ `Command(graph=, update=, resume=, goto=, PARENT)` | `invoke(Command(goto=...))` starts at the given nodes; parent commands are meaningful only inside subgraphs |
| Node metadata / input schema / destinations / name | ✅ `add_node_with_metadata` / `set_node_input_schema` / `set_destinations` (Python `add_node(..., metadata=, input_schema=, destinations=, name=)`); arbitrary node names | ✅ `add_node(..., metadata=, input_schema=, destinations=, name=)` | `input_schema` is stored as metadata, not enforced |
| Deferred nodes | ✅ `add_node_deferred` / `set_deferred` / Python `add_node(..., deferred=True)` | ✅ `add_node(defer=True)` | deferred nodes may not be entry points or have outgoing edges |
| Graph name / graph object export | ⚠️ `compile_with(CompileOptions::with_name)` / Python `compile(name=...)`, `get_graph()` returns a `Graph` object with `draw_mermaid()` and `to_dict()` (nodes/edges/conditional edges) | ✅ `compile(name=...)`, `get_graph()` | not a full LangGraph `Graph` object (no draw_ascii/draw_png) |

### Execution

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Supersteps + recursion limit | ✅ default 25, per compile/run | ✅ default 25 | |
| Parallel superstep execution | ✅ scoped OS threads | ✅ asyncio tasks | CPU parallelism vs cooperative concurrency |
| Deterministic channel merge | ✅ task-order merge | ✅ | |
| Async engine | ⚠️ separate `AsyncStateGraph` adapter: checkpoints/HITL/retry/error routing, stream modes, store/context, per-node timeouts, subgraphs (namespaces + nested interrupts), node cache, batch, subgraph streaming; Python async bindings (`AsyncStateGraph`/`ainvoke`/`aresume`) behind `python-async` | ✅ native `ainvoke`/`astream` | async graphs use the `AsyncCheckpointer`/`BlockingSaver` adapter or sync savers directly |
| Batch (`batch` / `abatch`) | ✅ sync `batch` / `batch_run`, async `batch` / `batch_run` (bounded by `max_concurrency`), Python `batch` and async `abatch` | ✅ | async `batch_run` runs sequentially (only the run report is collected) |
| `max_concurrency` / run control | ⚠️ `RunConfig::max_concurrency` bounds `batch` workers; `RunControl` supports cooperative `abort()` checked between supersteps (Rust/Python) | ✅ `max_concurrency`, `RunControl` | abort happens at superstep boundaries; no pause/resume control modes |
| Durability modes | ⚠️ `Durability::{Sync, Async, Exit}` (`Async` behaves like `Sync`) | ✅ `durability=` | `Exit` persists only initial/interrupt/final checkpoints |
| Managed values (`IsLastStep` / `RemainingSteps`) | ⚠️ `remaining_steps()` / `is_last_step()` accessors (Rust + Python) | ✅ managed values in state annotations | no `RemainingSteps` state annotation / `deferred` nodes |
| TimeoutPolicy | ⚠️ `TimeoutPolicy` with `run_timeout` enforced (sync helper thread, async `tokio::time::timeout`); `idle_timeout` is used as the per-attempt fallback when `run_timeout` is absent | ✅ `TimeoutPolicy` (`run_timeout`, `idle_timeout`) | sync timeout cannot cancel a blocking closure; idle is not reset by progress events |
| Deferred nodes (`deferred=True`) | ✅ sync + async (`add_node_deferred` / `set_deferred`, Python `add_node(..., deferred=True)`); run after all regular work, may interrupt/resume | ✅ | deferred nodes may not be entry points or have outgoing/conditional edges |
| Context / runtime injection (`Runtime`, `context_schema`) | ✅ `Runtime` + `current_runtime` / `current_context` / `current_thread_id` / `current_namespace` / `get_store`; `RunConfig::with_context`; Python `context=` with `current_context`/`current_runtime`; no `context_schema` validation | ✅ | |
| `get_config` / `get_stream_writer` | ✅ `get_config()` (thread/ns config) and `get_stream_writer()` (custom writer) in Rust and Python; `custom()` remains | ✅ context helpers | config carries thread/ns only |
| Managed values (`IsLastStep`, `RemainingSteps`) | ✅ `remaining_steps()` / `is_last_step()` accessors (Rust + Python); no state-annotation form | ✅ `langgraph.managed` | |
| Retry policy | ✅ `RetryPolicy` (attempts, exponential backoff, cap, deterministic jitter, `retry_on` message predicates; Python `RetryPolicy(jitter=, retry_on=)`) | ✅ `RetryPolicy(initial_interval, backoff_factor, max_interval, max_attempts, jitter, retry_on)` | jitter is deterministic, `retry_on` matches message substrings |
| Error handling of nodes | ✅ `add_error_edge` + `last_error()` | ✅ per-node `add_node(error_handler=...)` | equivalent capability, different API; our handler runs in the next superstep |
| Timeout policy (`TimeoutPolicy`, node/run/idle timeouts) | ✅ `TimeoutPolicy` with enforced `run_timeout` (sync helper thread, async `tokio::time::timeout`); `idle_timeout` stored only | ✅ | |
| Functional API (`@entrypoint` / `@task`) | ⚠️ `functional::entrypoint` compiles to a one-node graph; `functional::task` memoizes results in `checkpoint.task_results` (persisted with every checkpoint, restored on resume) and optionally in `BaseStore`; Python `lg.entrypoint` / `lg.task` mirror this | ✅ `langgraph.func` | tasks execute inline (no parallel task scheduling); task results travel inside checkpoint documents rather than separate pending-write records |
| Prebuilt agents (`ToolNode`, `create_react_agent`, `tools_condition`) | ✅ Rust `prebuilt::{ToolNode, ToolDefinition, tools_condition, create_react_agent}` and Python `lg.ToolNode` / `lg.tools_condition` / `lg.create_react_agent`; tool calls read from message `tool_calls`, results appended as `role=tool` messages | ✅ `langgraph.prebuilt` | minimal ReAct loop; no LangChain tool objects or model integration |

### Checkpointing and state management

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Memory checkpointer | ✅ `InMemorySaver` | ✅ `InMemorySaver` | |
| LangGraph checkpoint-tuple JSON interop | ⚠️ `interop::checkpoint_to_langgraph` / `checkpoint_from_langgraph`, `CompiledGraph::{export_checkpoint, import_checkpoint}` (Python `export_checkpoint`/`import_checkpoint`); `metadata.langgraph_rs` keeps pending tasks/interrupts | ✅ checkpoint tuple | JSON-level conversion; LangGraph's msgpack serializer/database formats are not read directly |
| Persistent checkpointers (sync) | ✅ `SqliteSaver`, `RedisSaver`, `PostgresSaver` (features) | ✅ SQLite/Postgres/Redis saver packages | ours store an internal JSON format; **not interoperable** with LangGraph's checkpoint format |
| Async checkpointers | ✅ `AsyncCheckpointer` port + `BlockingSaver` adapter (offloads a sync saver to a blocking pool on a private runtime); `AsyncStateGraph::with_async_checkpointer`; Python async graphs wrap `compile(checkpointer=...)` automatically; `Async*Saver` aliases remain | ✅ `AsyncSqliteSaver`, `AsyncPostgresSaver`, ... | I/O is offloaded rather than using native async DB drivers |
| Checkpoint namespaces (`checkpoint_ns`) | ✅ `RunConfig::with_checkpoint_ns`, `get_state_in` / `resume_in` / `continue_thread_in` / `update_state_in`; namespace-aware queries in all adapters (sync + async) | ✅ | subgraph-internal namespace trees are still unsupported |
| Pending writes / `put_writes` | ⚠️ pending `Task`s embedded in the checkpoint | ✅ separate writes + checkpoint tuple | resume replays pending tasks in both |
| `get_state` / history | ⚠️ `get_state(thread)` / `get_state_in(thread, ns)`, `get_state_history(before=, limit=)`; snapshots expose values/next/tasks/interrupts/metadata/parent id/checkpoint_ns | ✅ + `subgraphs=`, filter, metadata/config, tasks | no LangGraph `config` object and no interrupt/task ids |
| `update_state` (`as_node`, `task_id`) | ⚠️ `update_state(thread, values, as_node=)` / `update_state_with_task(..., task_id=)`; Python `update_state(..., task_id=)` infers `as_node` from the matching pending task | ✅ | recomputes next tasks from `as_node`'s edges, or keeps pending tasks; `task_id` is validated against pending tasks |
| Thread management | ✅ `threads()` list, `delete_thread`, `prune_thread(thread, keep_last)` (memory/sqlite/redis/postgres/BlockingSaver) and `copy_thread`, Rust and Python | ✅ `delete_thread`, `prune`, `copy_thread`, `list` | |
| Time travel | ⚠️ `resume_from(thread, checkpoint_id, value)` | ✅ `get_state_history` + replay from any config | history supports `before`/`limit`; snapshots expose LangGraph-style `config` |
| Checkpoint metadata / created_at semantics | ⚠️ `parent_id`, `created_at_millis`, `source`, plus `channel_versions` / `versions_seen` / `updated_channels` (versions are step-based, `versions_seen` simplified) | ✅ rich metadata, versions/versions_seen | our checkpoint model is simpler |

### Interrupts / HITL

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| `interrupt(value)` pause | ✅ `Error::Interrupt` | ✅ `GraphInterrupt` / `Interrupt` | node re-executes on resume in both |
| Resume with a value | ✅ `resume` / `Command::resume` | ✅ `Command(resume=...)` | |
| Multiple interrupts in one node | ✅ a node may call `interrupt()` repeatedly; resume values are replayed in call order across resumes | ✅ resume dict/list matched per interrupt |
| `interrupt(response_schema=...)` | ⚠️ Python `lg.interrupt(value, response_schema=...)` and Rust `interrupt_with_schema` validate resume values against `type`/`const`/`enum`/`anyOf`/`oneOf`/`allOf`/`not`/`required`/`properties`/`additionalProperties`/`items`/numeric bounds/`minLength`/`maxLength`/`minItems`/`maxItems`/`uniqueItems`/`minProperties`/`maxProperties`; schema is stored on the interrupt and exposed in snapshots | ✅ | `pattern` and `$ref` are not supported |
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
| v2 stream events | ⚠️ Python `stream(..., version="v2")` / `astream(..., version="v2")` return `{type, ns, data}` stream parts (values/updates/custom/messages/debug/tasks/checkpoints/interrupt/usage; subgraph chunks carry the namespace); messages parts carry `{langgraph_node, tags}` metadata | ✅ `version="v2"` | metadata is minimal (no full LangChain callback metadata); Rust exposes the typed `StreamChunk` enum |
| Stream writer from arbitrary helpers | ⚠️ thread-local `custom()` | ✅ `get_stream_writer()` | |

### Subgraphs, caching, async, extras

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| Subgraphs | ✅ `add_subgraph` (shared state) + `add_subgraph_mapped` (different state types); child checkpoint namespaces, nested interrupts/resume and subgraph streaming supported | ✅ first-class | the interrupt surfaces as the parent subgraph node |
| Node-level cache | ⚠️ `CachePolicy` + `NodeCache` port, `InMemoryCache`, sync and async graphs | ✅ `CachePolicy(key_func, ttl)` + `BaseCache` (`InMemoryCache`, `RedisCache`) passed to `compile(cache=...)` | ours: in-memory only, no size limits, no persistent cache adapters |
| Long-term memory (`BaseStore`) | ⚠️ `BaseStore` port + `InMemoryStore` + `SqliteStore` (`sqlite` feature, Rust/Python); `StateGraph::with_store` / `CompileOptions::with_store` / `RunConfig::with_store`; nodes read it with `get_store()` | ✅ `compile(store=...)`, `Runtime.store`, `PostgresStore`, ... | only memory/SQLite adapters; `search` supports prefix + substring only |
| Runtime / context | ⚠️ `Runtime`, `current_runtime()`, `current_context()`, `current_thread_id()`, `current_namespace()`, `get_store()`, `get_config()`; `RunConfig::with_context` carries a JSON context, validated against `CompileOptions::with_context_schema` / Python `compile(context_schema=...)` | ✅ `Runtime`, `context_schema`, `get_config`, `get_store` | validation covers the JSON-schema subset used by interrupts |
| Mermaid export | ✅ sync + async `to_mermaid()` (+ Python) | ✅ `draw_mermaid()`, PNG, ASCII, xray subgraph view | static topology export only |
| Per-node usage/report (`RunReport`, `TraceSink`) | ✅ first-class (extension) | ⚠️ callbacks / metadata / LangSmith | not a LangGraph feature; superset here |
| Custom reducers | ✅ Rust `Channel::reducer` / `StateSchema::with_reducer`; Python schema values may be callables (`StateGraph({"field": my_reducer})`) | ✅ `Annotated[..., reducer]` | Python reducers are called with the GIL from the engine thread |
| Python bindings | ⚠️ PyO3 bindings (`python` feature): `StateGraph`/`CompiledGraph`, `interrupt`/`resume` (including multiple pending interrupts via an `{id: value}` map and sequential interrupts in one node), `Send`, `Command` (node output incl. `graph=PARENT` in sync/async subgraphs / invoke input), `RetryPolicy`/`CachePolicy`/`error_handler`, `compile(cache=True, store=...)`, saver objects (`InMemorySaver`, `SqliteSaver` with `sqlite`, `RedisSaver`/`PostgresSaver` with the matching features), `InMemoryStore` + `get_store`, subgraphs (namespaces, nested interrupts, streaming), runtime context, functional API (`lg.entrypoint` / `lg.task`), `add_messages`, `update_state` (incl. `task_id=`), `prune_thread`/thread list/delete/copy, `RunControl`, `get_config`/`get_stream_writer`, `stream_mode`, `batch`, run reports | ✅ full Python API | async Python covers `AsyncStateGraph` (incl. `add_subgraph`), Python coroutine nodes (awaited on a private event loop with the task context installed), `ainvoke`, `aresume`, `abatch`, `astream` (stream modes + subgraph streaming), `get_state`, `get_state_history`, thread delete/copy, async checkpointers (`python-async` feature) |

## How the matrix was produced

Statuses are based on the public Rust/Python APIs of this crate, its integration tests
(`tests/graph.rs`, `tests/channels.rs`, `tests/delta_channels.rs`, `tests/messages.rs`, `tests/parent_command.rs`, `tests/subgraph_streaming.rs`, `tests/multi_interrupt.rs`, `tests/advanced.rs`, `tests/deferred.rs`, `tests/destinations.rs`, `tests/functional.rs`, `tests/managed.rs`, `tests/store.rs`,
`tests/checkpoint.rs`, `tests/state_model.rs`, `tests/usage.rs`,
`tests/resilience.rs`,
`tests/subgraph.rs`, `tests/cache.rs`, `tests/visualize.rs`, `tests/async_graph.rs`,
`tests/async_resilience.rs`, `tests/async_subgraph.rs`, `tests/persistence.rs`, `python/tests/test_bindings.py`) and
introspection of the installed LangGraph 1.2.22 API. Performance is measured separately in
[`RESULTS.md`](RESULTS.md).