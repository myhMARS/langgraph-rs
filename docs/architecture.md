# Architecture

English | [简体中文](architecture.zh-CN.md)

How `langgraph-rs` executes a graph: modules, the superstep loop, channel merging,
scheduling, checkpoints, interrupts and streaming. It describes the code in `src/` (crate
version 0.1.0) for contributors who need to reason about execution semantics.

## Module map

| Module | Responsibility |
| --- | --- |
| `kernel::error` | `Error`, `Interrupt`, `Result`. `Interrupt` is a `#[from]` variant and not a failure: the engine checkpoints and unwinds. |
| `kernel::channel` | `Channel` (`LastValue` / `Topic` / `Reducer`), the `Reducer` wrapper and `reducers::{append, add, messages}`. Owns merge policy. |
| `kernel::state` | `State`, `StateSchema` (strict or `dynamic`), `JsonState`, `ChannelUpdate`, `apply_updates` — the single point where writes become values. |
| `kernel::node` | `NodeFn`, `NodeOutput` (writes, `goto`, `Send`), `Send`, `Route`, `PathMap`, `ConditionalEdge`, `Task`. Pure data, no execution. |
| `kernel::context` | Thread-local `TaskContext` installed around each node call, plus `interrupt`, `custom`, `push_custom`, `current_node`. |
| `kernel::checkpoint` | `Checkpoint`, the `Checkpointer` trait, `InMemorySaver` / `MemorySaver`, id and timestamp helpers. |
| `kernel::command` | `Command`: an explicit resume description (value + optional patch). The `CompiledGraph` methods build the same intent in `executor::StartInput::Resume`. |
| `kernel::graph::builder` | `StateGraph`, the mutable builder, its edges and the validation rules. |
| `kernel::graph::compiled` | `CompiledGraph`, the immutable runnable form with `invoke` / `run` / `stream` / `resume` / state queries. |
| `kernel::graph::options` | `CompileOptions`, `RunConfig`, `Targets`, `DEFAULT_RECURSION_LIMIT`. |
| `kernel::graph::snapshot` | `StateSnapshot`, the typed view of a checkpoint. |
| `kernel::graph::visualize` | Mermaid renderer shared by the sync and async compiled graphs (`to_mermaid`). |
| `kernel::cache` | `NodeCache` port, `CachedOutput`, `CachePolicy` (TTL + custom key) for node-level caching. |
| `kernel::store` | `BaseStore` port, `StoreItem` / `Namespace`, `Runtime` and the thread-local run context (`get_store`, `current_context`, `current_runtime`). |
| `kernel::executor::mod` | `execute` and `StartInput`: the only loop in the crate; owns seeding, merge and control flow. |
| `kernel::executor::superstep` | `run_superstep` / `run_task`: inline or scoped-thread node execution, interrupt re-tagging, per-node timing. |
| `kernel::executor::scheduling` | Edge evaluation, `goto` override handling, `END`/`START` filtering and de-duplication. |
| `kernel::executor::checkpointing` | `save_checkpoint`: the only checkpoint writer. |
| `kernel::usage` | `Usage`, `NodeExecution`, `RunReport` and `RunOutcome`: per-node consumption/timing aggregation. |
| `kernel::stream` | `StreamChunk` and accessors, produced by the executor. |
| `adapters::python` | PyO3 bindings (feature `python`): `StateGraph` / `CompiledGraph` over `JsonState`, `interrupt`, usage reporting. |
| `adapters::sqlite` / `redis` / `postgres` | Persistent `Checkpointer` adapters behind the `sqlite`, `redis`, `postgres` features; each stores the JSON checkpoint payload with a thread index for `latest` / `history`. |
| `adapters::tokio_runtime` | Tokio async adapter (feature `async`): `AsyncStateGraph`, `AsyncCompiledGraph`, `AsyncContext`; reuses kernel state/scheduling. |
| `kernel::ports` | Ports: `Checkpointer` (persistence) and `TraceSink` (consumption records); the only contracts the kernel depends on. |
| `kernel::plugin` | Microkernel registry: `Kernel`, `KernelBuilder`, `Plugin`, `ChannelFactory`, `BuiltinChannels`. |
| `adapters::memory` | `InMemorySaver` / `MemorySaver`: the in-memory checkpointer adapter. |

Dependencies point inward: `adapters` → `kernel`, and inside the kernel `graph` → `executor` →
{`state`, `channel`, `checkpoint`, `context`, `node`, `stream`, `usage`} → `error`. The kernel
never references a concrete adapter; adapters implement `kernel::ports` traits. `lib.rs` is a pure
facade: it re-exports the public surface and defines `START` (`"__start__"`) and `END`
(`"__end__"`), so moving code between layers never breaks users.

## Microkernel and hexagonal architecture

- **Hexagonal core.** `kernel` is domain logic; `adapters` is infrastructure. Ports
  (`kernel::ports::Checkpointer`, `TraceSink`) are the seam. `InMemorySaver`
  (`adapters::memory`) and the PyO3 module (`adapters::python`) implement ports or translate
  into the domain.
- **Microkernel composition.** `Kernel` holds named `ChannelFactory`s and `Arc<dyn TraceSink>`
  adapters. `Kernel::builtin()` installs `BuiltinChannels` (`last_value`, `topic`, `append`,
  `add`, `messages`); `Kernel::builder().with_plugin(..)` composes downstream plugins.
  `Kernel::schema([(name, spec)])` turns plugin names into a `StateSchema`, so graph definitions
  refer to capabilities by name rather than constructor.
- **Compilation injects the kernel.** `CompileOptions::with_kernel(kernel)` copies the trace
  sinks into the compiled graph; the executor calls every sink per node execution. The Python
  bindings build their schema through the same built-in kernel, so both APIs resolve the same
  plugin names.
- **Extension rules.** To add a policy, implement `Plugin` and register a factory; to add
  persistence or tracing, implement the port and attach the adapter via `CompileOptions` or
  `Kernel::builder().with_trace_sink(..)`. No kernel file needs to change.

## The superstep algorithm

As implemented in `executor::execute`:

1. **Validate and compile.** `StateGraph::validate` rejects an empty graph, duplicate node
   registrations, reserved names, a missing or duplicated entry point, edges or path-map
   targets that are not nodes; `compile_with` also rejects interrupt hooks naming unknown
   nodes.
2. **Resolve the budget.** `config.recursion_limit` overrides the compiled limit, which
   defaults to `DEFAULT_RECURSION_LIMIT = 25`. A `thread_id` without a checkpointer fails
   with `MissingCheckpointer` before any work.
3. **Seed `(values, state, tasks, step)`.** `Values` serializes the typed input and builds
   entry tasks, filtering `END`. `Updates` starts from an empty channel map, applies the
   patch via `apply_updates` (node `"<input>"`) and deserializes. `Resume` loads the
   requested or latest checkpoint, keeps its id as parent, copies `paused_before` into a
   one-shot `skip_interrupt_before` flag, applies an optional patch (node `"<resume>"`),
   sets `tasks = checkpoint.next` and stamps each task with the resume value.
4. **Short circuit.** No tasks returns the state immediately — also why `continue_thread`
   on a finished thread is a no-op returning the final state.
5. **Root checkpoint.** A fresh run bound to a thread writes a step-0 checkpoint whose
   `next` is the entry-task list and whose `parent_id` is the thread's previous latest
   checkpoint, so history starts before the first node runs.
6. **Budget check.** `iteration >= budget` fails with `Error::RecursionLimit(budget)`. The
   counter measures supersteps, not node executions: a ten-node fan-out costs one.
7. **`interrupt_before` gate.** Unless skipped, the first hooked task pauses the run:
   checkpoint with `interrupted = true`, `paused_before = true`, emit an `Interrupt` chunk
   with a `null` payload and return `Error::Interrupt`.
8. **Run the superstep.** One task runs inline; two or more run on `std::thread::scope`
   handles joined in task order. `run_task` picks the node, chooses `task.input` (for
   `Send`) or a clone of the state, installs `TaskContext { node, resume_value, error }`,
   measures the wall-clock duration and calls the node. Node errors are retried according to
   the node's `RetryPolicy` (interrupts are never retried). A permanent failure becomes a
   `NodeFailure { node, message, attempts }`; if the graph has an `add_error_edge` handler the
   failure is queued as a handler task (visible through `last_error()`), otherwise the run
   aborts with `Error::Node`.
9. **Record usage.** Each result's `record_usage` / `record_tokens` calls and any
   `NodeOutput::with_usage` value are merged into one `Usage`; a `NodeExecution` (node, step,
   duration, usage) is appended to the run's `RunReport` and emitted as
   `StreamChunk::Usage`.
10. **Merge.** Results are folded in task order: `Updates` chunks per node, then
   `apply_updates` groups writes by channel and applies each policy. A merge error
   (`ConcurrentUpdate`, `UnknownChannel`, reducer failure) aborts before committing
   anything. Values are re-deserialized into `S` and a `Values` chunk is emitted.
11. **Schedule.** For each executed node that did not call `goto`, static edges are added,
    then conditional edges evaluated against the merged state; `goto` targets follow.
    `END`/`START` are dropped and names de-duplicated preserving first-seen order. `Send`
    tasks were queued in step 9 and bypass de-duplication.
12. **`interrupt_after` gate.** `step` increments, then the checkpoint is written with
    `interrupted = true` and `next = next_tasks`; the `Interrupt` chunk and error follow, so
    `continue_thread` resumes with the work after the paused node.
13. **Loop or finish.** Empty `next_tasks` returns the final typed state; otherwise they
    become the next iteration's `tasks`.

If a node raises an interrupt, the checkpoint holds the *pre-superstep* values and the whole
task list, so a resume replays the entire superstep (see "Interrupt and resume flow").

## Channel merging

`apply_updates` (`src/state.rs`) is the only place values change:

1. Writes are grouped into an `IndexMap<channel, (values, nodes)>` in result order. Since
   results are joined in task order, grouping and error messages are deterministic despite
   parallel execution.
2. Each channel is resolved through the schema: strict schemas reject undeclared channels
   with `Error::UnknownChannel`; `StateSchema::dynamic` (`JsonState`) treats them as
   last-value channels.
3. `LastValue` fails with `Error::ConcurrentUpdate { channel, nodes }` when a superstep
   wrote it more than once; otherwise the single write is kept.
4. `Topic` requires the current value to be absent, `null` or an array, and extends it with
   every write.
5. `Reducer` initializes from the first write when no value exists (`skip = 1`) and folds
   the rest left-to-right, mirroring LangGraph's `BinaryOperatorAggregate`. A reducer error
   aborts the merge.

**Why `LastValue` rejects concurrent writes.** Last-write-wins is order-dependent, and in a
parallel superstep "last" is decided by thread scheduling — a silent winner would make results
nondeterministic. Rejecting the write turns the ambiguity into a deterministic error naming the
offending nodes and pointing at the fix (declare a reducer). It also protects the checkpoint
contract: a checkpoint must represent one unambiguous state because it is replayed on resume.
Reducers must be pure and order-insensitive (append, additive merge, merge-by-id); reducers
sensitive to arrival order inherit task order and are stable only as long as the graph is.

## Scheduling and de-duplication

- Work is collected per *executed node name*, not per task. If three `Send` workers all have
  an edge to `finish`, it is collected three times and de-duplicated by a `HashSet` cursor,
  so it runs once per superstep (`parallel_fan_out_runs_nodes_and_joins_once` pins this).
- De-duplication is by node name only: distinct edges collapse, but two `Send` tasks to the
  same node do not, because each carries its own `arg`.
- A node reached by both a static edge and a `Send` runs twice in the same superstep (once
  with the state, once with the override); if both write a `LastValue` channel that is a
  `ConcurrentUpdate`.
- `NodeOutput::goto` is an override, not an addition: a node that returned a target has its
  static *and* conditional edges skipped for that step (`explicit_goto_overrides_static_edges`).
- Routers run after the merge, so they see post-superstep state, once per executed source node.

## `Send` fan-out

`Send { node, arg }` is produced by `NodeOutput::send` during a node. The executor validates the
target node and that `arg` is `null` or a JSON object (`Error::Routing` otherwise), then queues
`Task { node, input }` for the *next* superstep. `run_task` deserializes `task.input` into `S`
and passes it instead of the graph state, giving each map task its own input. Tasks run
concurrently and merge through normal channels, so the reduce channel must be a reducer. Queued
tasks are persisted in the checkpoint's `next` list, so an interrupt during a map/reduce step is
resumable.

## Checkpoint lifecycle and data model

A checkpoint is written only when both a `thread_id` and a checkpointer exist:

| Moment | `step` | `interrupted` | `paused_before` | `next` |
| --- | --- | --- | --- | --- |
| fresh run start, before any node | 0 | false | false | entry tasks |
| `interrupt_before` pause | current | true | true | task list including the paused node |
| node called `interrupt()` | current (pre-merge values) | true | false | whole task list of the superstep |
| after a completed superstep | current + 1 | false | false | next scheduled tasks |
| `interrupt_after` pause | current + 1 | true | false | tasks following the paused node |

Fields: `id` (`{thread_id}-step{step}-{sequence:06}`, sequence from a process-local
`AtomicU64`), `thread_id`, `step` (completed supersteps), `values` (all channels as one JSON
object), `next` (`Vec<Task>` with `node`, optional `input`, optional `resume`), `interrupted`,
`paused_before`, `parent_id` (previous checkpoint, forming the history chain) and
`created_at_millis`.

`InMemorySaver` keeps `Mutex<HashMap<thread_id, Vec<Checkpoint>>>` in insertion order: `latest`
returns the last element, `history` the reversed list (newest first), and a poisoned lock is
recovered with `into_inner`. `get_state` maps the latest checkpoint to `StateSnapshot<S>` (typed
values, `next` node names, `step`, `interrupted`, `created_at_millis`); `get_state_history` maps
the chain.

## Interrupt and resume flow

1. A node calls `interrupt(payload)`. `context::interrupt` inspects the thread-local
   `TaskContext`: a pending resume value is taken and returned (`resume_consumed = true`),
   otherwise it returns `Err(Error::Interrupt(Interrupt { node, value: payload }))`.
2. `run_task` keeps the interrupt variant, re-tagged with the executing node. The executor
   writes a checkpoint from the values *before* this superstep's merge with
   `interrupted = true`, emits `StreamChunk::Interrupt`, and returns the error.
3. The caller inspects `error.interrupt()` / `Error::is_interrupt()`, or calls `get_state`
   and sees `interrupted == true` with `next == [interrupted node]`.
4. `resume`, `resume_with_updates` or `resume_from` loads the checkpoint, applies any patch
   through the channels, stamps `resume` onto every pending task and re-enters the loop at
   the checkpoint's `step`.
5. **The interrupted node runs again from the top.** At the same `interrupt()` call the
   installed context returns the resume value instead of pausing. Everything the node did
   before that call therefore happens twice.
6. Execution continues normally from the node's writes.

**Sibling re-execution.** The checkpoint's `next` holds the whole task set of the interrupted
superstep, so a resume re-runs every task in it — including siblings that finished before the
interrupting node. Their writes were never merged (merging happens after all tasks join), so
state stays consistent, but their external side effects repeat. Keep workers idempotent, or put
human-in-the-loop interrupts in single-task supersteps.

**`paused_before`.** An `interrupt_before` pause checkpoints the pending tasks with
`paused_before = true`; on resume the flag becomes `skip_interrupt_before`, suppresses the hook
for exactly one iteration and is cleared, so the node finally runs instead of pausing forever.
An `interrupt_after` pause needs no flag: its checkpoint already holds the post-superstep values
and the following tasks, so `continue_thread` just runs them. Hook payloads are `Value::Null`;
only `interrupt()` carries a question. `continue_thread` passes no resume value, so using it on
a thread paused by `interrupt()` re-raises the interrupt.

## Streaming

`StreamChunk` is produced synchronously by the executor into a `FnMut(StreamChunk)` sink on
the calling thread:

| Variant | When | Payload |
| --- | --- | --- |
| `Usage { execution }` | per node, before its writes are merged | `NodeExecution` (node, step, duration, usage) |
| `Updates { step, node, update }` | per node that wrote channels, in task order | `{channel: write}` |
| `Custom { node, value }` | per `custom()` / `push_custom()`, when results are folded | arbitrary JSON |
| `Values { step, values }` | once per superstep, after the merge | full state object |
| `Checkpoint { checkpoint_id }` | every persisted checkpoint | id |
| `Interrupt { node, value }` | before returning `Error::Interrupt` | payload, `null` for hooks |

Ordering in a superstep is: per-node `Usage`/`Custom`/`Updates` (task order) → `Values` →
`Checkpoint` → optional `Interrupt`. `interrupt_before` yields `Checkpoint` then `Interrupt` with no
updates. `stream` collects chunks; `stream_with` returns the final state but returns `Err` on
interrupt *after* delivering the chunks (`stream_with_keeps_chunks_when_the_graph_interrupts`).
A `custom()` call before a node interrupts is lost: the failed task's result is discarded.

## Consumption tracking

Every node execution produces a `NodeExecution { node, step, duration_millis, usage }`:

- **Automatic duration.** `superstep::run_task` measures the wall-clock time around the node
  call, so timing works even when a node reports nothing.
- **Reported usage.** Inside a node, `record_usage(usage)` / `record_tokens(input, output)`
  push records into the thread-local `TaskContext`; `NodeOutput::with_usage` attaches one to the
  return value. All records of one execution are summed by `Usage::merge` (tokens and cost add,
  model is the first non-empty one, metadata keys merge).
- **Aggregation.** `RunReport` keeps executions in run order and offers `usage_by_node()`
  (`BTreeMap<node, Usage>`), `total_usage()`, `total_tokens()` and `total_cost_usd()`.
  `RunOutcome` pairs the final state with the report, returned by `CompiledGraph::run` /
  `run_updates`; `invoke` discards the report to keep its state-only signature.
- **Live streaming.** `StreamChunk::Usage` is emitted per node before its writes merge, so a
  frontend can show consumption while the graph runs. The report is not persisted in
  checkpoints, so a resumed run reports only the nodes it actually executes.

The Python bindings expose the same machinery through `__usage__` return keys and
`compiled.run(...)["report"]`.

## Python bindings

Enabled with the `python` feature (`pyo3`, `extension-module`). The bindings are a thin
translation layer over the Rust engine:

- **State.** Python uses `JsonState`: dicts in, dicts out. `StateGraph(schema)` maps channel
  names to the built-in policies; `None` means a dynamic (permissive) schema.
- **Nodes and routers.** `Py<PyAny>` callables are wrapped in the same `NodeFn` / router
  closures the Rust API uses. `node_from_python` builds the state dict, acquires the GIL with
  `Python::attach`, calls the callable and converts the returned dict (plus reserved
  `__goto__` / `__usage__` keys) back into `NodeOutput`.
- **GIL handling.** `CompiledGraph` methods call `py.detach(...)` around the engine, so the GIL
  is released while nodes run and re-acquired by each node's `Python::attach`. This avoids
  deadlocks between the calling thread and the scoped superstep threads.
- **Interrupts.** `langgraph_rs.interrupt(payload)` calls the Rust `context::interrupt`; a pause
  surfaces to Python as the `GraphInterrupt` exception carrying the payload, and `resume` feeds
  the answer back into the same re-executed node. `pyerr_to_error` maps the exception back to
  `Error::Interrupt`.
- **Errors and results.** Engine errors become `RuntimeError`; `run` returns
  `{"state": ..., "report": ...}`; `stream` returns chunk dicts; checkpoint queries return
  snapshot dicts.

## Retries, error routing and subgraphs

- **Retries.** `StateGraph::set_retry` / `add_node_with_retry` attach a `RetryPolicy`
  (`max_attempts`, `initial_backoff`, `multiplier`, `max_backoff`). `superstep::run_task`
  loops over attempts, sleeps the backoff between them, sums the usage of all attempts and
  records the attempt count on `NodeExecution`. `Error::Interrupt` short-circuits the retry
  loop.
- **Error routing.** `add_error_edge(from, handler)` registers a fallback node. When a node
  fails permanently, the executor records the failure (trace sink + `NodeExecution.error`),
  queues `Task { node: handler, error: Some(failure) }` for the next superstep and skips the
  failed node's outgoing edges. The handler sees the failure through the thread-local
  `last_error()`. Handler tasks are de-duplicated per superstep and take precedence over a
  normal edge to the same node; the failure travels through checkpoints, so resuming a run
  paused inside a handler keeps the error context.
- **Subgraphs.** `add_subgraph(name, Arc<CompiledGraph<S>>)` wraps the compiled graph in a
  normal node: it invokes the subgraph with the parent state, compares the serialized
  before/after states and emits `NodeOutput::replace` writes for every changed channel.
  `replace` bypasses reducers, so parent reducer channels are not appended twice; the
  executor applies replacements after the regular channel merge (and still validates that
  strict schemas declare them). Subgraphs share the state type and have no own checkpointer,
  state mapping or nested interrupts — an interrupt inside a subgraph surfaces as a node
  failure.

## Async runtime adapter (Tokio)

`src/adapters/tokio_runtime/` implements the `async` feature without touching the kernel:

- **Builder.** `AsyncStateGraph` mirrors the sync builder for async node closures
  `Fn(S, AsyncContext) -> impl Future<Output = Result<NodeOutput>>`; it validates the same
  structural rules (entry point, targets, reserved names, conditional path maps).
- **Supersteps.** All tasks of a superstep are awaited together with
  `futures::future::join_all`, so pending I/O overlaps; results are folded in task order and
  merged through the same `state::apply_updates` used by the sync engine. Scheduling reuses
  `executor::scheduling::collect_next_nodes` through closures, keeping one implementation of
  edge evaluation and de-duplication.
- **Context.** Futures interleave, so the sync thread-local context cannot be used.
  `AsyncContext` is an `Arc<Mutex<..>>` handle that nodes use to report usage
  (`record_usage` / `record_tokens`), emit `custom` values, and read `last_error()`.
- **Checkpoints and HITL.** The adapter owns the same checkpoint lifecycle as the sync
  executor: a step-0 checkpoint for fresh thread-bound runs, a checkpoint after every
  superstep, and an interrupted checkpoint (pre-superstep values + pending tasks) when a node
  calls `ctx.interrupt(..)`, when a node is paused by `interrupt_before`, or after
  `interrupt_after`. `resume(thread, value)` / `resume_with_updates` / `continue_thread`
  reload the latest checkpoint, re-stamp the pending tasks with the resume value and continue;
  `paused_before` suppresses the same hook exactly once.
- **Retries.** `add_node_with_retry` / `set_retry` reuse the kernel `RetryPolicy`; the async
  runner sleeps with `tokio::time::sleep` between attempts and never retries interrupts.
  Duration, attempt counts and usage of all attempts land in the report and trace sinks.
- **Error routing.** `add_error_edge` queues the handler as a task with the `NodeFailure`, so
  `ctx.last_error()` works; handler tasks are de-duplicated per superstep and take precedence
  over a regular edge to the same node.
- **Scope.** Nodes, static/conditional edges, goto, `Send`, streaming, usage/trace,
  checkpointing, interrupts, retries and error routing are supported. Nested subgraphs and
  Python bindings are not exposed for async graphs yet.

## Node-level caching

`kernel::cache` defines the `NodeCache` port (`get` / `put` / `clear`), the serializable
`CachedOutput` payload and `CachePolicy`. A policy is attached per node
(`StateGraph::set_cache`) together with a backend (`with_cache`, e.g. the `InMemoryCache`
adapter). `executor::superstep::run_task` computes the cache key from the node name and its
input channel values (or a custom key function), replays a hit as `NodeOutput::from_cached`
and records `NodeExecution.cached = true` with `attempts = 0`; on a miss it runs the node and
stores updates/replacements/goto/sends plus merged usage with the policy TTL. Errors are never
cached. The adapter currently keeps entries in memory (unbounded); the port is the extension
point for persistent caches.

## Visualisation

`kernel::graph::visualize::render_mermaid` takes a layout view (node names, edge tables,
entry, retries, error edges, interrupt hooks) so `CompiledGraph::to_mermaid` and the async
`AsyncCompiledGraph::to_mermaid` produce identical output. The renderer emits a
`graph TD` document: static edges as solid arrows, conditional edges through
`<node>_router_<n>{"router"}` diamonds with path-map labels (or an explanatory comment when
no path map exists), error edges labelled `on error`, and node labels annotated with
`retry xN` / `pause before` / `pause after`.

## Persistent checkpointers

The `Checkpointer` port stays small (`put` / `get` / `latest` / `history` / `threads`), so
adapters only translate a `Checkpoint` into their storage. The bundled adapters share one
strategy: encode the whole checkpoint as JSON, store it under `(thread_id, id)`, and keep a
per-thread index ordered by `step, created_at` for latest/history queries.

- `adapters::sqlite` (feature `sqlite`): one table with a `(thread_id, id)` primary key and
  an index for newest-first history. Needs a system SQLite library (`libsqlite3`).
- `adapters::redis` (feature `redis`): JSON strings under
  `langgraph_rs:checkpoint:{thread}:{id}`, a sorted-set index per thread and a set of thread
  ids; entries have no TTL unless the backend is configured for it.
- `adapters::postgres` (feature `postgres`): a table created on connect with the same
  ordering guarantees as SQLite.

Adapters are constructed before compiling (`SqliteSaver::open`, `RedisSaver::connect`,
`PostgresSaver::connect`) and passed to `CompileOptions::with_checkpointer`. They are
synchronous and hold a `Mutex` around their connection.

## Extension points

- **Custom checkpointer (adapter).** Implement `kernel::ports::Checkpointer` for `Send + Sync`
  storage: `put`, `get`, `latest`, `history`, optionally `threads`. The engine only appends and
  reads; `latest` must be the newest checkpoint and `history` newest-first. `resume_from` uses
  `get(thread, id)`, so retain old checkpoints.
- **Custom channel plugin (microkernel).** Implement `Plugin`, call
  `kernel.register_channel("my_spec", || Channel::reducer(...))`, then build a schema with
  `kernel.schema([(channel, "my_spec")])`. `BuiltinChannels` is just the first plugin.
- **Custom trace adapter (port).** Implement `TraceSink` (or pass a closure) and attach it with
  `CompileOptions::new().with_kernel(Kernel::builder().with_trace_sink(sink).build())`; it
  receives every `NodeExecution` — feed logs, metrics, OTel or a database.
- **Custom reducers.** `Reducer::new(|current, incoming| ...)` or
  `StateSchema::reducer("total", |current, incoming| ...)`, attached with `.with_reducer`.
  Keep them pure; returning `Err` fails the run at merge time.
- **Typed state vs `JsonState`.** `State` gives compile-time field access and a strict schema
  that catches typo'd channels; `JsonState` gives runtime-shaped channels and needs
  `StateGraph::with_schema` to declare non-`LastValue` behaviour. Both must serialize to a
  JSON object.
- **Nodes.** Any `Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static`; the builder stores it
  in an `Arc`, so graphs are cheap to clone. A node may call another compiled graph, an LLM,
  or bridge into a runtime it owns.
- **Sinks.** `stream_with` takes any `FnMut(StreamChunk)` — the natural seam for a frontend,
  progress bar or metrics collector.
- **Consumption.** `record_usage` / `record_tokens` / `NodeOutput::with_usage` feed any
  accounting system; use `RunReport::executions` for per-call granularity or `usage_by_node`
  for per-node totals.
- **Python.** Add methods to `PyStateGraph` / `PyCompiledGraph` in `src/adapters/python/graph.rs` (conversion helpers in `convert.rs`); the
  conversion helpers (`py_to_value`, `value_to_py`, `node_output_from_py`) keep the FFI layer
  thin.

## Known limitations and divergences

- **Sync engine is thread-local.** The default engine runs sync nodes; `interrupt()`,
  `custom()` and `current_node()` only work on the thread executing the node, so work spawned
  elsewhere in a node loses the context. Async nodes live in the separate Tokio adapter and
  use an explicit `AsyncContext` instead.
- **Async adapter scope.** The async adapter has no node caching yet (execution, streaming,
  checkpoints, HITL, retries and error routing are supported); its nodes receive the state by
  value (a clone per execution).
- **Persistent adapters are synchronous.** SQLite/Redis/Postgres adapters hold a
  `Mutex<Connection>` and block while writing; they are not surfaced through the async
  adapter yet, and Redis entries have no TTL by default.
- **Subgraphs share the state type.** `add_subgraph` embeds a compiled graph as a node with
  `replace` writes for changed channels; per-subgraph state mapping, checkpoint namespacing
  and nested interrupts are unsupported.
- **All-or-nothing supersteps.** An interrupt discards the superstep's writes and re-runs its
  whole task list on resume, so side effects must be idempotent.
- **JSON state.** No non-serializable handles; integer/float fidelity is JSON's.
  `state_to_values` rejects non-object state with `Error::NonObjectState`.
- **`invoke_updates` starts from the patch only**, not from `S::default()`, so fields absent
  from the patch must be serde-defaultable.
- **Coarse hooks and node-only `Send`.** Only the first matching task is reported, with a `null`
  payload; routers are pure, and `Route` carries plain names, so fan-out with per-task state
  happens in the node via `NodeOutput::send`.
- **Checkpoint ids are process-local** (`thread-step-sequence`, not UUIDs); `InMemorySaver` keeps
  everything for the process lifetime with no pruning, and there are no retries, caching,
  deferred nodes, checkpoint namespaces, per-node policies, graph visualisation or time-travel
  UI — time travel is `resume_from(thread, id, value)`.
- **Reducer errors are unwrapped**: they surface as `Error::Other` without the node name,
  unlike node errors, which are wrapped in `Error::Node`.
- **Retry/error timing.** Backoff sleeps happen on the task thread (a parallel superstep's
  other nodes have already joined before scheduling proceeds); only one failure per handler is
  routed per superstep, and handler failures are not retried by the handler's own policy.
- **Python specifics.** Only built-in reducers are available from Python; state is JSON
  (`JsonState`), `Send` is not exposed yet, and nodes/`__goto__`/`__usage__` are the supported
  return shape. Each Python node call re-acquires the GIL, so CPU-bound Python nodes do not
  benefit from superstep parallelism beyond what the GIL allows.
- **Usage is run-scoped, not checkpointed.** Resuming a thread starts a fresh report; usage
  reported before an interrupt is not replayed.