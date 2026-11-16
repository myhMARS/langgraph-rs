# Architecture

How `langgraph-rs` executes a graph: modules, the superstep loop, channel merging,
scheduling, checkpoints, interrupts and streaming. It describes the code in `src/` (crate
version 0.1.0) for contributors who need to reason about execution semantics.

## Module map

| Module | Responsibility |
| --- | --- |
| `error` | `Error`, `Interrupt`, `Result`. `Interrupt` is a `#[from]` variant and not a failure: the engine checkpoints and unwinds. |
| `channel` | `Channel` (`LastValue` / `Topic` / `Reducer`), the `Reducer` wrapper and `reducers::{append, add, messages}`. Owns merge policy. |
| `state` | `State`, `StateSchema` (strict or `dynamic`), `JsonState`, `ChannelUpdate`, `apply_updates` — the single point where writes become values. |
| `node` | `NodeFn`, `NodeOutput` (writes, `goto`, `Send`), `Send`, `Route`, `PathMap`, `ConditionalEdge`, `Task`. Pure data, no execution. |
| `context` | Thread-local `TaskContext` installed around each node call, plus `interrupt`, `custom`, `push_custom`, `current_node`. |
| `checkpoint` | `Checkpoint`, the `Checkpointer` trait, `InMemorySaver` / `MemorySaver`, id and timestamp helpers. |
| `command` | `Command`: an explicit resume description (value + optional patch). The `CompiledGraph` methods build the same intent in `executor::StartInput::Resume`. |
| `graph` | `StateGraph` builder, `CompiledGraph`, `CompileOptions`, `RunConfig`, `StateSnapshot`, `Targets`, `DEFAULT_RECURSION_LIMIT`; validation and public API. |
| `executor` | The engine: `execute`, `StartInput`, `run_superstep`, `run_task`, `save_checkpoint`. The only loop in the crate. |
| `stream` | `StreamChunk` and accessors, produced by the executor. |

Dependencies are acyclic: `graph` → `executor` → {`state`, `channel`, `checkpoint`, `context`,
`node`, `stream`} → `error`. `lib.rs` re-exports the public surface and defines `START`
(`"__start__"`) and `END` (`"__end__"`).

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
   `Send`) or a clone of the state, installs `TaskContext { node, resume_value }` and calls
   the node. `Error::Interrupt` is re-tagged with the node name; other errors become
   `Error::Node { node, message }`.
9. **Merge.** Results are folded in task order: `Updates` chunks per node, then
   `apply_updates` groups writes by channel and applies each policy. A merge error
   (`ConcurrentUpdate`, `UnknownChannel`, reducer failure) aborts before committing
   anything. Values are re-deserialized into `S` and a `Values` chunk is emitted.
10. **Schedule.** For each executed node that did not call `goto`, static edges are added,
    then conditional edges evaluated against the merged state; `goto` targets follow.
    `END`/`START` are dropped and names de-duplicated preserving first-seen order. `Send`
    tasks were queued in step 9 and bypass de-duplication.
11. **`interrupt_after` gate.** `step` increments, then the checkpoint is written with
    `interrupted = true` and `next = next_tasks`; the `Interrupt` chunk and error follow, so
    `continue_thread` resumes with the work after the paused node.
12. **Loop or finish.** Empty `next_tasks` returns the final typed state; otherwise they
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
| `Updates { step, node, update }` | per node that wrote channels, in task order | `{channel: write}` |
| `Custom { node, value }` | per `custom()` / `push_custom()`, when results are folded | arbitrary JSON |
| `Values { step, values }` | once per superstep, after the merge | full state object |
| `Checkpoint { checkpoint_id }` | every persisted checkpoint | id |
| `Interrupt { node, value }` | before returning `Error::Interrupt` | payload, `null` for hooks |

Ordering in a superstep is: per-node `Custom`/`Updates` (task order) → `Values` → `Checkpoint`
→ optional `Interrupt`. `interrupt_before` yields `Checkpoint` then `Interrupt` with no
updates. `stream` collects chunks; `stream_with` returns the final state but returns `Err` on
interrupt *after* delivering the chunks (`stream_with_keeps_chunks_when_the_graph_interrupts`).
A `custom()` call before a node interrupts is lost: the failed task's result is discarded.

## Extension points

- **Custom checkpointer.** Implement `Checkpointer` for `Send + Sync` storage: `put`, `get`,
  `latest`, `history`, optionally `threads`. The engine only appends and reads; `latest` must
  be the newest checkpoint and `history` newest-first. `resume_from` uses `get(thread, id)`,
  so retain old checkpoints.
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

## Known limitations and divergences

- **Synchronous and thread-local.** Nodes cannot be `async`; there is no tokio integration or
  cooperative cancellation, and `interrupt()`/`custom()`/`current_node()` only work on the
  thread executing the node, so work spawned elsewhere in a node loses the context.
- **No subgraphs.** A node may invoke another compiled graph manually, but state mapping,
  checkpoint namespacing and nested interrupts are unsupported.
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