# langgraph-rs

English | [简体中文](README.zh-CN.md)

**A Rust port of LangGraph's core runtime**: a synchronous, state-and-graph engine for
orchestrating LLM agents and workflows. Work is a `StateGraph` whose nodes are plain
`Fn(&State) -> Result<NodeOutput>` closures; nodes return *writes*, the engine merges them
through channels, routes edges, checkpoints between steps, and can pause for a human and
resume where it stopped. No async runtime, no hidden globals, and optional Python
bindings for teams that need them.

## Features

- **Channels and reducers** — `LastValue`, `Topic` or any custom fold over your own
  `Serialize + Deserialize` state type, validated against a `StateSchema`.
- **Static + conditional edges** — multi-target fan-out, routers returning node names, symbolic
  keys resolved through a `PathMap`, conditional entry points, `goto` overrides.
- **Parallel supersteps + `Send` map/reduce** — nodes scheduled together run on scoped threads in
  one wall-clock window; `Send` adds a task per item with its own state, reduced by channel.
- **Retries + error routing** — a per-node `RetryPolicy` (attempts, exponential backoff) and
  `add_error_edge` fallback handlers that read the failure with `last_error()`.
- **Subgraphs** — `add_subgraph` embeds a compiled graph as a node; it shares the state type
  and publishes the channels it changed.
- **Mermaid visualisation** — `graph.to_mermaid()` renders nodes, static/conditional edges,
  path-map labels, error edges, retries and interrupt hooks; also exposed from Python.
- **Node-level caching** — opt-in `CachePolicy` per node plus a `NodeCache` port; the
  `InMemoryCache` adapter supports TTLs and custom cache keys, and cached executions are
  marked in run reports.
- **Persistent checkpointers** — `SqliteSaver`, `RedisSaver` and `PostgresSaver` implement
  the same `Checkpointer` port behind the `sqlite` / `redis` / `postgres` features.
- **Tokio async runtime (feature `async`)** — `AsyncStateGraph` runs async nodes and awaits
  concurrent supersteps on a Tokio runtime; the sync path stays dependency-free.
- **Checkpointing + thread history** — `InMemorySaver`, typed snapshots, parent-linked
  history and `resume_from` time travel.
- **Human-in-the-loop** — `interrupt()` parks a run with a payload, `resume` re-executes the
  node and feeds the answer back, and `interrupt_before` / `interrupt_after` add coarse gates.
- **Streaming** — values, per-node updates, custom events, interrupts, checkpoint ids.
- **Per-node consumption tracking** — nodes report tokens/cost with `record_usage` or
  `NodeOutput::with_usage`; `run` returns a `RunReport` with per-node duration, usage and
  aggregated totals.
- **Python bindings** — the optional `python` feature exposes the same engine to Python
  (`StateGraph` / `CompiledGraph`, `interrupt` / `resume`, run reports) through PyO3.
- **Microkernel + hexagonal architecture** — a domain kernel (graph model, superstep
  engine, ports) with adapters for checkpointing, tracing and language bindings; channel
  merge policies are plugins registered in a `Kernel`.
- **Benchmarked against Python LangGraph** — same scenarios across Rust native, the PyO3
  bindings and LangGraph, plus a feature-parity matrix (see [benchmarks/](benchmarks/)).

## Why a Rust port?

LangGraph's model — state, supersteps, explicit writes, interrupts — is useful well beyond
Python. A faithful port should not be a syntax translation: the state is a compile-time-checked
type, a node writing an undeclared channel fails the run, and an ambiguous merge is impossible —
concurrent writes to a `LastValue` channel are an error instead of a race. Nodes are synchronous
closures, supersteps run on `std::thread::scope`, and `Checkpointer` is a trait, so persistence
is whatever database you operate.

## Architecture: microkernel + hexagonal

The crate is organised as a **kernel** (pure domain logic) surrounded by **adapters**
(infrastructure), wired through **ports** (traits):

```text
                 ┌────────────────────────── adapters ──────────────────────────┐
   Python ─────► │ adapters::python (PyO3)   adapters::memory (checkpoint saver) │
                 └───────────────▲───────────────────────────▲──────────────────┘
                                 │ implements ports          │
                 ┌───────────────┴───────────────────────────┴──────────────────┐
                 │ kernel::ports   Checkpointer, TraceSink                      │
                 │ kernel::plugin  Kernel / Plugin / ChannelFactory             │
                 │ kernel          graph model, channels, state, executor,      │
                 │                 usage, streaming, checkpoints, interrupts     │
                 └──────────────────────────────────────────────────────────────┘
```

- **Ports** (`kernel::ports`) are the only contracts the kernel depends on:
  `Checkpointer` for persistence and `TraceSink` for consumption records.
- **Adapters** (`adapters::*`) implement ports: `InMemorySaver` for checkpoints, the PyO3
  bindings for Python callers, and your own structs for Redis/Postgres/OTel/etc.
- **Microkernel** (`kernel::plugin`): the kernel resolves capabilities from a `Kernel`
  registry instead of constructing them. Built-in channel policies (`last_value`, `topic`,
  `append`, `add`, `messages`) are registered by `BuiltinChannels`; downstream crates add
  policies with a `Plugin`:

```rust
use langgraph_rs::{Channel, CompileOptions, Kernel, Plugin, StateGraph};

struct MyReducer;

impl Plugin for MyReducer {
    fn name(&self) -> &'static str {
        "my-reducer"
    }

    fn install(&self, kernel: &mut Kernel) -> langgraph_rs::Result<()> {
        kernel.register_channel("trimmed_append", || {
            Channel::reducer(|current, incoming| {
                let mut values = current.as_array().cloned().unwrap_or_default();
                if let Some(items) = incoming.as_array() {
                    values.extend(items.iter().cloned());
                }
                Ok(serde_json::Value::Array(values))
            })
        });
        Ok(())
    }
}

let kernel = Kernel::builder().with_plugin(MyReducer)?.build();
let schema = kernel.schema([("messages".to_string(), "trimmed_append".to_string())])?;
let graph = StateGraph::<Pipeline>::with_schema(schema)
    .add_node("node", |_state: &Pipeline| Ok(Default::default()))
    .set_entry_point("node")
    .set_finish_point("node")
    .compile_with(CompileOptions::new().with_kernel(kernel))
    .unwrap();
```

The kernel never names a concrete adapter: persistence, tracing and Python live outside it.

## Quick start

Add `langgraph-rs` (a path dependency until published), `serde` with `derive` and
`serde_json`. State is any `Clone + Default + Serialize + Deserialize` type whose
`State::schema` declares channels and merge rules.

```rust
use langgraph_rs::{NodeOutput, Route, RunConfig, State, StateGraph, StateSchema};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Counter {
    value: i64,
}

impl State for Counter {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

let graph = StateGraph::<Counter>::new()
    .add_node("increment", |state: &Counter| {
        Ok(NodeOutput::new().set("value", state.value + 1))
    })
    .set_entry_point("increment")
    .add_conditional_edges("increment", |state: &Counter| {
        Ok(if state.value < 3 {
            Route::node("increment")
        } else {
            Route::end()
        })
    })
    .compile()
    .unwrap();

let output = graph.invoke(&Counter::default(), &RunConfig::new()).unwrap();
assert_eq!(output.value, 3);
```

## State channel types

A channel is one field of the state and decides how a superstep's writes become the next value:

| Channel | Merge behaviour | Use it for |
| --- | --- | --- |
| `LastValue` | newest write wins; two writes in one superstep fail with `Error::ConcurrentUpdate` | scalars and control flags owned by one node per step |
| `Topic` | each write is appended into a JSON array | event logs and traces where order matters |
| `Reducer` | writes are folded left-to-right by your function | messages, counters, maps that several nodes update |
| `AnyValue` | newest write wins and concurrent writes are allowed | independent last-write channels |
| `EphemeralValue` | value is visible for one superstep, then cleared | transient signals consumed by routers |
| `UntrackedValue` | like `LastValue`, but excluded from checkpoints | runtime-only data and secrets |
| `NamedBarrierValue` | stores writer names until every expected name wrote, then becomes `true` | coordinating fan-in |

`NodeOutput::overwrite(channel, value)` writes directly to a channel, bypassing its reducer
(LangGraph's `Overwrite`). `DeltaChannel` and `LastValueAfterFinish` are not implemented yet.

Built-ins: `reducers::append()`, `reducers::add()`, `reducers::messages()` (merge by `id`).
`JsonState` is the schemaless alternative: a JSON map whose undeclared channels behave like
`LastValue` (build with `StateGraph::<JsonState>::with_schema` to register reducers).

## Conditional routing

Routers are pure `Fn(&State) -> Result<Route>` functions evaluated against the state *after* the
superstep's writes were merged. They may return one node, several nodes (`Route::nodes([...])`
fans out) or `Route::end()`. With a path map they return symbolic keys instead of node names; a
key missing from the map is a hard error.

```rust
let graph = StateGraph::<Counter>::new()
    .add_node("work", |state: &Counter| Ok(NodeOutput::new().set("value", state.value + 1)))
    .add_node("done", |_state: &Counter| Ok(NodeOutput::new()))
    .set_entry_point("work")
    .add_conditional_edges_mapped("work", |state: &Counter| {
        Ok(Route::node(if state.value >= 7 { "done" } else { "retry" }))
    }, [("done", "done"), ("retry", "work")])
    .set_finish_point("done")
    .compile()
    .unwrap();
```

`add_conditional_edges` routes to concrete node names directly, and a node can override its
edges with `NodeOutput::goto("target")` for the step.

## Parallel execution and `Send`

All nodes scheduled in one superstep share a state snapshot and run concurrently; the next
superstep sees their merged writes. Edges fan out statically, `Send` with per-task state.
Routers can also dispatch `Send` tasks: return `Route::sends([...])` (or combine
`Route::nodes([...]).with_sends([...])`) and the engine schedules the dynamic tasks for the
next superstep; conditional entry points support the same. From Python, a router may return a
single `lg.Send` or a list mixing node names and `lg.Send` objects.

```rust
use serde_json::json;
use langgraph_rs::{NodeOutput, RunConfig, Send, State, StateGraph, StateSchema, reducers};

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
struct Pipeline {
    value: i64,
    #[serde(default)]
    log: Vec<String>,
}

impl State for Pipeline {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value").with_reducer("log", reducers::append())
    }
}

// Map/reduce: `dispatch` spawns three workers with per-task state; writes reduce through `log`.
let map_reduce = StateGraph::<Pipeline>::new()
    .add_node("dispatch", |_s: &Pipeline| {
        Ok([1, 2, 3].into_iter().fold(NodeOutput::new(), |output, item| {
            output.send(Send::with_arg("worker", json!({ "value": item })))
        }))
    })
    .add_node("worker", |state: &Pipeline| {
        Ok(NodeOutput::new().set("log", vec![format!("worker-{}", state.value)]))
    })
    .set_entry_point("dispatch")
    .set_finish_point("worker")
    .compile()
    .unwrap();

let output = map_reduce.invoke(&Pipeline::default(), &RunConfig::new()).unwrap();
assert_eq!(output.log.len(), 3);
```

`Send::with_arg` replaces the target task's whole state (it must deserialize into `S`); `Send::to`
forwards the current state. Distinct `Send`s run concurrently and merge via reducers.

## Batch execution, concurrency and durability

`CompiledGraph::batch` runs one graph over many inputs and returns results in input order;
`RunConfig::with_max_concurrency(n)` bounds the number of concurrent invocations (scoped
threads). `batch_run` additionally returns `RunOutcome` reports. `RunConfig::with_durability`
selects `Durability::Sync` (checkpoint every superstep), `Async` (currently behaves like
`Sync`) or `Exit` (only the initial, interrupt and final checkpoints). `add_sequence` registers
a chain of nodes in one call and `add_edges_from` fans several sources into one target.

```rust
let results = graph.batch(inputs, &RunConfig::new().with_max_concurrency(4));
let outcomes = graph.batch_run(inputs, &RunConfig::new());
let config = RunConfig::new().with_durability(Durability::Exit);
```

## Reliability: retries and error routing

A node can be given a retry policy; the engine retries it automatically while it keeps
failing. Interrupts are never retried. Once attempts are exhausted the run fails — unless
an error edge routes the failure to a handler node, which runs in the next superstep and
can read the failure with `last_error()`.

```rust
use std::time::Duration;
use langgraph_rs::{Error, NodeOutput, RetryPolicy, StateGraph, WorkState, last_error};

let graph = StateGraph::<WorkState>::new()
    .add_node_with_retry(
        "call_model",
        RetryPolicy::new(3)
            .with_backoff(Duration::from_millis(200))
            .with_multiplier(2.0),
        |state: &WorkState| call_model(state),
    )
    .add_node("fallback", |_state: &WorkState| {
        let failure = last_error().expect("handler sees the failure");
        Ok(NodeOutput::new().set("answer", format!("degraded: {}", failure.message)))
    })
    .add_error_edge("call_model", "fallback")
    .set_entry_point("call_model")
    .set_finish_point("fallback")
    .compile()
    .unwrap();
```

Per-node timeouts use `TimeoutPolicy` (`add_node_with_timeout` / `set_timeout`). `run_timeout`
is enforced; sync graphs run the node on a helper thread so the run fails with
`Error::Timeout` when it expires (Rust cannot cancel a blocking closure), while async graphs
use `tokio::time::timeout`. `idle_timeout` is accepted for LangGraph API compatibility but is
not enforced yet. Managed values are available through `remaining_steps()` and
`is_last_step()` (also exposed from Python).

## Subgraphs

`add_subgraph` embeds a compiled graph as a node. The subgraph runs against the parent
state, and every channel it changed is written back as a **direct replacement** (bypassing
the parent reducer), so reducer channels are not appended twice.

```rust
use std::sync::Arc;
use langgraph_rs::{NodeOutput, RunConfig, StateGraph};

let inner = Arc::new(
    StateGraph::<WorkState>::new()
        .add_node("step", |state: &WorkState| Ok(NodeOutput::new().set("value", state.value + 1)))
        .set_entry_point("step")
        .set_finish_point("step")
        .compile()
        .unwrap(),
);

let outer = StateGraph::<WorkState>::new()
    .add_node("seed", |_state: &WorkState| Ok(NodeOutput::new().set("value", 10)))
    .add_subgraph("inner", inner)
    .set_entry_point("seed")
    .add_edge("seed", "inner")
    .set_finish_point("inner")
    .compile()
    .unwrap();
```

`add_subgraph` shares the parent's state type; `add_subgraph_mapped` accepts a subgraph
with a different state type plus input/output mapping closures. Subgraphs inherit the
parent thread and run in a child checkpoint namespace (`<parent_ns>|<name>`), so an
`interrupt()` inside a subgraph pauses the parent run and the next `resume` continues the
subgraph from its own checkpoint. Subgraph streaming (`subgraphs=True`) is not implemented
yet.

## Node-level caching

Caching is opt-in per node: attach a `NodeCache` backend to the graph and a `CachePolicy` to
the node. Before a node runs the executor looks the output up by a key derived from the node
name and its input state; after a successful run the output is stored. Interrupts and
failures are never cached, and cache hits are flagged as `cached` in the run report.

```rust
use std::sync::Arc;
use std::time::Duration;
use langgraph_rs::{CachePolicy, InMemoryCache, StateGraph};

let graph = StateGraph::<WorkState>::new()
    .add_node("call_model", |state: &WorkState| call_model(state))
    .set_entry_point("call_model")
    .set_finish_point("call_model")
    .with_cache(Arc::new(InMemoryCache::new()))
    .set_cache(
        "call_model",
        CachePolicy::new().with_ttl(Duration::from_secs(300)),
    )
    .compile()
    .unwrap();
```

The default key hashes the node name and the serialized input state (so only identical
inputs hit); `CachePolicy::with_key` installs a coarser custom key. `NodeCache` is a port, so
a SQLite/Redis/Postgres cache can be plugged in later.

## Long-term memory and runtime context

A `BaseStore` adds memory that survives runs and threads. Attach the built-in
`InMemoryStore` with `CompileOptions::with_store` (or `StateGraph::with_store`) and nodes read
it through `get_store()`; `RunConfig::with_context` carries per-run user data accessible via
`current_context()`. `current_runtime()` bundles thread id, namespace, store and context;
`current_thread_id()` and `current_namespace()` are available for subgraph-aware nodes. Both
the store and the context are inherited by subgraphs.

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, InMemoryStore, get_store, current_context, json};

let store = Arc::new(InMemoryStore::new());
let graph = StateGraph::<WorkState>::new()
    .add_node("remember", |state: &WorkState| {
        let store = get_store().expect("store attached");
        store.put(&["users".to_string()], "42", json!({ "seen": true }))?;
        let tenant = current_context().and_then(|ctx| ctx.get("tenant").cloned());
        Ok(NodeOutput::new().set("answer", tenant.unwrap_or_default()))
    })
    .set_entry_point("remember")
    .set_finish_point("remember")
    .compile_with(CompileOptions::new().with_store(store))
    .unwrap();

let config = RunConfig::new().with_context(json!({ "tenant": "acme" }));
```

## Visualisation

`CompiledGraph::to_mermaid()` (and `AsyncCompiledGraph::to_mermaid()`) render the compiled
topology as a Mermaid `graph TD` document: solid arrows for static edges, dashed arrows
through router nodes with path-map labels for conditional edges, `on error` edges for error
routing, and node labels that include retry counts and interrupt hooks.

```rust
let mermaid = graph.to_mermaid();
std::fs::write("graph.mmd", mermaid)?;
```

Paste the result into GitHub, GitLab, Notion or any Mermaid renderer. The Python binding
exposes the same method (`compiled.to_mermaid()`).

## Async with Tokio

Enable the `async` feature for an async runtime adapter. Nodes are async closures that
receive the state (by value) and an `AsyncContext`; supersteps are awaited concurrently
with `futures::future::join_all`.

```rust
use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{JsonState, NodeOutput};

async fn example() -> langgraph_rs::Result<()> {
    let graph = AsyncStateGraph::<JsonState>::new()
        .add_node("call", |_state: JsonState, ctx| async move {
            // do async I/O here, e.g. an HTTP request to an LLM
            ctx.record_tokens(120, 24);
            Ok(NodeOutput::new().set("answer", "42"))
        })
        .set_entry_point("call")
        .set_finish_point("call")
        .compile()?;

    let outcome = graph.run(&JsonState::default(), &Default::default()).await?;
    assert_eq!(outcome.report.total_tokens(), 144);
    Ok(())
}
```

The async adapter covers the same checkpoint, HITL, retry, error-routing and streaming
surface as the sync path (node caching and subgraphs remain sync-only):
checkpointing and thread history, human-in-the-loop (`ctx.interrupt(...)` +
`graph.resume(thread, value)`),
`interrupt_before` / `interrupt_after`, per-node `RetryPolicy`, error routing with
`ctx.last_error()`, goto/`Send`, streaming and usage/trace reporting. Checkpointing is
enabled with `.with_checkpointer(Arc::new(InMemorySaver::new()))` and needs a
`RunConfig::with_thread(...)`:

```rust
use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{InMemorySaver, RunConfig, json};
use std::sync::Arc;

# async fn example() -> langgraph_rs::Result<()> {
let graph = AsyncStateGraph::<JsonState>::new()
    .add_node("approve", |_state: JsonState, ctx| async move {
        let answer = ctx.interrupt(json!({ "question": "approve?" }))?;
        Ok(langgraph_rs::NodeOutput::new().set("approved", answer))
    })
    .set_entry_point("approve")
    .set_finish_point("approve")
    .with_checkpointer(Arc::new(InMemorySaver::new()))
    .compile()?;

let _ = graph
    .run(&JsonState::default(), &RunConfig::new().with_thread("demo"))
    .await
    .unwrap_err(); // GraphInterrupt-equivalent: Error::Interrupt

let state = graph.resume("demo", json!(true)).await?;
assert_eq!(state.get("approved"), Some(&json!(true)));
# Ok(())
# }
```

## Checkpointing and human-in-the-loop

Attach a saver with `CompileOptions` and bind each run to a `thread_id`. The run start, every
superstep and every pause are checkpointed.

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, InMemorySaver, NodeOutput, RunConfig, StateGraph, interrupt, json};

let saver = Arc::new(InMemorySaver::new());
let graph = StateGraph::<Pipeline>::new()
    .add_node("ask", |state: &Pipeline| {
        let answer = interrupt(json!({ "question": "how much to add?" }))?;
        Ok(NodeOutput::new()
            .set("value", state.value + answer.as_i64().unwrap_or_default())
            .set("log", vec!["ask".to_string()]))
    })
    .add_node("finish", |_s: &Pipeline| Ok(NodeOutput::new().set("log", vec!["finish".to_string()])))
    .set_entry_point("ask")
    .add_edge("ask", "finish")
    .set_finish_point("finish")
    .compile_with(CompileOptions::new().with_checkpointer(saver))
    .unwrap();

let paused = graph.invoke(&Pipeline::default(), &RunConfig::new().with_thread("demo")).unwrap_err();
assert!(paused.is_interrupt());

// The `ask` node runs again; this time `interrupt()` returns 41 instead of pausing.
let output = graph.resume("demo", json!(41)).unwrap();
assert_eq!(output.value, 41);
```

`get_state(thread)` reports `interrupted` and pending `next` tasks. `resume_with_updates` applies
human edits before continuing, `resume_command(thread, Command::resume(v).with_update([...]))`
is the explicit builder form of the same intent, `resume_from` replays from an older checkpoint,
and `get_state_history` returns the parent-linked chain newest first. For coarse gates, compile with
`.interrupt_before(["finish"])` or `.interrupt_after(["ask"])`: those pauses carry a `null`
payload and continue via `graph.continue_thread("demo")`.

### Durable checkpointers

`InMemorySaver` is handy for tests; production deployments can plug in a persistent
`Checkpointer` instead. All adapters store the same `Checkpoint` payload (state values,
pending tasks, interrupt flags, parent link) and implement `latest` / `history` / `threads`.

| Adapter | Feature | Connect |
| --- | --- | --- |
| `SqliteSaver` | `sqlite` (needs `libsqlite3`) | `SqliteSaver::open("checkpoints.db")` / `SqliteSaver::in_memory()` |
| `RedisSaver` | `redis` | `RedisSaver::connect("redis://127.0.0.1/")` |
| `PostgresSaver` | `postgres` | `PostgresSaver::connect("postgres://user:pass@host/db")` |

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, SqliteSaver}; // feature = "sqlite"

let saver = Arc::new(SqliteSaver::open("checkpoints.db")?);
let graph = builder
    .compile_with(CompileOptions::new().with_checkpointer(saver))?;
```

The SQLite adapter runs in CI; the Redis and Postgres tests are opt-in through
`LANGGRAPH_RS_REDIS_URL` / `LANGGRAPH_RS_POSTGRES_URL` and skip when the variables are
unset. On Debian/Ubuntu install `libsqlite3-dev` before building the `sqlite` feature.

Runs can be bound to a checkpoint namespace with `RunConfig::with_checkpoint_ns(...)`
(queries use `get_state_in` / `get_state_history_in` / `resume_in` / `update_state_in`), and
stored threads can be managed with `CompiledGraph::delete_thread` and `copy_thread` (also
exposed from Python).

## Streaming

`stream` collects every chunk; `stream_with` pushes them to a sink on the calling thread, so it
can forward to a channel, websocket or log without an async runtime. Chunks carry `Values`,
per-node `Updates`, `Custom` events, `Message` payloads (emitted with `message()`),
`Interrupt` payloads, `Checkpoint` ids, `Task` lifecycle records and `Debug` payloads.
`stream_modes(&input, &config, &[StreamMode::Values, ...])` filters to LangGraph's modes
(`values`, `updates`, `custom`, `messages`, `debug`, `tasks`, `checkpoints`); the Python
binding accepts the same names via `stream(..., stream_mode=...)`.

```rust
use langgraph_rs::{NodeOutput, RunConfig, StateGraph, StreamChunk, custom, json};

let graph = StateGraph::<Pipeline>::new()
    .add_node("emit", |_s: &Pipeline| {
        custom(json!({ "progress": 0.5 }))?;
        Ok(NodeOutput::new())
    })
    .set_entry_point("emit")
    .set_finish_point("emit")
    .compile()
    .unwrap();

let chunks = graph.stream(&Pipeline::default(), &RunConfig::new()).unwrap();
assert!(chunks.iter().any(|chunk| matches!(chunk, StreamChunk::Custom { .. })));
```


## Per-node usage and tracing

Every node execution is recorded: wall-clock duration plus whatever the node reported.
Nodes can report piecemeal (`record_tokens`, `record_usage`) or attach usage to their output;
multiple records from one execution are summed.

```rust
use langgraph_rs::{NodeOutput, RunConfig, StateGraph, Usage, record_tokens};

let graph = StateGraph::<Pipeline>::new()
    .add_node("llm", |state: &Pipeline| {
        record_tokens(512, 96)?;
        Ok(NodeOutput::new()
            .set("value", state.value + 1)
            .with_usage(Usage::new().with_model("gpt-4o").with_cost_usd(0.006)))
    })
    .set_entry_point("llm")
    .set_finish_point("llm")
    .compile()
    .unwrap();

// `run` returns the final state plus the report; `invoke` keeps the state-only shape.
let outcome = graph.run(&Pipeline::default(), &RunConfig::new()).unwrap();
let report = &outcome.report;
println!("{:?}", report.usage_by_node());   // BTreeMap<node, Usage>
println!("{}", report.total_tokens());      // summed across nodes
println!("{}", report.total_cost_usd());
for execution in report.executions() {
    println!("{} step={} {}ms", execution.node, execution.step, execution.duration_millis);
}
```

`RunReport` exposes `executions()` (one `NodeExecution` per node run), `usage_by_node()`,
`total_usage()`, `total_tokens()` and `total_cost_usd()`. `stream` additionally emits a
`StreamChunk::Usage` chunk per node, so a UI can display consumption as it happens.

## Python bindings

Enable the `python` feature (PyO3) to build a native extension module. With
[maturin](https://maturin.rs):

```bash
pip install maturin
maturin develop --features python     # installs `langgraph_rs` into the active venv
python python/examples/quickstart.py
```

Alternatively build the cdylib directly (`cargo build --features python`) and place
`liblanggraph_rs.so` / `langgraph_rs.dll` next to your test code as `langgraph_rs.pyd` /
`langgraph_rs.so`. The Python API uses `JsonState` and Python callables:

```python
import langgraph_rs as lg

graph = lg.StateGraph({"count": "last_value", "log": "append"})
graph.add_node("inc", lambda state: {"count": state["count"] + 1, "log": ["inc"]})
graph.add_edge("START", "inc")
graph.add_conditional_edges("inc", lambda state: "inc" if state["count"] < 3 else None)
compiled = graph.compile()
assert compiled.invoke({"count": 0, "log": []})["count"] == 3
```

- `graph.compile(checkpointer=True)` enables checkpoints; `invoke(..., thread_id="t")` pauses
  with `lg.GraphInterrupt` when a node calls `lg.interrupt(payload)`; `resume("t", answer)`
  continues and `get_state` / `get_state_history` inspect the checkpoint chain.
- LangGraph-style classes are available: `Send` (nodes can return a list of `Send` targets),
  `Command(update=..., goto=...)` returned from a node, `Command(resume=..., update=...)`
  passed to `invoke`, `RetryPolicy` / `CachePolicy` (via `add_node(..., retry_policy=...,
  cache_policy=..., error_handler="node")`), `compile(cache=True)`, and `lg.add_messages`.
- Checkpointer objects: `compile(checkpointer=lg.InMemorySaver())` (and
  `lg.SqliteSaver` / `lg.SqliteSaver.in_memory()` when built with the `sqlite` feature).
- Subgraphs: `parent.add_subgraph("child", child_compiled)` embeds a compiled graph. The
  child runs in a checkpoint namespace named after the node, a nested `interrupt()` pauses
  the parent, and `get_state(thread, checkpoint_ns="child")` /
  `get_state_history(thread, checkpoint_ns="child")` inspect the child chain (share one saver
  between parent and child to see both chains from the parent object).
- Long-term memory: `store = lg.InMemoryStore()` with `compile(store=store)`; nodes use
  `lg.get_store()` (`get` / `put` / `delete` / `search` / `namespaces`).
- Runtime context: `invoke(..., context={...})` (also `run` / `stream`); nodes read
  `lg.current_context()`, `lg.current_runtime()`, `lg.current_thread_id()` and
  `lg.current_namespace()`.
- `compiled.run(...)` returns `{"state": ..., "report": ...}` with the same per-node usage
  report; `compiled.stream(...)` returns chunk dicts.
- Nodes may return the reserved keys `__goto__` (node name or list) and `__usage__` (a dict
  with `model`, `input_tokens`, `output_tokens`, `cost_usd`, `metadata`).

## Benchmarks vs Python LangGraph

The repository ships a reproducible comparison against Python LangGraph 1.2.x: identical
graphs, node counts and iteration counts, one protocol for all engines. See
[benchmarks/README.md](benchmarks/README.md) for the scenarios and
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the current numbers; regenerate with
`python benchmarks/run.py`.

Snapshot (Windows 11, rustc 1.99.1, Python 3.12.10, LangGraph 1.2.22):

| Scenario | Rust native | Rust via Python binding | Python LangGraph | Rust vs LangGraph |
| --- | --- | --- | --- | --- |
| linear_10 (10-node pipeline) | 22.85 µs/op | 35.61 µs/op | 3,362.68 µs/op | **147.1×** |
| loop_50 (50-step loop) | 206.82 µs/op | 167.08 µs/op | 18,680.04 µs/op | **90.3×** |
| fanout_4 (1 → 4 parallel → join) | 704.23 µs/op | 468.25 µs/op | 3,785.87 µs/op | **5.4×** |
| checkpoint_5 (5 nodes + checkpoints) | 19.28 µs/op | 25.36 µs/op | 5,592.27 µs/op | **290.0×** |
| build_compile (10-node graph) | 11.47 µs/op | 22.45 µs/op | 1,442.46 µs/op | **125.8×** |

The full report also covers graph-size scaling (`scale_005` … `scale_100`), 8-thread
throughput (`mt_8x_linear_10`) and peak allocated memory (`mem_linear_10`, `mem_loop_50`);
see [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

These measure orchestration overhead only (no LLM or I/O calls): native Rust is one to two
orders of magnitude faster, the bindings pay Python callable dispatch per node, and the
fan-out scenario is the narrowest margin because native Rust spawns real threads per
superstep. The functional matrix — what reaches parity and what does not (functional API,
store, stream modes, subgraphs, async, Python surface) — is in
[benchmarks/features.md](benchmarks/features.md).

## Status / roadmap

Implemented and tested: microkernel + hexagonal layout (kernel ports, adapters, plugin registry);
strict or dynamic channel schemas; static, multi-target and conditional edges with path maps and
conditional entry points; parallel supersteps, `Send`, `goto`; the recursion limit (25 supersteps);
checkpointing, thread history, snapshots, `interrupt()` with before/after hooks, resume, updates,
time travel; retry policies with backoff and `last_error()` error routing; subgraphs; node-level
caching with TTLs and custom keys; Mermaid visualisation; persistent SQLite/Redis/Postgres
checkpointers; streaming; per-node duration/usage tracing with run reports and trace-sink
adapters; optional Tokio async adapter (`async` feature) with the full checkpoint/HITL/retry/
error-routing surface; validation and merge errors; optional Python bindings (PyO3); a
functional/performance benchmark against Python LangGraph, including scaling, threading and
memory scenarios.

Not implemented yet:

- external persistence of graph definitions, cross-process time travel, a time-travel UI,
  checkpoint namespaces, `Send` from routers
- async-adapter node caching, subgraph state mapping and per-subgraph checkpoint namespaces
- Python-defined custom reducers and async Python nodes (the bindings use built-in reducers and
  release the GIL around the engine)

## Differences from LangGraph

- **Sync-first, async opt-in.** The default engine runs `Fn(&S) -> Result<NodeOutput>` nodes on
  scoped threads; the `async` feature adds a Tokio adapter for async nodes. There is no event
  loop in the default path.
- **Thread-local node context.** `interrupt()`, `custom()` and `current_node()` read a
  thread-local installed per node call, so they only work on the executing thread.
- **State must serialize to a JSON object.** Channel values are `serde_json::Value`; non-object
  state fails with `Error::NonObjectState`, keeping checkpoints language-neutral.
- **Supersteps run nodes during the same wall-clock window via scoped threads** — real OS
  parallelism over a shared `&State`, not cooperatively scheduled tasks.
- **Routers are pure functions**, returning only destinations (or path-map keys).
- **Subgraphs share the state type** and publish changed channels as replacements; there is no
  per-subgraph state mapping or checkpoint namespace yet.
- **Python bindings use `JsonState`.** Python nodes exchange plain dicts; state is JSON, and
  custom reducers must be expressed with the built-in policies (`last_value`, `topic`, `append`,
  `add`, `messages`) for now. The GIL is released around the engine, so Python callables only
  run while a node executes.

## Build and test

```bash
cargo test                 # unit + integration tests
cargo test --features async   # async (Tokio) adapter tests
cargo test --features sqlite  # persistent checkpointer tests (needs libsqlite3)
cargo test --doc           # doctests, including the quick start above
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Python bindings
cargo build --features python
maturin develop --features python   # or: pip install maturin
python -m unittest discover -s python/tests -v

# Benchmarks vs Python LangGraph (requires `pip install langgraph`)
python benchmarks/run.py
```

Requires Rust 1.85+ (edition 2024). Design and internals: [docs/architecture.md](docs/architecture.md).

## License

MIT. This project is an independent port and is **not affiliated with LangChain** or the
LangGraph project; LangGraph is a trademark of its respective owners.