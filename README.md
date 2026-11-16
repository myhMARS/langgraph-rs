# langgraph-rs

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

## Streaming

`stream` collects every chunk; `stream_with` pushes them to a sink on the calling thread, so it
can forward to a channel, websocket or log without an async runtime. Chunks carry `Values`,
per-node `Updates`, `Custom` events, `Interrupt` payloads and `Checkpoint` ids.

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
| linear_10 (10-node pipeline) | 27.27 µs/op | 41.94 µs/op | 4,792.14 µs/op | **175.8×** |
| loop_50 (50-step loop) | 128.33 µs/op | 224.08 µs/op | 23,165.64 µs/op | **180.5×** |
| fanout_4 (1 → 4 parallel → join) | 372.24 µs/op | 700.43 µs/op | 3,455.19 µs/op | **9.3×** |
| checkpoint_5 (5 nodes + checkpoints) | 17.22 µs/op | 44.05 µs/op | 6,116.85 µs/op | **355.2×** |
| build_compile (10-node graph) | 8.08 µs/op | 27.90 µs/op | 1,038.64 µs/op | **128.5×** |

These measure orchestration overhead only (no LLM or I/O calls): native Rust is one to two
orders of magnitude faster, the bindings pay Python callable dispatch per node, and the
fan-out scenario is the narrowest margin because native Rust spawns real threads per
superstep. The functional matrix — what reaches parity and what does not (subgraphs, async
nodes, durable backends) — is in [benchmarks/features.md](benchmarks/features.md).

## Status / roadmap

Implemented and tested: microkernel + hexagonal layout (kernel ports, adapters, plugin registry);
strict or dynamic channel schemas; static, multi-target and conditional edges with path maps and
conditional entry points; parallel supersteps, `Send`, `goto`; the recursion limit (25 supersteps);
checkpointing, thread history, snapshots; `interrupt()` with before/after hooks; resume, updates,
time travel; streaming; per-node duration/usage tracing with run reports and trace-sink adapters;
validation and merge errors; optional Python bindings (PyO3).

Not implemented yet:

- async nodes/tokio integration, and subgraphs as first-class nodes (drive futures inside a node,
  or invoke a compiled graph inside one)
- durable checkpoint backends beyond memory (the `Checkpointer` trait is ready)
- external persistence of graph definitions, cross-process time travel, a time-travel UI,
  node-level retries/caching, checkpoint namespaces, `Send` from routers
- Python-defined custom reducers and async Python nodes (the bindings use built-in reducers and
  release the GIL around the engine)

## Differences from LangGraph

- **Sync-first.** Nodes are `Fn(&S) -> Result<NodeOutput>`; no async Python, no event loop.
- **Thread-local node context.** `interrupt()`, `custom()` and `current_node()` read a
  thread-local installed per node call, so they only work on the executing thread.
- **State must serialize to a JSON object.** Channel values are `serde_json::Value`; non-object
  state fails with `Error::NonObjectState`, keeping checkpoints language-neutral.
- **Supersteps run nodes during the same wall-clock window via scoped threads** — real OS
  parallelism over a shared `&State`, not cooperatively scheduled tasks.
- **Routers are pure functions**, returning only destinations (or path-map keys).
- **Python bindings use `JsonState`.** Python nodes exchange plain dicts; state is JSON, and
  custom reducers must be expressed with the built-in policies (`last_value`, `topic`, `append`,
  `add`, `messages`) for now. The GIL is released around the engine, so Python callables only
  run while a node executes.

## Build and test

```bash
cargo test                 # unit + integration tests
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