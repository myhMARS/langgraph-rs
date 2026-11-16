# Functional comparison with LangGraph

Compared against Python **LangGraph 1.2.22** (the reference implementation). "Parity"
means the same workflow can be expressed and behaves equivalently; "partial" means the
capability exists but with documented differences (see the notes column and
[`docs/architecture.md`](../docs/architecture.md#known-limitations-and-divergences)).

| Capability | langgraph-rs | LangGraph 1.2.22 | Notes |
| --- | --- | --- | --- |
| State schema + channel reducers | ✅ `LastValue` / `Topic` / custom reducer plugins | ✅ `Annotated[..., reducer]` | Rust channels are declared in `State::schema` or a `StateSchema`; custom policies are microkernel plugins |
| `add_messages` merge-by-id reducer | ✅ `reducers::messages()` | ✅ `add_messages` | same id-based replace/append semantics |
| Static edges, `START` / `END` | ✅ | ✅ | |
| Conditional edges | ✅ node names, lists, symbolic path maps | ✅ `path_map` | |
| Conditional entry point | ✅ | ✅ | |
| Parallel supersteps | ✅ scoped OS threads | ✅ asyncio tasks | Rust runs nodes on real threads; Python cooperatively |
| `Send` map/reduce fan-out | ✅ | ✅ | Rust `Send` carries a JSON-object state override |
| `Command(goto)` | ✅ `NodeOutput::goto` | ✅ `Command` | Rust supports goto + resume value via `Command` |
| Recursion limit | ✅ default 25, per compile/run | ✅ default 25 | |
| Checkpointing | ✅ `Checkpointer` port + `InMemorySaver` adapter | ✅ `MemorySaver` etc. | Rust has no durable backend shipped yet; the port is ready for SQLite/Redis/Postgres |
| Thread history / time travel | ✅ `get_state`, `get_state_history`, `resume_from` | ✅ | |
| Human-in-the-loop `interrupt` / `resume` | ✅ | ✅ | Rust re-executes the interrupted node; resume value returned by the same `interrupt()` call |
| `interrupt_before` / `interrupt_after` | ✅ | ✅ | hook payload is `null` |
| Streaming (`astream` / `stream`) | ✅ `StreamChunk` (values/updates/custom/interrupt/checkpoint/usage) | ✅ `stream_mode` values/updates/custom/messages/debug | Rust eager chunks, synchronous sink |
| Node-level consumption tracking | ✅ `Usage` + `NodeExecution` + `RunReport` + `TraceSink` | ⚠️ via callbacks/metadata, not first-class | Rust reports duration + tokens/model/cost per node automatically |
| Trace/metering adapters | ✅ `TraceSink` port | ✅ callbacks / LangSmith | plug any sink (logging, metrics, OTel) |
| Subgraphs | ✅ `add_subgraph` | ✅ first-class subgraphs | shares the parent state type; publishes changed channels as replacements; no state mapping, checkpoint namespacing or nested interrupts yet |
| Async nodes / runtime integration | ✅ Tokio adapter (`async` feature) with checkpoints, HITL, retries and error routing | ✅ native async | the default engine stays synchronous; async nodes use `AsyncContext`; async subgraphs/Python bindings are not exposed yet |
| Retry policies | ✅ `RetryPolicy` (attempts, exponential backoff, cap) | ✅ `RetryPolicy` | interrupts are never retried |
| Error routing / fallback nodes | ✅ `add_error_edge` + `last_error()` | ✅ conditional edges / error handling | handler runs in the next superstep and receives a `NodeFailure` |
| Python API | ✅ PyO3 bindings (`python` feature) | ✅ native | Rust exposes `StateGraph`/`CompiledGraph`, `interrupt`, `resume`, `run` reports; only built-in reducers from Python |
| Graph visualisation | ✅ `to_mermaid()` on sync and async graphs (+ Python) | ✅ `get_graph().draw_mermaid()` | static rendering from the compiled topology, including retries/error edges/interrupt hooks |
| Durable/remote checkpoint backends | ⚠️ trait only, memory saver ships | ✅ SQLite/Postgres/Redis | roadmap |
| Retries / caching / deferred nodes | ❌ | ✅ | roadmap |

Legend: ✅ supported · ⚠️ partial / documented difference · ❌ not implemented.

## How the matrix was produced

The row statuses are based on the public API of this crate and on the LangGraph 1.2.22
documentation/API surface, and are exercised by the Rust integration tests
(`tests/graph.rs`, `tests/checkpoint.rs`, `tests/usage.rs`, `tests/plugin.rs`) and the
Python binding tests (`python/tests/test_bindings.py`). Performance is measured
separately in [`README.md`](README.md) / [`RESULTS.md`](RESULTS.md).