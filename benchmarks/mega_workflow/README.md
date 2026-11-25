# Mega workflow: LangGraph vs Rust binding vs Rust native

One deliberately long workflow, written three times with the same topology, node bodies and
phase sequence, so the timings compare **orchestration cost** rather than different work.

```text
seed -> grow (loop x3) -> Send(worker) x3 -> join -> cached -> retry
     -> sub (embedded child graph) -> ask (interrupt) -> done
     -> defer (deferred node, terminal)   and   -> boom (always fails, error handler)
```

Features exercised per suite:

| Feature | Where |
| --- | --- |
| loop via conditional edges | `grow` routes back to itself until `count == 3` |
| `Send` fan-out | the `grow` router emits three `Send("worker", {"n": i})` |
| dynamic routing with `Command`-style goto | not used; `Send` covers dynamic fan-out |
| node cache policy | `cached` (hit on the second and later suites) |
| retry policy | `retry` fails once, succeeds on the retry (backoff set to 0 for timing) |
| error handler / error edge | `boom` always fails, `handled` writes the log entry |
| deferred node | `defer` (terminal: the Rust engine requires deferred nodes to have no out-edges) |
| embedded child graph | `sub` calls a separately compiled graph with explicit input/output mapping |
| interrupt + resume | `ask` pauses, the driver resumes with `"go"` |
| state APIs | `get_state`, `get_state_history(limit=5)` |
| `update_state` | `update_state({"count": 99}, as_node="done")` on the finished thread |
| stream | `stream(..., stream_mode="values")` on a fresh thread |
| batch | `batch([input, input])` on fresh threads |
| store | `seed` writes `("mega", "runs")` through `get_store()` |

Final state of every engine: `count=3 sink=6 child=4 log=15 flag=approved=go`.

## Files

| File | Role |
| --- | --- |
| `build_lg.py` | the reference implementation on real LangGraph 1.2 |
| `build_rs.py` | the same graph through the PyO3 binding (`import langgraph_rs`) |
| `bench.py` | runs all three, prints the table and writes `MEGA_BENCH.md` |
| `MEGA_BENCH.md` | the last generated report |
| `../mega_workflow_native.rs` | the same graph in pure Rust (`cargo run --release --example mega_workflow_native`) |

## Run

```powershell
# build the binding first (python feature) and expose it
cargo build --features "python python-async sqlite redis postgres msgpack"
Copy-Item target\debug\langgraph_rs.dll target\python\langgraph_rs.pyd -Force
$env:PYTHONPATH = (Resolve-Path .).Path + "\target\python"

python benchmarks/mega_workflow/bench.py --iterations 10
python benchmarks/mega_workflow/bench.py --iterations 10 --skip-native   # Python only
cargo run --release --example mega_workflow_native -- 10                # native only
```

The native example prints `FINAL ...` plus one `PHASE=<name>:<ns>` line per phase, which is
what `bench.py` parses.

## API differences that are **not** part of the measurement

These are the only places where the three sources differ, all of them API shape:

* **State schema** — LangGraph uses a `TypedDict` with `Annotated[..., operator.add]`;
  the binding uses a channel dict (`{"log": "append", "sink": "add", ...}`); native Rust uses
  `StateSchema::dynamic().with_reducer(...)`.
* **Deferred nodes** — LangGraph allows out-edges from a `defer=True` node; the Rust engine
  validates that deferred nodes are terminal, so `defer` is a terminal tail in all three.
* **Retry targeting** — LangGraph takes exception *types* (`retry_on=ValueError`), the binding
  takes exception *names* (`retry_on=["ValueError"]`), native takes `with_backoff(...)`.
* **Embedded child graph** — both Python versions call `child.invoke(...)` inside a node with
  an explicit state mapping; native Rust calls `child.invoke(&state, &config)`.
* **Adapter call shapes** — `invoke(input, config)` / `get_state(config)` in LangGraph versus
  `invoke(input, thread_id=...)` / `get_state(thread)` in the binding; `bench.py` hides the
  difference in `LangGraphDriver` / `BindingDriver`.

## Reading the numbers

Per-phase medians over N suites, measured after the graphs are compiled (build time is not
included). `invoke`/`stream`/`batch` include the whole 15-node workflow up to the interrupt;
`resume` continues from the pause; `state` covers `get_state` + five history entries;
`update_state` writes one update on the finished thread.

Retry backoff is forced to zero in all three engines so the table measures orchestration rather
than sleeping. Numbers are single-machine (Windows, Python 3.12, LangGraph 1.2.22) and are
medians of `--iterations` suites; treat small differences in `state`/`update_state` as noise.
## Async variant

`bench_async.py` runs the same 15-node workflow with every node defined as `async def` and every
call awaited (`ainvoke` / `aresume` / `astream` / `abatch` / `aget_state` / `aupdate_state`).
Sources: `build_lg_async.py`, `build_rs_async.py`, `../mega_workflow_async_native.rs`.
Report: `MEGA_BENCH_ASYNC.md`.

```powershell
python benchmarks/mega_workflow/bench_async.py --iterations 10
cargo run --release --example mega_workflow_async_native --features async -- 10
```

Last measured medians (10 suites, same machine):

| Phase | LangGraph async | binding async | native async |
| --- | ---: | ---: | ---: |
| invoke | 6.62 ms | 3.40 ms (1.94x) | 0.34 ms (19.5x) |
| resume | 2.18 ms | 1.78 ms (1.22x) | 0.24 ms (9.0x) |
| state | 0.82 ms | 0.93 ms (0.88x) | 0.16 ms (5.1x) |
| update_state | 0.31 ms | 0.47 ms (0.66x) | 0.04 ms (7.2x) |
| stream | 6.44 ms | 3.67 ms (1.76x) | 0.54 ms (12.0x) |
| batch | 12.32 ms | 4.32 ms (2.85x) | 0.72 ms (17.2x) |

Async notes:

* The binding used to drive every node coroutine through `asyncio.run`, which creates and
  tears down an event loop per node (~1 ms each). It now reuses one loop per OS thread
  (`src/adapters/python/async_support.rs`), which cut a 10-node `ainvoke` from ~14 ms to
  ~2.5 ms and moved the binding from 0.4-0.6x to 1.2-2.9x of LangGraph async.
* `state`/`update_state` are still slightly slower than LangGraph async because the binding
  converts snapshots into fresh Python dicts; they are dominated by that conversion, not by
  orchestration.
* An async parent cannot embed an async child in the binding (the child would need its own
  asyncio loop), so `build_rs_async.py` embeds a sync child graph; LangGraph awaits an async
  child. Native Rust awaits a sync child the same way.