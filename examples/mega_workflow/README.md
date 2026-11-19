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

python examples/mega_workflow/bench.py --iterations 10
python examples/mega_workflow/bench.py --iterations 10 --skip-native   # Python only
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