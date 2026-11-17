# Remaining differences audit

This file records the differences that are still intentionally documented after the
40-scenario parity matrix. They are grouped by whether they affect observable LangGraph
behavior or only the crate's internal representation / binding surface.

## Behavior covered by parity

The following areas have dedicated behavior + benchmark scenarios in
[`parity.py`](parity.py) and pass in [`PARITY.md`](PARITY.md):

- execution, conditional routing, loops, batch
- managed values (`remaining_steps` / `is_last_step`)
- v1/v2/messages streaming
- functional API, Send fan-out, deferred nodes, graph export
- Command(update/goto/PARENT)
- retry, error handler/propagation, node/Send timeouts
- memory/SQLite node cache, memory/SQLite/Redis/Postgres store
- SQLite/Postgres/Redis checkpoint persistence
- async execution, interrupt/resume, multi-interrupt, nested subgraph interrupt
- update_state (including `task_id`), checkpoint history, snapshot metadata, time travel
- pending writes in parallel branches

## Non-blocking differences

| Area | Difference | Impact |
| --- | --- | --- |
| Pending writes storage | Successful writes are stored inside the checkpoint JSON, not a separate `writes` table | Observable resume behavior matches; database layout differs from LangGraph |
| `versions_seen` values | Version suffix is deterministic rather than random | Shape matches LangGraph; scheduler uses pending tasks rather than versions |
| Checkpoint `step` | Snapshot `step` resets per run; LangGraph counts input checkpoints across the thread | Snapshot `source`/`next` match; numeric `step` differs |
| Messages callbacks | The binding needs an explicit `langchain_callback_handler`; LangGraph injects callbacks globally | Token stream content matches; callback metadata is narrower |
| Interrupt response schema | `pattern` and `$ref` are not supported | Supported JSON-schema subset covers the documented cases |
| Timed async loop | Timed Python nodes reuse a per-thread asyncio loop | With a 50ms deadline both engines are ~0.99x; short deadlines expose timer-resolution differences |
| SQLite/Postgres cache/store APIs | Binding constructors take a path/URL rather than a live connection/client | Behavior matches; Python object shape differs |

## Performance snapshot

Latest parity run:

- SQLite persistence: ~15–25x
- Postgres persistence: ~1.32x
- Redis persistence: ~2.39x
- Postgres store: ~1.42x
- Redis store: ~4.14x
- `send_timeout` (50ms deadline): ~0.99x

All numbers are from the same machine and the same Python binding process; they are
comparative indicators, not production benchmarks.