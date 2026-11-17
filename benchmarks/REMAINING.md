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
- update_state (including `task_id`), checkpoint history, snapshot metadata (source/next/step), time travel
- pending writes in parallel branches
- cross-language resume: LangGraph SqliteSaver reads a DB written by LangGraphSqliteSaver and resumes a single pending interrupt for dict (root-channel) state

## Cross-language compatibility scope

Verified so far: LangGraph reads a binding-written SQLite database, sees pending
`Interrupt` objects, and resumes them for dict/root-channel and typed channel schemas.
This covers a single pending task, multiple sequential interrupts, and parallel
pending interrupts resumed by interrupt id (the binding writes LangGraph-style
`xxh3` interrupt ids and derived task ids).

Not yet verified: subgraph namespaces against LangGraph readers.

## Non-blocking differences

No known non-blocking behavioral differences remain in the parity matrix.

Timed async nodes reuse a per-thread asyncio loop; with a 50ms deadline both engines are
~0.99x. SQLite/Postgres savers and stores accept live connections as well as paths/URLs.

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