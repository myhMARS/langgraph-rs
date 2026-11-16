# Benchmark results

- Generated: 2026-11-16 08:19 UTC
- Machine: Windows-11-10.0.26300-SP0 / Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Rust: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- Python: 3.12.10
- LangGraph: 1.2.22
- Protocol: same node counts and iteration counts per scenario for all engines;
  warm-up runs precede every measurement; times are wall-clock around the whole
  measured loop (graph construction excluded except for `build_compile`).

## Wall-clock cost per operation

| Scenario | Rust native | Rust via Python binding | Python LangGraph | Rust native vs LangGraph |
| --- | --- | --- | --- | --- |
| linear_10 | 27.27 µs/op | 41.94 µs/op | 4,792.14 µs/op | **175.8× faster** |
| loop_50 | 128.33 µs/op | 224.08 µs/op | 23,165.64 µs/op | **180.5× faster** |
| fanout_4 | 372.24 µs/op | 700.43 µs/op | 3,455.19 µs/op | **9.3× faster** |
| checkpoint_5 | 17.22 µs/op | 44.05 µs/op | 6,116.85 µs/op | **355.2× faster** |
| build_compile | 8.08 µs/op | 27.90 µs/op | 1,038.64 µs/op | **128.5× faster** |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

