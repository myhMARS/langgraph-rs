# Benchmark results

- Generated: 2026-11-17 17:03 UTC
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
| linear_10 | 30.01 µs/op | 6,855.95 µs/op | 2,420.26 µs/op | **80.7× faster** |
| loop_50 | 150.13 µs/op | 35,172.41 µs/op | 11,869.95 µs/op | **79.1× faster** |
| fanout_4 | 251.03 µs/op | 4,986.36 µs/op | 2,345.72 µs/op | **9.3× faster** |
| checkpoint_5 | 54.08 µs/op | 3,564.10 µs/op | 3,585.46 µs/op | **66.3× faster** |
| build_compile | 10.77 µs/op | 22.72 µs/op | 728.67 µs/op | **67.6× faster** |
| scale_005 | 20.57 µs/op | 3,468.41 µs/op | 1,319.05 µs/op | **64.1× faster** |
| scale_010 | 30.22 µs/op | 7,058.29 µs/op | 2,257.05 µs/op | **74.7× faster** |
| scale_025 | 76.57 µs/op | 17,621.88 µs/op | 5,141.58 µs/op | **67.1× faster** |
| scale_050 | 143.66 µs/op | 35,295.71 µs/op | 10,021.07 µs/op | **69.8× faster** |
| scale_100 | 282.40 µs/op | 68,236.98 µs/op | 19,079.99 µs/op | **67.6× faster** |
| mt_8x_linear_10 | 7.18 µs/op | 7,335.86 µs/op | 2,251.81 µs/op | **313.7× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 6.8 KiB peak | 1,228.3 KiB peak | 61.4 KiB peak |
| mem_loop_50 | 15.5 KiB peak | 1,611.3 KiB peak | 58.6 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

