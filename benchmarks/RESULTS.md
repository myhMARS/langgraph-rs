# Benchmark results

- Generated: 2026-11-17 19:44 UTC
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
| linear_10 | 27.15 µs/op | 7,207.55 µs/op | 2,054.22 µs/op | **75.7× faster** |
| loop_50 | 129.82 µs/op | 36,063.31 µs/op | 12,377.14 µs/op | **95.3× faster** |
| fanout_4 | 233.34 µs/op | 5,327.16 µs/op | 2,440.71 µs/op | **10.5× faster** |
| checkpoint_5 | 49.97 µs/op | 3,387.75 µs/op | 3,926.57 µs/op | **78.6× faster** |
| build_compile | 8.63 µs/op | 18.77 µs/op | 666.36 µs/op | **77.2× faster** |
| scale_005 | 17.52 µs/op | 3,530.53 µs/op | 1,201.75 µs/op | **68.6× faster** |
| scale_010 | 29.98 µs/op | 6,777.99 µs/op | 2,037.07 µs/op | **67.9× faster** |
| scale_025 | 75.02 µs/op | 17,151.63 µs/op | 4,647.31 µs/op | **62.0× faster** |
| scale_050 | 125.63 µs/op | 33,083.38 µs/op | 9,875.90 µs/op | **78.6× faster** |
| scale_100 | 255.21 µs/op | 68,451.85 µs/op | 18,210.23 µs/op | **71.4× faster** |
| mt_8x_linear_10 | 6.17 µs/op | 7,715.27 µs/op | 2,015.58 µs/op | **326.8× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 6.8 KiB peak | 1,223.4 KiB peak | 49.3 KiB peak |
| mem_loop_50 | 15.5 KiB peak | 1,594.2 KiB peak | 52.8 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

