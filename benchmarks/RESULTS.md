# Benchmark results

- Generated: 2026-11-16 09:05 UTC
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
| linear_10 | 22.85 µs/op | 35.61 µs/op | 3,362.68 µs/op | **147.1× faster** |
| loop_50 | 206.82 µs/op | 167.08 µs/op | 18,680.04 µs/op | **90.3× faster** |
| fanout_4 | 704.23 µs/op | 468.25 µs/op | 3,785.87 µs/op | **5.4× faster** |
| checkpoint_5 | 19.28 µs/op | 25.36 µs/op | 5,592.27 µs/op | **290.0× faster** |
| build_compile | 11.47 µs/op | 22.45 µs/op | 1,442.46 µs/op | **125.8× faster** |
| scale_005 | 20.07 µs/op | 16.80 µs/op | 2,200.83 µs/op | **109.6× faster** |
| scale_010 | 24.73 µs/op | 32.24 µs/op | 2,343.48 µs/op | **94.7× faster** |
| scale_025 | 59.73 µs/op | 79.22 µs/op | 6,702.80 µs/op | **112.2× faster** |
| scale_050 | 107.50 µs/op | 169.12 µs/op | 10,195.44 µs/op | **94.8× faster** |
| scale_100 | 280.13 µs/op | 378.65 µs/op | 21,398.79 µs/op | **76.4× faster** |
| mt_8x_linear_10 | 23.16 µs/op | 80.84 µs/op | 2,284.48 µs/op | **98.6× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 5.6 KiB peak | 0.3 KiB peak | 58.8 KiB peak |
| mem_loop_50 | 14.3 KiB peak | 0.3 KiB peak | 59.1 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

