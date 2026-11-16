# Benchmark results

- Generated: 2026-11-16 14:41 UTC
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
| linear_10 | 69.03 µs/op | 82.57 µs/op | 2,173.93 µs/op | **31.5× faster** |
| loop_50 | 326.52 µs/op | 455.50 µs/op | 10,888.65 µs/op | **33.3× faster** |
| fanout_4 | 499.27 µs/op | 614.23 µs/op | 2,621.89 µs/op | **5.3× faster** |
| checkpoint_5 | 60.89 µs/op | 40.07 µs/op | 3,672.57 µs/op | **60.3× faster** |
| build_compile | 19.03 µs/op | 25.49 µs/op | 860.28 µs/op | **45.2× faster** |
| scale_005 | 29.92 µs/op | 29.09 µs/op | 1,487.91 µs/op | **49.7× faster** |
| scale_010 | 52.78 µs/op | 59.93 µs/op | 2,345.78 µs/op | **44.4× faster** |
| scale_025 | 120.30 µs/op | 153.02 µs/op | 4,887.93 µs/op | **40.6× faster** |
| scale_050 | 284.37 µs/op | 264.99 µs/op | 9,584.17 µs/op | **33.7× faster** |
| scale_100 | 525.13 µs/op | 540.64 µs/op | 19,496.49 µs/op | **37.1× faster** |
| mt_8x_linear_10 | 13.95 µs/op | 82.01 µs/op | 2,118.06 µs/op | **151.8× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 6.5 KiB peak | 0.3 KiB peak | 62.5 KiB peak |
| mem_loop_50 | 15.1 KiB peak | 0.3 KiB peak | 53.3 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

