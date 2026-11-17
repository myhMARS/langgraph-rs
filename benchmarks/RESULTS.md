# Benchmark results

- Generated: 2026-11-17 18:59 UTC
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
| linear_10 | 28.80 µs/op | 8,596.98 µs/op | 2,610.01 µs/op | **90.6× faster** |
| loop_50 | 137.30 µs/op | 42,480.94 µs/op | 14,863.09 µs/op | **108.3× faster** |
| fanout_4 | 220.36 µs/op | 6,240.70 µs/op | 2,825.53 µs/op | **12.8× faster** |
| checkpoint_5 | 45.39 µs/op | 4,050.22 µs/op | 4,517.71 µs/op | **99.5× faster** |
| build_compile | 10.73 µs/op | 21.22 µs/op | 833.23 µs/op | **77.6× faster** |
| scale_005 | 19.87 µs/op | 3,977.20 µs/op | 1,393.47 µs/op | **70.1× faster** |
| scale_010 | 26.57 µs/op | 7,965.32 µs/op | 2,314.93 µs/op | **87.1× faster** |
| scale_025 | 71.65 µs/op | 20,021.90 µs/op | 6,498.87 µs/op | **90.7× faster** |
| scale_050 | 127.32 µs/op | 38,402.47 µs/op | 10,342.37 µs/op | **81.2× faster** |
| scale_100 | 239.82 µs/op | 84,357.48 µs/op | 20,520.27 µs/op | **85.6× faster** |
| mt_8x_linear_10 | 7.35 µs/op | 8,537.84 µs/op | 2,582.17 µs/op | **351.4× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 6.8 KiB peak | 1,225.6 KiB peak | 56.1 KiB peak |
| mem_loop_50 | 15.5 KiB peak | 1,601.3 KiB peak | 56.3 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

