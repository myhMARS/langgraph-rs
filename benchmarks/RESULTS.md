# Benchmark results

- Generated: 2026-11-16 10:35 UTC
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
| linear_10 | 37.33 µs/op | 46.43 µs/op | 2,268.19 µs/op | **60.8× faster** |
| loop_50 | 221.82 µs/op | 243.60 µs/op | 12,282.14 µs/op | **55.4× faster** |
| fanout_4 | 239.90 µs/op | 395.59 µs/op | 2,136.05 µs/op | **8.9× faster** |
| checkpoint_5 | 31.84 µs/op | 34.53 µs/op | 4,473.18 µs/op | **140.5× faster** |
| build_compile | 10.07 µs/op | 24.93 µs/op | 729.72 µs/op | **72.5× faster** |
| scale_005 | 19.67 µs/op | 24.82 µs/op | 1,353.16 µs/op | **68.8× faster** |
| scale_010 | 36.75 µs/op | 43.24 µs/op | 2,515.61 µs/op | **68.4× faster** |
| scale_025 | 97.69 µs/op | 113.41 µs/op | 4,754.59 µs/op | **48.7× faster** |
| scale_050 | 168.79 µs/op | 206.59 µs/op | 9,492.25 µs/op | **56.2× faster** |
| scale_100 | 415.92 µs/op | 402.71 µs/op | 18,657.50 µs/op | **44.9× faster** |
| mt_8x_linear_10 | 8.24 µs/op | 61.89 µs/op | 2,119.83 µs/op | **257.2× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 5.9 KiB peak | 0.3 KiB peak | 53.5 KiB peak |
| mem_loop_50 | 14.6 KiB peak | 0.3 KiB peak | 57.8 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

