# Benchmark results

- Generated: 2026-11-16 11:49 UTC
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
| linear_10 | 50.22 µs/op | 57.98 µs/op | 2,626.96 µs/op | **52.3× faster** |
| loop_50 | 256.96 µs/op | 287.01 µs/op | 12,841.48 µs/op | **50.0× faster** |
| fanout_4 | 451.33 µs/op | 460.65 µs/op | 2,400.45 µs/op | **5.3× faster** |
| checkpoint_5 | 34.82 µs/op | 32.17 µs/op | 3,486.97 µs/op | **100.1× faster** |
| build_compile | 13.24 µs/op | 26.60 µs/op | 752.00 µs/op | **56.8× faster** |
| scale_005 | 38.54 µs/op | 30.54 µs/op | 1,374.61 µs/op | **35.7× faster** |
| scale_010 | 50.38 µs/op | 52.23 µs/op | 2,361.61 µs/op | **46.9× faster** |
| scale_025 | 127.31 µs/op | 123.37 µs/op | 5,880.62 µs/op | **46.2× faster** |
| scale_050 | 279.96 µs/op | 219.52 µs/op | 12,169.67 µs/op | **43.5× faster** |
| scale_100 | 457.95 µs/op | 469.63 µs/op | 18,789.14 µs/op | **41.0× faster** |
| mt_8x_linear_10 | 29.98 µs/op | 76.54 µs/op | 2,074.73 µs/op | **69.2× faster** |

## Peak allocated memory

Rust figures come from a counting global allocator (allocation high-water mark
during the measured loop). Python figures use `tracemalloc`; allocations made
inside the Rust binding are invisible to tracemalloc, so binding numbers are a
lower bound on its real footprint.

| Scenario | Rust native | Rust via Python binding | Python LangGraph |
| --- | --- | --- | --- |
| mem_linear_10 | 6.0 KiB peak | 0.3 KiB peak | 56.0 KiB peak |
| mem_loop_50 | 14.7 KiB peak | 0.3 KiB peak | 56.9 KiB peak |

## Raw data

See [`results/latest.json`](results/latest.json) for every record and
[`README.md`](README.md) for the scenario definitions and caveats.

