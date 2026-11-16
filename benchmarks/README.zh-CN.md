# 基准测试：langgraph-rs vs Python LangGraph

[English](README.md) | 简体中文

本目录包含 **langgraph-rs** 与参考实现 **Python LangGraph** 的功能与性能对比。结果在你自己的
机器上重新生成；不要跨机器比较绝对数值。

## 场景

所有引擎实现相同的场景、相同的节点数与迭代次数。被测量的操作是一次完整图调用（`build_compile`
例外，它测量图的构建 + 编译）。checkpoint 场景每次运行使用新的 thread id，避免 saver 无限增长。

| 场景 | 节点 | 迭代次数 | 衡量内容 |
| --- | --- | --- | --- |
| `linear_10` | 10 个链式节点，`value += 1` | 1,000 | 简单流水线的单次运行编排开销 |
| `loop_50` | 单节点 + 条件循环直到 `value == 50` | 200 | 每步循环的 superstep/条件路由开销 |
| `fanout_4` | dispatch → 4 个并行 worker → join | 1,000 | 并行 superstep 调度 + 4 路写入的 channel 合并 |
| `checkpoint_5` | 5 个链式节点 + 内存 checkpointer，每次新 thread | 200 | 每个 superstep 的 checkpoint 写/读开销 |
| `build_compile` | 10 节点图定义 | 500 | builder + 校验 + 编译成本 |
| `scale_005` … `scale_100` | 5/10/25/50/100 节点的线性链 | 1,000 → 100 | 编排成本随图规模的增长 |
| `mt_8x_linear_10` | 8 线程并发运行 `linear_10` 图 | 共 1,000 | 多线程吞吐（原生：共享一个编译图；Python：每线程一个图） |
| `mem_linear_10`、`mem_loop_50` | 与 `linear_10` / `loop_50` 相同的图 | 200 | 内存分配高水位（Rust 另含每操作分配总量） |

实现：

- Rust 原生：[`benches/bench_rust.rs`](../benches/bench_rust.rs)，运行
  `cargo bench --bench bench_rust`。
- Python LangGraph：[`python/benchmarks/bench.py`](../python/benchmarks/bench.py)
  （`--engine langgraph`），针对 LangGraph 1.2.x。
- Rust Python 绑定：同一脚本的 `--engine binding`，导入 PyO3 扩展模块
  （`cargo build --release --features python`）。
- 扩展性场景在所有引擎上以 `nodes + 10` 作为递归上限，避免 50/100 节点链触发默认上限 25。

## 运行

```bash
# 一步完成：构建产物、运行三个引擎、写入结果
python benchmarks/run.py

# 复用现有产物，仅重跑测量
python benchmarks/run.py --skip-build
```

运行器会写出：

- [`RESULTS.md`](RESULTS.md) — 生成的对比表与环境信息；
- [`results/latest.json`](results/latest.json) — 所有原始记录（engine、scenario、iterations、
  总毫秒、ns/op、ops/s）。

## 方法与注意事项

- 仅测量循环的墙钟时间；先执行 10% 的 warm-up 并排除。结果为单次运行——小于约 10% 的差异视为噪声。
- Python LangGraph 通过同步 `invoke` 入口测量；所有引擎都不使用异步事件循环。LangGraph 内部可能
  仍会创建自己的任务机制。
- checkpoint 实现不同：Python 使用 `MemorySaver`，Rust 使用基于 `Checkpointer` 端口的
  `InMemorySaver`；两者都把 checkpoint 保存在内存。
- Rust 原生使用 release 等价的 `bench` profile；Python 绑定数字包含 Python 可调用派发（每节点
  获取 GIL）与每个 superstep 的 dict 转换。
- 这些基准衡量 **编排开销**，不包含 LLM 延迟、网络 I/O 或磁盘 I/O。
- **线程：** `mt_8x_linear_10` 使用 8 个 OS 线程共 1,000 次调用。原生引擎通过 `Arc` 共享一个
  编译图；Python 引擎每个线程创建独立编译图（LangGraph 与绑定未声明线程安全的共享对象）。
  Python 的 GIL 会限制其扩展性，因此该场景反映实际使用方式下的吞吐。
- **内存：** Rust 数字来自 bench 二进制中的计数全局分配器。计时场景关闭计数器，仅在 `mem_*`
  开启，报告分配高水位与每操作分配总量。Python 数字用 `tracemalloc` 包裹测量循环；
  `tracemalloc` 只能看到 Python 层分配，因此绑定的峰值是其真实占用的下界；请以 Rust 与
  LangGraph 的对比作为有效信号。
- 计数分配器为每次分配增加一次 relaxed 原子 load；跨运行的差异请以比例而非绝对值比较。
- 功能对比（哪些对等、哪些不同）见 [`features.md`](features.md)。

## 结果快照

下表由 `python benchmarks/run.py` 重新生成；当前数字与环境见
[`RESULTS.md`](RESULTS.md)。现代 x86-64 机器上的典型结果：对于同样的小图，Rust 原生比 Python
LangGraph 快一到两个数量级，PyO3 绑定位于两者之间（每个节点执行都增加 Python 可调用派发）。