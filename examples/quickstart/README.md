# quickstart — 三节点状态图 + 条件循环

这是最小可用的 LangGraph 程序：一个 `seed` 节点写入目标值，一个 `increment` 节点自增计数，
一条条件边在未达标时循环回 `increment`。三个实现完全等价：

| 文件 | 实现 | 说明 |
| --- | --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph（参考实现） | `from langgraph.graph import ...` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` Python binding | **只改 import**，其余代码逐行相同 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生 API | `cargo run --example quickstart_native` |
| [`compare.py`](compare.py) | 对比脚本 | 先校验三者结果一致，再测三者耗时 |

## 运行

```bash
# 1) 三条路径各自跑一次，输出必须相同
python examples/quickstart/langgraph_demo.py
python examples/quickstart/binding_demo.py
cargo run --release --example quickstart_native

# 2) 一致性 + 性能对比（会自己调用上面三条命令）
python examples/quickstart/compare.py --iterations 500 --runs 3

# 只做结果校验
python examples/quickstart/compare.py --check-only
```

`binding_demo.py` 与 `compare.py` 需要先构建扩展：

```bash
cargo build --release --features "python python-async sqlite redis postgres msgpack"
python python/build_package.py --release
set PYTHONPATH=%CD%\target\python        # PowerShell: $env:PYTHONPATH = "$PWD\target\python"
```

## 期望输出

```
{"count": 3, "steps": ["seed", "increment -> 1", "increment -> 2", "increment -> 3"], "target": 3}
```

三个实现逐字一致（`compare.py` 里按 JSON 解析后比较）。

## 性能对比

`compare.py` 的一次实际输出（本机，Windows，release 构建，`--iterations 500 --runs 2`）：

| 实现 | 每次 invoke | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 1502.38 µs | 1.00x | 0.03x |
| langgraph_rs binding | 42.80 µs | **35.10x** | 1.00x |
| Rust native | 22.66 µs | 66.31x | **1.89x** |

说明：

- 数字随机器负载波动，**以你自己跑出来的为准**，不要用上表做绝对结论；相对关系（binding ≫ LangGraph、
  native 约为 binding 的 2 倍快）在多轮测试中稳定。
- native 的耗时在二进制内部用 `Instant` 统计，已排除进程启动；binding 的耗时在 Python 进程内统计，
  已排除 import 和首次编译图的开销（先预热 50 次，再取 5 组中位数）。
- binding 的固定开销来自「每次调用跨 Python/Rust 边界 + 状态 JSON 转换」，节点越多、单节点越重，
  这个比例越低。

## 对应关系（代码映射）

| LangGraph | langgraph_rs binding | Rust native |
| --- | --- | --- |
| `StateGraph(State)` | 同 | `StateGraph::<DemoState>::new()` |
| `Annotated[list, append]` | 同 | `.with_reducer("steps", reducers::append())` |
| `graph.add_node("seed", seed)` | 同 | `.add_node("seed", \|state\| ...)` |
| `add_conditional_edges(...)` | 同 | `.add_conditional_edges(...)` |
| `compiled.invoke({...})` | 同 | `graph.invoke(&input, &RunConfig::new())` |

## 相关文档

- 兼容性说明：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)
- 性能总报告：[`docs/performance-report.md`](../../docs/performance-report.md)
- 基准测试目录：[`benchmarks/README.md`](../../benchmarks/README.md)