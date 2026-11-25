# subgraph — 子图与 `Send` 扇出

一张图里同时演示两种调度形态：

```text
START -> dispatch --Send(n)--> worker x4 --(append)--> reduce --> scaled(subgraph) -> END
```

1. **`Send` map**：`dispatch` 为每个数字发一个任务，每个 `worker` 只看到自己的参数，写入共享的
   `results` 通道（append reducer 负责合并）。
2. **子图**：`scaled` 是一个编译好的子图嵌进父图，在**独立命名空间**里运行，写回父状态。

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现（`Send` + `add_node("scaled", subgraph)`） |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同一份代码只改 import |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`Send::with_arg` + `add_subgraph` |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/subgraph/langgraph_demo.py
python examples/subgraph/binding_demo.py
cargo run --release --example subgraph_native
python examples/subgraph/compare.py --iterations 200 --runs 3
```

## 期望输出

```json
{"scaled": [220], "sorted_results": [1, 9, 36, 64], "total": 110}
```

三个实现一致：`3²+8²+1²+6² = 110`，子图再乘 `factor=2` 得到 `220`。

## 性能对比

`compare.py --iterations 200 --runs 2` 实测：

| 实现 | 每次 invoke | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 4576.55 µs | 1.00x | 0.04x |
| langgraph_rs binding | 179.76 µs | **25.46x** | 1.00x |
| Rust native | 79.37 µs | 57.67x | **2.26x** |

这一项比 quickstart 慢是预期的：一次 `invoke` 里有 4 个并行 `Send` 任务、一次归约和一次子图调度，
状态跨父/子命名空间映射。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `Send("worker", {"number": n})` | 同（`langgraph_rs.types.Send`） | `Send::with_arg("worker", json!({...}))` |
| `add_node("scaled", subgraph)` | 同（会自动识别编译后的子图） | `.add_subgraph("scaled", Arc::new(child))` |
| `Annotated[list, add]` | 同 | `.with_reducer("results", reducers::append())` |
| `graph.compile()` | 同 | `.compile()` |

## 子图能暂停（interrupt）时的注意事项

子图里调用 `interrupt()`、由父图 `resume()` 继续，这一路径已经可用（见
[`docs/compatibility-audit.md`](../../docs/compatibility-audit.md) 第 62、63 轮），但有一个前提：

> 子图**可以不传** `checkpointer`：父图编译时的 saver 会自动继承给子图（与 LangGraph 一致），
> 嵌套多层同样成立。

暂停发生在字图时，父图的 `resume(thread, value)` 会把值传进子图、在子图原来的命名空间里继续，
然后父图接着跑后续节点。

## 相关文档

- 跨引擎恢复（含子图/命名空间）：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)
- 子图实现与已知差异：[`docs/compatibility-audit.md`](../../docs/compatibility-audit.md)