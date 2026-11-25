# checkpointing — 持久化、时间旅行与断点续跑

一个节点、一个线程、四次检查点操作：

```text
run 1 ─► count=1   run 2 ─► count=2   run 3 ─► count=3
                                   │
                    update_state(count=42)  ← 不跑节点，直接改状态（分叉）
                                   │
                              run 4 ─► count=43
```

| 操作 | 作用 |
| --- | --- |
| 多次 `invoke` 同一 `thread_id` | 每次都从上一个检查点继续 |
| `update_state(thread, {...})` | 分叉：不执行节点，直接写入并生成新检查点 |
| `get_state_history(thread)` | 列出该线程的全部检查点（本 demo 打印到 stderr） |

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`InMemorySaver` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同一流程 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`CompileOptions::with_checkpointer` + `update_state` + `get_state_history` |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/checkpointing/langgraph_demo.py
python examples/checkpointing/binding_demo.py
cargo run --release --example checkpointing_native
python examples/checkpointing/compare.py --iterations 150 --runs 3
```

## 期望输出

stdout（三者一致）：

```json
{"final": 43, "log": ["count=1", "count=2", "count=3", "count=43"]}
```

stderr（仅用于观察，不参与对拍）：

```text
history checkpoints: 13
  count=43
  count=42
  ...
```

## 性能对比

`compare.py --iterations 150 --runs 2` 实测：

| 实现 | 每次（4 轮 + 分叉 + 历史） | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 11379.96 µs | 1.00x | 0.02x |
| langgraph_rs binding | 171.67 µs | **66.29x** | 1.00x |
| Rust native | 143.27 µs | 79.44x | **1.20x** |

这是所有功能里 absolute 耗时差距最大的一项：每次迭代要写 4 个检查点、做一次分叉并列出全部历史。

## 一个 API 差异（native 侧）

`binding` 和 LangGraph 的 `invoke({"count": n})` 是**部分写入**，会走通道 reducer，所以只更新
`count`、`log` 继续累积。native 的 `invoke(&state)` 接收的是**完整状态**，因此 demo 里先
`get_state` 再合并 `count`（等价于 LangGraph 的续跑语义）。

binding 侧没有这个差异 —— 因为 Python 侧把传入的 dict 当部分写入处理。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `compile(checkpointer=InMemorySaver())` | 同 | `.compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))` |
| `invoke(input, config)`（部分写入） | 同 | `invoke(&full_state, &RunConfig::new().with_thread(t))`（完整状态） |
| `update_state(config, {"count": 42})` | `update_state(thread, {"count": 42})` | `update_state(thread, [("count", json!(42))], None)` |
| `get_state_history(config)` | `get_state_history(thread)` | `get_state_history(thread)` |
| `get_state(config)` | `get_state(thread)` | `get_state(thread)` |

## 相关文档

- 检查点格式与 `update_state` 语义：[`docs/architecture.md`](../../docs/architecture.md)
- 跨引擎共存（同一 SQLite 文件两引擎交替续跑）：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)