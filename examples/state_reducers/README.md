# state_reducers — 状态通道与 reducer

两个节点在**同一个超步**里并行执行，并同时写同一批通道。能不能写、写进去怎么合并，由每个通道的
reducer 决定：

| 通道 | 类型 | reducer | 并行写入的结果 |
| --- | --- | --- | --- |
| `items` | 列表 | `operator.add` / `reducers::append()` | 两份写入都保留，拼接 |
| `total` | 整数 | 自定义 `merge_max` | 取较大值，不报冲突 |
| `note` | 字符串 | 默认 last-value | 只有 `summarize` 写，无冲突 |

图结构：

```text
START ──┬──> left  ──┐
        └──> right ──┴──> summarize ──> END
```

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现（`Annotated[list, add]`、`Annotated[int, merge_max]`） |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同一份代码只改 import |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`reducers::append()` + `Reducer::new(...)` |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/state_reducers/langgraph_demo.py
python examples/state_reducers/binding_demo.py
cargo run --release --example state_reducers_native
python examples/state_reducers/compare.py --iterations 300 --runs 3
```

## 期望输出

```json
{"items": ["left-1", "left-2", "right-1", "right-2"], "note": "4 items, total=7", "total": 7}
```

三个实现一致。注意 `items` 在输出前做了排序：**并行写入的落地顺序属于实现细节**，真正要保证的是
「两份写入都保留、`total` 取最大值」这个语义，所以跨引擎比较时排序后再比。

## 性能对比

`compare.py --iterations 300 --runs 2` 实测：

| 实现 | 每次 invoke | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 2182.08 µs | 1.00x | 0.05x |
| langgraph_rs binding | 103.20 µs | **21.14x** | 1.00x |
| Rust native | 43.24 µs | 50.46x | **2.39x** |

比 quickstart 慢是预期的：这里每个超步有 2 个节点，且要合并两个 reducer 通道。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `Annotated[list, add]` | 同 | `.with_reducer("items", reducers::append())` |
| `Annotated[int, merge_max]` | 同 | `.with_reducer("total", Reducer::new(\|a, b\| ...))` |
| 未标注字段 = last-value | 同 | `.last_value("note")` |
| `add_edge(START, "left")` 两次 = 并行扇出 | 同 | `.add_edge(START, "left").add_edge(START, "right")` |

## 相关文档

- 通道语义与 reducer 列表：[`docs/architecture.md`](../../docs/architecture.md)
- 大规模并行/归约基准：[`benchmarks/mega_workflow/`](../../benchmarks/mega_workflow/)