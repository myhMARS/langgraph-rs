# routing — 条件路由与 `Command(goto)`

两种「下一步去哪」的写法放在同一张图里：

```text
START -> classify --(条件边)--> fast | slow --> polish --(Command goto)--> END
```

| 方式 | 写法 | 说明 |
| --- | --- | --- |
| 条件边 | `add_conditional_edges("classify", router)` | 路由函数返回节点名 |
| `Command(goto=...)` | 节点返回 `Command(update=..., goto=END)` | 节点自己决定后继，同时带回状态更新 |

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现（条件边 + `Command`） |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同一份代码只改 import |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`add_conditional_edges` + `NodeOutput::goto(END)` |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/routing/langgraph_demo.py
python examples/routing/binding_demo.py
cargo run --release --example routing_native
python examples/routing/compare.py --iterations 400 --runs 3
```

## 期望输出

```json
{"answer": "RESEARCHED ANSWER", "route": "slow", "trace": ["classify", "slow", "polish"]}
```

问题长度 20 字符 ≥ 20，所以走 `slow`，最后由 `polish` 用 `Command(goto=END)` 结束。三者逐字一致。

## 性能对比

`compare.py --iterations 400 --runs 2` 实测：

| 实现 | 每次 invoke | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 882.36 µs | 1.00x | 0.04x |
| langgraph_rs binding | 32.89 µs | **26.82x** | 1.00x |
| Rust native | 19.23 µs | 45.89x | **1.71x** |

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `add_conditional_edges("c", router)` | 同 | `.add_conditional_edges("c", \|s\| Ok(Route::node(..)))` |
| 路由函数返回 `"fast"` / `END` | 同 | `Route::node("fast")` / `Route::end()` |
| `Command(update={...}, goto=END)` | 同（`langgraph_rs.types.Command`） | `NodeOutput::new().set(..).goto(END)` |
| `Command(goto="node")` 动态跳转 | 同 | `.goto("node")` / `.goto_parent("node")`（子图内） |

## 相关文档

- 路由与调度语义：[`docs/architecture.md`](../../docs/architecture.md)
- `Command` 的全部字段（`graph` / `resume` / `goto`）：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)