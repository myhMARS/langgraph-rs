# async — 异步执行

三个异步节点串成一条链，三种实现都走 `await` 语义：

```text
START -> retrieve -> generate -> judge -> END
```

| 实现 | 异步载体 |
| --- | --- |
| LangGraph | asyncio 协程 + `ainvoke` |
| binding | `AsyncStateGraph` + `ainvoke`（协程由 Rust/Tokio 驱动） |
| Rust native | `AsyncStateGraph` + `Graph::invoke`（`#[tokio::main]`） |

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`asyncio` + `ainvoke` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs.AsyncStateGraph`，同样的协程节点 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生，`AsyncStateGraph` + Tokio；额外演示消费统计 |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/async/langgraph_demo.py
python examples/async/binding_demo.py
cargo run --release --example async_native
python examples/async/compare.py --iterations 400 --runs 3
```

## 期望输出

stdout（三者一致）：

```json
{"answer": "answer based on: context for `What is a superstep?`",
 "trace": ["retrieve", "generate", "judge"]}
```

native 版还会往 **stderr** 打印消费统计（不影响结果对拍）：

```text
usage: 708 tokens, $0.000000, 0 ms
  generate   tokens=608
  judge      tokens=36
  retrieve   tokens=64
```

## 性能对比

`compare.py --iterations 400 --runs 2` 实测：

| 实现 | 每次 `ainvoke` | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 1422.57 µs | 1.00x | 0.22x |
| langgraph_rs binding | 317.73 µs | **4.48x** | 1.00x |
| Rust native | 18.98 µs | 74.95x | **16.74x** |

**这是所有功能里 binding 与 native 差距最大的一项（16.7 倍）**：异步链上每次 `await` 都要跨
Python↔Rust 事件循环桥接（协程调度、状态转换、异常映射），而 native 直接复用同一个 Tokio 运行时。
同步场景下这个差距通常只有 1.4–2.4 倍（见 quickstart / state_reducers / interrupt）。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `async def node(state)` | 同 | `\|state, ctx\| async move { ... }` |
| `StateGraph(...)` + async 节点 | `AsyncStateGraph(...)` | `AsyncStateGraph::<S>::new()` |
| `await graph.ainvoke(input)` | 同 | `graph.invoke(&input).await?` |
| — | — | `graph.run(&input, &cfg).await?` 额外返回 `report`（token/耗时） |
| `record_tokens(...)` | `langgraph_rs.record_tokens(in, out)` | `record_tokens(in, out)?` |

## 相关文档

- 异步执行与并发超步：[`docs/architecture.md`](../../docs/architecture.md)
- 消费统计与性能总报告：[`docs/performance-report.md`](../../docs/performance-report.md)