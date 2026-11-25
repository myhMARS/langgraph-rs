# interrupt — 人机协同（暂停 / 恢复）

`review` 节点调用 `interrupt()` 暂停整条图，等人工给出答复后再继续：

```text
START -> draft -> review --(interrupt, 等待人工)--> publish -> END
```

三个实现的关键差异只有「暂停信号怎么返回」：

| 实现 | 暂停时的返回 | 恢复方式 |
| --- | --- | --- |
| LangGraph | `invoke()` 返回值里带 `__interrupt__` | `invoke(Command(resume=...), config)` |
| binding | 同样返回 `__interrupt__`（不抛异常） | `invoke(Command(resume=...), thread_id)` 或 `resume(thread, value)` |
| native | `Err(Error::Interrupt)` | `graph.resume(thread, value)` |

> `interrupt()` 不是失败，是「暂停」。恢复时被中断的节点会**从头重新执行**，那次 `interrupt()` 调用
> 返回人工给的值；因此节点里 `interrupt()` 之前不要做副作用。

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`InMemorySaver` + `Command(resume=...)` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同一流程 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`interrupt(...)?` → `Err(Error::Interrupt)` → `graph.resume` |
| [`compare.py`](compare.py) | 三者结果一致性 + 完整「暂停+恢复」一轮的耗时对比 |

## 运行

```bash
python examples/interrupt/langgraph_demo.py
python examples/interrupt/binding_demo.py
cargo run --release --example interrupt_native
python examples/interrupt/compare.py --iterations 100 --runs 3
```

## 期望输出

```json
{"approved": true,
 "draft": "Quarterly report v1 [published]",
 "feedback": "Looks good, ship it!",
 "paused_at": "review",
 "payload": {"options": ["approve", "reject"], "question": "Approve `Quarterly report v1`?"}}
```

三者逐字一致：包括暂停时**待执行节点名**（`paused_at`）和中断载荷（`payload`）。

## 性能对比

`compare.py --iterations 100 --runs 2` 实测（每次迭代是一整轮 暂停 → 恢复，含 checkpoint 写入）：

| 实现 | 每轮暂停+恢复 | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 4000.14 µs | 1.00x | 0.03x |
| langgraph_rs binding | 112.40 µs | **35.59x** | 1.00x |
| Rust native | 80.78 µs | 49.52x | **1.39x** |

native 与 binding 的差距比纯计算场景小，因为这一轮的主要成本是 checkpoint 读写与跨边界传状态。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `from langgraph.types import interrupt` | `from langgraph_rs import interrupt` | `use langgraph_rs::interrupt;` |
| `interrupt(payload)` | 同 | `interrupt(json!(...))?` |
| `compile(checkpointer=InMemorySaver())` | `compile(checkpointer=InMemorySaver())` | `.compile_with(CompileOptions::new().with_checkpointer(...))` |
| `invoke({"__interrupt__": ...})` 暂停 | 同 | `Err(Error::Interrupt)` |
| `invoke(Command(resume=v), config)` | 同（或 `compiled.resume(thread, v)`） | `graph.resume(thread, v)?` |

## 相关文档

- 中断语义与恢复规则：[`docs/architecture.md`](../../docs/architecture.md)
- 跨引擎恢复（LangGraph 与 langgraph_rs 互相恢复对方的中断）：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)
- 更多中断场景（并行中断、`Send` 扇出、多轮中断）：[`python/tests/test_bindings.py`](../../python/tests/test_bindings.py)