# streaming — 流式输出

三个节点依次执行，每次 `stream(..., stream_mode="updates")` 产出一条 `{node: writes}` 更新：

```text
START -> plan -> work -> report -> END
        └ chunk 1 ┘ chunk 2 ┘ chunk 3
```

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`compiled.stream(..., stream_mode=...)` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs` binding，同样的调用形态 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`graph.stream(&input, &config)` → `Vec<StreamChunk>`，取 `Updates` 变体 |
| [`compare.py`](compare.py) | 逐 chunk 对拍 + 耗时对比 |

## 运行

```bash
python examples/streaming/langgraph_demo.py
python examples/streaming/binding_demo.py
cargo run --release --example streaming_native

# values 模式（每个超步的完整状态）
python examples/streaming/langgraph_demo.py --mode values
python examples/streaming/binding_demo.py --mode values

python examples/streaming/compare.py --iterations 300 --runs 3
```

## 期望输出

```json
{"chunks": [{"plan": {"log": ["plan"], "plan": "3 steps"}},
            {"work": {"log": ["work"], "result": "did 3 steps"}},
            {"report": {"log": ["report"], "summary": "did 3 steps / 2 events"}}],
 "mode": "updates"}
```

三条实现**逐 chunk 完全一致**（顺序、节点名、写入内容都相同）。

## 支持的 stream mode

| mode | 含义 | binding | native（`StreamChunk` 变体） |
| --- | --- | --- | --- |
| `values` | 每个超步后的完整状态 | ✅ | `StreamChunk::Values { step, values }` |
| `updates` | 每个节点产生的写入 | ✅ | `StreamChunk::Updates { step, node, update }` |
| `custom` | 节点内 `custom(...)` 主动推送 | ✅ | `StreamChunk::Custom { node, value }` |
| `messages` | 节点内 `message(...)` 推送的消息 | ✅ | `StreamChunk::Message { node, message }` |
| `debug` | 调度/检查点调试事件 | ✅ | `StreamChunk::Debug { .. }` / `Task { .. }` |

## 性能对比

`compare.py --iterations 300 --runs 2` 实测（每次迭代消费完整 update 流并序列化为 JSON）：

| 实现 | 每次 run | 相对 LangGraph |
| --- | ---: | ---: |
| LangGraph | 825.14 µs | 1.00x |
| langgraph_rs binding | 38.00 µs | **21.71x** |
| Rust native | 50.08 µs | 16.48x |

**这里的 native 比 binding 慢，不是引擎慢**：这个 demo 的耗时大头是「把每个 chunk 组装成 JSON」——
binding 侧由 Python 的 `json.dumps` 统一序列化一次，而 native demo 用 `json!` 宏逐 chunk 分配
`serde_json::Value`。引擎本身的产流速度见 `benchmarks/`（大工作流 + criterion 微基准），那里对
「纯执行」和「含序列化」是分开测的。

结论：**流式场景两者都在 50µs 以内、比 LangGraph 快 16x 以上**；若要比较纯引擎吞吐，看
[`benchmarks/README.md`](../../benchmarks/README.md)。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `compiled.stream(input, stream_mode="updates")` | 同 | `graph.stream(&input, &config)?` |
| `for chunk in ...` | 同 | `for chunk in chunks { match chunk { StreamChunk::Updates {..} => ... } }` |
| `stream_mode="values"` | 同 | 过滤 `StreamChunk::Values` |
| 节点内 `get_stream_writer()` | `langgraph_rs.config.get_stream_writer` | `langgraph_rs::custom(...)` / `message(...)` |

## 相关文档

- 流式协议与 chunk 形状：[`docs/architecture.md`](../../docs/architecture.md)
- `StreamPart` TypedDict 与 LangGraph 的对应：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)