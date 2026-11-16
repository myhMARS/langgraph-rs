# langgraph-rs

[English](README.md) | 简体中文

**LangGraph 核心运行时的 Rust 重写**：一个同步的、状态 + 图引擎，用于编排 LLM Agent 与工作流。
工作单元是一个 `StateGraph`，节点是普通的 `Fn(&State) -> Result<NodeOutput>` 闭包；节点返回
*写入*，引擎通过 channel 合并写入、路由边、在步骤之间做 checkpoint，并且可以暂停等待人工输入后
从断点恢复。没有异步运行时、没有隐藏的全局状态，并提供可选的 Python 绑定。

## 特性

- **Channel 与 reducer** — `LastValue`、`Topic` 或针对自定义 `Serialize + Deserialize` 状态
  类型的任意折叠函数，并由 `StateSchema` 校验。
- **静态 + 条件边** — 多目标扇出、router 返回节点名、通过 `PathMap` 解析符号键、条件入口、
  `goto` 覆盖。
- **并行 superstep + `Send` map/reduce** — 同一 superstep 的节点在 scoped 线程上并行执行；
  `Send` 为每个元素创建带独立状态的任务，并通过 channel 归约；router 也可返回
  `Route::sends([...])`（或 `Route::nodes([...]).with_sends([...])`）在下一 superstep 派发动态
  `Send` 任务，条件入口点同样支持；Python router 可返回单个 `lg.Send` 或包含节点名与 `lg.Send`
  的列表。
- **重试 + 错误路由** — 每节点 `RetryPolicy`（次数、指数退避）以及通过 `last_error()` 读取失败
  信息的 `add_error_edge` 兜底 handler。
- **子图** — `add_subgraph` 把已编译图作为节点嵌入；共享状态类型，仅发布发生变化的 channel。
- **Mermaid 可视化** — `graph.to_mermaid()` 渲染节点、静态/条件边、path-map 标签、错误边、
  重试与中断钩子；Python 绑定同样可用。
- **节点级缓存** — 按节点 opt-in 的 `CachePolicy` + `NodeCache` 端口；`InMemoryCache` 适配器
  支持 TTL 与自定义 key，命中会在运行报告中标记。
- **持久化 Checkpointer** — `SqliteSaver`、`RedisSaver`、`PostgresSaver` 实现同一个
  `Checkpointer` 端口，分别对应 `sqlite` / `redis` / `postgres` feature。
- **Tokio 异步运行时（`async` feature）** — `AsyncStateGraph` 在 Tokio 上运行异步节点并并发
  等待 superstep；同步路径不引入运行时依赖。
- **Checkpoint + 线程历史** — `InMemorySaver`、类型化快照、父子链历史、`resume_from` 时间旅行。
- **Human-in-the-loop** — `interrupt()` 携带 payload 暂停运行，`resume` 重新执行节点并注入答复；
  `interrupt_before` / `interrupt_after` 提供粗粒度闸门。
- **流式输出** — 值、节点级更新、自定义事件、中断、checkpoint id。
- **节点级消耗追踪** — 节点通过 `record_usage` 或 `NodeOutput::with_usage` 上报 token/成本；
  `run` 返回包含每节点耗时、usage 与聚合总量的 `RunReport`。
- **Python 绑定** — 可选 `python` feature 通过 PyO3 暴露同一引擎（`StateGraph` /
  `CompiledGraph`、`interrupt` / `resume`、运行报告）。
- **Microkernel + Hexagonal 架构** — 领域内核（图模型、superstep 执行器、端口）与适配器
  （checkpoint、追踪、语言绑定）分离；channel 合并策略是注册在 `Kernel` 中的插件。
- **与 Python LangGraph 的基准对比** — Rust 原生、PyO3 绑定、LangGraph 三引擎同场景对比，
  并提供功能矩阵（见 [benchmarks/](benchmarks/)）。

## 为什么用 Rust 重写？

LangGraph 的模型——状态、superstep、显式写入、中断——在 Python 之外同样有价值。忠实的重写不应
只是语法翻译：状态是编译期检查的类型，节点写入未声明的 channel 会直接失败，模糊合并不可能发生
——对 `LastValue` 的并发写入是错误而不是竞态。节点是同步闭包，superstep 运行在
`std::thread::scope` 上，`Checkpointer` 是 trait，持久化后端由你决定。

## 架构：Microkernel + Hexagonal

crate 由 **kernel**（纯领域逻辑）与 **adapters**（基础设施）组成，通过 **ports**（trait）连接：

```text
                 ┌────────────────────────── adapters ──────────────────────────┐
   Python ─────► │ adapters::python (PyO3)   adapters::memory (checkpoint saver) │
                 └───────────────▲───────────────────────────▲──────────────────┘
                                 │ implements ports          │
                 ┌───────────────┴───────────────────────────┴──────────────────┐
                 │ kernel::ports   Checkpointer, TraceSink                      │
                 │ kernel::plugin  Kernel / Plugin / ChannelFactory             │
                 │ kernel          graph model, channels, state, executor,      │
                 │                 usage, streaming, checkpoints, interrupts     │
                 └──────────────────────────────────────────────────────────────┘
```

- **端口**（`kernel::ports`）是内核唯一依赖的契约：`Checkpointer`（持久化）与 `TraceSink`
  （消耗记录）。
- **适配器**（`adapters::*`）实现端口：`InMemorySaver`（内存 checkpoint）、PyO3 绑定，以及
  下游自定义的 Redis/Postgres/OTel 等实现。
- **微内核**（`kernel::plugin`）：内核从 `Kernel` 注册表解析能力，而不是自行构造。内建 channel
  策略（`last_value`、`topic`、`append`、`add`、`messages`）由 `BuiltinChannels` 注册；下游通过
  `Plugin` 扩展：

```rust
use langgraph_rs::{Channel, CompileOptions, Kernel, Plugin, StateGraph};

struct MyReducer;

impl Plugin for MyReducer {
    fn name(&self) -> &'static str {
        "my-reducer"
    }

    fn install(&self, kernel: &mut Kernel) -> langgraph_rs::Result<()> {
        kernel.register_channel("trimmed_append", || {
            Channel::reducer(|current, incoming| {
                let mut values = current.as_array().cloned().unwrap_or_default();
                if let Some(items) = incoming.as_array() {
                    values.extend(items.iter().cloned());
                }
                Ok(serde_json::Value::Array(values))
            })
        });
        Ok(())
    }
}

let kernel = Kernel::builder().with_plugin(MyReducer)?.build();
let schema = kernel.schema([("messages".to_string(), "trimmed_append".to_string())])?;
let graph = StateGraph::<Pipeline>::with_schema(schema)
    .add_node("node", |_state: &Pipeline| Ok(Default::default()))
    .set_entry_point("node")
    .set_finish_point("node")
    .compile_with(CompileOptions::new().with_kernel(kernel))
    .unwrap();
```

内核不引用任何具体适配器：持久化、追踪与 Python 都在内核之外。

## 快速开始

添加 `langgraph-rs`（发布前先用 path 依赖）、`serde`（derive）与 `serde_json`。状态是任意
`Clone + Default + Serialize + Deserialize` 类型，`State::schema` 声明 channel 与合并规则。

```rust
use langgraph_rs::{NodeOutput, Route, RunConfig, State, StateGraph, StateSchema};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Counter {
    value: i64,
}

impl State for Counter {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

let graph = StateGraph::<Counter>::new()
    .add_node("increment", |state: &Counter| {
        Ok(NodeOutput::new().set("value", state.value + 1))
    })
    .set_entry_point("increment")
    .add_conditional_edges("increment", |state: &Counter| {
        Ok(if state.value < 3 {
            Route::node("increment")
        } else {
            Route::end()
        })
    })
    .compile()
    .unwrap();

let output = graph.invoke(&Counter::default(), &RunConfig::new()).unwrap();
assert_eq!(output.value, 3);
```

## 状态 channel 类型

channel 是状态的一个字段，决定一个 superstep 的写入如何成为下一个值：

| Channel | 合并行为 | 适用场景 |
| --- | --- | --- |
| `LastValue` | 最新写入获胜；同一 superstep 两次写入会以 `Error::ConcurrentUpdate` 失败 | 标量与单节点每步拥有的控制标志 |
| `Topic` | 每次写入追加到 JSON 数组 | 对顺序敏感的事件日志与 trace |
| `Reducer` | 写入按左到右折叠 | 多方更新的消息、计数器、映射 |
| `AnyValue` | 最新写入生效，且允许多个节点并发写入 | 独立的 last-write channel |
| `EphemeralValue` | 值只在一个 superstep 内可见，之后清空 | 供路由使用的瞬时信号 |
| `UntrackedValue` | 类似 `LastValue`，但不写入 checkpoint | 仅运行时数据、敏感值 |
| `NamedBarrierValue` | 记录写入者名称，全部到齐后变为 `true` | 汇合协调 |
| `DeltaChannel` | 写入按 reducer 折叠，保存累计值 | token / 部分结果流 |
| `LastValueAfterFinish` | 保留最后一次写入，但只在图结束时发布 | 循环中产生的最终答案 |

`Send::with_timeout` 可为单个动态任务设置超时（覆盖节点 `TimeoutPolicy`）；Python 使用
`lg.Send(node, arg, timeout=秒)`。用 `add_node_deferred` / `set_deferred` 把清理/汇总节点标记为
deferred，在全部常规节点结束后执行；deferred 节点支持中断/resume，但不能作为入口或拥有出边。

`NodeOutput::overwrite(channel, value)` 直接写入 channel、绕过 reducer（对应 LangGraph 的
`Overwrite`）。`Channel::delta(reducer)` 接受 snapshot frequency 参数以兼容 API；本引擎始终快照完整值。

内置 reducer：`reducers::append()`、`reducers::add()`、`reducers::messages()`（按 `id` 合并，
`RemoveMessage` 可删除消息）。`MessageGraph`（`StateGraph<MessagesState>` 类型别名）与
`lg.message_graph()` 提供消息专用图便捷入口；`MessagesState` 是聊天记录的便捷状态类型；
Python 提供 `lg.MessagesState` 与 `lg.RemoveMessage`。
`JsonState` 是无 schema 的替代方案：JSON map，未声明的 channel 按 `LastValue` 处理（用
`StateGraph::<JsonState>::with_schema` 注册 reducer）。

## 条件路由

router 是纯函数 `Fn(&State) -> Result<Route>`，在 superstep 写入合并 *之后* 的状态上求值。可返回
单个节点、多个节点（`Route::nodes([...])` 扇出）或 `Route::end()`。配合 path map 时返回符号键而
不是节点名；缺失的键是硬错误。

```rust
let graph = StateGraph::<Counter>::new()
    .add_node("work", |state: &Counter| Ok(NodeOutput::new().set("value", state.value + 1)))
    .add_node("done", |_state: &Counter| Ok(NodeOutput::new()))
    .set_entry_point("work")
    .add_conditional_edges_mapped("work", |state: &Counter| {
        Ok(Route::node(if state.value >= 7 { "done" } else { "retry" }))
    }, [("done", "done"), ("retry", "work")])
    .set_finish_point("done")
    .compile()
    .unwrap();
```

`add_conditional_edges` 直接路由到具体节点名；节点也可用 `NodeOutput::goto("target")` 覆盖当前步骤
的出边。

## 并行执行与 `Send`

同一 superstep 调度的节点共享状态快照并并发执行，下一个 superstep 看到合并后的写入。边可以静态
扇出，`Send` 则携带每任务独立状态。

```rust
use serde_json::json;
use langgraph_rs::{NodeOutput, RunConfig, Send, State, StateGraph, StateSchema, reducers};

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
struct Pipeline {
    value: i64,
    #[serde(default)]
    log: Vec<String>,
}

impl State for Pipeline {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value").with_reducer("log", reducers::append())
    }
}

// Map/reduce：`dispatch` 派发三个带独立状态的任务，写入通过 `log` 归约。
let map_reduce = StateGraph::<Pipeline>::new()
    .add_node("dispatch", |_s: &Pipeline| {
        Ok([1, 2, 3].into_iter().fold(NodeOutput::new(), |output, item| {
            output.send(Send::with_arg("worker", json!({ "value": item })))
        }))
    })
    .add_node("worker", |state: &Pipeline| {
        Ok(NodeOutput::new().set("log", vec![format!("worker-{}", state.value)]))
    })
    .set_entry_point("dispatch")
    .set_finish_point("worker")
    .compile()
    .unwrap();

let output = map_reduce.invoke(&Pipeline::default(), &RunConfig::new()).unwrap();
assert_eq!(output.log.len(), 3);
```

`Send::with_arg` 替换目标任务的全部状态（必须能反序列化为 `S`）；`Send::to` 转发当前状态。不同的
`Send` 并发执行并通过 reducer 合并。

节点可声明显式的目标允许列表（`set_destinations` / `add_node_with_destinations`）：
静态边、条件路由、`goto` 与 `Send` 若指向列表之外的目标，会在运行时以 routing 错误失败
（`END` 始终允许）。异步适配器同样提供 `batch` / `batch_run`。

## 批量执行、并发与持久化策略

`CompiledGraph::batch` 在一组输入上运行同一张图，结果按输入顺序返回；
`RunConfig::with_max_concurrency(n)` 限制并发调用数（scoped 线程）。`batch_run` 额外返回
`RunOutcome` 报告。`RunConfig::with_durability` 选择 `Durability::Sync`（每个 superstep 写
checkpoint）、`Async`（当前等同 `Sync`）或 `Exit`（只写初始、中断和最终 checkpoint）。
`add_sequence` 一次注册链式节点，`add_edges_from` 支持多源汇入同一目标。

```rust
let results = graph.batch(inputs, &RunConfig::new().with_max_concurrency(4));
let outcomes = graph.batch_run(inputs, &RunConfig::new());
let config = RunConfig::new().with_durability(Durability::Exit);
```

## Functional API

`langgraph_rs::functional` 在同一引擎上提供 `entrypoint` / `task` 编程模型。
`entrypoint(function)` 返回带单个 `entrypoint` 节点的 `StateGraph`（checkpoint、interrupt/resume、
流式、超时与 store 均可用）；`task(name, function)` 包装函数，其结果以「任务名 + 序列化输入」
为键记忆化到运行的 `BaseStore` 中。`interrupt()` 后恢复 entrypoint 时，已完成任务直接从 store
返回而不会重复执行。任务仍为内联执行（无并行任务调度、无逐任务 checkpoint 记录）。

```rust
use langgraph_rs::functional::{entrypoint, task};

let fetch = task("fetch", |id: i64| Ok(id + 1));
let graph = entrypoint(move |state: MyState| {
    let value = fetch.call(state.id)?;
    Ok(MyState { id: value })
})
.with_store(store)
.compile()?;
```

## Prebuilt helpers

`prebuilt` 提供工具调用 agent 常用的小型助手：

```rust
use langgraph_rs::prebuilt::{ToolDefinition, ToolNode, tools_condition, create_react_agent};

let tools = [ToolDefinition::new("add", |args| {
    Ok(serde_json::json!(args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0)))
})];
let graph = create_react_agent(model_node, tools).compile()?;
```

`tools_condition(state)` 在最后一条消息含 `tool_calls` 时路由到 `"tools"`，否则 `END`；
`ToolNode` 执行调用并追加 `role="tool"` 结果消息。Python 提供 `lg.ToolNode`、
`lg.tools_condition`、`lg.create_react_agent` 与 `lg.END`。

## 可靠性：重试与错误路由


节点可配置重试策略；失败时引擎自动重试。中断永不重试。重试耗尽后运行失败——除非配置了错误边，
失败会路由到 handler 节点，在下一个 superstep 执行，并可通过 `last_error()` 读取失败信息。

```rust
use std::time::Duration;
use langgraph_rs::{Error, NodeOutput, RetryPolicy, StateGraph, WorkState, last_error};

let graph = StateGraph::<WorkState>::new()
    .add_node_with_retry(
        "call_model",
        RetryPolicy::new(3)
            .with_backoff(Duration::from_millis(200))
            .with_multiplier(2.0),
        |state: &WorkState| call_model(state),
    )
    .add_node("fallback", |_state: &WorkState| {
        let failure = last_error().expect("handler sees the failure");
        Ok(NodeOutput::new().set("answer", format!("degraded: {}", failure.message)))
    })
    .add_error_edge("call_model", "fallback")
    .set_entry_point("call_model")
    .set_finish_point("fallback")
    .compile()
    .unwrap();
```

节点级超时通过 `TimeoutPolicy`（`add_node_with_timeout` / `set_timeout`）配置：`run_timeout`
会被强制执行——同步图在辅助线程上运行节点，超时以 `Error::Timeout` 失败（Rust 无法取消阻塞
闭包），异步图使用 `tokio::time::timeout`；`idle_timeout` 仅为 LangGraph API 兼容而保存，暂不
强制执行。managed values 通过 `remaining_steps()` 与 `is_last_step()` 获取（Python 同样可用）。

## 子图

`add_subgraph` 把已编译图作为节点嵌入。子图针对父状态运行，其发生变化的 channel 以 **直接替换**
写回（绕过父级 reducer），因此 reducer channel 不会被重复追加。

```rust
use std::sync::Arc;
use langgraph_rs::{NodeOutput, RunConfig, StateGraph};

let inner = Arc::new(
    StateGraph::<WorkState>::new()
        .add_node("step", |state: &WorkState| Ok(NodeOutput::new().set("value", state.value + 1)))
        .set_entry_point("step")
        .set_finish_point("step")
        .compile()
        .unwrap(),
);

let outer = StateGraph::<WorkState>::new()
    .add_node("seed", |_state: &WorkState| Ok(NodeOutput::new().set("value", 10)))
    .add_subgraph("inner", inner)
    .set_entry_point("seed")
    .add_edge("seed", "inner")
    .set_finish_point("inner")
    .compile()
    .unwrap();
```

`add_subgraph` 共享父状态类型；`add_subgraph_mapped` 支持不同状态类型的子图并提供输入/输出映射
闭包。子图继承父线程，并在子 checkpoint 命名空间（`<parent_ns>|<name>`）中运行，因此子图内
`interrupt()` 会暂停父运行，下一次 `resume` 从子图自己的 checkpoint 继续；子图流式
（`subgraphs=True`）可把子图 chunk 带命名空间输出。子图节点返回
`NodeOutput::goto_parent("parent_node")`（Python `Command(..., graph=Command.PARENT)`）时，
子图中止、写入应用到父状态并调度父图目标节点（同步图与异步图均支持）。

## 节点级缓存

缓存按节点 opt-in：给图挂 `NodeCache` 后端，给节点挂 `CachePolicy`。节点执行前，执行器用「节点名 +
输入状态」派生的 key 查找缓存；成功后写入。中断与失败永不缓存，命中会在运行报告中标记 `cached`。

```rust
use std::sync::Arc;
use std::time::Duration;
use langgraph_rs::{CachePolicy, InMemoryCache, StateGraph};

let graph = StateGraph::<WorkState>::new()
    .add_node("call_model", |state: &WorkState| call_model(state))
    .set_entry_point("call_model")
    .set_finish_point("call_model")
    .with_cache(Arc::new(InMemoryCache::new()))
    .set_cache(
        "call_model",
        CachePolicy::new().with_ttl(Duration::from_secs(300)),
    )
    .compile()
    .unwrap();
```

默认 key 哈希节点名与序列化后的输入状态（只有完全相同的输入才命中）；`CachePolicy::with_key` 可
安装更粗粒度的自定义 key。`NodeCache` 是端口，后续可接入 SQLite/Redis/Postgres 缓存。

## 长期记忆与运行时上下文

`BaseStore` 提供跨运行、跨线程的长期记忆。通过 `CompileOptions::with_store`（或
`StateGraph::with_store`）挂载内置 `InMemoryStore`，节点通过 `get_store()` 读取；
`RunConfig::with_context` 携带每次运行的用户上下文，节点通过 `current_context()` 访问。
`current_runtime()` 汇总线程 id、命名空间、store 与 context；`current_thread_id()` /
`current_namespace()` 供子图感知节点使用。store 与 context 会被子图继承。

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, InMemoryStore, get_store, current_context, json};

let store = Arc::new(InMemoryStore::new());
let graph = StateGraph::<WorkState>::new()
    .add_node("remember", |state: &WorkState| {
        let store = get_store().expect("store attached");
        store.put(&["users".to_string()], "42", json!({ "seen": true }))?;
        let tenant = current_context().and_then(|ctx| ctx.get("tenant").cloned());
        Ok(NodeOutput::new().set("answer", tenant.unwrap_or_default()))
    })
    .set_entry_point("remember")
    .set_finish_point("remember")
    .compile_with(CompileOptions::new().with_store(store))
    .unwrap();

let config = RunConfig::new().with_context(json!({ "tenant": "acme" }));
```

## 可视化

`CompiledGraph::to_mermaid()`（以及 `AsyncCompiledGraph::to_mermaid()`）把编译后的拓扑渲染为
Mermaid `graph TD`：静态边为实线，条件边经过 router 菱形节点并以 path-map 标签标注，错误路由为
`on error` 虚线，节点标签包含重试次数与中断钩子。

```rust
let mermaid = graph.to_mermaid();
std::fs::write("graph.mmd", mermaid)?;
```

渲染结果可粘贴到 GitHub、GitLab、Notion 或任意 Mermaid 渲染器。Python 绑定同样提供
`compiled.to_mermaid()`。

`compile_with(CompileOptions::new().with_name("demo"))` 可为编译图命名，`graph_spec()` 返回
JSON 描述（节点、静态/条件边）。Python 对应 `compile(name=...)` 与 `compiled.get_graph()`
（`draw_mermaid()` / `to_dict()`）。

## Tokio 异步

启用 `async` feature 获得异步运行时适配器。节点是异步闭包，接收按值传递的状态与 `AsyncContext`；
superstep 通过 `futures::future::join_all` 并发等待。

```rust
use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{JsonState, NodeOutput};

async fn example() -> langgraph_rs::Result<()> {
    let graph = AsyncStateGraph::<JsonState>::new()
        .add_node("call", |_state: JsonState, ctx| async move {
            // 在这里执行异步 I/O，例如调用 LLM 的 HTTP 请求
            ctx.record_tokens(120, 24);
            Ok(NodeOutput::new().set("answer", "42"))
        })
        .set_entry_point("call")
        .set_finish_point("call")
        .compile()?;

    let outcome = graph.run(&JsonState::default(), &Default::default()).await?;
    assert_eq!(outcome.report.total_tokens(), 144);
    Ok(())
}
```

async 适配器覆盖与同步路径相同的 checkpoint、HITL、重试、错误路由与流式能力（节点缓存与子图仍仅
同步路径支持）：checkpoint 与线程历史、HITL（`ctx.interrupt(...)` + `graph.resume(thread, value)`）、`interrupt_before` / `interrupt_after`、每节点 `RetryPolicy`、
`ctx.last_error()` 错误路由、goto/`Send`、流式与 usage/trace 上报。checkpoint 通过
`.with_checkpointer(Arc::new(InMemorySaver::new()))` 启用，并需要 `RunConfig::with_thread(...)`：

```rust
use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{InMemorySaver, RunConfig, json};
use std::sync::Arc;

async fn example() -> langgraph_rs::Result<()> {
    let graph = AsyncStateGraph::<JsonState>::new()
        .add_node("approve", |_state: JsonState, ctx| async move {
            let answer = ctx.interrupt(json!({ "question": "approve?" }))?;
            Ok(langgraph_rs::NodeOutput::new().set("approved", answer))
        })
        .set_entry_point("approve")
        .set_finish_point("approve")
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .compile()?;

    let _ = graph
        .run(&JsonState::default(), &RunConfig::new().with_thread("demo"))
        .await
        .unwrap_err(); // 对应 GraphInterrupt：Error::Interrupt

    let state = graph.resume("demo", json!(true)).await?;
    assert_eq!(state.get("approved"), Some(&json!(true)));
    Ok(())
}
```

异步适配器同样支持子图（`AsyncStateGraph::add_subgraph`，沿用子 checkpoint 命名空间与嵌套
interrupt/resume 语义）、节点级缓存（`with_cache` / `set_cache`）、长期记忆 store 与运行时上下文
（`with_store`、`RunConfig::with_context`、`ctx.store()`）以及节点级 `TimeoutPolicy`。

## Checkpoint 与 Human-in-the-loop

通过 `CompileOptions` 挂载 saver，并让每次运行绑定 `thread_id`。运行开始、每个 superstep 与每次
暂停都会写 checkpoint。

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, InMemorySaver, NodeOutput, RunConfig, StateGraph, interrupt, json};

let saver = Arc::new(InMemorySaver::new());
let graph = StateGraph::<Pipeline>::new()
    .add_node("ask", |state: &Pipeline| {
        let answer = interrupt(json!({ "question": "how much to add?" }))?;
        Ok(NodeOutput::new()
            .set("value", state.value + answer.as_i64().unwrap_or_default())
            .set("log", vec!["ask".to_string()]))
    })
    .add_node("finish", |_s: &Pipeline| Ok(NodeOutput::new().set("log", vec!["finish".to_string()])))
    .set_entry_point("ask")
    .add_edge("ask", "finish")
    .set_finish_point("finish")
    .compile_with(CompileOptions::new().with_checkpointer(saver))
    .unwrap();

let paused = graph.invoke(&Pipeline::default(), &RunConfig::new().with_thread("demo")).unwrap_err();
assert!(paused.is_interrupt());

// `ask` 节点重新执行，这次 `interrupt()` 返回 41 而不是暂停。
let output = graph.resume("demo", json!(41)).unwrap();
assert_eq!(output.value, 41);
```

`get_state(thread)` 报告 `interrupted` 与待执行的 `next` 任务。`interrupt_with_schema(payload,
Some(schema))`（Python `lg.interrupt(payload, response_schema=...)`）会在 interrupt 上保存 JSON
schema 并对 resume 值做校验（`type`/`enum`/`required`/`properties`/`items`），schema 在快照中可见。
`resume_with_updates` 在继续前应用
人工编辑；`resume_command(thread, Command::resume(v).with_update([...]))` 是同一意图的显式 builder
形式；`resume_from` 从旧 checkpoint 重放；`get_state_history` 返回按父链组织、最新在前的历史。
`export_checkpoint(thread)` 以 LangGraph checkpoint tuple JSON 导出最新 checkpoint，
`import_checkpoint(thread, doc)` 可导入该文档（JSON 级转换，非 LangGraph msgpack 数据库格式）。
粗粒度闸门可用 `.interrupt_before(["finish"])` 或 `.interrupt_after(["ask"])`，这类暂停携带
`null` payload，并通过 `graph.continue_thread("demo")` 继续。

### 持久化 Checkpointer

`InMemorySaver` 适合测试；生产环境可换成持久化 `Checkpointer`。所有适配器存储相同的 `Checkpoint`
payload（状态值、待执行任务、中断标志、父链）并实现 `latest` / `history` / `threads`。

| 适配器 | Feature | 连接方式 |
| --- | --- | --- |
| `SqliteSaver` | `sqlite`（需要 `libsqlite3`） | `SqliteSaver::open("checkpoints.db")` / `SqliteSaver::in_memory()` |
| `RedisSaver` | `redis` | `RedisSaver::connect("redis://127.0.0.1/")` |
| `PostgresSaver` | `postgres` | `PostgresSaver::connect("postgres://user:pass@host/db")` |

```rust
use std::sync::Arc;
use langgraph_rs::{CompileOptions, SqliteSaver}; // feature = "sqlite"

let saver = Arc::new(SqliteSaver::open("checkpoints.db")?);
let graph = builder
    .compile_with(CompileOptions::new().with_checkpointer(saver))?;
```

SQLite 适配器在 CI 中运行测试；Redis/Postgres 测试通过 `LANGGRAPH_RS_REDIS_URL` /
`LANGGRAPH_RS_POSTGRES_URL` 选择性开启，未设置时跳过。Debian/Ubuntu 上启用 `sqlite` feature 前
请安装 `libsqlite3-dev`。

运行可绑定 checkpoint 命名空间：`RunConfig::with_checkpoint_ns(...)`（查询使用
`get_state_in` / `get_state_history_in` / `resume_in` / `update_state_in`）；已存储的线程可用
`CompiledGraph::delete_thread` 与 `copy_thread` 管理（Python 同样暴露）。

同一 superstep 内的多个待处理中断会一起出现在 `get_state(...).interrupts` 中（每个带唯一
`id`）；恢复时传入 `{interrupt_id: value}` 映射：`resume(thread_id, json!({ id: value, ... }))`。
单中断仍可接受任意 payload（包括 JSON object）。

## 流式

`stream` 收集全部 chunk；`stream_with` 在调用线程把 chunk 推给 sink，因此无需异步运行时即可转发
到 channel、WebSocket 或日志。chunk 包括 `Values`、节点级 `Updates`、`Custom` 事件、`Message`
（由 `message()` 发出）、`Interrupt` payload、`Checkpoint` id、`Task` 生命周期记录与 `Debug`
payload。`stream_modes(&input, &config, &[StreamMode::Values, ...])` 按 LangGraph 的
`values/updates/custom/messages/debug/tasks/checkpoints` 过滤；Python 绑定通过
`stream(..., stream_mode=...)` 使用相同名称。

设置 `RunConfig::with_stream_subgraphs(true)`（Python `stream(..., subgraphs=True)`）可把子图内
发出的 chunk 包装为 `StreamChunk::Subgraph { namespace, chunk }`，嵌套命名空间用 `|` 连接。

```rust
use langgraph_rs::{NodeOutput, RunConfig, StateGraph, StreamChunk, custom, json};

let graph = StateGraph::<Pipeline>::new()
    .add_node("emit", |_s: &Pipeline| {
        custom(json!({ "progress": 0.5 }))?;
        Ok(NodeOutput::new())
    })
    .set_entry_point("emit")
    .set_finish_point("emit")
    .compile()
    .unwrap();

let chunks = graph.stream(&Pipeline::default(), &RunConfig::new()).unwrap();
assert!(chunks.iter().any(|chunk| matches!(chunk, StreamChunk::Custom { .. })));
```

## 节点级消耗与追踪

每次节点执行都会被记录：墙钟耗时加上节点上报的内容。节点可以分多次上报（`record_tokens`、
`record_usage`）或把 usage 附加到输出；同一次执行的多次记录会求和。

```rust
use langgraph_rs::{NodeOutput, RunConfig, StateGraph, Usage, record_tokens};

let graph = StateGraph::<Pipeline>::new()
    .add_node("llm", |state: &Pipeline| {
        record_tokens(512, 96)?;
        Ok(NodeOutput::new()
            .set("value", state.value + 1)
            .with_usage(Usage::new().with_model("gpt-4o").with_cost_usd(0.006)))
    })
    .set_entry_point("llm")
    .set_finish_point("llm")
    .compile()
    .unwrap();

// `run` 返回最终状态 + 报告；`invoke` 保持仅返回状态的形态。
let outcome = graph.run(&Pipeline::default(), &RunConfig::new()).unwrap();
let report = &outcome.report;
println!("{:?}", report.usage_by_node());   // BTreeMap<node, Usage>
println!("{}", report.total_tokens());      // 跨节点求和
println!("{}", report.total_cost_usd());
for execution in report.executions() {
    println!("{} step={} {}ms", execution.node, execution.step, execution.duration_millis);
}
```

`RunReport` 提供 `executions()`（每个节点一次 `NodeExecution`）、`usage_by_node()`、
`total_usage()`、`total_tokens()` 与 `total_cost_usd()`。`stream` 还会为每个节点发出
`StreamChunk::Usage`，便于 UI 实时展示消耗。

## Python 绑定

启用 `python` feature（PyO3）构建原生扩展模块。使用 [maturin](https://maturin.rs)：

```bash
pip install maturin
maturin develop --features python     # 安装 `langgraph_rs` 到当前 venv
maturin develop --features "python python-async"   # 增加 AsyncStateGraph/ainvoke/aresume
python python/examples/quickstart.py
```

也可以直接构建 cdylib（`cargo build --features python`），把 `liblanggraph_rs.so` /
`langgraph_rs.dll` 放到测试代码旁并命名为 `langgraph_rs.pyd` / `langgraph_rs.so`。Python API 使用
`JsonState` 与 Python 可调用对象：

```python
import langgraph_rs as lg

graph = lg.StateGraph({"count": "last_value", "log": "append"})
graph.add_node("inc", lambda state: {"count": state["count"] + 1, "log": ["inc"]})
graph.add_edge("START", "inc")
graph.add_conditional_edges("inc", lambda state: "inc" if state["count"] < 3 else None)
compiled = graph.compile()
assert compiled.invoke({"count": 0, "log": []})["count"] == 3
```

- `graph.compile(checkpointer=True)` 启用 checkpoint；`invoke(..., thread_id="t")` 在节点调用
  `lg.interrupt(payload)` 时以 `lg.GraphInterrupt` 暂停；`resume("t", answer)` 继续，
  `get_state` / `get_state_history` 检查 checkpoint 链。
- 提供 LangGraph 风格类：`Send`（节点可返回 Send 列表）、节点返回
  `Command(update=..., goto=...)`、`invoke` 传入 `Command(resume=..., update=...)`、
  `RetryPolicy` / `CachePolicy`（通过 `add_node(..., retry_policy=..., cache_policy=...,
  error_handler="node")`）、`compile(cache=True)` 以及 `lg.add_messages`。
- Checkpointer 对象：`compile(checkpointer=lg.InMemorySaver())`；启用 `sqlite` feature 时支持
  `lg.SqliteSaver` / `lg.SqliteSaver.in_memory()`，启用 `redis` / `postgres` feature 时支持
  `lg.RedisSaver(url)` / `lg.PostgresSaver(url)`。
- 异步绑定（`python-async` feature）：`lg.AsyncStateGraph` 与同步 builder 对应（含
  `add_subgraph`），`await compiled.astream(...)` 收集流式 chunk（支持 `stream_mode=` 与
  `subgraphs=`），`abatch(...)` 并发执行多组输入，
  `await compiled.ainvoke(...)` / `await compiled.aresume(...)` 返回由 Tokio 适配器驱动的
  awaitable（`compiled.get_state(...)` 保持同步）。
- Functional API：`task = lg.task(fn, name="...")` 将结果记忆化到运行的 store；
  `lg.entrypoint(fn)` 返回带单个 `entrypoint` 节点的 `StateGraph`，checkpoint、interrupt 与
  store 均可继续使用；resume 后已完成任务不会重复执行。
- 子图：`parent.add_subgraph("child", child_compiled)` 把已编译图嵌入为节点；子图在按节点名
  派生的 checkpoint 命名空间中运行，子图内 `interrupt()` 暂停父图，可用
  `get_state(thread, checkpoint_ns="child")` / `get_state_history(thread, checkpoint_ns="child")`
  查看子图 checkpoint 链（父子共享同一个 saver 对象即可从父对象看到两条链）。同步子图中的
  Python 节点可返回 `lg.Command(..., graph=lg.Command.PARENT)` 跳转父图节点。
- 长期记忆：`store = lg.InMemoryStore()` + `compile(store=store)`；节点通过 `lg.get_store()`
  使用 `get` / `put` / `delete` / `search` / `namespaces`。
- 运行时上下文：`invoke(..., context={...})`（`run` / `stream` 同样支持）；节点读取
  `lg.current_context()`、`lg.current_runtime()`、`lg.current_thread_id()` 与
  `lg.current_namespace()`。
- `compiled.run(...)` 返回 `{"state": ..., "report": ...}`，包含与 Rust 相同的节点消耗报告；
  `compiled.stream(...)` 返回 chunk 字典。
- 节点可返回保留键 `__goto__`（节点名或列表）与 `__usage__`（包含 `model`、`input_tokens`、
  `output_tokens`、`cost_usd`、`metadata` 的字典）。

## 与 Python LangGraph 的基准

仓库提供可复现的对比：相同的图、节点数与迭代次数，所有引擎使用同一协议。场景说明见
[benchmarks/README.md](benchmarks/README.md)（中文：[benchmarks/README.zh-CN.md](benchmarks/README.zh-CN.md)），
当前数据见 [benchmarks/RESULTS.md](benchmarks/RESULTS.md)，可用 `python benchmarks/run.py` 重新生成。

快照（Windows 11，rustc 1.99.1，Python 3.12.10，LangGraph 1.2.22）：

| 场景 | Rust 原生 | Rust Python 绑定 | Python LangGraph | Rust 相对 LangGraph |
| --- | --- | --- | --- | --- |
| linear_10（10 节点流水线） | 50.22 µs/op | 57.98 µs/op | 2,626.96 µs/op | **52.3×** |
| loop_50（50 步循环） | 256.96 µs/op | 287.01 µs/op | 12,841.48 µs/op | **50.0×** |
| fanout_4（1 → 4 并行 → join） | 451.33 µs/op | 460.65 µs/op | 2,400.45 µs/op | **5.3×** |
| checkpoint_5（5 节点 + checkpoint） | 34.82 µs/op | 32.17 µs/op | 3,486.97 µs/op | **100.1×** |
| build_compile（10 节点图） | 13.24 µs/op | 26.60 µs/op | 752.00 µs/op | **56.8×** |

完整报告还包含图规模扩展（`scale_005` … `scale_100`）、8 线程并发（`mt_8x_linear_10`）与峰值
内存（`mem_linear_10`、`mem_loop_50`），见 [benchmarks/RESULTS.md](benchmarks/RESULTS.md)。

这些数字只衡量编排开销（无 LLM 或 I/O 调用）：Rust 原生快一到两个数量级，绑定在每个节点上付出
Python 可调用派发成本，扇出场景差距最小，因为原生 Rust 每个 superstep 会启动真实线程。功能矩阵
（哪些达到对等、哪些为已知差异）见
[benchmarks/features.md](benchmarks/features.md)。

## 状态 / 路线图

已实现并测试：Microkernel + Hexagonal 布局（kernel ports、adapters、插件注册表）；严格或动态
channel schema（last value、topic、append/add、messages、ephemeral、any、untracked、named
barrier、delta、last-value-after-finish、overwrite）；静态、多目标与条件边（含 path map 与条件
入口）；并行 superstep、`Send`（router 也可发起）、`goto`；递归上限（25 superstep）；带
checkpoint 命名空间的 checkpoint、线程历史、类型化快照、带唯一 id 的 interrupt（同一 superstep
多中断、同一节点重复 interrupt）、resume、更新、时间旅行；带退避的 `RetryPolicy` 与
`last_error()` 错误路由；子图（同状态/映射状态、独立 namespace、嵌套中断、流式、同步图
`Command(graph=PARENT)`）；节点级缓存（TTL 与自定义 key）；deferred 节点；节点级超时；节点
metadata、input-schema metadata 与 destination allow-list；batch、`max_concurrency` 与
durability 模式；Mermaid/JSON 图导出；SQLite/Redis/Postgres 持久化 checkpointer；长期记忆
（`BaseStore`/`InMemoryStore`）与运行时上下文；functional API（`entrypoint`/`task`）；全部
stream modes（`values`/`updates`/`custom`/`messages`/`debug`/`tasks`/`checkpoints`，含子图流与
Python v2 stream parts）；节点级耗时/usage 追踪与运行报告、trace-sink 适配器；可选 Tokio 异步
适配器（`async` feature，覆盖完整 checkpoint/HITL/重试/错误路由、子图、缓存与 batch）；可选
Python 绑定（同步与异步，含 store、saver、functional API 与流式）；以及包含扩展性、线程与内存
场景的功能/性能基准。

已知限制（详见 `benchmarks/features.md`）：图定义与 checkpoint 与 LangGraph 不保证字节互操作；
async checkpointer 通过 `BlockingSaver` 把同步 saver 的 I/O 卸载到阻塞池（无原生 async DB 驱动）；functional API 内联执行任务，已完成任务结果持久化在 `checkpoint.task_results`（可选 `BaseStore`）
并用 `BaseStore` 记忆化，而非逐任务 checkpoint；Python 节点可以是普通 callable 或 coroutine；
schema 支持 Python callable reducer。

## 与 LangGraph 的差异

- **同步优先，异步可选。** 默认引擎在 scoped 线程上运行 `Fn(&S) -> Result<NodeOutput>` 节点；
  `async` feature 增加 Tokio 适配器支持异步节点。默认路径没有事件循环。
- **节点上下文。** 同步图中 `interrupt()`、`custom()` 与 `current_node()` 读取每次节点调用安装的
  thread-local；异步图通过随 future 传递的 `AsyncContext` 提供同样的能力。
- **状态必须序列化为 JSON 对象。** channel 值是 `serde_json::Value`；非对象状态以
  `Error::NonObjectState` 失败，从而保持 checkpoint 语言中立。checkpoint 是 JSON 文档，并非
  LangGraph 的 checkpoint 格式。
- **superstep 通过 scoped 线程在同一墙钟窗口执行节点** —— 对共享 `&State` 的真实 OS 并行，而不是
  协作式调度任务；异步适配器则用 Tokio task 运行 superstep。
- **router 是纯函数**，只返回目标（或 path-map 键）。
- **子图**支持同状态或映射状态、子图独立 checkpoint 命名空间、嵌套中断、流式，以及同步图中的
  `Command(graph=PARENT)`（同步与异步图均支持）。
- **Python 绑定使用 `JsonState`。** Python 节点交换普通 dict，状态是 JSON；schema 值可使用内置策略
  （`last_value`、`topic`、`append`、`add`、`messages`）或 Python callable reducer。Python 节点可以是
  普通 callable 或 coroutine，异步图会驱动 coroutine 并在任务上下文安装的情况下运行。引擎执行期间
  释放 GIL，因此
  Python 可调用对象只在节点执行时运行。

## 构建与测试

```bash
cargo test                 # 单元 + 集成测试
cargo test --features async   # async（Tokio）适配器测试
cargo test --features sqlite  # 持久化 checkpointer 测试（需要 libsqlite3）
cargo test --doc           # doctest，包括上面的快速开始
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Python 绑定
cargo build --features python
maturin develop --features python   # 或：pip install maturin
python -m unittest discover -s python/tests -v

# 与 Python LangGraph 的基准（需要 `pip install langgraph`）
python benchmarks/run.py
```

需要 Rust 1.85+（edition 2024）。设计与内部实现：[docs/architecture.md](docs/architecture.md)（中文摘要：
[docs/architecture.zh-CN.md](docs/architecture.zh-CN.md)）。

## License

MIT。本项目是独立重写，与 LangChain 或 LangGraph 项目**无关联**；LangGraph 是其各自所有者的商标。