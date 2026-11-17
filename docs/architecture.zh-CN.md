# 架构（中文摘要）

[English](architecture.md) | 简体中文

本文是 [`architecture.md`](architecture.md) 的中文摘要，覆盖当前实现的结构与关键机制；英文文档
包含更详细的算法步骤与数据模型。

## 模块地图

| 路径 | 职责 |
| --- | --- |
| `src/lib.rs` | facade：模块声明与公开 re-export |
| `src/kernel/` | 纯领域层，不依赖具体基础设施 |
| ├ `kernel/graph/` | `StateGraph` builder、`CompiledGraph`、entry/编译选项、快照、Mermaid 渲染 |
| ├ `kernel/executor/` | 唯一的 superstep 循环：`superstep`（节点执行/重试/缓存）、`scheduling`（边求值与去重）、`checkpointing`（落盘） |
| ├ `kernel/state.rs` | `State` / `JsonState`、`StateSchema`、`ChannelUpdate`、写入合并 |
| ├ `kernel/channel.rs` | `LastValue` / `Topic` / `Reducer` 与内置 reducer |
| ├ `kernel/node.rs` | 节点函数类型、`NodeOutput`、`ConditionalEdge`、`Send`、`Task` |
| ├ `kernel/cache.rs` | `NodeCache` 端口、`CachedOutput`、`CachePolicy`（TTL + 自定义 key） |
| ├ `kernel/checkpoint.rs` | `Checkpoint` 数据结构与 id 生成 |
| ├ `kernel/retry.rs` | `RetryPolicy`（次数、指数退避、上限） |
| ├ `kernel/usage.rs` | `Usage`、`NodeExecution`、`RunReport`、`RunOutcome` |
| ├ `kernel/stream.rs` | `StreamChunk` |
| ├ `kernel/ports.rs` | `Checkpointer`、`TraceSink` 端口 |
| └ `kernel/plugin.rs` | 微内核：`Kernel` / `KernelBuilder` / `Plugin` / `ChannelFactory` |
| `src/adapters/` | 端口适配器 |
| ├ `adapters/memory.rs` | `InMemorySaver`、`InMemoryCache` |
| ├ `adapters/sqlite.rs` / `redis.rs` / `postgres.rs` | 持久化 `Checkpointer`（feature 开关） |
| ├ `adapters/python/` | PyO3 绑定（`python` feature） |
| └ `adapters/tokio_runtime/` | Tokio 异步适配器（`async` feature），拆分为 builder / compiled / context |

依赖方向严格为 `adapters → kernel`；内核只认识端口，不引用任何适配器。

## Superstep 算法

1. 由入口（静态或条件入口）生成首批 `Task`。
2. 循环：检查递归上限 → `interrupt_before` 钩子 → 并发执行本步任务（`std::thread::scope`，
   单任务时内联）→ 若出现中断则保存 pre-superstep checkpoint 并返回 `Error::Interrupt`。
3. 处理结果：成功任务按任务顺序确定性合并写入；`Send` 生成下一批任务；`goto` 覆盖静态出边；
   子图的直接替换写入在 channel 合并之后应用。
4. 求值边（静态边 + 条件边，条件边对合并后的状态求值），去重后生成下一批任务。
5. 保存 checkpoint（含待执行任务、中断标志、`paused_before`），处理 `interrupt_after`，继续或结束。

恢复时从 checkpoint 读取状态与待执行任务，重新执行整个 superstep；`paused_before` 保证
`interrupt_before` 钩子只触发一次。

## 重试与错误路由

- 每个节点可配置 `RetryPolicy`；`superstep::run_task` 在失败后按指数退避重试，中断永不被重试。
- 重试耗尽后产生 `NodeFailure { node, message, attempts }`；若配置了 `add_error_edge`，handler 任务
  在下一个 superstep 执行，并通过 thread-local `last_error()` 读取失败信息。
- 失败与重试次数写入 `NodeExecution.attempts` / `.error`；handler 任务随 checkpoint 持久化。

## 子图

`add_subgraph` 包装一个已编译图：以父状态运行子图，比较前后序列化状态，仅对发生变化的 channel
发出 `NodeOutput::replace`（绕过 reducer，避免重复追加）。同类型子图共享父状态类型；mapped 子图支持不同状态类型、独立 checkpoint namespace 与嵌套中断。

## 节点级缓存

`StateGraph::with_cache` 挂载 `NodeCache` 后端，`set_cache` 为节点配置 `CachePolicy`。执行器以
「节点名 + 输入 channel 序列化」（或自定义 key 函数）查找；命中时通过 `NodeOutput::from_cached`
回放，不执行节点，`NodeExecution.cached = true`、`attempts = 0`，usage 保留。中断与失败不缓存；
适配器错误作为节点失败上报。

## 持久化 Checkpointer

`Checkpointer` 端口只有 `put` / `get` / `latest` / `history` / `threads`。三个适配器共用同一策略：
把整个 `Checkpoint` 序列化为 JSON，按 `(thread_id, id)` 存储，并为每个 thread 维护按
`step, created_at` 排序的索引。适配器是同步的，内部用 `Mutex` 保护连接。

## 可视化

`kernel::graph::visualize::render_mermaid` 接收布局视图（节点名、边表、入口、重试、错误边、中断
钩子），同步与异步编译图都通过它输出 Mermaid `graph TD`：静态边实线、条件边经过 router 菱形并带
path-map 标签、错误边标注 `on error`、节点标签包含重试次数与暂停钩子。

## 异步适配器（Tokio）

- `AsyncStateGraph`：异步节点 `Fn(S, AsyncContext) -> impl Future<Output = Result<NodeOutput>>`
  （future 需 `Send`），结构校验与同步 builder 一致。
- 一个 superstep 的所有任务用 `futures::future::join_all` 并发等待；结果按任务顺序合并，复用
  `collect_next_nodes` 与 `apply_updates`，保证与同步引擎语义一致。
- `AsyncContext` 提供 `interrupt` / `record_usage` / `record_tokens` / `custom` / `last_error`；
  future 交错执行，因此不使用 thread-local 上下文。
- 与同步共享同一 Pregel 主循环，具备 checkpoint/HITL/重试/错误路由/流式/usage/节点缓存/子图/Python 暴露能力；AsyncBackend 仅负责 future 调度与异步持久化。

## 已知限制

- 同步引擎的 `interrupt()` / `custom()` / `current_node()` 依赖 thread-local，只能在执行线程内使用。
- mapped-state 子图目前只暴露在同步 builder 上。
- SQLite/Redis/Postgres 同步 saver 写入时阻塞；异步图可使用 AsyncPostgresSaver / AsyncRedisSaver；Redis 条目默认无 TTL。
- 状态必须序列化为 JSON 对象；Python 绑定使用 `JsonState` 与内置 reducer。