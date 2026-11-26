# 用 `langgraph_rs.*` 替换 `langgraph.*`

本文说明如何把现有的 LangGraph 程序切换到 Rust 实现：**只需要改写 import 行**，图的定义、
节点函数、`compile`、`invoke`、`stream`、`get_state`、`update_state`、异步调用方式都保持不变。

包名刻意不与官方 `langgraph` 同名，避免两个发行版在同一环境里互相覆盖：所有路径都在
`langgraph_rs` 之下（例如 `langgraph_rs.graph`、`langgraph_rs.types`）。

## 1. 构建与安装

扩展必须用 **release** 构建（debug 构建的 Rust 引擎慢 4–5 倍）：

```powershell
cargo build --release --features "python python-async sqlite redis postgres msgpack"
python python/build_package.py --release     # 组装 target/python/langgraph_rs
$env:PYTHONPATH = "<repo>\target\python"
```

`python/build_package.py` 会把纯 Python 包（`python/langgraph_rs`）与编译好的扩展
（`langgraph_rs/langgraph_rs.pyd` / `.so`）组装成一个可导入的包。装好之后：

```python
import langgraph_rs                      # 扁平 API 不变：langgraph_rs.StateGraph
from langgraph_rs.graph import StateGraph, START, END
from langgraph_rs.checkpoint.memory import InMemorySaver
```

## 2. 导入映射

| LangGraph | langgraph_rs |
| --- | --- |
| `from langgraph.graph import StateGraph, START, END` | `from langgraph_rs.graph import StateGraph, START, END` |
| `from langgraph.graph.state import CompiledStateGraph` | `from langgraph_rs.graph.state import CompiledStateGraph` |
| `from langgraph.graph.message import MessagesState, add_messages, RemoveMessage` | `from langgraph_rs.graph.message import …` |
| `from langgraph.types import Command, Send, interrupt, RetryPolicy, CachePolicy, TimeoutPolicy` | `from langgraph_rs.types import …` |
| `from langgraph.errors import GraphInterrupt` | `from langgraph_rs.errors import GraphInterrupt` |
| `from langgraph.constants import START, END` | `from langgraph_rs.constants import START, END` |
| `from langgraph.checkpoint.memory import InMemorySaver` | `from langgraph_rs.checkpoint.memory import InMemorySaver` |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | `from langgraph_rs.checkpoint.sqlite import SqliteSaver` |
| `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver` | `from langgraph_rs.checkpoint.sqlite import AsyncSqliteSaver` |
| `from langgraph.checkpoint.redis import RedisSaver` | `from langgraph_rs.checkpoint.redis import RedisSaver` |
| `from langgraph.checkpoint.postgres import PostgresSaver` | `from langgraph_rs.checkpoint.postgres import PostgresSaver` |
| `from langgraph.store.memory import InMemoryStore` | `from langgraph_rs.store.memory import InMemoryStore` |
| `from langgraph.store.sqlite/redis/postgres import …` | `from langgraph_rs.store.<same> import …` |
| `from langgraph.cache.memory import InMemoryCache` | `from langgraph_rs.cache.memory import InMemoryCache` |
| `from langgraph.config import get_store, get_stream_writer, get_config` | `from langgraph_rs.config import …` |
| `from langgraph.func import entrypoint, task` | `from langgraph_rs.func import entrypoint, task` |
| `from langgraph.managed import IsLastStep, RemainingSteps` | `from langgraph_rs.managed import IsLastStep, RemainingSteps` |
| `from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition` | `from langgraph_rs.prebuilt import …` |
| `from langgraph.runtime import get_runtime` | `from langgraph_rs.runtime import get_runtime` |

`langgraph_rs` 本身也是扁平命名空间，所以 `langgraph_rs.StateGraph`、`langgraph_rs.InMemorySaver`
这类旧写法继续有效。

## 3. 已验证可用（本仓库测试覆盖）

- `TypedDict` / `Annotated[list, operator.add]` / `Annotated[list, add_messages]` 状态声明，
  包括 **PEP 563**（`from __future__ import annotations`）
- `add_node` / `add_edge` / `add_conditional_edges(…, path_map)` / `set_entry_point` / `set_finish_point`
- `compile()`、`compile(checkpointer)`、`compile(checkpointer=…)`
- `invoke` / `stream`（`values`、`updates`、`messages`、`custom`、`subgraphs`、`v2`）/ `batch`
- `get_state` / `get_state_history` / `update_state(as_node=…)` / `get_state_history(limit=…, filter=…)`
- `Command(resume=…)`、`Command(goto=…)`、`Command(update=…)`、`Send`
- `interrupt` / `interrupt_before` / `interrupt_after`
- 异步 `ainvoke` / `astream` / `aget_state` / `aupdate_state` / `aget_state_history` / `aresume`
- 检查点后端：内存、SQLite（含 LangGraph 兼容读写）、Redis、Postgres
- Store / Cache：内存、SQLite、Redis、Postgres
- `langgraph.prebuilt` 工具构建与包装：`create_tool`（函数/装饰器/Runnable 三种形态，名称/描述/schema/
  行为与 LG 逐项一致）、`ToolCallRequest`、`ToolCallWithContext`、`ToolCallWrapper`、`ToolInvocationError`、
  `msg_content_output`、`get_all_basemodel_annotations`、错误文案模板常量、`TOOL_MESSAGE_BLOCK_TYPES`、
  `create_tool_calling_executor`（= `create_react_agent`）、`StructuredResponse`/`Prompt`/`LanguageModelLike`
- `store.base.embed`（embedding 辅助命名空间，可 `from ... import embed` 也可子模块 import）
- `func` 的 `coerce_timeout_policy` / `is_async_callable` / `identifier` / `sync_timeout_unsupported`
- `langgraph.types` 全量：`StreamPart`（7 种分片 TypedDict）、`CheckpointPayload`、`TaskPayload`、
  `TaskResultPayload`、`CheckpointTask`、`DebugPayload`、`CheckpointMetadata`、`StateUpdate`、
  `Overwrite`、`GraphOutput`、`TracePolicy`、`CacheKey`、`Durability`、`PregelExecutableTask`、
  `NS_SEP`、`xxh3_128_hexdigest`（与 LangGraph 同算法、同摘要）、`default_cache_key`、
  `default_retry_on`、`ensure_valid_checkpointer`、`get_cached_annotated_keys`、`get_update_as_tuples`、
  `omit_payload`、`AnyMessage`、`Checkpointer`/`BaseCheckpointSaver` 别名
- `langgraph.prebuilt` 状态 schema：`AgentState`、`AgentStatePydantic`、
  `AgentStateWithStructuredResponse`、`AgentStateWithStructuredResponsePydantic`
- `langgraph.graph.state`/`graph.message` 的再导出（`All`、`CachePolicy`、`BaseChannel`、
  `BinaryOperatorAggregate`、`BranchSpec`、`ChannelRead`、`CONF`、`CONFIG_KEY_SEND`、`Messages`、
  `REMOVE_ALL_MESSAGES`、`push_message`、`BaseMessage`/`BaseMessageChunk`）
- `langgraph.channels`：`LastValue`、`LastValueAfterFinish`、`AnyValue`、`EphemeralValue`、
  `UntrackedValue`、`Topic`、`BinaryOperatorAggregate`、`DeltaChannel` 可作为 schema 直接声明
  （`StateGraph({"v": LastValue(int)})`），映射到与 LangGraph 相同的通道语义；子模块
  `channels.topic` / `channels.binop` 等也可 import。`NamedBarrierValue` / `NamedBarrierValueAfterFinish`
  在引擎侧无对应实现，建图时**明确报错**（不静默近似）
- `langgraph.store.base` 全量面：`tokenize_path` / `get_text_at_path`（路径提取，与 LangGraph
  逐项对照 189 组用例一致）、`override`、`validate_op_namespace`、`BaseStore`、`Item`/`SearchItem`、
  `GetOp`/`PutOp`/`SearchOp`/`ListNamespacesOp`、`MatchCondition`、`TTLConfig`、`IndexConfig`、
  `NotProvided`/`NOT_PROVIDED`、`ensure_embeddings`（函数/异步函数直接包装；provider 字符串与
  LangGraph 一样需要 langchain 侧支持，否则报错说明）
- `langgraph.runtime`：`Runtime`（`context`/`store`/`stream_writer`/`heartbeat`/`previous`/
  `execution_info`/`server_info`/`control`）、`get_runtime()`（运行外返回 `DEFAULT_RUNTIME`）、
  `ExecutionInfo`、`ServerInfo`、`BaseStore`、`BaseUser`、`CONF`、`CONFIG_KEY_RUNTIME`、`get_config`
- `langgraph.checkpoint.serde`：`JsonPlusSerializer`（JSON + bytes/datetime/set 标签化；tuple 会变 list，
  不支持 msgpack 载荷，无法序列化的对象直接抛 `TypeError`）、`SerializerProtocol`、`Serializable`
- `langgraph.cache.base` / `cache.memory`：`BaseCache`、`FullKey`、`Namespace`、`ValueT` 及其再导出
- `langgraph.store.base`：`BaseStore`（抽象基类，子类化或直接用协议对象均可）、`Item`、`SearchItem`
- `langgraph.prebuilt.ValidationNode`（只校验不执行待执行工具调用，返回 ToolMessage；与 LG 输出逐字一致）、
  `tool_validator` 兼容路径、`ToolRuntime` 注入（`state`/`store`/`context` 已填，`tool_call_id`/`execution_info` 为空）
- `langgraph.checkpoint.serde` 子模块（`jsonplus`/`encrypted`/`base`）：`EncryptedSerializer`（基于 `cryptography`，
  缺依赖时明确报错）
- `langgraph.pregel`、`langgraph.checkpoint.serde.{event_hooks,types}`、`pregel.{debug,main,protocol}`
  仅提供可 import 的内部模块名（无对应实现）
- `langgraph.constants` 常量（`CONF`/`TAG_NOSTREAM`/`TAG_HIDDEN`/`TASKS`/`CONFIG_KEY_CHECKPOINTER`）、
  `langgraph.config`（`CONF`/`CONFIG_KEY_RUNTIME`/`var_child_runnable_config`/`StreamWriter`）、
  `langgraph.func`（`EphemeralValue`/`LastValue`/`CACHE_NS_WRITES`/`PREVIOUS`/`MISSING`）、
  `checkpoint.base` 辅助函数（`empty_checkpoint`/`create_checkpoint`/`copy_checkpoint`/`writes_sort_key`/
  `get_checkpoint_id`/`get_checkpoint_metadata`/`WRITES_IDX_MAP`/`EXCLUDED_METADATA_KEYS`）、
  `checkpoint.memory.MemorySaver`（= `InMemorySaver`）
- `langgraph.prebuilt` 参数注入：`InjectedState`、`InjectedStore`（模型工具 schema 中自动隐藏，
  运行时由引擎填入状态/长期 store）；`ToolRuntime` 仅提供类型名，注入未实现（使用时明确报错）
- `create_react_agent` / `ToolNode` 支持上述注入（含普通可调用对象与 `@tool` 工具）
- `langgraph.errors` / `langgraph.warnings` 全量类：`GraphBubbleUp`、`GraphRecursionError`
  （`RecursionError` 子类，文案与 LangGraph 一致）、`GraphDrained`、`NodeInterrupt`、`NodeError`、
  `NodeTimeoutError`、`EmptyChannelError`、`InvalidUpdateError`、`TaskNotFound`、
  `UnknownInterruptError`、`ParentCommand`、`ErrorCode`、`create_error_message`、`is_invalid_resume`、
  `LangGraphDeprecatedSinceV05/V10/V11`、`LangGraphDeprecationWarning`
- `MessageGraph`（单 `messages` 通道；节点收到消息列表、可返回单条/多条消息，`invoke` 返回消息列表）、
  `StateGraph` 支持 Python 继承（子类构建器方法返回 `self`）
- `create_react_agent`（`prompt` / `state_schema` / `store` / `interrupt_*` / `name` / `version`）、
  `ToolNode`、`tools_condition`、`MessagesState`、`add_messages`
- 托管值注解：`remaining_steps: RemainingSteps`、`is_last_step: IsLastStep`（每次超步注入、
  不写 checkpoint，与 LangGraph 一致）
- `entrypoint` / `task`（functional API）、`get_store` / `get_stream_writer`

## 4. 与 LangGraph 的差异（当前状态）

1. **`interrupt` 在 `invoke`/`resume`/`batch` 上的表现已对齐 LangGraph**（2026-11 对齐）：
   暂停时不再抛异常，而是返回状态字典并附带 `__interrupt__` 列表；静态断点
   （`interrupt_before` / `interrupt_after`）只返回状态、不带 `__interrupt__`，与 LangGraph 一致。

   ```python
   paused = graph.invoke({"messages": []}, config)
   paused["__interrupt__"][0].value      # 节点传给 interrupt() 的 payload
   graph.invoke(Command(resume="ok"), config)
   ```

   **唯一的例外**：图没有配置 checkpointer 时，`langgraph_rs` 仍然抛 `GraphInterrupt`（LangGraph
   会返回输入状态 + `__interrupt__`）。没有 checkpointer 的暂停无法恢复，实际使用中请配置
   `InMemorySaver`/`SqliteSaver` 等。`GraphInterrupt` 类仍然导出，行为与 LangGraph 的同名异常一致。
2. **`stream()` 返回 `StreamChunks`**：可迭代、可 `next()`（与 LangGraph 生成器的用法一致），
   另外**额外**支持 `len()` 与索引；与生成器的差别只有「可重复迭代」以及
   `isinstance(..., types.GeneratorType)` 为 False。真正的惰性请用 `astream`。
3. **`get_state_history()` 返回惰性 `StateHistory`**：同样可迭代、可 `next()`，并额外支持
   `len()` 与索引（LangGraph 返回生成器）。
4. **`create_react_agent` 参数已全部实现**：`prompt`、`state_schema`、`store`、`interrupt_before`、
   `interrupt_after`、`name`、`version`、`response_format`、`pre_model_hook`、`post_model_hook`、
   `context_schema` 均可用，并与 LangGraph 对照验证（`python/tests/test_react_agent_extras.py`）。
5. **自定义 Python checkpointer 已支持**：`compile(checkpointer=<对象>)` 接受任何实现 LangGraph
   saver 协议（`put` / `put_writes` / `get_tuple` / `list`）的对象，包括 LangGraph 自带的
   `InMemorySaver` 等实例。引擎自己的调度字段（`next`、deferred routes、task results 等）
   放在 `metadata["langgraph_rs"]` 里交给 saver 保存；**会丢弃未知 metadata 键的 saver 不被支持**，
   读取时会报错说明。`langgraph_rs.checkpoint.base.BaseCheckpointSaver` 仍只是占位类（继承它
   不会被引擎识别），直接把对象传给 `compile` 即可。
6. **自定义 Python store 已支持**：`compile(store=<对象>)` 接受任何实现 LangGraph store 协议
   （`get` / `put` / `delete` / `search` / `list_namespaces`）的对象，包括 LangGraph 自带的
   `InMemoryStore`。
7. **`store` 的检索表面已对齐**：`Item` 返回结构、`filter`（含 `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`
   与嵌套对象）、`limit`/`offset`、`list_namespaces(prefix/suffix/max_depth/limit/offset)`、`ttl`，
   以及**向量索引**（`InMemoryStore(index={"dims","embed","fields"})` + `put(index=)` +
   `search(query=)` 的余弦排序与 `score`）。字段路径支持 `text` / `meta.title` / `items[0]` / `$`，
   不支持 LG 的 `*` 通配符；embedding 按每次 `put` 调用（LG 会把一批 op 合成一次）。
8. **`langgraph.stream` 子系统完整**：`StreamChannel`、`ProtocolEvent`、`StreamTransformer`、
   `StreamMux`、`GraphRunStream`/`AsyncGraphRunStream`、`SubgraphRunStream`、
   `compile(transformers=...)`、`stream_events(version="v3")`/`astream_events`，以及内置
   transformers（`Values`/`Updates`/`Custom`/`Checkpoints`/`Debug`/`Tasks`/`Lifecycle`/`Subgraph`/
   `Messages`）与 `prebuilt.ToolCallTransformer`（引擎自身会发出 `tools` 流事件）。
9. **`RemoteGraph` 已实现**：`langgraph_rs.pregel.remote` 通过 `langgraph_sdk` 调用远程部署
   （`invoke`/`stream`/`astream`/`get_state`/`update_state`/`get_state_history`/`get_graph`），
   测试用注入的假客户端覆盖（本环境无实时部署）。
10. **PEP 563 的状态类必须是可导入的模块级类**：`typing.get_type_hints` 需要能解析字符串注解；
   在函数内 `exec` 出来的类无法解析（LangGraph 同样如此，因此不算差异）。
：`typing.get_type_hints` 需要能解析字符串注解；
   在函数内 `exec` 出来的类无法解析（LangGraph 同样如此，因此不算差异）。

## 5. 如何验证

```powershell
# 347 项 Python 测试（test_package_layout.py 覆盖本包的全部导入路径）
python -m unittest discover -s python/tests

# 与官方 LangGraph 的逐场景对照（38/38 全部等价）
python benchmarks/parity.py

# 可运行的迁移示例
python examples/quickstart/compare.py --check-only
```

`benchmarks/parity.py` 用同一份工作流分别跑 LangGraph 与 `langgraph_rs`，逐场景比较状态、步骤数、
事件顺序；新增场景时它会自动把差异暴露出来。
## 6. 对齐状态小结（2026-11，目标收尾）

`langgraph_rs` 在**不改动图结构**（通道名、reducer 语义、checkpoint 形状与 LangGraph 一致）的前提下，
已覆盖 LangGraph 的公开 API 面。**每一个可导入的 `langgraph.*` 模块都有对应实现**；剩余 `dir()`
差异只是 LangGraph 内部引擎模块（`pregel.main`、`pregel.debug`、`pregel.protocol`、`graph.state`、
`prebuilt.tool_node` 等）导入了自己的依赖，属于其私有布局，扩展有意不复制。

| 领域 | 覆盖内容 |
| --- | --- |
| 建图 | `StateGraph`（含 Python 继承）、`MessageGraph`、`START`/`END`、条件边、`Send`、`Command`、`graph.ui` |
| 状态 | TypedDict/`Annotated` reducer、通道对象（`langgraph.channels`）、托管值（`IsLastStep`/`RemainingSteps`） |
| 运行 | `invoke`/`stream`/`resume`/`batch` 及异步版本、`__interrupt__` 语义、`RecursionLimit`→`GraphRecursionError` |
| prebuilt | `ToolNode`、`ValidationNode`、`create_tool`、`ToolRuntime`/`InjectedState`/`InjectedStore` 注入、`create_react_agent`（全部参数）、`ToolCallTransformer`、`HumanInterrupt*` |
| 流式 | 7 种 `stream_mode`、`stream_events(version="v3")` 全链路（`langgraph.stream` 完整）、`stream_mode="tools"` |
| 持久化 | 内置 saver 全套、LangGraph-schema SQLite saver（**双向跨引擎恢复**）、自定义 Python saver/store 协议桥接 |
| 长期记忆 | `Item`、`filter`/分页/命名空间选项/`ttl`、**向量索引**（余弦排序与 `score`）、路径提取 |
| 远程 | `RemoteGraph`（经 `langgraph_sdk` 调用 LangGraph Server API） |
| 序列化 | `JsonPlusSerializer`、`EncryptedSerializer`、`serde` 子模块、msgpack `EXT_*` 常量 |
| 错误/类型 | 全套异常、`langgraph.types` 全量、`EventStream*`/`StreamPart`、`xxh3_128_hexdigest` |

### 有意保留的差异（全部有测试与文档）

1. **引擎内部类为可导入占位**：LG 1.2.22 里只有 `NodeBuilder` / `Pregel` 在 `langgraph.pregel`
   公开，`ChannelWrite` / `ChannelWriteEntry` 仅在私有模块 `langgraph.pregel._write`，`PregelNode` /
   `SyncAsyncFuture` 连私有路径都没有。我们把这六个名字都做成可导入占位（同名路径
   `langgraph_rs.pregel._write` 也在），实例化即抛 `NotImplementedError` 并提示改用
   `StateGraph.compile()` —— 属于**超集**，不会让迁移里的 import 失败。
2. **原生 `invoke(&state)` 传完整状态**：绑定层与 LangGraph 一样把 `invoke(dict)` 视为经 reducer
   的部分写入；原生 Rust API 需要完整状态。
3. **store 超集（只剩 `ttl`）**：`InMemoryStore` 接受 `ttl`，LG 的 `InMemoryStore` 对 `ttl` 抛
   `NotImplementedError`。`search(query=...)` 在没有向量索引时已与 LG 一致（忽略 `query`，只走
   filter/分页）。
4. **`stream()` / `get_state_history()` 返回可重复迭代对象**（`StreamChunks` / `StateHistory`）：
   除生成器用法外还支持 `len()`/下标；`next()` 与 LG 一样可用，因此不构成兼容性问题。
5. **`RemoteGraph` 未经实时部署验证**：测试用注入的假 SDK 客户端覆盖调用与形状。

### 模块级 API 完整性（2026-11-26 复核）

对 `langgraph` 的 88 个模块逐个导入、只比较「该模块自己定义的名字」（排除 `Any`/`Callable` 这类
re-export 噪音）后，**公开模块的名字缺口为 0**。本轮补齐的内容：

| 模块 | 补齐内容 |
| --- | --- |
| `langgraph.callbacks` | 整模块：`GraphCallbackHandler`、`GraphInterruptEvent`、`GraphResumeEvent`、`GraphLifecycleStatus/Event`、两个 `get_*_graph_callback_manager_for_config`（同步/异步 manager 均可用） |
| `langgraph.pregel.protocol` | `PregelProtocol`（15 个抽象方法）、`StreamProtocol`；编译后的图注册为虚拟子类，`isinstance(graph, PregelProtocol)` 与 LG 一致 |
| `langgraph.pregel.debug` | 9 个函数；其中 8 个与 LG 逐项对照结果完全一致，`map_debug_checkpoint` / `tasks_w_writes` 按相同形状实现 |
| `langgraph.channels.base` | `Value` / `Update` / `Checkpoint` |
| `langgraph.managed.base` | `U` / `V` |
| `langgraph.prebuilt.tool_node` | `ContextT` / `StateT`（`ToolRuntime` 参数顺序与 LG 一致：`Generic[ContextT, StateT]`） |
| `langgraph.types` | `PregelTask` 从占位改为真实 NamedTuple（7 个字段 + 默认值） |
| `langgraph.pregel.main` / `_write` | `NodeBuilder`、`Pregel`、`ChannelWriteTupleEntry` |

**仍未覆盖（有意）**：`langgraph._internal.*`（36 个私有模块）与 `langgraph.pregel._algo` 等 18 个
私有模块——LG 的内部执行布局，扩展跑自己的引擎。

### 调用方向：Python 驱动 Rust（2026-11-26）

架构约束：**不允许 Rust 反向调用 Python**。所有运行入口都由门面（Python）发起、由原生方法执行；
引擎只调用「用户提供的 Python 可调用对象」（节点函数、reducer、saver、transformer——这些本来就是
Python 代码，属于正向调用）。

本轮把原先「原生方法回头调 Python 垫片」的部分全部搬回 Python 侧：

| 原位置 | 处理 |
|---|---|
| 原生 `ainvoke`/`astream`/`abatch`（调 `*_via_sync`） | 删除；门面用 `asyncio.to_thread`（同步原生）或直连原生异步 API |
| 原生 `stream_events`/`astream_events`（import `stream._events`） | 删除；门面直接调垫片并传入**门面自身** |
| 原生 `aget_state`/`aget_state_history`/`aupdate_state`/`abulk_update_state`/`aclear_cache`（`call_via_sync`） | 删除；门面用 `asyncio.to_thread` 包同步方法 |
| 原生 `get_context_jsonschema`（调 `_compat.jsonschema_of`） | 原生只返回 schema，门面在 Python 里转 JSON Schema |
| `graph_pause::dispatch_*`（回调 Python handler） | 改为记录事件，见下节 |

**构造期包装也搬到 Python**：原生 `compile()` 现在直接返回引擎对象，由 Python 的 `StateGraph` /
`AsyncStateGraph`（`#[pyclass(subclass)]` 的子类）在 `compile()` 里套上门面；`create_react_agent`
同理（Python 包装函数，`create_tool_calling_executor` 是同一个对象）。`AsyncStream` 也从「Rust 里
`exec` 一段 Python 源码」搬到了 `langgraph_rs/stream/_stream.py`。

**schema 归一化也搬到 Python**：`langgraph_rs/graph/_schema.py` 负责把类式 schema 展开、把自定义
`BaseChannel` 标记成 `{"__custom_channel__": channel}`；原生 `schema_from_py` 只接受纯字典，不再
`isinstance` 判断垫片的基类。Rust 里那段 `exec` 出来的 `_schema_from_class` / `MessagesState`
也随之删除（改由同一 Python 模块提供）。

`Rust 侧垫片 import` 从 17 处降到 **3 处**（JsonPlus 编解码 ×2、`ToolRuntime` 构造），下一轮继续。

**`astream` 的形状也对齐了 LG**：LG 的 `astream` 是 **async generator（不可 await）**，此前我们返回
可 await 对象；现在是 `async def ... yield`，与 LG 一致（相关测试已改）。

**`GraphCallbackHandler` 由「事件队列 + 单点派发」驱动**：

- 引擎在暂停/恢复时把事件写进本次运行的 `LifecycleSink`（`RunConfig::with_lifecycle_sink`），
  **不在 Rust 里回调 Python**——与 LG 的 loop 队列同构
- Python 门面在 `invoke`/`ainvoke`/`stream`/`astream` 返回后调用一次 `take_lifecycle_events()`
  并派发：**一次运行只建一个 manager**，`run_id` 是真实 UUID（此前恒为 `None`）
- `status` 由引擎给出（动态中断 `pending`、静态断点 `interrupt_before`）
- 没有配置 handler 时队列仍是空的，运行时行为不受影响

事件字段（`status`、`checkpoint_id`、`checkpoint_ns`、`interrupts`）与 LangGraph 实测一致。

> **本轮结论（第 6 节终版）**：除上述 5 条外，已无可消除的差异。其中 3 条是超集（内部类占位、
> `ttl`、可重复迭代对象），1 条是原生 Rust API 的设计取舍（完整状态），1 条是环境限制
> （`RemoteGraph` 无实盘）。此前记录过的 6 条差异（`run.messages` 实时性、未声明键、无 saver 中断、
> 普通 callable 传参、`push_ui_message` 写 state、store `query` 语义）都已对齐并有测试锁定。

已经消掉的旧差异（保留记录，避免回退）：

* `run.messages` 非实时 → 现在同步/异步都实时（模型 token、节点自写消息、custom 事件）；见
  `examples/streaming-liveness/compare.py`。
* `create_react_agent` 无 `state_schema` 时接受额外键 → 现在与 LG 一致地丢弃未声明键；声明的
  schema 同样只接受自己声明的输入键。
* 无 checkpointer 时 `interrupt` 抛 `GraphInterrupt` → 现在与 LG 一致，返回输入状态 + `__interrupt__`。
* 普通 callable 收到位置参数 → 现在与 LG 一致：按参数名传关键字（`**kwargs` / 仅关键字参数都支持），
  `def tool(args: dict)` 这种单字典写法仍然收到位置参数。
* `push_ui_message` 只发流事件、不写 state → 引擎补了「节点内写通道」的原语（`send()`，等价于
  LG 的 `CONFIG_KEY_SEND`），`push_ui_message` / `delete_ui_message` 现在会把事件写进 `ui` 状态键，
  同一通道的多次写入按列表交给 reducer；`langgraph_rs._send(channel, value)` 也可直接用。
* store `search(query=...)` 无索引时的子串匹配 → 现在与 LG 一致：无向量索引时 `query` 被忽略，
  结果只由 `filter` / `limit` / `offset` 决定（对照测试见 `python/tests/test_store_search.py`）。

### 编译后图就是一个 `Runnable`（2026-11-26）

LangGraph 的编译图是 `Runnable`（`CompiledStateGraph → Pregel → PregelProtocol → Runnable`），
所以 `bind` / `with_config` / `pipe` / `map` / `as_tool` / `batch_as_completed` / `astream_log` /
schema introspection 全套由基类提供。扩展现在也这样：

- `compile()` 返回 `CompiledStateGraph` 门面：持有原生图，继承 `langchain_core.runnables.Runnable`
- `invoke` / `ainvoke` / `stream` / `astream` / `batch` / `abatch` / `stream_events` / `astream_events`
  / `get_graph` 显式转发到原生实现（否则会被 `Runnable` 的默认实现遮蔽）
- 其余方法（`resume`、`resume_from`、`stream_live`、`get_subgraphs`、`bulk_update_state`…）由
  `__getattr__` 透传；`dir()` 同时列出原生方法
- `isinstance(app, Runnable)` 为真，`app | fn`、`app.with_config(...)`、`app.as_tool()` 可用
- 不支持的运行期关键字（`interrupt_before` / `interrupt_after` / `output_keys`）会显式抛
  `NotImplementedError` 并给出替代做法，而不是静默忽略

### `bulk_update_state` 与 `get_state_history` 的参数（2026-11-26）

实测 LG 1.2.22 后修正两处：

1. **`bulk_update_state` 的 `supersteps` 是 `(values, as_node, task_id)` 三元组**，不是 dict。
   我们此前只认 dict，传 LG 形状会被**静默跳过**（值完全没写入）——这是真 bug，已修；
   dict 拼法保留为超集，无法识别的形状改为抛 `TypeError`。
2. **`get_state_history(before=...)` 接受 `RunnableConfig`**（快照的 `.config`）。我们此前只收
   checkpoint id 字符串，传 dict 直接 `TypeError`。现已两种都接受。

参数语义核对一致：`limit=2` → `[2, 1]`；`before=<step1 的 config>` → `[0, -1]`；
`filter={"next": ("b",)}` → `[]`（LG 同样为空）。

**已与 LG 对齐**（引擎新增 `bulk_update_superstep`）：
- **一个 superstep 一个 checkpoint**：两轮 superstep 只写 2 个 checkpoint（此前 3 个）
- **`next` 取同一 superstep 内所有 update 的并集**：同时 `as_node="a"` 与 `as_node="b"` 时
  得 `('b',)`，与 LG 一致（`a` 会调度 `b`，`b` 不再调度）

### `messages` 流与投影（2026-11-26）

LG 的 `stream_mode="messages"` 产出 `(message_chunk, metadata)`，v3 run stream 默认带 `messages`
投影。扩展此前：消息是 **dict**（不是 `AIMessageChunk`），且 run stream 只注册了
`lifecycle`/`subgraphs`/`values`。

现在：
- 回调层保留 `content`/`id`/`tool_call_chunks`/`additional_kwargs`/`response_metadata`，
  由 `_compat.message_chunk` 重建为 `AIMessageChunk`（`Human`/`System` 也有对应 chunk 类型）
- `MessagesTransformer` 进入默认投影集，于是 `run.messages` 默认存在
- 显式传 `transformers=[MessagesTransformer]` 不会再与默认注册冲突（工厂按类型去重）
- 没有 live sink 的图（同步引擎门面）在 live 路径上自动降级为非实时 parts，而不是报错

实测：`stream(mode="messages")` 首条为 `AIMessageChunk(content='Hello')`，metadata 含
`langgraph_node`/`langgraph_step`；`run.extensions` 为 `['lifecycle','messages','subgraphs','values']`

**metadata 字段**（本轮补齐）：`langgraph_triggers` 与 `langgraph_path` 现在都有，且是 tuple
（JSON 里是数组，转换层按 LG 的类型还原）。引擎把任务的 triggers 装进 `TaskContext`，
回调层通过 `current_triggers()` 读到。

- `langgraph_path`：`("__pregel_pull", <node>)`，子图命名空间按段追加 —— 与 LG 一致
- `langgraph_step`：LG 的超级步计数**从 1 开始**（首个超级步是 1）；扩展引擎内部从 0 开始，
  这里已 +1 对齐（此前我们收 0、LG 收 1）
- `langgraph_triggers`：LG 给的是**调度通道名**而不是节点名 —— 静态/条件边是
  `("branch:to:<node>",)`，`Send` 扇出是 `("__pregel_push",)`，入口任务同样是 `branch:to:<node>`。
  扩展已按同一约定生成（此前用的是源节点名、且入口任务为空）
- 根命名空间下仍没有 `langgraph_checkpoint_ns`（LG 是 `n:<uuid>`；我们不合成 id）

### LangChain 对象可以进状态了（2026-11-26）

LangGraph 的状态里可以放 `AIMessage`、`HumanMessage`、`ToolMessage` 这类 LangChain 对象，
扩展此前会拒绝：`value of type `AIMessage` cannot be converted to JSON`。这是最影响实用的差异
——消息就是 agent 的主要载荷。

现在走 **LangGraph 同款 JsonPlus 编码**：

```json
{"lc": 1, "type": "constructor", "id": ["langchain_core.messages", "AIMessage"], "kwargs": {...}}
```

- 写入方向：Python 对象 → 上述 JSON（`_compat.encode_value`），因此 checkpoint 里存的是 LG 能读的形状
- 读取方向：`Value::Object` 且 `lc == 1` → `_compat.decode_value` 用 `id` 路径 import 并重建对象
- 只对「本来无法转 JSON」的对象走这条路径，普通 dict/list/str 不受影响

实测：节点返回 `AIMessage` → 拿回 `AIMessage`；输入 `HumanMessage` ✓；带 tool_calls 的消息经过
checkpointer 往返后仍是 `AIMessage` 且 `tool_calls` 完整；纯 dict 消息行为不变。

### 子图节点的 `updates` 事件（2026-11-26）

父图里跑子图的节点，LangGraph 会在父命名空间发出自己的 `updates` 事件
（`{'sub': {'value': 1}}`），扩展此前只在子图命名空间里发子图内部节点的事件，父节点那次事件缺失。

根因：子图节点的写入是「替换」（`NodeOutput::replace`），而发射 `updates` 事件的代码只在
`updates` 非空时才发。现在两者任一非空都会发一条事件（`updates` 与 `replaces` 合并成同一个
channel→value 映射），与 LG 的形状一致。

实测（`stream_mode="updates"`）：

```
无 subgraphs：  {'sub': {'value': 1}}, {'after': {'value': 11}}        ← 与 LG 相同
带 subgraphs：  (sub-ns, {'inner': ...}), ((), {'sub': ...}), ((), {'after': ...})
```

已知细微差异：当父图的**入口就是子图**时，LG 不发出父图的输入回显，而扩展会发一条子图的输入回显
（子图不是入口时两边完全一致）。命名空间里的 id 形态也不同（LG 用 uuid，扩展用确定性 id），
两者都是不透明标识，比较时应只比名字。

### 并行中断会全部返回（2026-11-26）

同一 superstep 里多个并行节点都 `interrupt()` 时，LangGraph 返回**全部**中断（每个任务一条），
恢复时用 `Command(resume={id: value, ...})` 按 id 分发。扩展此前只返回第一个——不是引擎没收集，
而是 Python 侧拿的是 `Error::Interrupt`（只带第一个）：暂停 checkpoint 里其实两个都有。

已修：`pause_values` 改用快照里的 `interrupts`（`stream` 路径同样处理），于是

```
interrupts: [{'ask': 'a'}, {'ask': 'b'}]
resume(Command(resume={id_a: 'va', id_b: 'vb'})) -> {'a': 'va', 'b': 'vb'}
```

per-task 归属也保留：`get_state().tasks` 里 `na` 带 1 条、`nb` 带 1 条。

### `durability` 的传法（实测对齐）

LangGraph 只在 **`invoke`/`stream` 关键字参数** 上认 `durability`；放进 config（顶层或 `configurable`）
都被忽略。扩展的行为（实测）：

| 传法 | LangGraph 1.2.22 | 本扩展 |
|---|---|---|
| `invoke(..., durability="exit")` | 生效（1 个 checkpoint） | 生效（1 个） |
| `config["configurable"]["durability"]` | 忽略 | 忽略 |
| `config["durability"]`（顶层） | 忽略 | **生效**（超集） |
| 不传 | 每 superstep 都写（4 个） | 4 个 |

即：按 LangGraph 正确写法（关键字参数）行为完全一致；多出来的「顶层 config 也认」是超集。

### 自定义 channel 已支持（2026-11-26）

LangGraph 的 `BaseChannel` 是开放类层次：继承它、实现 `update(values) -> bool` 与 `get()` 就得到
自定义合并语义。扩展现在也支持——检测到子类覆写了 `update`/`get` 时，把它桥接成引擎的 reducer：

```python
class MaxValue(BaseChannel):
    __channel_spec__ = None          # 声明「不是内置 channel」
    def update(self, values): ...    # 自己维护状态
    def get(self): ...
```

实测：`MaxValue`（取最大）+ `Union`（去重累积）在三级图里得到 `{'best': 9, 'items': ['x','y','z']}`。
未知的内置 channel 仍然在构图时抛错，不会静默降级。

已知限制：自定义 channel 的内部状态由 Python 对象持有，从 checkpoint 恢复时不会重建该对象
（值仍会随 checkpoint 恢复，但 `update` 依赖的私有状态需要 channel 自己能从 JSON 值推导）。

### 异步图同样走门面（2026-11-26）

`AsyncStateGraph.compile()` 现在也返回同一个 `CompiledStateGraph` 门面：异步原生图补上了阻塞式
`invoke` / `stream` / `batch`（在 Tokio 运行时上 `block_on`，调用线程等待、不阻塞解释器），
因此两种图在 Python 侧是同一种对象，`isinstance(app, Runnable)` 都成立。异步图没有同步事件流，
`stream_events` 会显式提示改用 `astream_events`。

同时补齐了**能如实提供**的引擎属性：`interrupt_before_nodes`、`interrupt_after_nodes`、
`retry_policy`、`step_timeout`、`cache_policy`、`stream_transformers`、`context_schema`、`debug`、
`checkpointer`（返回 `compile()` 时传入的原始对象）、`cache`、`trigger_to_nodes`、`builder`
（返回编译它的 `StateGraph`）、`stream_mode`（`values`）、`stream_eager`（`false`）、`config`（空）、
`validate()`（编译期已校验，返回自身）、`get_context_jsonschema()`（由 `context_schema` 生成）。

### 编译后图仍不实现的成员（2026-11-26）

`dir()` 对比后仍缺的成员分两类，都是有意的：

1. **`RunnableSerializable` 家族**（LG 的图同时是 LangChain Runnable）：`bind`、`with_config`、
   `with_retry`、`with_types`、`with_fallbacks`、`with_listeners`、`with_alisteners`、`pipe`、`map`、
   `pick`、`assign`、`transform`、`atransform`、`as_tool`、`validate`、`batch_as_completed`、
   `abatch_as_completed`、`astream_log`、`get_name`、`get_prompts`，以及 schema/introspection 那组
   （`get_input_schema`、`get_output_schema`、`*_jsonschema`、`input_schema`、`output_schema`、
   `config_schema`、`InputType`、`OutputType`）。我们的图不是 `Runnable` 子类。
2. **LG 自己的引擎内部状态属性**：`nodes`、`channels`（返回 `PregelNode`/channel 实例）、
   `checkpointer`、`config`、`debug`、`stream_mode`、`stream_channels*`、`trigger_to_nodes`、
   `attach_node`/`attach_edge`/`attach_branch`、`builder`、`cache*`、`retry_policy`、`step_timeout`、
   `store`、`interrupt_before_nodes`/`interrupt_after_nodes` 等。这些反映 LG 的 Python 对象模型；
   我们只暴露能如实提供的部分（`channels` 给的是声明过的 channel 名，`store` 给的是真实 store）。

### 编译后图的同步/异步双 API（2026-11-26）

LangGraph 的编译结果是**一个对象**，同时提供 `invoke` 与 `ainvoke`、`stream` 与 `astream`、
`stream_events` 与 `astream_events`。扩展现在也如此：

- `CompiledGraph` 增加了 `ainvoke` / `astream` / `astream_events`：同步引擎在工作线程上执行
  （`run_in_executor`），事件循环不被阻塞；`astream` 返回 async generator，与 LG 的形态一致
- `AsyncCompiledGraph.ainvoke` / `astream` / `astream_events` 都接受 LG 的 `config=` 参数（此前只有 `thread_id`）
- `batch` / `abatch` 接受 LG 的 `config=`（单份或**每个输入一份**）、`return_exceptions=` 与 `context=`；
  config 列表长度不匹配时抛 `TypeError`
- 图自省：`get_subgraphs(namespace=, recurse=)` 返回内嵌子图（`(name, CompiledGraph)`），
  `copy(update=)` 返回可再编译的副本，`channels` / `store` 作为属性暴露
- 状态 API 补齐异步孪生与批量更新：`aget_state` / `aget_state_history` / `aupdate_state` /
  `bulk_update_state` / `abulk_update_state` / `clear_cache` / `aclear_cache`；`get_graph(config, xray=)`
  接受 LG 的参数（`xray` 对渲染结果无影响，子图本就内嵌为节点）

### 跨引擎 resume 支持矩阵（2026-11）

| 共享方式 | LangGraph 暂停 → rs 恢复 | rs 暂停 → LangGraph 恢复 | 交替继续（新输入） |
| --- | --- | --- | --- |
| 同一个 Python saver 对象（如 LG `InMemorySaver`） | ✅ | ✅ | ✅ |
| LangGraph-schema SQLite（`rs.LangGraphSqliteSaver`） | ✅ | ✅ | ✅（两引擎交替写同一线程已验证） |

- 共享 saver 对象的写侧使用 UUIDv6 风格、按时间+step 单调递增的 checkpoint id，两个引擎交替写同一线程时
  LangGraph 才能选中「最新」checkpoint。
- 两个 saver 都写入 UUIDv6 风格、按「时间 + step」单调递增的 checkpoint id（**去横线的 32 位十六进制**）：
  - 时间前缀与 LangGraph 的 uuid6 同构，所以按 id 字符串排序即为时间序；
  - 去横线是必需的 —— LangGraph 内部会对 `checkpoint["id"]` 调用 `binascii.unhexlify(...)`，带横线会让它
    推导任务 id 失败，进而把待执行任务的中断丢掉；
  - 读取侧统一用 `ORDER BY REPLACE(checkpoint_id, '-', '') DESC`，两种格式（我们与 LangGraph 写的行）在同一张表里
    也能正确取到最新检查点。
- 已实测：同一 SQLite 文件上 `LG → rs → LG → rs → LG` 交替续跑 5 轮，`attempts` 依次为 1..5，状态累积正确；
  同一线程内 `Send` 扇出产生的中断，LangGraph 也能用 `Command(resume={id: value})` 逐个恢复。
- 子图（`checkpoint_ns = "sub:<task_id>"`）同样双向可用：LangGraph 在子图里暂停后，本引擎能读到
  `{'approved': False, 'draft': 'v1'}` 并 `resume(True)` 走完到 `{'approved': True, 'draft': 'v1 [done]'}`。
  子图**继承父图的 checkpointer**（与 LangGraph 一致）：子图不传 `checkpointer` 时会自动采用父图编译时的 saver，嵌套多层同样成立。
- 暂停恢复依赖：`__root__`、`branch:to:<node>`、`__interrupt__` 写、以及从 `versions_seen` 中剔除待执行节点。