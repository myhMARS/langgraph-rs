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
- `langgraph.prebuilt` 参数注入：`InjectedState`、`InjectedStore`、`ToolRuntime`（模型工具 schema 中
  自动隐藏，运行时由引擎填入状态/长期 store/运行 config/custom 流写入器）
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

**JsonPlus 编解码改成原生实现**：`encode_object` 只用对象自己的 `model_dump`/`to_json` 加
`type().__module__`/`__qualname__` 组装 `{"lc":1,...}`；`decode_object` 用 `importlib` 还原类并
`cls(**kwargs)`。两边都不再 import 垫片。

**`ToolRuntime` 改成原生 `#[pyclass]`**：引擎构造注入句柄时不再 import 垫片的 dataclass，垫片反向
re-export 原生类（`langgraph_rs.prebuilt.tool_node.ToolRuntime` 仍是同一个对象）。

**结论：Rust 侧对 `langgraph_rs.*` 垫片的 import 现在是 0 处。** 剩余的 Python 交互只有两类必要形式：
调用用户可调用对象（节点/reducer/saver/transformer/工具）、以及构造 Python 对象作为返回值。

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

### 未声明的 key 会被丢弃（2026-11-27）

LG 只保留 state schema 里声明过的通道：输入里多给的 key、节点返回里多写的 key 都**静默丢弃**。扩展此前会
把它们当成隐式通道带进状态：

```
输入 {"value": 0, "typo": 5}   LG: {'value': 1}   扩展改前: {'typo': 5, 'value': 1}
节点返回 {"typo": 5}            LG: 丢弃          扩展改前: 保留
```

现在：只要 schema **声明了通道**，未声明的读写都被忽略（与 LG 相同）；完全动态的 schema（无任何声明通道，
`JsonState` 那类用法）仍保持宽松。

顺带记录：LG **不接受** dict 形式的 schema（`StateGraph({"value": "last_value"})` 直接
`TypeError: unhashable type: 'dict'`），这是扩展的便利超集。

### state 里可以放任意 Python 对象（2026-11-27）

LG 的 state 不限于 JSON：`invoke({"point": Point(1)})`、节点返回 `set`/自定义类都能跑（有无 checkpointer 都一样）。
扩展此前直接拒绝：`TypeError: value of type 'Point' cannot be converted to JSON`。

现在无 JSON 形式的对象按引用暂存，引擎看到占位符，取回时还原成原对象：

```
no saver:    {'point': Point(x=1), 'tag': {'set'}}         ← 与 LG 相同
with saver:  {'point': Point(x=2), 'tag': {'set'}}         ← 状态里读回来也是原对象
```

同一个对象重复传入会复用同一个占位符（按对象地址去重），所以 `is` 与相等性都保持。
可调用对象与模块仍然**明确拒绝**（它们是代码而非数据，静默暂存会掩盖 bug）。

**已知限制**：对象按引用保存于**本进程**，因此跨进程恢复检查点（如落盘 sqlite 后换进程读）只能拿到占位符；
LG 依靠 msgpack 序列化能跨进程。同一进程内（内存 saver、sqlite 同进程）不受影响。

### `context_schema` 与 `context=` 参数（2026-11-27）

LG 允许 `StateGraph(state_schema, context_schema=Ctx)`，运行期 `invoke(..., context=Ctx(user="ann"))`，
节点里 `runtime.context` 拿到的就是**那个对象**。扩展此前两条都不行：

| | LG | 改前 | 现在 |
|---|---|---|---|
| `StateGraph(..., context_schema=Ctx)` | 支持 | `TypeError: unexpected keyword argument 'context_schema'` | 支持（构建期声明，编译时带上） |
| `invoke(context=Ctx(...))` | 节点收到原对象 | `TypeError: value of type Ctx cannot be converted to JSON` | 节点收到原对象 |
| `get_context_jsonschema()` | 输出 schema | 一致 | 一致 |

实现方式：context 无法 JSON 化时，引擎把 **Python 对象按引用暂存**（`{"__langgraph_rs_object__": id}` 占位）
并在取回时还原成原对象；这类按引用的 context 跳过引擎内的 JSON schema 校验（Python 侧已经校验过）。
暂存表上限 4096 条，避免长跑进程无限增长。

### `Runtime` 注入与同步构建器里的异步节点（2026-11-27）

**1. `Runtime` 注入**：LG 允许节点声明 `runtime: Runtime` 参数，运行期注入上下文对象。扩展此前只传
state，节点会直接 `TypeError: node() missing 1 required positional argument: 'runtime'`。现在同步/异步节点都能拿到
`Runtime`（`context` / `store` / `stream_writer` 均可用）。

已知差异：`runtime.store` 是**可用的**存储句柄，但不是编译时那个对象本身（`isinstance(runtime.store,
InMemoryStore)` 为 `False`），因为引擎回传的是自己的包装。

**2. 同步构建器 + 异步节点**：LG 里 `StateGraph` 可以注册 `async def` 节点，用 `ainvoke` 跑。扩展此前
`ainvoke` 会报 `InvalidUpdateError: Expected dict, got <coroutine object ...>`。现在同步节点路径遇到可等待对象会
驱动它到完成（复用异步构建器的私有事件循环）。

超集：LG 的同步 `invoke` 对异步节点会抛 `TypeError: No synchronous function provided to "n"`，扩展在 `invoke` 里
也能跑（更宽容，不会让从 LG 迁来的代码失败）。

### `messages` 流：完整消息不再被包成 chunk（2026-11-27）

LG 的 `stream_mode="messages"` 里，模型**没有逐 token 流式**时会给出**消息本体**（`AIMessage`），
逐 token 时才是 `AIMessageChunk`。扩展此前一律返回 `AIMessageChunk`。

现在按回调事件区分（`on_llm_new_token` → chunk；`on_llm_end` 且未流过 token → 完整消息）：

```
非流式模型：('AIMessage', 'hello')      ← 与 LG 相同
逐 token  ：('AIMessageChunk', 'Hello') / ('AIMessageChunk', ' ') / ...
```

顺带补上 `langgraph_rs.checkpoint.base.RunnableConfig`（LG 从该模块导出，之前缺）。

### `StateGraph` 的辅助方法与 store 索引属性（2026-11-27）

补齐上一轮 `dir()` 对比里剩下的两处（纯门面改动，不涉及引擎）：

| 成员 | 行为（与 LG 一致） |
|---|---|
| `add_sequence(nodes)` | 依次注册节点并串成链（支持 `callable` 或 `(name, callable)`），返回 `self`；入口仍需自己连 |
| `set_node_defaults(retry_policy=…, cache_policy=…, error_handler=…, timeout=…)` | 对**之后**注册的节点生效 |
| `validate(interrupt=None)` | 做一次编译期校验（非法时抛 `ValueError`），成功返回 `self` |
| `InMemoryStore.embeddings` / `index_config` | 暴露构造时的索引配置（无索引时为 `None`） |

实测：`set_node_defaults(retry_policy=RetryPolicy(max_attempts=2))` 后注册的节点确实重试 2 次。

**已对齐**：`StateSnapshot` 现在是 **NamedTuple**（字段与 LG 完全相同：`values`/`next`/`config`/
`metadata`/`created_at`/`parent_config`/`tasks`/`interrupts`），`isinstance(snapshot, tuple)`、索引、
解包都成立；扩展额外保留了字典式取值的便利（`snapshot["values"]`、`.step`、`.thread_id`、
`.interrupted`）。

### `Send` / `Interrupt` / `TimeoutPolicy` 的属性面（2026-11-27）

按类做 `dir()` 对比后补齐的可读属性：

| 类 | 补上 | 说明 |
|---|---|---|
| `Send` | `arg`、`timeout` | `send.arg` 此前读不到；`send.timeout` 现在与 LG 一样是 **`TimeoutPolicy`**（传 0.5 会被规范化） |
| `Interrupt` | `interrupt_id`、`from_ns(value, ns)` | `interrupt_id` 是 `id` 的别名；`from_ns` 给出稳定的命名空间 id |
| `TimeoutPolicy` | — | `repr` 改成 LG 的单引号形式：`TimeoutPolicy(run_timeout=0.5, idle_timeout=None, refresh_on='auto')` |

**仍未对齐（已记录）**：`StateSnapshot` 是 dict 子类（LG 是 NamedTuple）——字段属性可读，但
`dir()` 形状与 `isinstance(snapshot, tuple)` 不同；`InMemoryStore` 缺 `embeddings` / `index_config`
属性；`InMemorySaver` 缺 `put` / `put_writes`（引擎内部写入路径）；`StateGraph` 缺
`add_sequence` / `set_node_defaults` / `validate`。

### 无检查点器时的行为、`Command` 字段、缓存 TTL（2026-11-27）

三处一起修：

**1. 有 `thread_id` 但没编译 checkpointer**：LG 直接**忽略**这个配置照常运行；扩展此前报
`RuntimeError: a run bound to a thread_id requires a checkpointer`。现在与 LG 一致 —— 只有
`Command(resume=...)` 这种真正需要历史状态的情况才报错：
`RuntimeError: Cannot use Command(resume=...) without checkpointer`。
`get_state()` 这种情况下两边都报 `ValueError: No checkpointer set`。

**2. `Command` 的字段**：LG 是 dataclass，`update` / `resume` / `goto` / `graph` 都可读；扩展此前
只能读 `PARENT`，`cmd.resume` 直接 `AttributeError`。现在四个字段都是可读属性。

**3. `CachePolicy(ttl=0.2)`**：LG 的 `ttl` 是**秒（可为小数）**；扩展只接受整数，
`TypeError: 'float' object cannot be interpreted as an integer`。现在按秒接受浮点，TTL 过期语义与 LG
一致（`calls == [1, 1]`）。

### 弃用警告（2026-11-27）

LG 对 `ValidationNode`、`MessageGraph` 发出 `LangGraphDeprecatedSinceV10` 警告（这正是让
`warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)` 生效的关键）。扩展此前不警告。

现在两者逐字一致：

```
ValidationNode  → ValidationNode is deprecated. Please use `create_agent` from `langchain.agents`
                  with custom tool error handling. Deprecated in LangGraph V1.0 to be removed in V2.0.
MessageGraph    → MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0.
                  Please use StateGraph with a `messages` key instead. ...
```

另外记录：LG 1.2 已移除 `create_tool_calling_executor`（`ImportError`），扩展仍保留该别名（超集）。

### 图构建错误：`ValueError` + LG 文案（2026-11-27）

LG 把所有图构建期的问题都报成 `ValueError`；扩展此前一律 `RuntimeError: graph validation failed: ...`，
`except ValueError` 完全接不住。现在两者一致：

| 情况 | 异常与文案（与 LG 逐字相同） |
|---|---|
| 没有入口 | `ValueError: Graph must have an entrypoint: add at least one edge from START to another node` |
| 边指向不存在的节点 | `ValueError: Found edge ending at unknown node \`b\`` |
| 分支目标不存在 | `ValueError: At 'a' node, 'condition' branch found unknown target 'other'` |
| 重复注册节点 | `ValueError: Node \`a\` already present.` |

`Error::Validation` 现在映射为 Python 的 `ValueError`（不再带 `graph validation failed:` 前缀）。

### `ToolRuntime` 注入（2026-11-27）

LG 允许工具声明 `runtime: ToolRuntime`，运行时注入一个带 `state` / `config` / `store` /
`stream_writer` / `tool_call_id` / `context` 等字段的句柄，并在模型可见的 schema 里**隐藏**这个参数。

扩展此前：注入本身能跑，但 `config` 是空的，且 `tool_call_schema` 直接抛
`PydanticInvalidForJsonSchema`（因为 `runtime` 没有被标记为注入参数）。

现在与 LG 一致：

```
schema 参数：['query']                       # runtime 被隐藏
工具内：state=['messages'] cfg=True writer=True
```

实现要点：`ToolRuntime` 改为继承 langchain 的 `_DirectlyInjectedToolArg`（由 langchain 的
`@tool` 判定"隐藏此参数"），并用自定义 pydantic schema 接受引擎在 Rust 侧构造的句柄 ——
这样既隐藏了参数、又不必让 Rust 反向导入 shim 的类。

### 子图 + `debug` 流：整条记录进元组（2026-11-27）

`stream_mode="debug", subgraphs=True` 时 LG 产出 `(namespace, chunk)`，其中 chunk 是**完整的** debug
记录（`type` / `timestamp` / `step` / `payload`）。扩展此前只把 `payload` 放进去，`type` 等字段丢失：

```
LG: [([], 'task'), (['sub:...'], 'task'), (['sub:...'], 'task_result'), ([], 'task_result')]
改前: [([], None), (['sub:...'], None), ...]         ← type 不在
现在: 与 LG 完全一致，且每条都能取到 4 个字段
```

顺带核对（已一致）：`stream_mode="tasks", subgraphs=True` 的名称与命名空间序列；`astream_events(version="v2")`
的事件序列（`on_chain_start` / `on_chain_stream` / `on_chain_end`，含 `LangGraph` 根节点）逐条相同。

**`get_graph()` 的边顺序已对齐**：门面在 `add_edge` / `add_conditional_edges` 时记录声明顺序，并传给编译后的
图，`edges` 与 `draw_mermaid()` 与 LG **逐字相同**（实测整段 mermaid 文本相等）。

**仍有的表层差异**：`Graph.nodes[...].data` 为 `None`（LG 是内部 `PregelNode` 的 repr 字符串）。
编译后的图对象公共面仅缺 `nodes` / `attach_node` / `attach_edge` / `attach_branch` 这组 Pregel 内部成员。

### `RetryPolicy(retry_on=<callable>)`（2026-11-27）

LG 的 `retry_on` 接受**三种**形式：异常类型、类型序列、以及 `(exc) -> bool` 的谓词。扩展此前只把
`retry_on` 解析成"异常名列表"再按名字匹配，于是：

| 形式 | LG | 扩展（改前） | 现在 |
|---|---|---|---|
| `retry_on=lambda exc: isinstance(exc, ValueError)` | 3 次 | **1 次**（`<lambda>` 匹配不到） | 3 次 |
| `retry_on=lambda exc: False` | 1 次 | 1 次 | 1 次 |
| `retry_on=ValueError` | 3 次 | 3 次 | 3 次 |
| `retry_on=KeyError`（不匹配） | 1 次 | 1 次 | 1 次 |
| `retry_on=[ValueError, KeyError]` | 3 次 | 3 次 | 3 次 |

实现：Rust 的 `RetryPolicy` 新增 `condition: Option<RetryCondition>`（`Arc<dyn Fn(&str) -> bool>`），
由 Python 适配层注入 —— 引擎只用**错误消息**回调，消息里带着异常对象的标记，适配层据此取回原异常
再调用用户谓词。这不违反"Rust 不反向调用 shim"的约定：回调是用户提供的可调用对象。

`repr(RetryPolicy(retry_on=...))` 也与 LG 逐字一致：
`retry_on=<class 'ValueError'>` / `retry_on=[<class 'ValueError'>, <class 'KeyError'>]` /
`retry_on=<function <lambda> at 0x...>`。

### `RemoteGraph` 的公共面与 LG 完全一致（2026-11-27）

`RemoteGraph`（连 LangGraph Server 的客户端）此前只有 `invoke`/`stream`/`get_state`/`update_state` 等
14 个方法，缺 `batch`/`bind`/`as_tool`/`with_retry` 这些 `Runnable` 协议方法，也缺
`bulk_update_state`。

现在直接继承 `langchain_core.runnables.Runnable`（LG 就是这么做的），Runnable 的方法全部继承过来，
另补 `bulk_update_state` / `abulk_update_state`。与 LG 做集合对比：

```
missing vs LG: []
extra:         []
```

`get_name()` / `input_schema` / `output_schema` / `config_specs` 也都能取到（`input_schema` 比 LG
更稳：LG 在无真实客户端时会 `SchemaError`，扩展正常生成模型）。

### `bulk_update_state` 的入参与返回值（2026-11-27）

LG 的 `bulk_update_state(config, supersteps)` 里每个 superstep 是
`(values, as_node, task_id)` **元组**列表，返回**新检查点的 config**；扩展此前只接受
`{"values": ..., "as_node": ...}` 字典形式，返回 `StateSnapshot`。

现在两种入参都接受（元组是 LG 的原生形式，字典是扩展的便利形式），返回值改成 config：

```
bulk_update_state(cfg, [[({"value": 7}, "a", None)]])
→ {'configurable': {...}}；get_state(cfg): values {'value': 7}，next ('b',)
```

顺带核对 `StateSnapshot` 的属性面：`values` / `next` / `config` / `metadata` / `created_at` /
`parent_config` / `tasks` / `interrupts` 八项都能取到，与 LG 一致（扩展额外支持字典式下标访问）。

### `update_state` 的作者推断与返回值（2026-11-27）

LG 的 `update_state(config, values)` 在没给 `as_node` 时**自动推断作者**（按 `versions_seen` 里
最新的节点），返回**新检查点的 config**；扩展此前直接报
`Ambiguous update, specify as_node`，而且返回的是 `StateSnapshot`。

现在两边一致：

```
默认调用        → 返回 {'configurable': {...}}，next: ()，values: {'log': ['a','b'], 'value': 99}
as_node="a"    → next: ('b',)（按 a 的出边重新调度）
```

推断修复的关键：扩展的 `versions_seen` 按**通道**记录（一次写入会让 `log`、`value` 共享同一版本），
而 LG 按**节点**记录一条。旧代码把重复版本当成"两个节点同时写"而判定歧义；现在先按节点取最大版本
再比较。

### 函数式 API：`entrypoint` 的取值语义与签名（2026-11-27）

LG 的 `@entrypoint` 是**值进值出**：装饰的函数收到输入值本身，返回值就是结果；签名是
`entrypoint(checkpointer=None, store=None, cache=None, context_schema=None, cache_policy=None,
retry_policy=None, timeout=None)`。扩展此前只接受 `entrypoint(function)`，且把整份 **state dict**
交给函数 —— `entrypoint(checkpointer=...)` 直接 `TypeError`，`entrypoint(fn).invoke(3)` 报
`graph state must be a dict`。

现在与 LG 一致：

```
@task
def double(x): return x * 2

@entrypoint(checkpointer=InMemorySaver())
def workflow(x): return double(x).result() + 1

workflow.invoke(3, cfg)                    -> 7
interrupt()                                -> __interrupt__ = [{'q': 'value?'}]
invoke(Command(resume=7))                  -> 7
```

实现要点：

- 适配层把值包进单通道状态，再把输出解回来（`invoke` / `ainvoke` / `stream` / `astream` /
  `get_state` / `update_state` / `resume`），`Command` 输入原样透传
- `@task` 的返回值统一带 `.result()` / `.aresult()`（LG 返回 future；引擎可能已即时算完），
  函数返回结构里夹带的 future 会被自动解开
- 顶层 `entrypoint` / `task` 现在指向门面实现（`from langgraph_rs.func import ...` 同样可用）

### saver 的其余 API 面（2026-11-27）

`InMemorySaver` 补齐到 LG 的 `BaseCheckpointSaver` 形状：

| 方法 / 属性 | 行为（与 LG 一致） |
|---|---|
| `get(config)` / `aget` | 返回**原始 checkpoint dict**（`id`/`channel_values`/`channel_versions`/`v`/`ts`/`versions_seen`） |
| `get_delta_channel_history(config=..., channels=[...])` | `{channel: {'writes': [...], 'seed': 0}}` |
| `config_specs` | `[]` |
| `serde` | `JsonPlusSerializer` 实例 |
| `with_allowlist(extra)` | 返回自身 |
| `prune` / `aprune` / `delete_for_runs` / `adelete_for_runs` | `NotImplementedError`（LG 的 `InMemorySaver` 就是这样） |

**一处已知差异**：`get_delta_channel_history` 的 `writes` 在扩展里通常为空 —— 引擎按步
**快照完整值**（不使用 delta 日志），只有真正的 pending writes（例如中断时）才会出现在这里。

### 消息对象：`add_messages` 通道返回 LangChain 消息对象（2026-11-27）

LG 的 `add_messages` 通道里装的是 **LangChain 消息对象**；扩展此前返回 dict
（`{'role': 'user', 'content': ...}`），`state["messages"][-1].content` 这类写法直接崩。

现在与 LG 一致：

```
输入 {'messages': [{'role': 'user', 'content': 'hi'}]}
节点内：HumanMessage，.content == 'hi'
返回：  [HumanMessage('hi'), AIMessage('hello')]
         .type == 'human' / 'ai'
```

实现：`StateSchema` 标记 `add_messages` 通道 → 运行时登记通道名（全局注册表，异步图在别的线程
转换）→ `json_state_to_py` 用 `langchain_core.messages.convert_to_messages` 转成对象。

过渡开关：`langgraph_rs._set_messages_objects(False)` 可退回 dict 表示（默认开）。
普通通道不受影响，仍是原样 JSON 值。

### `create_react_agent` 的参数面（2026-11-27）

LG 的 `create_react_agent` 有一份明确的参数表（`model, tools, prompt, response_format,
state_schema, config_schema, checkpointer, store, interrupt_before, interrupt_after, debug, version,
name`）；扩展此前是 `(*args, **kwargs)`，`inspect.signature` 看不到任何参数，传错名字也不会报错。

现在参数表与 LG 一致：`model / tools / prompt / response_format / state_schema / config_schema /
checkpointer / store / interrupt_before / interrupt_after / debug / version / name`。
`config_schema` 映射到引擎的 `context_schema`（显式传 `context_schema` 时以它为准）。

行为也逐条核对过：假模型 + 一个工具时，两边都是
`[human, ai(tool_call), tool('3'), ai('3')]`，图节点都是 `__start__ / agent / tools / __end__`。

### `get_config()` 的形状（2026-11-27）

LG 在节点里调用 `get_config()` 拿到的是完整 RunnableConfig 视图：

```
keys: callbacks / configurable / metadata / tags
metadata: {'m': 1, 'thread_id': 'cfg', 'langgraph_step': 1, 'langgraph_node': 'n',
           'langgraph_triggers': ('branch:to:n',), 'langgraph_path': ('__pregel_pull', 'n'),
           'langgraph_checkpoint_ns': 'n:<uuid>'}
```

扩展此前只返回 `{'configurable': {'thread_id', 'checkpoint_ns'}}`：`tags`、`metadata` 全丢，
`config["metadata"]` 直接 KeyError。

现在四个键齐全，`metadata` 里合并了调用方的 metadata 与六个 `langgraph_*` 键，
`langgraph_path` / `langgraph_triggers` 与 LG 一样是**元组**。
顺带补上 `langgraph_rs.config.RunnableConfig`（LG 从这里导出的类型别名）。

### `handle_tool_errors=<异常类型/元组>` 按类型过滤（2026-11-27）

LG 允许只处理特定异常：`handle_tool_errors=ValueError` 只接住 `ValueError`，其它异常照常抛出。扩展此前
把类型当成"总是处理"，任何异常都会被吞成 `ToolMessage`。

现在按 `isinstance` 判断，四种形态与 LG 逐条一致：

```
ValueError                     → Error: ValueError('kaboom')  (被处理)
(ValueError, KeyError)         → Error: ValueError('kaboom')  (被处理)
KeyError                       → ValueError: kaboom           (向上抛)
(KeyError,)                    → ValueError: kaboom           (向上抛)
```

### `handle_tool_errors` 的可调用对象拿到原异常（2026-11-27）

LG 把**异常本体**交给 `handle_tool_errors` 的回调；扩展此前构造一个 `RuntimeError` 代替，于是
`type(error).__name__` 从 `ValueError` 变成了 `RuntimeError`。

现在工具错误同样带着异常对象（消息里附不可见标记，表里存对象），回调拿到的是
`ValueError('kaboom')`，与 LG 相同；而默认消息仍是 `Error: ValueError('kaboom')\n Please fix your
mistakes.`（标记在格式化时被剥掉，不会泄漏给模型）。

### 失败节点的 `snapshot.tasks[i].error`（2026-11-27）

LG 在一次失败之后仍能从 `get_state()` 读到"哪个任务失败了、为什么"：

```
values: {'value': 1} | next: ('b',) | tasks: [('b', "ValueError('kaboom')")]
history steps: [1, 0, -1]      # 失败不新增检查点
```

扩展此前 `tasks[0].error` 是 `None`（失败不写检查点，快照里没有这条错误）。现在在失败路径上把错误
挂回**上一条检查点**的待执行任务（复用同一 checkpoint id，所以历史条数不变），并把 error 渲染成
LG 的形式 —— 异常本体的 `repr`：

```
rs: [('b', "ValueError('kaboom')")]
```

实现细节：为了让快照能拿到 repr，异常表的条目在"重新抛出给调用方"之后**不再删除**（表有 64 条上限）。

### 节点异常按原样抛出（2026-11-27）

LG 把节点里抛出的异常**原样**抛给调用方：`ValueError("kaboom")` 就是 `ValueError: kaboom`（重试耗尽
后也一样）。扩展此前一律包成
`RuntimeError: node 'b' failed: python node raised ValueError: ValueError: kaboom`，用户代码里的
`except ValueError` / `except ValidationError` 全部失效。

实现方式：`pyerr_to_error` 把异常对象存进全局表并在消息里留一个不可见标记，`to_pyerr` 在最后一步
把标记换回原异常对象。三条路径都验证过：

```
ValueError("kaboom")  → ValueError: kaboom
重试耗尽（max_attempts=2） → ValueError: kaboom（不是包装文本）
KeyError("missing")   → KeyError: 'missing'
```

顺带修正：`Send(..., timeout=...)` 这类**任务级**超时此前没进 `NodeTimeoutError` 映射（只看了节点策略），
现在同样抛 `NodeTimeoutError: Node 'worker' exceeded its run timeout of 0.050s ...`。

**仍缺（已记录）**：节点失败后 `snapshot.tasks[i].error` 里 LG 会留错误字符串
（`"ValueError('kaboom')"`），扩展目前是 `None`（失败不写检查点，快照里没有这条任务错误）。

### `interrupt(response_schema=...)` 与非法节点输出（2026-11-27）

**`response_schema` 可以传类**：LG 接受 pydantic / TypedDict 类或 JSON schema 字典。扩展此前只认
字典 —— 传类会在 `interrupt()` 里报 `TypeError: value of type 'type' cannot be converted to JSON`。
现在类会先转成 JSON schema 再存进 interrupt 记录（`snapshot.interrupts[0].response_schema` 能看到
`{'title': 'Answer', ...}`）。

**非法节点输出**：节点返回非映射时，LG 抛
`InvalidUpdateError: Expected dict, got ['not-a-dict']` 加排错链接；扩展抛
`RuntimeError: node 'b' failed: ...`。现在与 LG 一致（消息逐字，含值本身）。

**一处已记录的差异**：`resume` 值不满足 schema 时，LG 抛 pydantic 的 `ValidationError`
（`1 validation error for Answer ...`）；扩展由引擎按 JSON schema 校验，抛 `RuntimeError: resume
value is missing required response_schema property 'answer'`。两者都会失败，只是异常类型不同。

### 工具里的 `interrupt()`（2026-11-27）

LG 允许在**工具内部**调用 `interrupt()`，效果与节点内相同：图暂停、`__interrupt__` 带上值、`resume`
后工具从暂停处继续。扩展此前把 `interrupt()` 抛出的 `GraphInterrupt` 当成"工具执行失败"处理，
`handle_tool_errors` 于是把它变成一条错误 `ToolMessage` —— 图不暂停，agent 会一直重复调用工具，
直到触到递归上限（表现为"卡住"）。

两处修复：

1. `tool_definitions` 里把 `GraphInterrupt` 交回 `pyerr_to_error`（→ `Error::Interrupt`），不再转成
   repr 文本
2. `ToolNode.execute` 遇到 `Error::Interrupt` **直接向上传播**，不走 `handle_tool_errors` 策略

实测与 LG 一致：

```
第一次 invoke → __interrupt__ = [{'question': 'why?'}]
resume('yes') → 消息尾部 ['answer=yes', 'done']
```

### `add_node(defer=True)`（2026-11-27）

LG 用 `defer=True` 声明延迟节点（等没有常规任务时才跑），并且允许：

- 延迟节点**作为入口**
- 延迟节点**有出边** / 条件边

扩展此前：关键字叫 `deferred`（LG 是 `defer`），且把上面两种情况都当校验错误
（`deferred node ... cannot be an entry point` / `cannot have outgoing edges`）。

现在 `defer` 与 `deferred` 都接受，两条校验限制已移除（调度器本来就在没有常规任务时才跑延迟
任务）。实测与 LG 一致：

```
输入 {'log': [], 'value': 0}
输出 {'log': ['e'], 'value': 1}   # 延迟节点 d 在 e 之后执行
```

顺带核对了 `Command(update=..., goto=...)`：两边都是 `{'value': 11}`。

### 检查点历史里的 `source="input"` 条目（2026-11-27）

LG 的 `get_state_history` 会给出四条记录，其中 `metadata.source == "input"`、`step == -1` 的那条
**通道值是空的**：

```
step: 2 | source: loop  | values: {'value': 2} | next: ()
step: 1 | source: loop  | values: {'value': 1} | next: ('b',)
step: 0 | source: loop  | values: {'value': 0} | next: ('a',)
step: -1| source: input | values: {}            | next: ('__start__',)
```

扩展把原始输入写进了这条记录（`values: {'value': 0}`）。现在与 LG 一致 —— 输入落在紧随其后的
post-input 检查点里，`input` 检查点只有 `__start__` 这个待执行任务。

`durability` 三种取值（`sync` / `async` / `exit`）的检查点数量也与 LG 相同（4 / 4 / 1）。

### 错误类型与"新线程上的 Command"（2026-11-27）

**并发写冲突**：LG 抛 `InvalidUpdateError`，消息是
`At key 'value': Can receive only one value per step. Use an Annotated key to handle multiple
values.` 加上排错链接；扩展此前抛裸 `RuntimeError: conflicting concurrent writes to channel ...`。
现在映射为 `InvalidUpdateError`，消息与 LG 逐字一致。

**新线程上的 `Command`**：LG 不会报错 —— 没有检查点时它**从起点开始跑**，`resume` 被忽略，
`update` 作为初始输入：

```
invoke(Command(resume=5))        → 节点看到 {}           （resume 被忽略）
invoke(Command(update={"value": 9})) → 节点看到 {'value': 9}
```

扩展此前抛 `RuntimeError: checkpoint failure: no checkpoint found for thread ...`。现在与 LG 一致。

### 顶层导出的 saver / store 是"兼容类"（2026-11-27）

`langgraph_rs.InMemorySaver` 之前指向**原生类**（只有 `delete_thread` / `get_next_version`），而带
完整表面（`get_tuple` / `list` / `aget_tuple` / `alist` / `adelete_thread`）的兼容子类藏在
`langgraph_rs.checkpoint.memory` 里 —— 同一路径（`from langgraph.checkpoint.memory import
InMemorySaver` → `langgraph_rs.checkpoint.memory`）能拿到的方法，走顶层导入就没有。现在顶层导出
统一指向兼容类（`InMemorySaver` / `MemorySaver` / `AsyncInMemorySaver` / `InMemoryStore` / `Store`）。

补上的 saver 表面：

- 异步孪生 `aget_tuple` / `alist` / `aput` / `aput_writes` / `adelete_thread` / `acopy_thread`
- `copy_thread(source, target, *, checkpoint_id=None)`（原生侧新增，实测能复制线程状态）

**已补齐**：`put(config, checkpoint, metadata, new_versions)` 与 `put_writes(config, writes, task_id)`
（原生 `put_document` 接收 LangGraph 形状的检查点文档并落盘，`put_writes` 在其上追加 pending writes；与 LG
实测一致：id 保持、pending writes 追加、历史长度不变）。引擎自己的写入仍走内部路径。

**仍缺（已记录）**：`prune` /
`with_allowlist` / `get_delta_channel_history` / `delete_for_runs` 以及 `serde` / `storage` /
`writes` / `blobs` 等内部属性。

### `CachePolicy.key_func` 与 `InMemoryStore` 的批量/异步面（2026-11-27）

**`CachePolicy.key_func`**：LG 用 `key_func(val)` 计算节点缓存键（一个参数，节点输入状态），扩展此前
只是把字段存下来、**从不调用**。现在真正接入引擎的缓存键计算：

```
run1 → node 调用一次
run2 → 缓存命中（key_func 两次都以 {'value': 1} 调用，与 LG 相同）
```

**`InMemoryStore`**：

- 补齐异步孪生方法 `aget` / `aput` / `adelete` / `asearch` / `alist_namespaces` / `abatch`
- 补齐 `batch(ops)`（按 `GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp` 分派）
- 补齐 `supports_ttl` / `ttl_config`
- `Item.namespace` 改为**元组**，`created_at` / `updated_at` 改为 **datetime**（此前是 list / 字符串）

`batch` 还有一个语义细节：LG 的读操作看到的是**批处理开始前**的状态（同一批里的 `PutOp` 对后续
`GetOp` / `ListNamespacesOp` 不可见），已按此实现。

### 策略对象的参数与 `repr`（2026-11-27）

LG 的三个策略都是 dataclass，构造参数与 `repr` 是用户可见的接口。扩展此前：

- `RetryPolicy` 没有 `__repr__`（打印成 `<builtins.RetryPolicy object at 0x...>`），`jitter` 只收
  float（LG 是 **bool**，默认 `True`），`retry_on` 只收字符串名（LG 收 **异常类/元组**）
- `CachePolicy` 缺 `key_func`（LG 的核心字段之一）
- `TimeoutPolicy` 缺 `refresh_on`（`"auto"` / `"heartbeat"`）

现在全部对齐，`repr` 也按 LG 的字段顺序输出：

```
RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=())
CachePolicy(key_func=None, ttl=60)
TimeoutPolicy(run_timeout=0.5, idle_timeout=None, refresh_on="auto")
```

`retry_on=(ValueError, KeyError)` 会按名字参与匹配（已端到端验证 `max_attempts=2` 重试成功）。

### `snapshot.tasks` / `snapshot.interrupts` 的元素类型（2026-11-27）

LG 的 `snapshot.tasks[i]` 是 **`PregelTask` named tuple**（字段 `id, name, path, error,
interrupts, state, result`），`snapshot.interrupts[i]` 是 **`Interrupt` 对象**。扩展两者都是普通
dict，于是 `task.name` / `interrupt.value` 这类属性访问直接报错。

现在构建快照时转成对象：

```
task: PregelTask(name='ask', path=('__pregel_pull', 'ask'), error=None, ...)
interrupt: Interrupt(id='...', value='q?')
```

实现上 `snapshot_to_py` 拆成「原始 dict」与「最终对象」两步：`get_state(subgraphs=True)` 需要先把
子图状态写进 task dict（named tuple 不可变），完成后再统一转成 `PregelTask`。

### `StateSnapshot` 的字段类型（2026-11-27）

LG 的 `StateSnapshot` 里 `next` / `interrupts` / `tasks` 都是**元组**，`metadata` 一定带
`parents`（顶层是空字典）。扩展此前三个字段是 list，`metadata` 也缺 `parents`：

```
LG: tuple tuple tuple | metadata keys: parents, source, step
我们: list  list  list  | metadata keys: source, step
```

现在一致。改动还牵出一个隐藏问题：`get_state(..., subgraphs=True)` 里
`embed_subgraph_states` 用 `cast::<PyList>()` 取 `tasks`，字段变成元组后子图快照就**静默不再嵌入**
（`child_task["state"]` 变成 `None`）—— 已改为迭代器，两种容器都能处理。

顺带核对了这几项与 LG 一致：`response_format`（`structured_response`）、`prompt=`（system 消息
前置）、`store=`（工具里 `runtime.store` 读写）、`context_schema` + `context=`（工具里
`runtime.context`）。

### `create_react_agent` 的完整回环（2026-11-27）

用 `GenericFakeChatModel`（先返回工具调用、再返回答案）对拍，发现**工具调用被吞掉**：

```
LG: [human, ai(tool_calls), tool('42'), ai('the answer is 42')]
我们: [user, ai('the answer is 42')]        # 少了 ai(tool_calls) 和 tool
```

根因：`runnable_model_node` 里**显式调用 `model.stream()`**（上一轮之前为了让 token 进流而加的
workaround），它把假模型的第一条消息吃掉了；累积失败后回退到 `model.invoke()`，于是消费了第二条
消息 —— 节点只看到最终答案。

LG 的做法是**只调用一次 `invoke`**：它安装的 `_StreamingCallbackHandler` 会让 LangChain 自己把
`invoke` 切到流式路径。现在我们把「是否流式」做成运行期开关：

- `stream_mode="messages"`、`stream_live`（events API）→ 用 `StreamingLangChainCallbackHandler`
- 普通 `invoke` → 用不带流式标记的处理器（与 LG 一致）

修完后：

```
rs: [user, ai(tool_calls), tool('42'), ai('the answer is 42')]   # 与 LG 相同
```

**一处已记录的差异**：LG 的消息对象 `type` 是 `human` / `ai` / `tool`，扩展的状态里消息是字典、
用 `role: user / ai / tool`。parity 场景里做了别名折叠，文档记录该差异。

### `stream_mode="messages"` 的 token 流（2026-11-27）

LG 在 `messages` 模式下按 **token** 上报模型输出，元数据是一整套 `langgraph_*` / `ls_*` 键。扩展
此前只把节点的最终消息上报一条（`'hello world'`），元数据也缺 `checkpoint_ns` / `lc_versions` /
`ls_*`。

两处根因：

1. 我们的 LangChain 回调处理器**不是** `_StreamingCallbackHandler`，于是 LangChain 的
   `BaseChatModel._should_stream()` 返回 False，节点里的 `model.invoke(...)` 不会走流式路径，
   自然没有 token。改成同时继承 `BaseCallbackHandler` 与
   `langchain_core.tracers._streaming._StreamingCallbackHandler` 后，`invoke` 与 LG 一样自动流式。
2. 元数据里少了 `checkpoint_ns`。LG 用的是**任务级**命名空间 `<node>:<task id>`，而
   `langgraph_path` 只追加**子图**段 —— 两者不是同一个值，现在分别取用。

实测两边一致：

```
tokens: ['hello', ' ', 'world']
meta:   checkpoint_ns, langgraph_checkpoint_ns, langgraph_node, langgraph_path,
        langgraph_step, langgraph_triggers, lc_versions, ls_integration,
        ls_model_type, ls_provider
```

顺带把 `tags` 从消息元数据里去掉（LG 的元数据里没有这个键）。

### `subgraphs=True` 的命名空间包装（2026-11-27）

LG 在 `subgraphs=True` 时把**每个** chunk 都包成 `(namespace, payload)`，顶层用空元组 `()`：

```
[(('sub:<uuid>',), {'inner': {'value': 7}}), ((), {'sub': {'value': 7}}), ((), {'tail': {'value': 9}})]
```

扩展只包装了来自子图的 chunk，顶层 chunk 是裸的 —— 调用方按 `for ns, chunk in stream` 解包会
`ValueError`。现在顶层也用 `()` 包装，`updates` / `values` / `astream` 三条路径一致。

顺带确认 `stream_mode="checkpoints"`（4 条 chunk、键集合）与 LG 相同，无需改动。

### `stream_mode="tasks"` 的载荷（2026-11-27）

LG 的 task 载荷字段是 `PregelTask` 的形状：`{id, name, input, triggers}`。扩展此前：

- 多一个 `step` 键
- `input` 常为 `None`（只有 `Send` 任务才带输入）
- `triggers` 是 list 而非 **tuple**

现在完全一致（`debug` 模式里的 `task` 记录本就用同一份数据，一并修正）：

```
{"id": "...", "name": "a", "input": {"value": 0}, "triggers": ("branch:to:a",)}
{"id": "...", "name": "b", "input": {"value": 1}, "triggers": ("branch:to:b",)}
```

`input` 的语义按 LG：任务收到的状态 —— 没有显式输入（普通调度）时就是当前通道值。

### `ToolNode` 的构造面与工具调用中间件（2026-11-27）

LG 的 `ToolNode` 构造签名是
`(tools, *, name="tools", tags=None, handle_tool_errors=..., messages_key="messages",
wrap_tool_call=None, awrap_tool_call=None)`，实例上还有 `tools_by_name`、`get_name()`、`invoke()`。
扩展此前只有 `(tools, messages_key=...)` 和 `__call__`。

现在补全：

- `name` / `tags` / `tools_by_name` / `get_name()` / `invoke()` / `ainvoke()`
- **`wrap_tool_call` / `awrap_tool_call`**：LG 的中间件语义 —— 包装函数收到
  `ToolCallRequest(tool_call, tool, state, runtime)` 和一个 `handler`，调用 `handler(request)`
  才真正执行工具，返回值即最终的 `ToolMessage`

实测两边一致：

```
wrap(request, handler) 收到 ("ToolCallRequest", "add")
handler(request) -> status="success", content="3", name="add", tool_call_id="c1"
```

实现上，未配置包装函数时仍走原生节点（零开销），配置后才在 Python 侧逐个调用并包装结果。

**顺带修了 parity 的假失败**：`send_interrupts` 场景原来按**下标**取 interrupt id 来 resume，
而并行 `Send` 任务顺序本身不固定，于是 LG 有时恢复的是另一个任务。现在按 interrupt 自身的
`ask` 值定位，三次运行稳定 **56/56**。

### `ValidationNode` 与超时的时间类型（2026-11-27）

**`ValidationNode`**：LG 逐条校验 tool_calls 而不执行工具，扩展此前的输出缺 `status`、不校验参数类型、
未知工具只是写一条错误消息。现与 LG 一致：

```
有效调用   -> status="success", content = '{"query":"x"}'
参数不合法 -> status="success", content = '1 validation error for Validation\ncount\n  Input should be ...'
未知工具   -> 抛 KeyError（节点失败）
```

参数校验按 LG 的做法：把可调用对象的签名转成名为 `Validation` 的 pydantic 模型，
所以注解真的会被检查（此前只 `inspect.signature().bind()`，`"nope"` 能当 `int` 通过）。

**超时的时间类型**：LG 的 `TimeoutPolicy` / `Send(timeout=...)` / `add_node(timeout=...)` 都接受
「秒数 / `timedelta` / `TimeoutPolicy`」；扩展此前只吃 `float`，传 `timedelta` 会报
`TypeError: must be real number`。现在三处都接受 `timedelta`（内部换算成秒）。

### `ToolNode(handle_tool_errors=...)`（2026-11-26）

LG 的 `ToolNode` 用 `handle_tool_errors` 决定工具抛异常后的行为，扩展此前连这个参数都不接受
（`TypeError: ToolNode.__new__() got an unexpected keyword argument`），而且失败的工具消息没有
`status`。

实测四种形态，现已逐条对齐：

| `handle_tool_errors` | 结果 `ToolMessage` |
|---|---|
| `True`（默认） | `status="error"`，内容 `Error: ValueError('kaboom')\n Please fix your mistakes.` |
| 字符串 | `status="error"`，内容就是该字符串 |
| 可调用对象 | `status="error"`，内容是 `handler(exc)` 的返回值 |
| `False` | 异常向上抛出，节点失败 |

两处细节也一并修正：

- 工具异常文本改用**异常本体的 `repr()`**（`ValueError('kaboom')`），这正是 LG 写进消息的形式
- 交给可调用对象的异常，其 `str()` 是**原始消息**（`kaboom`），与 LG 一致

另外修了一个更基础的问题：`ToolNode` 现在会先把 LangChain **消息对象**摊平成
`role`/`content`/`tool_calls` 字典再执行，否则 `AIMessage(...).tool_calls` 在引擎的 JSON 形态里
看不到（此前的测试都用字典消息，没暴露这一点）。

### `idle_timeout` 的语义（2026-11-26）

LG 的 `idle_timeout` 在节点**停止上报进度**时触发，抛
`NodeTimeoutError: Node 'x' exceeded its idle timeout of 0.050s without making progress (elapsed: ...)`；
只要节点持续 `custom()` / `message` / usage 写入，计时就重置。扩展此前**完全忽略** `idle_timeout`
（节点跑完就成功）。

原因：Python 协程跑在私有事件循环里，Tokio 侧的看门狗（`run_with_idle_timeout`）根本没机会轮询。
现在：

- 引擎暴露 `_progress_counter()`（AsyncContext 与同步 TaskContext 的进度计数之和，
  `custom()` / `message` / usage / `get_stream_writer()` 都会累加）
- 异步节点安装一个 Python 侧看门狗：每 `idle` 轮询一次进度，无进展就取消协程并抛
  `TimeoutError("langgraph-rs idle timeout")`
- 该标记被映射回引擎的 `Error::IdleTimeout`，最终成为 `NodeTimeoutError(kind="idle")`

验证：静默节点 → `NodeTimeoutError`（消息与 LG 同格式）；每 20ms 写一次 `custom` 的节点
（`idle=50ms`）正常完成 —— 与 LG 相同。

### 超时策略的语义（2026-11-26）

对照 LG 修正两处：

1. **同步节点不允许超时**：LG 在 `compile()` 就抛
   `ValueError: Node timeouts are only supported for async nodes because sync Python execution
   cannot be safely cancelled in-process. Node 'x' is sync.`
   扩展此前会「超时放弃」并报错，但线程仍在跑（正是 LG 拒绝的原因）。现在编译期同样拒绝。
2. **超时错误类型**：LG 抛 `NodeTimeoutError`（`langgraph.errors`），消息形如
   `Node 'x' exceeded its run timeout of 0.050s (elapsed: 0.063s).`；扩展此前抛
   `RuntimeError: node 'x' failed: python node raised TimeoutError: ...`。

   现在引擎的 `Error::Timeout`（含经重试包装后的形态）统一映射为 `NodeTimeoutError`，
   消息格式与 LG 相同；`elapsed` 用配置的超时值近似（LG 是实测耗时）。

另外 `add_node(timeout=...)` 现在接受**秒数 / `timedelta` / `TimeoutPolicy`**（LG 的三种形态），
普通数值会被包成 `TimeoutPolicy(run_timeout=...)`。

### `Command(graph=...)` 的语义（2026-11-26）

LG 只有 `Command.PARENT`（字符串 `"__parent__"`）表示「回到外层图」；而 `graph="<子图名>"` 只是在
跨图边界时才被解读，**在父图节点里返回时等于没写**。扩展此前把「任何 `graph=`」都当成父图路由，
于是父图里的 `Command(goto="child", graph="child")` 直接报
`Command(graph=PARENT) can only be returned from inside a subgraph`。

现在只有 `graph=Command.PARENT` 触发父图路由，其余 `graph=` 值按 LG 的行为被忽略：

```
Command(goto="child", graph="child")  -> {'value': 7}   # 两边相同
Command(goto="child")                 -> {'value': 7}   # 两边相同
```

**一处有意的不同**：`Command(goto="<不存在的节点>")` 扩展会报
`invalid route from ...: unknown destination node`，LG 只是在日志里告警并忽略该写入。这条严格性
是本项目既有设计（`test_bindings` 有对应断言），保留。

### `astream_events` 的 v2 / v3 形状（2026-11-26）

LG 的 `astream_events` 是**普通 `def`**：`version="v3"` 返回可 await 的 `AsyncGraphRunStream`，
`version="v2"` 直接返回**异步生成器**，逐条产出 LangChain 风格的回调事件。

扩展此前只实现 v3，且 `astream_events` 是 `async def`——于是 `version="v2"`（以及任何 v2 调用）
会拿到协程，`async for` 直接报错。现在：

- 门面改成普通 `def`，按版本返回「协程」或「异步生成器」
- 新增 v2 发射器：由引擎的 `updates`/`values` 流合成 LG 的 `on_chain_*` 序列

实测两边事件序列逐条相同：

```
on_chain_start(LangGraph) → on_chain_start(a) → on_chain_stream(a)
→ on_chain_end(a) → on_chain_stream(LangGraph) → on_chain_end(LangGraph)
```

每个事件的键也一致（`event` / `name` / `run_id` / `tags` / `metadata` / `data` / `parent_ids`），
节点事件的 `metadata` 带 `langgraph_node`（子图时再带 `langgraph_checkpoint_ns`）。

### Checkpointer 的读取协议（2026-11-26）

LG 的 saver 自己承担 `get_tuple` / `list` / `get` / `delete_thread` / `get_next_version`（异步有
`aget_tuple` / `alist` / `adelete_thread`）。扩展此前把这些能力放在**图**上
（`threads()` / `get_state_history()`），saver 对象只有构造与 `repr`。

现在 `InMemorySaver` / `AsyncInMemorySaver` 也提供读取协议：

| 方法 | 行为 |
|---|---|
| `get_tuple(config)` | 返回 LG 的 `CheckpointTuple`（NamedTuple：`config` / `checkpoint` / `metadata` / `parent_config` / `pending_writes`），属性访问与 LG 一致 |
| `get(config)` | `get_tuple(...).checkpoint` |
| `list(config, *, before, limit, filter)` | 按最新优先遍历，`filter` 匹配 metadata 键 |
| `delete_thread(thread_id)` | 删除整条线程 |
| `get_next_version(current, channel)` | 返回 LG 形状的版本串 `"<counter:032x>.0.<timestamp>"` |
| 异步 `aget_tuple` / `alist` / `adelete_thread` | 对应异步版本 |

`checkpoint` 体的键与 LG 相同（`v` / `id` / `ts` / `channel_values` / `channel_versions` /
`versions_seen` / `updated_channels`）。`put` / `put_writes` 仍由引擎内部完成（不经 saver 对象）。

### `path_map` 的列表形式 / `Send` 扇出中断 / 子图中断（2026-11-26）

三处对照结果：

1. **`add_conditional_edges(..., path_map=[...])`**：LG 接受「节点名列表」，扩展此前只认 dict
   （`path_map must be a dict`）。现在列表会展开成 `{name: name}`，dict 形式不变。
2. **`Send` 扇出各自中断**：两边都暂停并返回两个中断 `[{ask:0},{ask:1}]`；用
   `Command(resume={id: value})` 逐条恢复得到 `['0:a'] → ['0:a','1:b']`，中断载荷同步减少。
3. **子图内的 `interrupt()`**：父图 `invoke` 直接看到子图的中断载荷，父图
   `Command(resume=...)` 即可继续，恢复后 `next` 为空、状态正确 —— 两边相同。

### 多轮 resume 与单节点多次 `interrupt()`（2026-11-26）

对照 LG 验证了两件事，结论是**行为已经一致**（本轮没有改引擎代码，只补了回归场景）：

1. **多轮 resume**（每次恢复一个值，节点循环询问三次）：
   两边都是 `['one'] → ['one','two'] → ['one','two','three']`，每轮中断载荷
   `{'ask': n}` 与历史步数 `[3, 2, 1, 0, -1]` 完全相同。
2. **一个节点内连续两次 `interrupt()`**：resume `"A"` 后暂停在第二个问题上，
   再 resume `"B"` 得到 `['A', 'B']` —— 说明恢复后重跑节点时会**重放已消费的旧答案**，
   两边一致。

调查中曾误判为 bug：LG 侧用了 `log: list`（last-value 通道）而扩展侧用了 `append` 通道，
节点返回整份列表时 append 会重复累加。**对齐 reducer 后结果逐项相同** —— 这也说明对照实验里
通道语义必须一致，否则会得到假的差异。

### 任务 id 与子图命名空间是 UUID 形状（2026-11-26）

LG 的任务 id 是 UUID（`1c0e7fd3-4c28-a50a-0899-e8d6c649ab7a`），子图命名空间是
`<节点名>:<task_id>`，因此形如 `('child:3820a0d1-...',)`。扩展此前用
`<节点名>-<毫秒>-<序号>`，命名空间是 `child:child-1795688892058-8`，用户在 tasks 流与
`subgraphs=True` 的命名空间里会看到完全不同的形状。

现在 `new_task_id` 用 时钟/进程号/计数器 经 SplitMix64 生成 128 位，按 UUID v4 的形状输出（无额外依赖），
命名空间随之对齐：

```
tasks 流 id:     0e06feea-ef23-47a8-8fe7-edae83fac2a2
子图命名空间:     ('child:92a7f918-320b-477c-9cf6-aeb5e03b66b3',)
```

LG 的 id 每次运行都不同（随机），扩展同样不是固定值，只保证同一任务内一致。

### `get_graph()` 返回真正的 LangChain 图（2026-11-26）

LG 的 `get_graph()` 返回 `langchain_core.runnables.graph.Graph`，带 `Node` / `Edge`（NamedTuple）以及
`draw_mermaid()` / `draw_png()` / `draw_ascii()`。扩展此前返回原生 `Graph` 对象，只有 `to_dict()`。

现在：原生只提供图描述（`to_dict()`），门面在 Python 里用 **langchain-core 的 `Graph` API** 重建，
因此 `type(...)` 与 LG 相同、`graph.nodes` / `graph.edges` / 绘图方法都能直接用。`to_dict()` 作为兼容
便利保留在返回对象上。

为此图描述本身也补齐了两处（LG 有、我们此前没有）：

- `nodes` 含 `__start__` / `__end__`
- **`__start__ -> 首个节点` 的入口边**（条件入口按 path map 展开）
- 条件边带上具体目标（`conditional: true`），而不是只列源节点

`draw_ascii()` 依赖 `grandalf`，缺包时与 LG 一样抛 `ImportError`（测试改用 `draw_mermaid()`）。

### 可调用 `error_handler` 与内省属性（2026-11-26）

实测 LG：`add_node(..., error_handler=<callable>)` 会**合成一个节点** `__error_handler__<node>`，
把失败路由过去；处理器收到的是**状态**（不是异常对象），其返回值作为更新写入，**失败节点的出边不再
执行**。扩展此前只接受「节点名」形式。现在可调用形式按同样机制实现：Python builder 注册
`__error_handler__<node>` 节点并把名字交给引擎，二者可观测行为一致：

```
{'value': 0, 'note': "handled:{'value': 0}"}   # 两边完全相同
```

内省属性补齐（LG 有、我们此前缺失的）：

| 属性 | 取值 |
|---|---|
| `input_channels` | `'__start__'` |
| `output_channels` / `stream_channels` / `stream_channels_list` / `stream_channels_asis` | 声明的输出通道，否则全部通道 |
| `node_error_handler_map` | 节点 → 处理器（可调用对象或节点名） |
| `schema_to_mapper` | `{schema 类: None}` |
| `_repr_mimebundle_` | 有 graphviz 时给 `image/png`，否则退回 `text/plain` |

另外 `get_graph().to_dict()` 的 `nodes` 现在包含 `__start__` / `__end__`（LG 的图描述含这两个合成节点，
边也引用它们，此前我们只列真实节点）。

### `stream_mode="debug"` 的形状（2026-11-26）

实测 LG 后对齐三点：

1. **只有存在 checkpointer 时才有 `checkpoint` 条目**（无 checkpointer 时只有 `task` /
   `task_result`）。扩展此前无条件发出 `checkpoint`。
2. **`checkpoint` 载荷的键**是 `config` / `metadata` / `next` / `parent_config` / `tasks` /
   `values` —— 我们多了一个 `executed`。
3. **`step` 用「展示值」**：LG 的原始输入 checkpoint 报 `-1`，随后是 `0, 1, 2`；且 `metadata.step`
   与条目 `step` 相同。扩展此前整体大 1。

现在两条流都逐条一致：

```
无 checkpointer:  task 1, task_result 1, task 2, task_result 2
有 checkpointer:  checkpoint -1, checkpoint 0, task 1, task_result 1,
                  checkpoint 1, task 2, task_result 2, checkpoint 2
```

### `durability="exit"` 的暂停历史（2026-11-26）

实测 LG：`durability="exit"` 下运行暂停时**只保留一个 checkpoint**（`source="loop"`，携带 `next`），
恢复后总共两个。扩展此前会在暂停时把「本回合输入 checkpoint」也补写出来，导致暂停历史多一条。

现在暂停只补写 post（loop）checkpoint，与 LG 一致：

```
暂停后  [('loop', 1, ('gate',))]
恢复后  [('loop', 2, ()), ('loop', 1, ('gate',))]
```

`durability="sync"` / `"async"` 的历史（`loop 1 / loop 0 / input -1`）本来就一致。

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