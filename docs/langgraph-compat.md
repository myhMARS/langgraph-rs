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

1. **引擎内部类**（`PregelNode`、`ChannelWrite`、`ChannelWriteEntry`、`SyncAsyncFuture`）保持
   可导入的占位：使用即抛错并提示改用 `StateGraph.compile()`。
2. **原生 `invoke(&state)` 传完整状态**：绑定层与 LangGraph 一样把 `invoke(dict)` 视为经 reducer
   的部分写入；原生 Rust API 需要完整状态。
3. **`create_react_agent` 无 `state_schema` 时接受额外键**：LangGraph 会丢弃它们。
4. **普通（非 `@tool`）callable 收到位置参数**：LangGraph 传关键字参数映射。
5. **store 超集**：`InMemoryStore` 接受 `ttl`（LG 会抛 `NotImplementedError`）；`search(query=...)`
   在没有向量索引时做子串匹配（LG 忽略 `query`）。
6. **`run.messages` 已是分片级，但还不是生成期实时**：`create_react_agent` 现在会调用模型的
   `stream()` 逐块产出（并按 LangChain 的 `+` 合并成最终消息），`MessagesTransformer` 把每个分片
   作为 `content-block-delta` 派发，因此句柄在**第一个分片**就出现、`handle.text` 逐块产出
   （例如 `["Hello", " ", "world"]`）。**剩余差距**：引擎按任务收集消息分片、节点结束后才下发
   （实测：节点内 3 个分片各间隔 0.2 秒，第一个句柄在 0.6 秒出现），且 `stream()` 在返回前已收集
   完毕；要做到生成期实时，需要引擎「边产出边下发」+ 绑定层改成拉取式迭代器。
7. **`stream()` / `get_state_history()` 返回可重复迭代对象**（`StreamChunks` / `StateHistory`），
   而非生成器。
8. **无 checkpointer 时 `interrupt` 抛 `GraphInterrupt`**：LangGraph 返回输入状态 + `__interrupt__`
   （无 saver 的暂停本也无法恢复）。
9. **`graph.ui.push_ui_message` 只发流事件**：写入 state key 需要 `CONFIG_KEY_SEND`，引擎未暴露。
10. **`RemoteGraph` 未经实时部署验证**：测试用注入的假 SDK 客户端覆盖调用与形状。
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