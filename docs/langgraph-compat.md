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

### `AsyncStateGraph.astream` 仍是收集式（已知限制，2026-11-27）

实测对比（两个各 0.2s 的 `async def` 节点）：

```
StateGraph（LG 兼容面）      .astream -> [0.22, 0.43]   ← 逐 superstep
AsyncStateGraph（扩展自有 API）.astream -> [0.42, 0.42]   ← 收集式
```

**这不算 LangGraph 一致性差异** —— LG 1.2.22 没有 `AsyncStateGraph`，它的 `StateGraph` 同时支持同步与异步，正是我们
上面那一行已经对齐的形态。`AsyncStateGraph` 是本扩展自己的异步构建器。

根因已定位：原生异步 `astream` 把运行 spawn 到 pyo3-async-runtimes 的运行时，运行与接收端的 future 在该运行时下
不会交错推进（同步路径用独立工作线程，所以正常）。修它需要给异步运行单独的线程/运行时模型，风险大于收益。

**规避方式**：用 `StateGraph`（LG 兼容面）—— 它支持 `async def` 节点，且 `astream` 逐 superstep 交付。

### 子图中断的恢复流已对齐（2026-11-27，第 135 轮）

第 134 轮剩的另一半修掉了：恢复时 LG 会先发内层 `('sub:…', {'ask': …})`，再发父级 `((), {'sub': …})`，扩展此前只有
父级块。

根因：子图恢复走的是 `resume_in_streaming`，它内部调用的是 `execute_quiet`（`emit=false`），**不产生流式块**。
换成会发射块的 `pregel::execute` 后：

```
LG:   [(('sub:…',), {'ask': …}), ((), {'sub': …})]
扩展: [(('sub:…',), {'ask': …}), ((), {'sub': …})]
```

至此第 124 轮记录的"子图中断两处差异"全部关闭（暂停两块 + 恢复补内层块）。

### 子图中断的暂停流已对齐（2026-11-27，第 134 轮）

第 124 轮记录的两处差异，本轮修掉第一处：用 `subgraphs=True` 时，暂停应当发出**两块** —— 内层命名空间块 +
父级块。实测现在已经一致：

```
LG:   [(('sub:…',), {'__interrupt__': …}), ((), {'__interrupt__': …})]
扩展: [(('sub:…',), {'__interrupt__': …}), ((), {'__interrupt__': …})]
```

两个原因，都在 live 路径的过滤逻辑里：

1. 中断块此前会被**聚合**成一个（为了匹配非命名空间流的形状）；命名空间流应当逐块上报，所以聚合在
   `wrap_namespaces` 时跳过。
2. 中断块此前依赖"中断绕过模式过滤"的旁路；改造后需要显式写成 `!chunk.is_interrupt() && !matches(...)`，
   否则命名空间流里会被 `updates` 模式过滤掉。

**仍剩**：恢复时 LG 会补发内层 `('sub:…', {'ask': …})`，扩展只发父级块 —— 子图恢复时的 chunk 转发还没接上。

### 单元素模式列表的形状已对齐（2026-11-27，第 133 轮）

第 132 轮验证了方向、但脚本改坏了语法。本轮改为**逐处手工编辑 + 每处独立编译**，一次通过：

- `convert_stream_chunks_with` 保持不变（内部按 `requested.len() > 1`），新增 `convert_stream_chunks_full(...,
  multi_mode)` 供"知道用户是否传了列表"的调用方使用；
- 三处调用点各自计算 `multi_mode`：`stream()` 用 `PyList`/`PyTuple` 判定；`astream` 与 live 路径沿用长度判定
  （单元素列表本来就不走 live）。

实测三种形状：

```
stream(stream_mode=["updates"])  -> [('updates', {'a': {'value': 1}})]   ← 与 LG 相同
stream(stream_mode="updates")    -> [{'a': {'value': 1}}]
astream(["updates"])             -> [('updates', {'a': {'value': 1}})]
```

### 单元素模式列表的形状：尝试与回退（2026-11-27，第 132 轮）

目标：`stream(stream_mode=["updates"])` 应返回 `("updates", payload)` 对（LG 对**列表**总是给成对形式，哪怕只有一个
元素；只有字符串才给裸载荷）。

改动方向是对的：把"用户是否传了列表"作为参数传给转换层（`convert_stream_chunks_with(..., multi_mode)`），而不是在
Python 侧包裹。改完后形状确实正确：

```
单元素列表: [('updates', {'a': {'value': 1}})]   ← 与 LG 一致
标量:       [{'a': {'value': 1}}]
双元素列表: [('values', …), ('updates', …), ('values', …)]
```

但**回退**了：需要用脚本给 5 个调用点补参数，脚本改坏了 `graph_compiled.rs` 的语法（两次），修复耗时超过本轮预算。
工作树已完全恢复（671 测试、100/100 parity、clippy 0）。

结论与教训：**这一项不适合用脚本批量改** —— 调用点分散在 `stream`/`stream_live`/`astream` 里，每处都需要各自的
`multi_mode` 定义。下一轮用逐处手工编辑（每处独立编译验证）完成，预计 15 分钟。

### 暂停快照的任务顺序已对齐（2026-11-27，第 131 轮）

第 103 轮审计的 F8 修掉了：暂停快照里的任务顺序此前是"未完成在前、已完成在后"（`[b, a]`），LG 用的是
**superstep 的声明顺序**（`[a, b]`）。

修法：`Task` 增加 `schedule_index`（暂停时记录各任务在原调度列表里的位置），快照把"待执行 + 已完成"合并后按
该索引稳定排序。

```
改前: [('b', None), ('a', {'seen': ['a']})]
现在: [('a', {'seen': ['a']}), ('b', None)]   ← 与 LG 相同
```

### `PregelTask.result` 已对齐（2026-11-27，第 130 轮）

第 129 轮锁定的疑点是对的：`finalize_tasks`（把 dict 改写成 `PregelTask` 对象的那一步）**只传了 6 个字段，
漏掉了 `result`**。补上第 7 个参数后，配合"历史按子行取写入"的合并，结果立刻正确：

```
LG   step 0: [('a', {'seen': ['a']}), ('b', {'seen': ['b']})]
扩展 step 0: [('a', {'seen': ['a']}), ('b', {'seen': ['b']})]
```

暂停场景同样正确：已完成的兄弟任务带着自己的写入（`a: {'seen': ['a']}`），未完成的是 `None`。

**仍有一处小差异**：暂停快照里的任务顺序是 `[b, a]`，LG 是声明顺序 `[a, b]`（第 103 轮审计的 F8，独立问题）。

### `PregelTask.result`：第三个疑点（2026-11-27，第 129 轮）

本轮把读取侧的合并接进 Python 懒历史（`history.rs`，按"子行"取 `task_writes`），并用探针确认：

```
[probe] row 0 own=1 child=0 tasks=0
[probe] row 1 own=0 child=1 tasks=1
[probe] task=a9568e44-… keys=["a9568e44-…"]   ← 任务 id 与写入键匹配，lookup 成功
step 0 [('a', None)]                          ← 但 Python 侧仍是 None
```

也就是说：**合并执行了、键也匹配、赋值也发生了**，但转换后的对象里依然是 `None`。疑点因此移到转换层 ——
`convert::snapshot_to_py` 先由 `snapshot_to_py_raw` 生成 dict 形式的 tasks，再由 `finalize_tasks` 改写成
`PregelTask`；问题很可能在 `finalize_tasks` 重建对象时没有保留 `result`。

本轮把读取侧改动回退（保留第 128 轮已验证的写入链路），669 测试全绿。下一轮从 `finalize_tasks` 入手。

### `PregelTask.result`：探针定位到真正断点（2026-11-27，第 128 轮）

改用**探针法**（`LG_PROBE_TASKS=1` 打印关键点计数）而不是继续猜，一次就定位了：

```
[probe] save store=0      ← 第一个检查点（superstep 之前，符合预期）
[probe] recorded=1 store=1 ← superstep 之后，运行上下文里已记录 1 条写入
[probe] save store=1      ← 之后的检查点确实带上了写入
```

结论分三层：

1. **记录是有效的**（`TaskResult.output.updates_map()` 非空，写入成功入存储）；
2. **保存也带上了**（检查点的 `task_writes` 非空）；
3. 但**读取路径没走我改的分支** —— Python 的 `get_state_history` 走的是"懒序列 + `snapshot_from_handle`"，
   不经过 `get_state_history_filtered_in`，所以"子行合并"从未执行（探针在读取侧一次都没打印）。

因此本轮的工作是**打通并验证了写入链路**（新增独立 `task_writes` 字段与运行上下文存储，不动恢复语义），
并把读取侧的确切接入点定位到"Python 懒历史需要把子行传进去"。这一步留给下一轮完成，避免在收尾阶段做大改。

### `PregelTask.result`：第二次尝试与结论（2026-11-27，第 127 轮）

按第 126 轮的结论做了完整实现：新增独立的 `task_writes` 存储（不碰恢复用的 `task_results`）、superstep 后按
task 记录写入、暂停时合并、历史快照与"子行"合并。

结果：**检查点里的 `task_writes` 始终为空**。定位到两个环节之一有问题（未能区分）：

- `TaskResult.output.updates_map()` 在记录点可能已被消费为空；
- 同步执行器保存检查点的路径可能不在我改的 `save_checkpoint_via` 分支上（`fresh.rs` 另有一条）。

两次尝试都完整回退（669 测试全绿）。结论：这不是"加一个字段"的改动，需要把"任务写入"作为**一等数据**在
superstep 结算处收集，并在两条保存路径（同步/异步）上都接上。工作量与风险都超出单轮可安全完成的范围。

这是当前唯一**已知根因但未实现**的功能项，其余项都已修复或属线程模型限制。

### `PregelTask.result` 的实现结论：需要跨行关联（2026-11-27，第 126 轮）

本轮尝试实现"按 task 记录写入"，做完后仍拿不到结果，但把**真实结构**摸清了：

- 检查点 N（含 `next: [a, b]`）是在 superstep **执行之前**写入的，因此它的 `task_writes` 必然为空；
- a/b 的写入随检查点 N+1 一起落盘（N+1 的 `next` 已经空了）。

所以 `tasks[].result` 不是"同一行内的字段"，而是**行 N 的 tasks ← 行 N+1 的 task_writes** 的跨行关联。LG 能直接给出来，
是因为它把每一步的 pending writes 挂在"将被修正"的那个检查点上（`pending_writes`），快照时再合并。

结论：实现路径是「写 checkpoint N+1 时把 N 的 tasks 结果带上（或快照时按 `parent_id` 取子行合并）」，而不是本轮试的
同快照查找。试验代码已完整回退（669 测试全绿），避免留下半成品字段。

### `PregelTask.result` 为何恒为 `None`（2026-11-27，第 125 轮实测）

审计里"`PregelTask.result` 恒为 `None`"这条，本轮查清了根因：

- 引擎的 `checkpoint.task_results` 只记录**函数式 API 任务**和**暂停时**的零星结果；
  普通节点的写入是直接进 channel 的，不会按 task 记录。
- 因此快照里的 `tasks[].result` 只能拿到空。

对照实测（两个并行节点，各自写 `seen`）：

```
LG 历史 step 0: [('a', {'seen': ['a']}), ('b', {'seen': ['b']})]
扩展开    step 0: [('a', None), ('b', None)]
```

顺带确认另一处：暂停快照里任务顺序是 `[b, a]` 而 LG 是 `[a, b]`（声明顺序）。

修法需要在 superstep 执行后把每个任务的写入按 `task_id` 记进运行上下文，再随检查点持久化。这条会牵动
"恢复时不重跑已完成任务"的去重逻辑（`load_task_results`），风险不小，因此本轮只做定位、不擅自改。

### 暂停的 `values` 块形状修正（2026-11-27，第 124 轮）

第 120 轮审计的 F17（暂停时 `values` 模式多出一个重复块）已修：live 路径把 `__interrupt__` 单独作为一块上报，
而 LG 是把它**挂在该 superstep 的 values 块上**。现在 `_LiveIterator` 把"只有 `__interrupt__` 的块"折回上一块：

```
改前: [{'answer': None}, {'answer': None}, {'__interrupt__': (…)}]   ← 3 块
现在: [{'answer': None}, {'answer': None, '__interrupt__': (…)}]     ← 2 块，与 LG 相同
```

**仍剩两项（都在子图中断的传播路径上，需要引擎内改造）**：

1. 子图内中断时，LG 在暂停流里发**两块**（内层命名空间块 + 父级块），扩展只发内层。
2. 恢复时 LG 会补发内层 `('sub:…', {'ask': …})`，扩展只有父级块。

状态本身是对的（暂停/恢复/历史都一致），只是流里的命名空间块缺失。

### 超时诊断信息修正 + `AsyncStateGraph.astream` 缓冲确认（2026-11-27，第 123 轮）

**诊断修正**：idle 超时的错误消息此前在同步构建器路径上报 `0.000s`（数值丢失，因为 Python 侧看门狗抛的
`TimeoutError` 没带时长）。现在标记串带上秒数（`langgraph-rs idle timeout 0.12`），映射层解析后：

```
改前: Node 's' exceeded its idle timeout of 0.000s without making progress (elapsed: 0.000s).
现在: Node 's' exceeded its idle timeout of 0.120s without making progress (elapsed: 0.120s).
```

两个引擎现在一致。

**`AsyncStateGraph.astream` 仍是收集式**（两个 0.2s 节点在 0.41s 一起到）。已确认不是包装层问题：直接调用原生
`_AsyncStreamChunks` 也是缓冲的，说明异步执行器里"spawn 运行 + 通道"的组合在当前 tokio 运行时下不会交错推进
（同步路径用独立工作线程，因此正常）。修它需要给异步运行单独的运行时/线程模型，风险大于收益，因此记录在案。

### `subgraphs=True` 的流也逐 superstep（2026-11-27，第 122 轮）

第 120 轮审计的 F1 已修：`subgraphs=True` 的流现在走 live 路径。关键是命名空间包装放在**转换层**而不是
Python 侧 —— 给 `PyLiveStream` 增加 `wrap_namespaces` 标志（`stream_live(wrap_subgraphs=...)`），
`poll` 时用 `convert_stream_chunks_with(..., wrap)`。这样 `stream_events` 等内部消费者不受影响。

实测（子图内 0.2s 节点 + 父图 0.2s 尾节点）：

```
扩展: [(('sub:…',), {...}), ((), {...}), ((), {...})]  到达 [0.20, 0.40, 0.40]
LG:   同样三块                                        到达 [0.21, 0.42, 0.42]
```

形状与增量交付都与 LG 一致。

**仍剩一项**：`AsyncStateGraph.astream` 还是收集式（两个 0.2s 节点在 0.41s 一起到）—— 原生异步执行器的
sink 似乎也在运行结束时才回调，需要单独排查。

### `astream` 不再阻塞事件循环，且可取消（2026-11-27，第 121 轮）

第 120 轮审计里最严重的一项已修：`astream` 的每一块 `next()` 现在都跳到工作线程执行
（`loop.run_in_executor`），不再在事件循环里忙等。

实测（两个各 0.2s 节点，同时跑一个 20Hz 心跳协程）：

```
改前: 心跳在 0.4s 运行期间只跳 1 次（事件循环被冻结），取消无效（CancelledError 不抛出）
现在: 心跳跳 7 次，块到达 [0.20, 0.40]，取消在 0.1s 立即生效（0 块已交付）
```

同时补上 `close()` / `aclose()`：`_LiveIterator` 现在可关闭，`astream` 的 `finally` 会关闭底层迭代器。

*注：关闭/放弃流只是让调用方停止消费，运行本身仍会在工作线程里跑完（线程无法中断）；要中止运行需要用
`invoke` 路径上的 `RunControl.abort()`。*

### 第七次审计：流式/超时/取消的剩余缺口（2026-11-27，第 120 轮）

又做了一轮针对**最近改动区域**（流式交付、超时、取消、`RunControl`）的审计：71 项检查，40 通过、22 项功能差异、
6 项表面差异。**同步、非子图、单/多模式的流式路径是扎实的**（块序列、增量交付、中断、流后状态/历史全部与 LG 一致），
差异集中在四处：

| 缺口 | 影响 |
|---|---|
| `astream` 在调用方事件循环里同步迭代 live 流 | **阻塞事件循环**（20Hz 心跳在 0.4s 运行期间只跳 1 次），且**取消无效** |
| `subgraphs=True` / `AsyncStateGraph.astream` 仍是收集式 | 流式退化为批量交付 |
| 放弃/关闭流不停止运行；`_LiveIterator` 没有 `close()` | 弃流后节点继续跑并写检查点 |
| 多模式下 `debug` + `tasks` 会**丢掉全部 `tasks` 块**；单元素模式列表丢失 `(mode, payload)` 包装 | 静默数据丢失 / 形状错误 |

本轮先修了两项安全的小问题（`control=` 关键字重复导致的 `TypeError`、`debug`/`tasks` 走 eager 避免丢块），
其余（尤其 `astream` 阻塞与取消）需要改动异步门面的迭代方式，本轮尝试后发现牵动多个事件流测试，故回退并记录。

### `idle_timeout` 在同步构建器上已修复（2026-11-27，第 119 轮）

第 118 轮定位的缺口已补上：同步构建器的 Python 节点包装现在会在节点返回协程时套用同一个 Python 侧 idle
看门狗（`node_from_python_with_idle` + `wrap_awaitable_idle`）。

实测两个引擎现在完全一致：

| 场景（`async def` 节点，`idle_timeout=0.12`） | `StateGraph` | `AsyncStateGraph` |
|---|---|---|
| 静默 0.4s | `NodeTimeoutError` ✓ | `NodeTimeoutError` ✓ |
| 每 50ms `custom()` 报告进度 | 正常完成 ✓ | 正常完成 ✓ |

实现要点：看门狗必须装在 Python 里（协程跑在私有事件循环上，引擎无法直接计时）；包装器要**调用**才得到协程
（第一版忘了调用，直接报 `TypeError`）。

### 超时语义：`idle_timeout` 在同步构建器上不生效（2026-11-27，第 118 轮实测）

第 106 轮审计提到的"异步执行器超时语义不同"，本轮测清楚了，方向**与预期相反**：

| 场景（`async def` 节点） | `StateGraph`（同步引擎） | `AsyncStateGraph`（异步引擎） |
|---|---|---|
| `run_timeout=0.15`，节点 0.4s | `NodeTimeoutError` ✓ | `NodeTimeoutError` ✓ |
| `idle_timeout=0.1`，节点静默 0.4s | **无超时，正常运行** ✗ | `NodeTimeoutError` ✓ |

即：**异步引擎是对的，同步引擎缺了 `idle_timeout`**。原因是 idle 看门狗用 Python 包装实现
（`wrap_async_idle`，因为协程跑在私有事件循环上），而这个包装只装在异步构建器
（`async_builder.rs`）里；同步构建器的通用 Python 节点包装
（`convert::node_from_python`）没有装。

修法需要把超时策略传进 `node_from_python` 并在返回协程时套用同一个包装 —— 涉及同步构建器的节点包装签名，
本轮时间不够做完整验证，因此**记录为已知差异**（规避方式：需要 `idle_timeout` 的异步节点用 `AsyncStateGraph`）。

### 取消传播：`RunControl.abort()` 真正生效（2026-11-27，第 117 轮）

第 106 轮审计指出"`task.cancel()` 后节点仍跑完并写检查点"。本轮发现**更基础的缺陷**：
`RunControl.abort()` 一直是**空操作** —— 引擎从不检查 `is_aborted()`（文档却写着"在下一个 superstep 边界中止"）。

修法两步：

1. 循环开头检查 `is_aborted()`：已中止则不再开始新的 superstep。
2. `run_superstep` 返回后、写检查点前再检查一次：**丢弃本次 superstep 的结果**，线程停留在上一个检查点。

门面在 `ainvoke` 的 await 被取消时调用 `abort()`，实测：

```
两个节点（a 立即完成、b 耗时 0.5s），0.15s 时 cancel
改前: 节点全跑完，state={'value': 2}, next=()
现在: 节点 b 仍在飞行（线程无法被杀），但结果被丢弃
      state={'value': 1}, next=('b',)   ← 停留在上一个检查点，可从 b 继续
```

与 LG 的差异：LG 能真正中断在飞的节点（协程取消），我们的同步执行器在**工作线程**里跑，无法中断；但我们保证
"已完成的 superstep 一致保留、未完成的丢弃"，比"跑完并提交"正确得多。

### `astream_events(v2)` 支持异步节点（2026-11-27，第 116 轮）

第 106 轮异步审计列出的 `astream_events(version="v2")` + `async def` 节点直接报
`Cannot run the event loop while another loop is running` —— 因为 v2 的收集器在**调用方事件循环线程**上驱动
同步 `stream()`。现在收集器放在工作线程，通过线程安全队列桥接到异步迭代器，事件循环不再被占用。

实测（`async def` 节点 + `version="v2"`）：

```
扩展: on_chain_start, on_chain_start, on_chain_stream, on_chain_end, on_chain_stream, on_chain_end
LG:   on_chain_start, on_chain_start, on_chain_stream, on_chain_end, on_chain_stream, on_chain_end
```

**`subgraphs=True` 的流式仍是 eager**：本轮尝试接入 live 时又出现一批形状/事件解析问题
（`EventTaskRunsTest`、`SubgraphUpdateChunkTest` 等 6 个测试），说明命名空间包装的语义还没吃透，因此保持
eager（形状正确、只是非增量）。

### 多模式流也逐 superstep 交付（2026-11-27，第 115 轮）

第 114 轮的"有界接入"本只覆盖单模式，本轮把范围扩到**多模式**（`["updates","values"]` 等）：

```
两个各 0.2s 节点
单模式 updates : [0.20, 0.40]
多模式 [updates, values]: [0.00, 0.20, 0.20, 0.40, 0.40]   ← 逐 superstep
```

扩范围时修掉两个真问题：

1. **`values` 去重过猛**：第 113 轮的去重只看载荷，会把「相邻两个 superstep 恰好产生相同 values」误删。现在按
   `(step, 载荷)` 一起去重，只丢弃**同一步**里的重复块。
2. **`Command` 输入必须走 eager**：live 路径不处理 `Command(resume=...)`（会把恢复当成普通输入），实测"恢复后仍
   处于暂停状态"。现在带 `Command` 的流一律走 eager 路径，恢复语义正确。

剩下的边界：`subgraphs=True` 的流仍走 eager —— live 路径不给父图块加 `((), payload)` 包装，形状不一致。

### `stream()` / `astream()` 现在逐 superstep 交付（2026-11-27，第 114 轮，已接线）

第 109-113 轮铺垫的 live 流**终于接上了门面**，并且采用**有界接入**：只在「单个 `stream_mode`（`updates` /
`values` / `custom` / `debug` / `tasks`）+ 不带 `subgraphs` + 无 `version="v2"`」时走 live，其余保持 eager
（多模式、子图、v2 的形状另有测试覆盖）。

实测（两个各 0.25s 节点）：

```
stream(stream_mode="updates")  块到达 [0.25, 0.50]   ← 逐 superstep（LG 行为）
astream(stream_mode="updates") 块到达 [0.21, 0.41]   ← 异步同样生效
多模式 ["updates","values"]     仍是 [0.50 × 5]      ← 有界接入的边界，已在文档标注
带 checkpointer 的流式运行      history 3 行 + 值正确  ← 与 LG 完全一致
```

返回值同时兼容两种用法：`for chunk in ...` 逐块（真流式），以及 `len(...)` / `[i]`（内部会按需拉完），所以既有
调用方无需改动。

### live 流的块形状已与 eager 一致（2026-11-27，第 113 轮）

第 112 轮剩的"多一个中间块 + `__interrupt__` 是 tuple"本轮修掉了，两处都在 `PyLiveStream.poll`：

1. **中断块聚合成一个条目**：interrupt 块先攒着，等流结束时一次性转换，得到与 eager 相同的单个
   `{"__interrupt__": [...]}`（此前每块单独转换）。
2. **重复的 `values` 块丢弃**：eager 是"整段列表一起转换"，天然会把连续相同的 values 载荷合并；live 逐块转发，
   于是多出一个空 `{}`。现在用"上一个 values 载荷"做去重。

实测（中断图，`stream_mode="values"`）：

```
eager: [{'__empty__'}, {'__interrupt__': [...]}]
live : [{'__empty__'}, {'__interrupt__': [...]}]   ← 完全一致
非中断图（两节点）也逐块一致
```

**门面接线仍未保留**：接上后仍有一批问题（`subgraphs=True` 的 debug 块形状、`astream` 与 live 迭代器的交互、
lifecycle、命名空间前缀）。下一轮改用**有界接入**：只在"单 stream_mode 且无 subgraphs"这个最常见、形状已验证一致的
场景走 live，其余继续走 eager —— 把影响面限制在可验证的范围内。

### live 流的中断/lifecycle 已补齐（2026-11-27，第 112 轮）

第 111 轮剩余两项，本轮都修了：

1. **暂停时的 lifecycle 事件**：eager 路径在运行结束后会记录 `pending` 中断事件（`record_interrupt` +
   `store_lifecycle`），live 路径完全没有 —— 所以 `stream` 上的 `GraphCallbackHandler` 收不到中断通知。现在工作线程
   在运行结束后做同样的事（并且因为 live 的 `RunConfig` 上没有 sink，会自建一个）。实测：live 运行后
   `take_lifecycle_events()` 从 0 变成 1。
2. **中断块形状**：`PyLiveStream.poll` 改用与 eager 完全相同的转换函数（不再自行包装命名空间），避免把
   `stream_events(v3)` 的原始事件口径改坏。

**门面接线仍未保留**：live 路径在一个中断图上会比 eager 多一个中间 `{}` 块，`__interrupt__` 也还是 tuple 而非 list；
这两处是转换/去重细节，需要下一次专注处理。`stream_live` 现在的实测能力已经是完整的：逐 superstep、正确落盘、
config 完整转发、暂停事件可上报。

### `stream_live` 现在能接收完整 config（2026-11-27，第 111 轮）

第 110 轮列的 4 项前置条件，本轮修掉两项：

1. **`stream_live` 接受并解析完整 config**（`config=` 参数，走 `config_from_py`）—— 此前它自建一个最小
   `RunConfig`，**丢掉调用方的 callbacks / metadata / durability / recursion_limit**。转发 config 后，
   `stream` 路径上的 lifecycle 事件与回调才有机会触发。
2. **`subgraphs` 的越权修正已回退**：把命名空间包装塞进 `PyLiveStream` 会破坏 `stream_events(v3)` 的原始
   事件口径（`debug` 变成空串），事件管道自己已经处理命名空间，live 流不该再包装。

**仍剩两项**（门面接线未保留，树保持全绿）：

- 中断块的**形状**：live 路径比 eager 多出一个中间 `{}` 块，且 `__interrupt__` 从 list 变 tuple —— 需要统一。
- lifecycle 事件在 `stream` 上仍不派发（`invoke` 上正常）—— 需要把"运行结束后 drain"接到 live 迭代器。

修完这两项即可安全接线；`stream_live` 本身现在已经是"config 完整 + 逐 superstep + 正确落盘"的了。

### `stream_live` 的线程解析缺陷已修（2026-11-27，第 110 轮）

本轮先把 `stream_live`（后台线程 + 通道的原生流）修对：它此前从 `thread_id` 参数**只读顶层**，传
`{"configurable": {"thread_id": ...}}` 时会拿到空字符串，于是**运行时什么都没持久化**（`get_state_history` 为空）。
改用 `config_from_py` 解析后：

```
两个各 0.2s 节点 + checkpointer
原生 live 到达: [0.25, 0.48]   ← 逐 superstep
history: 4                     ← 正常落盘
```

这就是门面接上 live 之前必须修的底层缺陷（第 109 轮没发现，因为当时没带 checkpointer 测）。

**门面接线仍未保留**：接上后还剩 4 个具体问题（都不难，但需要成组处理）：

1. 缺 `thread_id` 时的错误类型：live 走工作线程，异常被包成 `RuntimeError`；应为 `ValueError`（门面需要在启动前校验）。
2. `subgraphs=True` 的 debug 块形状（3 元组）与 `updates` 命名空间前缀在 live 路径上不一致。
3. 中断/恢复时 lifecycle 事件的派发时机（应在运行结束后派发，而不是启动时）。
4. `stream(thread_id=...)` 直接传关键字参数时的线程解析路径要与 config dict 等价。

修完这 4 项再接线；接线后 `stream()` 与 `astream()` 同时是真流式。

### `stream()` 缓冲的真因：门面提前收集（2026-11-27，第 109 轮）

上一轮判断"引擎把块攒到运行结束才交"，**这个判断是错的**。本轮把门面的 `stream()` 临时接到原生
`stream_live`（后台线程跑图 + 通道转发）后，实测块到达时间立刻变成：

```
两个各 0.3s 节点，stream_mode="updates"
改前（门面收集）: [0.60, 0.60]
改用 stream_live: [0.30, 0.60]   ← 与 LG 完全一致
```

所以引擎**本来就是每个 superstep 结束就回调 sink**，缓冲发生在门面：它调用 `_inner.stream(...)`（返回收集好的
列表）后一次性交出。原生 `stream_live` 早就存在且计时正确。

**为什么本轮没有保留这个改动**：切换后 8 个测试失败，都是可以修的兼容问题，但需要一次完整的改造：

1. `stream()` 的返回类型从"列表"变成"迭代器"，有些调用方直接下标取值 —— 需要 `_LiveIterator` 支持
   `__getitem__`/`__len__`（或在测试里改成 `list(...)`）。
2. "缺 thread_id 时报错"变成惰性（迭代时才报）—— 与 LG 一致（LG 的 `stream` 也是生成器），但我们的测试期望
   调用即报错，需要同步调整。
3. 多模式下 `custom` 块在 live 路径上缺失，需要先查清（`lg.custom()` 走 pending sink 的那条路径）。

这三项是下一轮的第一件事，完成后 `stream()` 就是真流式（异步路径同样受益，因为 `astream` 的迭代器已经就位）。

### `stream()` 的块是"跑完再给"（2026-11-27，第 108 轮实测）

第 107 轮列出的三个前置条件里，第 1 条（增量交付）本轮做了实现：`astream` 现在返回真正的异步迭代器
（`_AsyncStreamChunks`，基于 mpsc 通道，`__aiter__`/`__anext__`），`AsyncStream` 也支持原生迭代器。

**但实测块仍然在运行结束时才到达**，原因不在包装层，而在**发射时机**：

```
两个各 0.2s 的节点，stream_mode="updates"
扩展（同步）: 块到达 [0.60, 0.60]
扩展（异步）: 块到达 [0.42, 0.42]
LG        : 块到达 [0.30, 0.60] / [0.20, 0.40]   ← 每个 superstep 结束就交一块
```

也就是说引擎把该轮产生的块攒到运行结束才交给 sink（`emit`/collector 的排空时机），包装层只是"如实转发了一个迟到的
序列"。这解释了为什么 parity 一直没发现：parity 比对的是块的**内容序列**，不是**到达时刻**。

修法在第 9 轮会从引擎的发射路径入手（superstep 结束即 drain 并立即回调），而不是继续改包装层。

### 异步统一的尝试与结论（2026-11-27，第 107 轮）

尝试把 `StateGraph.compile()` 的产物改成"同时支持同步与异步"（LG 的形态）：新增
`AsyncStateGraph::from_sync(builder)`（把同步节点包进 async 块、复制 schema/边/条件边/重试/缓存/超时/元数据等全部
配置），`PyCompiledGraph` 因此带上了 `_async_inner`（同一份定义编译出的原生异步图）。

**结果是：暂不接线。** 把门面的 `ainvoke`/`astream` 指向孪生图后，两个关于超时的测试立刻失败 —— 异步执行器的
超时语义与同步执行器不同（run/idle 超时的判定与计时起点不一致）。也就是说，问题不在"有没有原生异步入口"，而在
**两个执行器的语义还没有收敛**。切换前必须先让异步执行器与同步执行器在以下三点对齐：

1. **分块增量交付**：实测两条路径都是"跑完再吐"（两个 0.3s 节点，块到达时间 `[0.63, 0.63]`，LG 是 `[0.32, 0.63]`）。
   `astream` 目前把块收进 `Vec` 后一次性返回，要改成 mpsc + 异步迭代器。
2. **取消向节点传播**：`task.cancel()` 后节点仍跑完并写检查点；LG 会真正中断（`values` 保持原值、`next` 保留）。
3. **超时语义**：异步执行器的 run/idle 超时行为与同步执行器不一致（本轮实测暴露）。

孪生图现在挂在 `compiled._inner._async_inner`（可用），但门面仍走同步引擎，因此**行为与上一轮完全相同**，只是暴露了
通向统一所需的接口。上述三点完成后再切换，才能真正对齐 LG。

### 异步 API：`AsyncStateGraph` 是真异步，`StateGraph.compile()` 是线程模拟（已知差异，2026-11-27）

第六次审计专攻**异步 API**（此前从未审计）。结论分两半：

**真异步路径（`AsyncStateGraph(...).compile()`）**：`_inner` 是原生 `AsyncCompiledGraph`，`ainvoke` / `astream` /
`aget_state` / `Command(resume=...)` 都是原生实现。实测：`astream` 逐块返回（0.21s 的节点在 0.21s 时而不是运行结束时
给块）、异步节点、同步节点、混用、取消后同线程可复用、并发 `gather` 全部正常。

**线程模拟路径（`StateGraph(...).compile()` 与 `create_react_agent`）**：`_inner` 是同步 `CompiledGraph`，门面的
`a*` 方法走 `asyncio.to_thread(同步调用)`。结果是**结果正确但异步语义不完整**：

| 症状 | 原因 |
|---|---|
| `astream` 仍然"攒完再吐"（两块同时到达） | `to_thread` 里跑完整个同步流后再逐个 yield |
| `asyncio.wait_for` / `task.cancel()` 后节点**仍然跑完并写检查点** | 线程里的同步运行无法被取消 |
| `async def` 节点里 `await model.ainvoke()` 的 token 不出现（`stream_mode="messages"` 为空） | 模型回调在第三个线程触发，Rust 引擎的线程内上下文不在那里 |
| `astream_events(version="v2")` + 异步节点直接报 `Cannot run the event loop while another loop is running` | v2 在调用方事件循环线程上驱动同步 `stream()` |
| `aget_state_history` 是协程而非异步迭代器 | 同样是 `to_thread` 包装 |

**规避方式**：需要真异步语义时用 `AsyncStateGraph`（或等下一轮把 `StateGraph.compile()` 统一到异步实现）。
LG 只有一个 `StateGraph`，编译产物同时支持同步与异步 —— 这是两边结构的差异，不是参数差异。

### 子图中断的两种接入方式（已知差异）

在父节点里**手工调用**带 checkpointer 的子图（`child.invoke(...)`，且不传 config）：LG 会把父图的 config 传进嵌套
运行，因此子图的中断能让**父图暂停**（`next=('child',)`），`Command(resume=...)` 后子图拿到答案；扩展没有把
config 传入嵌套运行，所以会抛 `ValueError: Checkpointer requires ... thread_id`（2026-11-27 之前是静默跑完并给出
`answer: None` —— 现在至少是显式失败而不是静默错误结果）。

**改用 `parent.add_node("child", compiled_subgraph)` 这种接入方式则完全一致**（暂停载荷、`next`、任务路径、恢复都
相同），这也是 LG 文档推荐的形式。

根因是嵌套调用不继承父运行的 config（LG 用 contextvar 传递）。要做对需要命名空间感知的 resume 管道，牵涉子图独立的
checkpoint 线程，风险大，故记录为已知差异。

另外：`interrupt()` 写在条件边路由器里，LG 允许（暂停 + 恢复），扩展会在运行时拒绝 —— 这是未文档化的边缘用法。

### 聚合通道的默认值出现在开局检查点（2026-11-27）

LG 的聚合通道（`Annotated[list, operator.add]`）在**任何写入之前**就持有默认值 `[]`，所以快照里能看到
`{'trail': []}`。扩展此前只存被写过的通道，`values['trail']` 会 `KeyError`。

现在通道带上了默认值（`append`/`messages`/`topic` → `[]`，`add` → `0`，可通过 `Reducer::with_default`
指定），`source="input"` 与 `source="loop"`（进入节点前）两个检查点都会带上它。

**仍有一处差异**：最后一次 `invoke` 的返回值与随后那个（step 最大的）检查点仍不含默认值 —— 因为它们由运行结束时
的状态直接序列化，而通道"从未被写入"时状态里没有该键。要完全对齐需要让通道在"空"时也返回默认值，会影响
`update_state`、节点输入等更多语义，风险不成比例，因此记录在案。

### drop-in 压测发现的两处真实缺陷（2026-11-27，第五轮审计）

第五次审计换了视角：写四个"真实用户会写"的程序（ReAct 智能体多轮记忆、supervisor + 子图 handoff + `Send` 扇出 +
返回列表的条件边、四种流模式、持久化/fork/durability/中断），只改根包名跑两边。**结论：除了根包名，一处 API 都没
需要改** —— 导入路径、调用形式、参数名全部通用。但发现两处真实缺陷：

| # | 问题 | LG | 改前 | 现在 |
|---|---|---|---|---|
| 1 | 有 checkpointer 但 `invoke` 没给 `thread_id` | `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id` | **静默成功、什么都没存**（数据丢失） | 同样的 `ValueError` |
| 2 | `create_react_agent` 产出的 `AIMessage.additional_kwargs` | `{}` | `{'invalid_tool_calls': [], 'usage_metadata': None}` | `{}` |

第 2 条的原因：模型输出被 `model_dump()` 成扁平 dict 存进 `messages` 通道，复活时两个非构造参数落进
`additional_kwargs`。现在改用与其他 LangChain 对象相同的 `{"lc": 1, ...}` 封装；同时把 `tool_calls` 的读取改成
两种形状都支持（`message_field` 辅助函数），因为状态里两种形状本来就会共存。

### `Interrupt` 对象语义与按键恢复错误类型（2026-11-27）

审计还发现两处 API 契约差异，已修复：

| 项 | LG | 改前 | 现在 |
|---|---|---|---|
| `Interrupt` 相等性 | dataclass 字段相等（`Interrupt("v") == Interrupt("v")` 为真，可 `in` 列表） | 身份比较（为假） | 字段相等 |
| `Interrupt` 可哈希性 | `TypeError: unhashable type: 'Interrupt'` | 可哈希（身份哈希） | 同样不可哈希 |
| `Interrupt` repr | `Interrupt(value='v', id='placeholder-id', response_schema=None)` | `Interrupt(id="", value='v')` | 与 LG 一致 |
| 多中断并行时用普通值 resume | `RuntimeError`（含文档链接的提示） | `ValueError` | `RuntimeError`，文案一致 |

`dataclasses.is_dataclass(Interrupt(...))` 仍为 `False`（原生 pyclass），这是已知差异 —— 其余 dataclass 行为
（`==`、`in`、不可哈希、repr）已对齐。

### `interrupt(response_schema=...)` 校验归属重做（2026-11-27，第四轮审计）

第四次独立审计专攻 **human-in-the-loop（interrupt/resume）**。核心流程（单次/连续/并行中断、工具内中断、编译期
`interrupt_before/after`、`update_state`、历史行、各种错误路径）全部一致，但 `response_schema` 的校验归属错了：

| 场景 | LG | 改前 | 现在 |
|---|---|---|---|
| 非法 resume（pydantic 模型） | `pydantic.ValidationError`，`is_invalid_resume(e)` 为真，**保持暂停** | `RuntimeError`，标记为假 | 与 LG 一致 |
| pydantic 可强转的值（`{"approved": 1}`） | 强制转换后通过 | 被 JSON Schema 类型检查拒绝 | 通过 |
| 原始 JSON-Schema `dict` | **完全不校验**（原样接受） | 按 schema 校验并拒绝 | 原样接受 |
| 工具内 `interrupt` + 非法 resume | 抛出、暂停保留、可再答 | **被当成工具错误**，图直接跑完 | 抛出、暂停保留、可再答 |

根因：引擎先用自研的 JSON-Schema 子集校验 resume（`Error::other` → `RuntimeError`），pydantic 永远轮不到。现在
Python 侧改走 `interrupt_unvalidated`，由 pydantic 独占校验（非法时打上 LG 同名的
`_langgraph_invalid_resume` 标记），`ToolNode` 见到该标记直接上抛而不是转成 `ToolMessage`。

### `add_messages` 可调用、输入归一化、序列化往返（2026-11-27，第三轮审计）

第三轮审计剩余三项，均已修复：

| 项 | LG | 改前 | 现在 |
|---|---|---|---|
| `add_messages` | 可调用函数（`add_messages(left, right)`） | 导出的是**字符串** `"messages"`，调用报 `TypeError` | 真函数（支持按 id 合并/替换、`RemoveMessage`、`__remove_all__`、缺 id 时报错），并带 `__channel_spec__` 标记供建图识别 |
| 输入归一化 | 单条消息 → 包成 list；`("user", "hi")` 元组 → `HumanMessage` | 原样存进 channel（存成裸消息 / 裸 list） | 与 LG 相同（`_set_messages_objects(False)` 的旧模式仍跳过归一化） |
| `JsonPlusSerializer.loads_typed` | 还原成 `HumanMessage` 等对象 | 返回 dict | 通过 `langchain_core.load` 还原（往返后类型与 id 都保留） |

### 消息变更语义（2026-11-27，第二轮审计）

第三次独立审计专攻消息语义，又发现 4 处真实差异，都已修复：

| 场景 | LG | 改前 | 现在 |
|---|---|---|---|
| 引擎自己写入的消息（`ToolNode`、dict 输入、`update_state`） | 有 id | **无 id**，导致"原样返回"时重复 | 由 reducer 补 id |
| `RemoveMessage(id)` + 同 id 新消息 | `[a2, b]`（原位替换） | `[b, a2]`（先删后追加） | `[a2, b]` |
| 同一 update 里重复删除同一 id | `[b]`（幂等） | `ValueError` | `[b]` |
| `RemoveMessage("__remove_all__")` | 清空 | `ValueError: ... doesn't exist` | 清空 |

修法：reducer 改成 LG 的顺序 —— 先收集删除意图、再应用写入（同 id 原位替换）、最后只删除**没有被本次写入替换**的旧消息；
重复删除幂等；`__remove_all__` 走单独分支。消息 id 在 reducer 入口统一补齐（覆盖引擎侧写入路径）。

### `add_messages` 按 id 去重 + 消息自动分配 id（2026-11-27）

调试 `pre_model_hook` 时发现更底层的问题：**`add_messages` 从来不按 id 去重**。LG 会给每条消息分配 `id`，并在
reducer 里按键替换，所以"节点把当前消息列表原样返回"不会产生重复；扩展此前是纯追加：

```
输入 [hi]，节点返回 state["messages"] + [new]
改前: ['hi', 'hi', 'new']        ← 重复
改后: ['hi', 'new']              ← 与 LG 相同
```

两处原因，都已修：

1. 消息编码时如果 `id` 为空，就按 LG 的做法**分配一个 uuid 并写回对象**（所以"原样返回"能带上同一个 id）
2. reducer 读 id 时同时支持两种形状 —— 顶层 `id`（引擎的 JSON 形式）和 `kwargs.id`（LangChain 序列化形式）；
   此前只看顶层，因此真实消息对象永远匹配不到

### 审计后续三项（2026-11-27）

1. **`context_schema` 接受裸类**：LG 允许 `context_schema=Ctx` 传任意类；扩展此前抛
   `TypeError: context_schema must be a JSON schema dict or a pydantic/TypedDict class`。现在无 schema 可推导时
   按"空 schema"接受（LG 对裸类也不做校验），`runtime.context` 仍是原对象。
2. **重复 `add_node` 立即报错**：LG 在**注册时**抛 `ValueError: Node \`a\` already present.`，扩展此前要等到
   `compile()`。现在与 LG 同时机。
3. **`add_node(metadata=..., trace_policy=...)`**：LG 支持这两个关键字；扩展此前直接
   `TypeError`。`metadata` 会存进节点元数据，`trace_policy` 接受但不参与（引擎不区分 trace 策略）。

另外修掉 `get_config()` 的差异：LG 只在运行**带 tags** 时才返回 `tags` 键，扩展此前总是带上。

### 对抗性审计发现的四处行为差异（2026-11-27）

让独立审计逐条验证文档里的"已对齐"声明，查出四处**真实行为差异**（非文案）：

| # | 问题 | 改前 | 现在 |
|---|---|---|---|
| 1 | `tools_condition` **判断反了** | 真实 `AIMessage` → `__end__`，dict → `tools`（移植的图会静默跳过工具节点） | 与 LG 相同：看最后一个消息的 `tool_calls` 属性；空消息抛 `ValueError` |
| 2 | `stream()` 默认模式 | `values`（还多发一个初始整状态 chunk） | **`updates`**（LG 的默认） |
| 3 | `RetryPolicy()` 默认重试规则 | 空谓词 = 全部重试（`ValueError` 也重试 3 次） | LG 的 `default_retry_on`：连接类错误重试，`ValueError`/`TypeError`/`OSError` 等不重试 |
| 4 | `create_react_agent` 默认 schema + `ValidationNode` 返回 **dict 消息** | `messages[-1].content` 直接 `AttributeError` | 都是 LangChain 消息对象（`AIMessage`/`ToolMessage`，带 `status`） |

`create_react_agent` 的原因是没有把它的 `messages` 通道标记为消息通道；`ValidationNode` 是直接返回了 dict。

**另一处相关**：`context_schema` 仍只接受 pydantic/TypedDict/JSON schema，裸类（LG 接受）会被拒绝 —— 待处理。

### 空闲超时（idle_timeout）语义修正（2026-11-27）

独立审计发现的真实缺陷：`TimeoutPolicy(idle_timeout=...)` 在**阻塞执行路径**上被当成**整体运行时限**，
于是"一直在报进度"的节点也会被杀掉：

```
节点每 20ms custom() 一次，总耗时 0.3s，idle_timeout=0.05
改前: NodeTimeoutError: exceeded its run timeout of 0.050s
改后: 正常完成（空闲看门狗只在真的没进度时触发）
```

两处修正：

1. `effective_timeout()` 只认 `run_timeout` —— 阻塞路径看不到进度，拿空闲值当整体时限会误杀正常节点
2. 异步看门狗 `run_with_idle_timeout` 改用**合并后**的进度计数（`custom()` / `message()` 在同步侧
   自增，此前看门狗只读异步侧），并且失败时返回 `IdleTimeout`（文案为 LG 的
   `exceeded its idle timeout of 0.050s without making progress`）

### `get_delta_channel_history` 的语义（2026-11-27）

引擎按检查点快照**完整值**，没有 LG 的增量 blob 存储，所以：`writes` 为空、`seed` = 该通道在检查点里的值。
之前 `seed` 固定返回 `0`（类型都不对）。现在按"把 writes 叠加到 seed 上"的读取方式，两边都能还原出正确状态 ——
parity 场景 `delta_history` 就是按这个语义比对（LG 把值拆成 seed + writes，我们全放 seed）。

### 内部协议类型与工具输出增量（2026-11-27）

类扫描从 5 处收窄到 **1 处**：

| 项 | 变化 |
|---|---|
| `PregelExecutableTask` | 按 LG 的 dataclass 字段补齐（13 个字段，含 `cache_key` / `path` / `retry_policy` / `subgraphs` / `timeout`） |
| `ChannelProtocol` | 补上（`ValueType` / `UpdateType` / `get` / `update` / `checkpoint` / `from_checkpoint` / `consume`） |
| `NodeBuilder` | 补上 10 个方法（`subscribe_to` / `read_from` / `write_to` / `meta` / `add_cache_policy` / `add_retry_policies` / `set_timeout` / `subscribe_only` / `do` / `build`） |
| `ToolRuntime.emit_output_delta(delta)` | **原生实现**：把增量推到 `stream_mode="tools"` 通道（`tool-output-delta` 事件）；不在运行期时静默无操作，与 LG 的"无 writer"行为一致 |

唯一剩余：`RetryPolicy` 的 `count` / `index`（LG 是 NamedTuple，扩展是原生类；字段访问与重试行为实测一致，
不为此重写类型）。

### 编译图就是 `Pregel`、通道协议补全（2026-11-27）

**1. `isinstance(compiled, Pregel)` 现在是 `True`。** LG 的编译图是 `Pregel` 的子类；扩展此前没有这个关系。
`Pregel` 补成拥有 LG 的 69 个公开成员的类型标记，并把 `CompiledStateGraph` **虚拟注册**为它的子类
（`ABCMeta.register`）—— 不进入 MRO，所以原生方法的转发不受影响（第一版直接继承，把
`clear_cache` / `get_subgraphs` 等遮蔽成了 `NotImplementedError`，被测试逮到）。

**2. `BaseChannel` 协议补全**：`get` / `update` / `is_available` / `consume` / `finish` / `checkpoint` /
`from_checkpoint` / `copy`。空通道的 `checkpoint()` 返回 `MISSING`（与 LG 一致），`copy()` 因此可用。

（后续章节补齐了那 5 处：`emit_output_delta`、`PregelExecutableTask`、`ChannelProtocol`、`NodeBuilder`；
现在只剩 `RetryPolicy.count/index` 一处，见上文。）

### 最后一批小成员（2026-11-27）

| 项 | 变化 |
|---|---|
| `TimeoutPolicy.coerce(value)` | 补上（数字 / `timedelta` / `TimeoutPolicy` → 规范化策略，`None` → `None`） |
| `JsonPlusSerializer.with_msgpack_allowlist` / `with_allowlist` | 补上（返回新的序列化器；无可加项时返回自身） |
| `ToolNode.name` / `tools_by_name` | 补上类级声明（LG 是 dataclass 字段，`dir()` 里能看到） |
| `checkpoint.base.DeltaChannelHistory` | 改为 TypedDict（`writes` / `seed`） |

**类扫描剩下的差异（8 处，均在内部协议层）**：`Pregel`（含 69 个成员，主要是 `Runnable` 组合）、
`NodeBuilder`、`BaseChannel` / `ChannelProtocol`（自定义通道协议）、`PregelExecutableTask`、
`RetryPolicy` 的 `count`/`index`（LG 是 NamedTuple，我们是原生类；字段访问与重试行为已实测一致）、
`ToolRuntime.emit_output_delta`。

### 小类型与运行时辅助（2026-11-27）

按类扫描继续补齐：

| 项 | 变化 |
|---|---|
| `StateUpdate` / `CacheKey` | dataclass → **NamedTuple**（可解包、`_fields`） |
| `DeltaChannelHistory` | 补上（TypedDict：`writes` / `seed`） |
| `langgraph_rs.types.BaseCheckpointSaver` | 之前是 `Any`，现在是真类（24 个成员） |
| `channels.base.EmptyChannelError` | 补上 re-export |
| `Runtime.merge` / `override` / `patch_execution_info` | 补上（语义与 LG 一致，未设 `execution_info` 时 `patch` 抛 `RuntimeError`） |
| `Runtime.drain_requested` / `drain_reason` | 补上（读 `control`） |
| `RunControl.request_drain` / `drain_requested` / `drain_reason` | 原生类补上（Rust 侧加了 `drain_reason` 存储） |
| `ExecutionInfo.patch` | 补上 |
| `BaseUser` | 从普通 Protocol 属性改为 `@property` 协议（`identity` / `display_name` / `is_authenticated` / `permissions`） |

### `ToolNode` / `ValidationNode` 是 `Runnable`（2026-11-27）

LG 的 `ToolNode` 是 LangChain `Runnable`，因此可以放进 LCEL 链、能用 `batch` / `stream` / `as_tool` 等。扩展此前是
普通类，`dir()` 上少 38 个成员。现在两者都继承 `langchain_core.runnables.Runnable`，`get_name(suffix, name=)`
签名也对齐。

`as_tool()` 两边都**不可用**：LG 自己会抛
`PydanticUserError: tools_input is not fully defined`（同一处 pydantic 限制），我们是等价错误 —— 行为一致。

### store 的 op 类与 Item 形状（2026-11-27）

| 项 | LG | 改前 | 现在 |
|---|---|---|---|
| `GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp` / `MatchCondition` | NamedTuple（可解包、`_fields`） | dataclass | NamedTuple |
| `TTLConfig` / `IndexConfig` | TypedDict | dataclass | TypedDict |
| `store.get()` 返回类型 | `Item` | 原生 `StoreItem` | `Item` |
| `store.search()` 返回类型 | `SearchItem` | 原生 `StoreItem` | `SearchItem` |
| `Item.namespace` | tuple | list（shim 侧） | tuple |
| 时间戳时区 | UTC-aware | **本地时区 naive** | UTC-aware |
| `Item.dict()` / `SearchItem.dict()` | 有（SearchItem 多 `score`） | 无 | 有，键集合一致 |

时区那条影响最实际：同一时刻我们给 `2026-11-27T22:33:28`，LG 给 `2026-11-27T14:33:29+00:00`。

### `BaseStore` 也是完整协议基类（2026-11-27）

与上一轮同一个问题：LG 的 `BaseStore` 子类只需实现 `batch`（可选 `abatch`），其余
`get` / `put` / `delete` / `search` / `list_namespaces` 及异步孪生都由基类派生，并带 `supports_ttl` /
`ttl_config` 与命名空间校验。扩展此前是 4 个 `@abstractmethod` + 返回空元组的 `list_namespaces`。

现在成员与 LG **完全相同**（14 个），实测「只实现 `batch` 的自定义 store」可直接使用全部同步/异步方法；
删除按 LG 的编码（`PutOp(value=None, ttl=None)`），TTL 在不支持的 store 上抛 `NotImplementedError`。

### `BaseCheckpointSaver` 现在是完整的协议基类（2026-11-27）

LG 的 `BaseCheckpointSaver` 是"自带检查器"的基类：子类只实现 `get_tuple` / `put` / `put_writes` / `list`，
其余（`get`、异步孪生、`get_next_version`、`with_allowlist`、`config_specs`、`serde`）由基类提供。
扩展此前是个空占位类（`dir()` 里一个成员都没有）。

现在成员与 LG **完全相同**（24 个），语义也一致：`get` 基于 `get_tuple`、`get_next_version` 默认自增从 1 开始、
未实现的方法抛 `NotImplementedError`、异步孪生委托同步版本。

### `InMemoryStore` 的 TTL 与 batch 错误类型（2026-11-27）

两处对齐 LG：

| 场景 | LG | 改前 | 现在 |
|---|---|---|---|
| `store.put(..., ttl=1)` | `NotImplementedError`（`supports_ttl = False`） | 静默接受并生效 | `NotImplementedError`，文案一致 |
| `store.batch([{...}])`（非 op 对象） | `ValueError: Unknown operation type: <class 'dict'>` | `TypeError: unknown store op` | `ValueError`，文案一致 |

顺带确认一致：`list_namespaces` 的 `prefix` / `suffix` / `max_depth` / `limit` / `offset` 四种用法、
`delete` / `aget` / `aput` / `asearch` / `alist_namespaces` / `abatch([])`、`get` 缺失返回 `None`、
`search` 的 `filter` 与 `query`。

### Rust 不再反向导入 Python 包（2026-11-27）

按架构要求，Rust 侧**不得** import `langgraph_rs.*`（那是 shim）。此前有 9 处反向导入，现已全部消除：

| 原来 | 现在 |
|---|---|
| `py.import("langgraph_rs.errors")` ×6（`InvalidUpdateError` / `NodeTimeoutError`） | shim 在 `errors.py` 导入时**注册异常类**（`_register_error_classes`），Rust 查表构造 |
| `py.import("langgraph_rs._compat").message_chunk` | shim 注册 `message_chunk` 钩子（`_register_shim_hook`） |
| `py.import("langgraph_rs.types").PregelTask` | 同上，注册 `pregel_task` 类 |

现在 `grep 'import("langgraph_rs' src/` 为空；异常对象仍是 shim 定义的类，`isinstance` / `except` 行为不变。

原先 `src/adapters/python/mod.rs` 里还有 220 行**以 Rust 字符串内嵌的 Python 回调处理器**。现已移到
`python/langgraph_rs/_callbacks.py`，由 shim 在导入时注册工厂（`_register_shim_hook("callback_handler", ...)`），
Rust 只查表调用。至此 Rust 里既无 `import langgraph_rs`，也无 `PyModule::from_code` 内嵌 Python。

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

### `astream_events` 里子图事件嵌套 + 图级 chunk 内容（2026-11-27）

两处继续对齐：

**1. 子图事件**：LG 在父图节点是子图时，会**嵌套**发出子图自己的 `LangGraph` 生命周期事件（父节点 start →
子图 start → 子图节点事件 → 子图 end → 父节点 stream/end）。扩展此前只有父图那一层。
现在事件序列与 `parent_ids` 深度逐条相同。

**2. 图级 chunk 的内容**：LG 的 `on_chain_stream`（图级）装的是**该超步的 updates**（`{'a': {'value': 1}}`），
不是完整 state；扩展此前装完整 values。现在一致。

**metadata 已基本齐平**：节点事件现在带 `ls_integration`、`langgraph_step`（超步序号，从 1 开始）、
`langgraph_node`、`langgraph_path`（`["__pregel_pull", <node>]`）、`langgraph_checkpoint_ns`
（`<node>:<task-id>`，子图带 `|` 前缀）。并行 `Send` 到同一节点时**每个任务各自一条 run**（run_id 与 ns 都不同）。

**仍未对齐（记录）**：

* `langgraph_triggers`：LG 是 `('branch:to:<node>',)` 或 `('__pregel_push',)`，取决于任务由边还是 `Send` 触发；
  引擎目前不在流块里带这个信息。
* 同一超步内多个并行任务的**事件交错顺序**：LG 先发完所有 start 再发 stream（
  `start#1, start#2, stream#1, stream#2, end#1, end#2`），扩展按任务顺序发（`start#1, stream#1, end#1, start#2…`）。
* 子图 `LangGraph` 的 `on_chain_end` 里 LG 多一个 `input` 字段。

### `astream_events` 的图级事件按超步发出（2026-11-27）

LG 在 `astream_events` 里，**每个超步结束**都会发一次图级 `on_chain_stream`（两个节点 → 两次 + 一次 start/end）。
扩展只在最后发一次：

```
LG : start LangGraph | start a | stream a | end a | stream LangGraph | start b | ... | stream LangGraph | end LangGraph
改前: ... 只有结尾那一个 stream LangGraph
```

现在数量与顺序逐条相同（含"图级 chunk 在节点事件之后"这个顺序细节）。

顺带确认已一致：递归上限错误（`GraphRecursionError` 文案逐字相同）、节点超时
（`NodeTimeoutError`，仅 elapsed 实测值不同）、`astream_events` 的事件字段集合。

### 错误文案与空线程快照（2026-11-27）

三处小差异，均已对齐：

| 场景 | LG | 改前 | 现在 |
|---|---|---|---|
| 节点返回非 dict | `Expected dict, got not a dict` + `INVALID_GRAPH_NODE_RETURN_VALUE` | 带引号 `'not a dict'`，且错误码是 `INVALID_CONCURRENT_GRAPH_UPDATE` | 逐字一致 |
| `get_state()` 查不存在的线程 | `config={'configurable': {'thread_id': ...}}`（回显请求） | `config={}` | 回显请求 |
| 裸调 `interrupt()` | `RuntimeError: Called get_config outside of a runnable context` | `RuntimeError: interrupt() can only be called from inside a running graph node` | **保留我们的文案**（LG 那句暴露内部实现，信息量更低；异常类型一致） |

顺带确认已一致：并行 `interrupt` + `Command(resume={id: value})` 按 id 恢复、`Command(graph=Command.PARENT)`
子图交回父图、`update_state` 的历史行（step/source/values/next）。

### `source="input"` 检查点带上一轮的值（2026-11-27）

LG 的历史里，每一轮开头的 `source="input"` 检查点存的是**本轮开始前**线程的值（首轮为空）。扩展此前一律存空，
所以连跑两轮后历史对不上：

```
step 2  input   LG: {'value': 1}    扩展改前: {}
```

现在一致，完整历史逐条相同（`step` / `source` / `values` / `next`）：

```
(4, loop,  {'value': 11}, ())
(3, loop,  {'value': 10}, ('a',))
(2, input, {'value': 1},  ('__start__',))
(1, loop,  {'value': 1},  ())
(0, loop,  {'value': 0},  ('a',))
(-1, input, {},           ('__start__',))
```

另外确认已一致：`durability` 三种模式的历史长度（sync 4 / async 4 / exit 1）、时间旅行
（`invoke(None, older_config)`）、`Command(update=...)` 与 `update_state` 里未声明的 key 丢弃。

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

**这四项后来都补齐了**（见后文）：`StateSnapshot` 改成 NamedTuple；`InMemoryStore` 有
`embeddings` / `index_config`；`InMemorySaver` 有 `put` / `put_writes`；`StateGraph` 有
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

现在参数表与 LG 1.2.22 一致：`model / tools / prompt / response_format / pre_model_hook /
post_model_hook / state_schema / context_schema / checkpointer / store / interrupt_before /
interrupt_after / debug / version / name / deprecated_kwargs`（`config_schema` 作为旧拼写保留，映射到
`context_schema`）。

**`pre_model_hook` / `post_model_hook`** 两个钩子都作为图中节点执行（与 LG 相同），钩子返回
"原消息 + 新消息"时结果也一致 —— 前提是 `add_messages` 按 `id` 去重（见下一条：这一条原先不一致，已修复）。

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

**已补齐**：节点失败后 `snapshot.tasks[i].error` 现在与 LG 一致，都是 `"ValueError('kaboom')"`
（失败记录写回上一个检查点，见后文"失败任务"一节）。

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

**之后的补齐**：`with_allowlist`、`get_delta_channel_history`（`seed` 带真实通道值）、`delete_for_runs`
（与 LG 一样抛 `NotImplementedError`）、`config_specs`、`serde` 都已提供。`prune` 与 LG 的
`InMemorySaver` 一致，同样抛 `NotImplementedError`。`storage` / `writes` / `blobs` 是 LG 内部属性，未提供。

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

### 编译后图的成员（已对齐，2026-11-27 复核）

**这块早期是缺的，现已补齐**：编译图是 `Runnable` 子类（`bind` / `with_config` / `with_retry` /
`pipe` / `batch_as_completed` / `as_tool` / schema introspection 等全部可用），并且
`isinstance(app, Pregel)` 为真。与 LangGraph 的 `Pregel` 逐成员比对后**没有缺失项**。

仍是差别的只有 3 个**Pregel 内部成员**：`nodes`、`attach_node` / `attach_edge` / `attach_branch`
不存在 —— 它们操作的是 LG Python 侧的 `PregelNode` 对象图，扩展的节点表在 Rust 引擎里，无法如实暴露。

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