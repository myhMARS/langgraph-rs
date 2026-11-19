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
- `create_react_agent`（`prompt` / `state_schema` / `store` / `interrupt_*` / `name` / `version`）、
  `ToolNode`、`tools_condition`、`MessagesState`、`add_messages`
- 托管值注解：`remaining_steps: RemainingSteps`、`is_last_step: IsLastStep`（每次超步注入、
  不写 checkpoint，与 LangGraph 一致）
- `entrypoint` / `task`（functional API）、`get_store` / `get_stream_writer`

## 4. 已知差异（迁移前请确认）

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
2. **`stream()` 返回 `list`**（LangGraph 返回生成器）：迭代结果相同，但不是惰性的，也没有
   `.close()`。需要惰性时用 `astream` 或自行包一层生成器。
3. **`get_state_history()` 返回惰性 `StateHistory`**（支持 `len()`、索引、迭代），不是生成器。
4. **`create_react_agent` 已对齐常用参数**：`prompt`（字符串 / `SystemMessage` / `callable`）、
   `state_schema`、`store`、`interrupt_before`、`interrupt_after`、`name`、`version`（`v1`/`v2`）
   均支持；`response_format`、`pre_model_hook`、`post_model_hook`、`context_schema`
   会抛 `NotImplementedError` 并说明，不会静默忽略。
5. **`langgraph.checkpoint.base` / `langgraph.store.base` 未实现**：对应的 Rust 只提供具体实现，
   不支持在 Python 里自定义 saver/store。`langgraph_rs.checkpoint.base` 等模块可以 import，
   但实例化或继承会抛 `NotImplementedError` 并说明原因。
6. **PEP 563 的状态类必须是可导入的模块级类**：`typing.get_type_hints` 需要能解析字符串注解；
   在函数内 `exec` 出来的类无法解析（LangGraph 同样如此，因此不算差异）。

## 5. 如何验证

```powershell
# 124 项 Python 测试，其中 test_package_layout.py 覆盖本包的全部导入路径
python -m unittest discover -s python/tests

# 与官方 LangGraph 的逐场景对照（38/38 全部等价）
python benchmarks/parity.py

# 可运行的迁移示例
python python/examples/dropin_from_langgraph.py
```

`benchmarks/parity.py` 用同一份工作流分别跑 LangGraph 与 `langgraph_rs`，逐场景比较状态、步骤数、
事件顺序；新增场景时它会自动把差异暴露出来。