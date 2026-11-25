# tools — ToolNode 与参数注入

一张图、一个 `ToolNode`、两个工具：输入的最后一条消息带 `tool_calls`，节点执行工具并把结果作为
tool 消息追加回 `messages`。

| 工具 | 参数来源 | 说明 |
| --- | --- | --- |
| `add(a, b)` | 模型给的参数 | 普通工具 |
| `lookup(query)` | 模型给 `query`；`state` 由运行时注入 | `InjectedState` 的参数不出现在模型看到的 schema 里 |

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`@tool` + `ToolNode` + `InjectedState` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs.prebuilt`，同样用 `@tool` + `InjectedState` |
| [`native_demo.rs`](native_demo.rs) | Rust 原生：`ToolDefinition` / `with_state`（等价于注入 state） |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/tools/langgraph_demo.py
python examples/tools/binding_demo.py
cargo run --release --example tools_native
python examples/tools/compare.py --iterations 300 --runs 3
```

## 期望输出

```json
{"results": ["5", "q|ada"]}
```

`add(2, 3)` → `"5"`；`lookup("q")` 里 `state.user == "ada"` → `"q|ada"`。三者逐字一致。

## 性能对比

`compare.py --iterations 300 --runs 2` 实测：

| 实现 | 每次 invoke | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 4694.68 µs | 1.00x | 0.02x |
| langgraph_rs binding | 110.87 µs | **42.34x** | 1.00x |
| Rust native | 53.49 µs | 87.76x | **2.07x** |

## 两种消息形态

- **LangGraph** 的 `ToolNode` 只认 `AIMessage` 对象（会 `isinstance` 检查），所以参考实现用
  `langchain_core.messages.AIMessage`。
- **binding** 用 dict 消息（`{"role": "assistant", "tool_calls": [...]}`）即可，`ToolNode` 读
  `tool_calls` 字段；native 同样用 JSON 消息。

两者结果一致，只是消息对象不同 —— 跨引擎比较时对比的是**工具结果**，不是消息对象本身。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `@tool def add(...)` | 同（`langchain_core.tools.tool`） | `ToolDefinition::new("add", \|args\| ...)` |
| `state: Annotated[dict, InjectedState]` | 同（`langgraph_rs.prebuilt.InjectedState`） | `ToolDefinition::with_state(name, \|args, state\| ...)` |
| `ToolNode([add, lookup])` | 同 | `ToolNode::new([..])` |
| `graph.add_node("tools", ToolNode(...))` | 同 | `.add_node("tools", \|s\| tools.execute(s))` |
| store 注入 `InjectedStore` | 同 | 通过 `langgraph_rs::get_store()` 读取 |

> **普通函数（未加 `@tool`）的调用约定不同**：`langgraph_rs` 把 `tool_calls.args` 当作**一个位置
> 参数**传给普通函数，所以 `def add(a, b)` 要写成 `def add(args)`；而 LangChain 的 `@tool` 装饰器
> 会按关键字绑定参数（`add(a, b)` 直接可用）。因此两个 Python demo 都用 `@tool`——它两边行为一致，
> 也是 LangGraph 生态里的常规写法。这条差异记录在 `docs/langgraph-compat.md` 的差异清单里。

## 相关文档

- 工具与 `create_react_agent` 的完整表面：[`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)
- 注入参数的行为对照测试：[`python/tests/test_tool_injection.py`](../../python/tests/test_tool_injection.py)