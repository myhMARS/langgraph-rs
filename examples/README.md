# examples — 每个功能一个独立 demo

每个功能目录都包含**三份等价实现 + 一个对比脚本 + 一篇说明**：

| 文件 | 作用 |
| --- | --- |
| `langgraph_demo.py` | 该功能的 LangGraph 参考实现（`from langgraph...`） |
| `binding_demo.py` | 用 `langgraph_rs` Python binding 实现的同一功能（通常只改 import） |
| `native_demo.rs` | 同一功能的 Rust 原生实现（`cargo run --example <feature>_native`） |
| `compare.py` | 校验三者结果一致，并给出 binding vs native 的耗时对比 |
| `README.md` | 该功能的说明、运行方式、期望输出、对比结论、代码映射表 |

所有 demo 都是自包含的：不依赖 `benchmarks/`，也不互相引用。

## 功能清单

| 功能 | 目录 | LangGraph 参考 | binding | native | 文档 |
| --- | --- | :---: | :---: | :---: | :---: |
| 快速开始：状态图 + 条件循环 | [`quickstart/`](quickstart/) | ✅ | ✅ | ✅ | [README](quickstart/README.md) |
| 状态与 reducer（append / 自定义 / MessageGraph） | `state_reducers/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 条件路由与 `Command(goto)` | `routing/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 人机协同：`interrupt` / resume | `interrupt/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 流式输出：`stream` / `astream` 各模式 | `streaming/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 持久化：savers / 时间旅行 / 断点续跑 | `checkpointing/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 子图与 `Send` 扇出 | `subgraph/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 工具：`ToolNode` / `create_react_agent` / 参数注入 | `tools/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 异步：`ainvoke` / `astream` / 并发超步 | `async/` | ⏳ | ⏳ | ⏳ | ⏳ |
| 长期记忆：store 读写与命名空间 | `store/` | ⏳ | ⏳ | ⏳ | ⏳ |

图例：✅ 已完成 · ⏳ 计划中（按上表顺序补齐）

## 运行前准备

```bash
# 1) 构建 release 扩展并组装 Python 包
cargo build --release --features "python python-async sqlite redis postgres msgpack"
python python/build_package.py --release

# 2) 让 Python 找到它（PowerShell）
$env:PYTHONPATH = "$PWD\target\python"

# 3) 任何 demo 都可以直接跑
python examples/quickstart/langgraph_demo.py
python examples/quickstart/binding_demo.py
python examples/quickstart/compare.py
```

LangGraph 参考实现需要 `langgraph`/`langchain_core` 已安装；未安装时 `compare.py` 会明确报错而不是
给出伪造结果。

## 基准测试不在这里

压测脚本、parity 校验、跨引擎大工作流基准都在 [`benchmarks/`](../benchmarks/README.md)：

- `benchmarks/parity.py` —— 38 个场景的双引擎行为对拍
- `benchmarks/mega_workflow/` —— 大工作流的 bindings / native / LangGraph 三方对比
- `benchmarks/rust/` —— criterion 微基准与 Rust 侧的 native 基线程序

`examples/` 只放「看得懂、跑得通、能对照」的功能级 demo。