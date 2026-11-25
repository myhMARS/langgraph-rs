# store — 长期记忆（命名空间键值存储）

store 不属于某一次图运行，而是按**命名空间元组**寻址的长期记忆。演示三个核心操作：

| 操作 | 作用 |
| --- | --- |
| `put(namespace, key, value)` | 写入（同名覆盖） |
| `get(namespace, key)` | 读一个条目 |
| `search(namespace_prefix)` | 按命名空间前缀列出条目 |
| `list_namespaces()` / `namespaces()` | 列出已有命名空间 |

## 文件

| 文件 | 说明 |
| --- | --- |
| [`langgraph_demo.py`](langgraph_demo.py) | LangGraph 参考实现，`InMemoryStore` |
| [`binding_demo.py`](binding_demo.py) | `langgraph_rs.InMemoryStore`，同样的调用 |
| [`native_demo.rs`](native_demo.rs) | Rust 原生，`InMemoryStore` + `BaseStore` trait |
| [`compare.py`](compare.py) | 三者结果一致性 + 耗时对比 |

## 运行

```bash
python examples/store/langgraph_demo.py
python examples/store/binding_demo.py
cargo run --release --example store_native
python examples/store/compare.py --iterations 1000 --runs 3
```

## 期望输出

```json
{"names": ["Ada", "Grace"],
 "namespaces": ["teams/core", "users/1", "users/2"],
 "read": {"first": "Ada", "last": "Lovelace"},
 "search": ["users/1:email", "users/1:name", "users/2:name"]}
```

三者逐字一致。

## 性能对比

`compare.py --iterations 1000 --runs 2` 实测（每次迭代新建一个 store、写 4 条、读 1 条、搜 1 次）：

| 实现 | 每次迭代 | 相对 LangGraph | 相对 binding |
| --- | ---: | ---: | ---: |
| LangGraph | 22.09 µs | 1.00x | 0.69x |
| langgraph_rs binding | 15.20 µs | **1.45x** | 1.00x |
| Rust native | 7.37 µs | 3.00x | **2.06x** |

这是所有功能里差距最小的一项：操作本身是纯内存读写，跨语言边界的那点开销占比不高。

## 两个注意点

1. **值必须是 JSON 对象**。LangGraph 1.x 的 `InMemoryStore` 在 `search` 时会对非 dict 值做
   `dict(value)` 转换，写字符串会直接抛 `ValueError` —— 所以 demo 里所有 value 都是 dict。
2. **`search` 返回 `Item` 对象**（`item.namespace` / `item.key` / `item.value`），不是裸值。
   这一点 LangGraph 1.x 与 `langgraph_rs` 一致。

## 代码映射

| LangGraph | binding | Rust native |
| --- | --- | --- |
| `InMemoryStore()` | 同（`langgraph_rs.InMemoryStore`） | `InMemoryStore::new()` |
| `store.put(("users","1"), "name", {...})` | 同 | `store.put(&ns, "name", json!({...}))` |
| `store.get(ns, "name")` → `Item` | 同 | `store.get(&ns, "name")? -> Option<Value>` |
| `store.search(("users",))` | 同 | `store.search(&ns, None, None)? -> Vec<StoreItem>` |
| `store.list_namespaces()` | 同 | `store.namespaces()?` |
| `InjectedStore` 注入到工具 | 同 | `langgraph_rs::get_store()` |

## 相关文档

- store 与检索语义（`tokenize_path` / `get_text_at_path` 与 LangGraph 189 组差分一致）：
  [`docs/langgraph-compat.md`](../../docs/langgraph-compat.md)
- 后端选择（内存 / SQLite / Redis / Postgres）：[`docs/architecture.md`](../../docs/architecture.md)