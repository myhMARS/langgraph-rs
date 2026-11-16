//! Long-term memory (`BaseStore`) and runtime context tests.

use std::sync::Arc;

use langgraph_rs::{
    BaseStore, CompileOptions, InMemorySaver, InMemoryStore, NodeOutput, RunConfig, State,
    StateGraph, StateSchema, current_context, current_namespace, current_runtime,
    current_thread_id, get_store, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct MemoryState {
    value: i64,
    tenant: String,
}

impl State for MemoryState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value").last_value("tenant")
    }
}

#[test]
fn in_memory_store_round_trip() {
    let store = InMemoryStore::new();
    assert!(store.get(&["users".to_string()], "42").unwrap().is_none());
    store
        .put(&["users".to_string()], "42", json!({ "name": "ada" }))
        .unwrap();
    store
        .put(
            &["users".to_string(), "prefs".to_string()],
            "theme",
            json!("dark"),
        )
        .unwrap();
    assert_eq!(
        store.get(&["users".to_string()], "42").unwrap(),
        Some(json!({ "name": "ada" }))
    );
    let users = store.search(&["users".to_string()], None, None).unwrap();
    assert_eq!(users.len(), 2);
    let filtered = store
        .search(&["users".to_string()], Some("ada"), None)
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].key, "42");
    assert_eq!(
        store.namespaces().unwrap(),
        vec![
            vec!["users".to_string()],
            vec!["users".to_string(), "prefs".to_string()]
        ]
    );
    store.delete(&["users".to_string()], "42").unwrap();
    assert!(store.get(&["users".to_string()], "42").unwrap().is_none());
}

#[test]
fn graph_reads_and_writes_long_term_memory_across_threads() {
    let store = Arc::new(InMemoryStore::new());
    let graph = StateGraph::<MemoryState>::new()
        .add_node("remember", |_state: &MemoryState| {
            let store = get_store().expect("store is available");
            let previous = store
                .get(&["mem".to_string()], "counter")
                .ok()
                .flatten()
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            let next = previous + 1;
            store
                .put(&["mem".to_string()], "counter", json!(next))
                .unwrap();
            Ok(NodeOutput::new().set("value", next))
        })
        .set_entry_point("remember")
        .set_finish_point("remember")
        .compile_with(CompileOptions::new().with_store(store.clone()))
        .unwrap();

    let first = graph
        .invoke(&MemoryState::default(), &RunConfig::new())
        .unwrap();
    let second = graph
        .invoke(&MemoryState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(first.value, 1);
    assert_eq!(second.value, 2);
    assert_eq!(
        store.get(&["mem".to_string()], "counter").unwrap(),
        Some(json!(2))
    );
}

#[test]
fn runtime_exposes_thread_namespace_and_context() {
    let graph = StateGraph::<MemoryState>::new()
        .add_node("inspect", |_state: &MemoryState| {
            assert_eq!(current_thread_id().as_deref(), Some("runtime-thread"));
            assert_eq!(current_namespace(), "sub");
            assert_eq!(current_context(), Some(json!({ "tenant": "acme" })));
            let runtime = current_runtime().expect("runtime available");
            assert_eq!(runtime.thread_id.as_deref(), Some("runtime-thread"));
            assert_eq!(runtime.namespace, "sub");
            assert_eq!(runtime.context, Some(json!({ "tenant": "acme" })));
            Ok(NodeOutput::new().set("tenant", "acme"))
        })
        .set_entry_point("inspect")
        .set_finish_point("inspect")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    let output = graph
        .invoke(
            &MemoryState::default(),
            &RunConfig::new()
                .with_thread("runtime-thread")
                .with_checkpoint_ns("sub")
                .with_context(json!({ "tenant": "acme" })),
        )
        .unwrap();
    assert_eq!(output.tenant, "acme");
}

#[test]
fn subgraphs_inherit_the_parent_store_and_context() {
    let store = Arc::new(InMemoryStore::new());
    let child = Arc::new(
        StateGraph::<MemoryState>::new()
            .add_node("write", |state: &MemoryState| {
                let store = get_store().expect("inherited store");
                store
                    .put(&["child".to_string()], "value", json!(state.value))
                    .unwrap();
                let context = current_context().expect("inherited context");
                Ok(NodeOutput::new().set(
                    "tenant",
                    context
                        .get("tenant")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default(),
                ))
            })
            .set_entry_point("write")
            .set_finish_point("write")
            .compile()
            .unwrap(),
    );
    let parent = StateGraph::<MemoryState>::new()
        .add_node("seed", |_state: &MemoryState| {
            Ok(NodeOutput::new().set("value", 11))
        })
        .add_subgraph("child", child)
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("child")
        .compile_with(CompileOptions::new().with_store(store.clone()))
        .unwrap();

    let output = parent
        .invoke(
            &MemoryState::default(),
            &RunConfig::new().with_context(json!({ "tenant": "acme" })),
        )
        .unwrap();
    assert_eq!(output.tenant, "acme");
    assert_eq!(
        store.get(&["child".to_string()], "value").unwrap(),
        Some(json!(11))
    );
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_graphs_expose_store_and_context() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let store = Arc::new(InMemoryStore::new());
    let graph = AsyncStateGraph::<MemoryState>::new()
        .add_node("work", |state: MemoryState, _ctx| async move {
            let store = get_store().expect("async store");
            store
                .put(&["async".to_string()], "value", json!(state.value + 1))
                .unwrap();
            let context = current_context().expect("async context");
            Ok(NodeOutput::new().set(
                "tenant",
                context
                    .get("tenant")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default(),
            ))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .with_store(store.clone())
        .compile()
        .unwrap();

    let output = graph
        .run(
            &MemoryState::default(),
            &RunConfig::new().with_context(json!({ "tenant": "async-acme" })),
        )
        .await
        .unwrap()
        .state;
    assert_eq!(output.tenant, "async-acme");
    assert_eq!(
        store.get(&["async".to_string()], "value").unwrap(),
        Some(json!(1))
    );
}
