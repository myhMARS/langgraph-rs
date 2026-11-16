#![cfg(feature = "sqlite")]

//! SQLite long-term memory store.

use langgraph_rs::{BaseStore, SqliteStore, json};

#[test]
fn sqlite_store_roundtrip_search_and_namespaces() {
    let store = SqliteStore::in_memory().unwrap();
    let namespace = vec!["users".to_string(), "42".to_string()];
    assert!(store.get(&namespace, "name").unwrap().is_none());
    store
        .put(&namespace, "name", json!({"first": "Ada"}))
        .unwrap();
    store
        .put(&namespace, "email", json!("ada@example.com"))
        .unwrap();
    store
        .put(
            &["users".to_string(), "7".to_string()],
            "name",
            json!("Grace"),
        )
        .unwrap();

    assert_eq!(
        store.get(&namespace, "name").unwrap(),
        Some(json!({"first": "Ada"}))
    );
    let items = store
        .search(&["users".to_string()], Some("first"), None)
        .unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].key, "name");
    assert_eq!(items[0].namespace, namespace);
    assert_eq!(store.search(&[], None, None).unwrap().len(), 3);
    assert_eq!(
        store.namespaces().unwrap(),
        vec![
            vec!["users".to_string(), "42".to_string()],
            vec!["users".to_string(), "7".to_string()],
        ]
    );

    store.delete(&namespace, "name").unwrap();
    assert!(store.get(&namespace, "name").unwrap().is_none());
}

#[test]
fn sqlite_store_is_visible_from_nodes() {
    use langgraph_rs::{
        CompileOptions, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, get_store,
    };

    let store = std::sync::Arc::new(SqliteStore::in_memory().unwrap());
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("remember", |_state: &JsonState| {
            let store = get_store().expect("store configured");
            store
                .put(&["mem".to_string()], "greeting", json!("hello"))
                .unwrap();
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .set_entry_point("remember")
        .set_finish_point("remember")
        .compile_with(CompileOptions::new().with_store(store.clone()))
        .unwrap();

    graph.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(
        store.get(&["mem".to_string()], "greeting").unwrap(),
        Some(json!("hello"))
    );
}
