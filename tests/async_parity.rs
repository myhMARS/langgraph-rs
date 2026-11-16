#![cfg(feature = "async")]

//! Async parity APIs: update_state, history filters, checkpoint export/import.

use std::sync::Arc;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    Durability, InMemorySaver, JsonState, NodeOutput, RunConfig, StateSchema, json,
};

fn graph(saver: Arc<InMemorySaver>) -> langgraph_rs::tokio_runtime::AsyncCompiledGraph<JsonState> {
    AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or(0);
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .with_checkpointer(saver)
        .compile()
        .unwrap()
}

#[tokio::test]
async fn update_state_and_filtered_history() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new().with_thread("async-update"),
        )
        .await
        .unwrap();

    let snapshot = graph
        .update_state("async-update", [("value", json!(42))], None)
        .await
        .unwrap();
    assert_eq!(snapshot.values.get("value"), Some(&json!(42)));

    let history = graph
        .get_state_history_filtered("async-update", None, Some(1))
        .unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].values.get("value"), Some(&json!(42)));

    let latest_id = snapshot.config["configurable"]["checkpoint_id"]
        .as_str()
        .unwrap()
        .to_string();
    let after = graph
        .get_state_history_filtered("async-update", Some(&latest_id), Some(10))
        .unwrap();
    assert!(
        after
            .iter()
            .all(|snapshot| snapshot.config["configurable"]["checkpoint_id"] != json!(latest_id))
    );
}

#[tokio::test]
async fn checkpoint_export_and_import_roundtrip() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new().with_thread("async-export"),
        )
        .await
        .unwrap();
    let document = graph
        .export_checkpoint("async-export")
        .unwrap()
        .expect("checkpoint");
    assert_eq!(document["checkpoint"]["channel_values"]["value"], json!(1));
    let version = document["checkpoint"]["channel_versions"]["value"]
        .as_str()
        .expect("string version");
    assert_eq!(version.split('.').next().unwrap().len(), 32);
    assert!(version.ends_with(".0.1"));

    graph
        .import_checkpoint("async-imported", &document)
        .unwrap();
    let snapshot = graph
        .get_state("async-imported")
        .unwrap()
        .expect("imported checkpoint");
    assert_eq!(snapshot.values.get("value"), Some(&json!(1)));
}

#[tokio::test]
async fn durability_exit_writes_the_final_checkpoint() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = graph(saver);
    let output = graph
        .run(
            &JsonState::from(json!({"value": 0}).as_object().unwrap().clone()),
            &RunConfig::new()
                .with_thread("async-exit")
                .with_durability(Durability::Exit),
        )
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(1)));
    let latest = graph
        .get_state("async-exit")
        .unwrap()
        .expect("final checkpoint");
    assert_eq!(latest.values.get("value"), Some(&json!(1)));
}

#[tokio::test]
async fn builder_conveniences_and_name() {
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .with_name("async-demo")
        .add_node("a", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .add_node("b", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().update("value", json!(2)))
        })
        .set_entry_point("a")
        .add_edges_from(["a"], "b")
        .set_finish_point("b")
        .compile()
        .unwrap();
    assert_eq!(graph.name(), Some("async-demo"));
    let output = graph.invoke(&JsonState::new()).await.unwrap();
    assert_eq!(output.get("value"), Some(&json!(2)));
}

#[tokio::test]
async fn subgraph_states_are_listed() {
    let saver = Arc::new(InMemorySaver::new());
    let child = Arc::new(
        AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
            .add_node("inc", |state: JsonState, _ctx| async move {
                let value = state
                    .get("value")
                    .and_then(|value| value.as_i64())
                    .unwrap_or_default();
                Ok(NodeOutput::new().update("value", json!(value + 1)))
            })
            .set_entry_point("inc")
            .set_finish_point("inc")
            .with_checkpointer(saver.clone())
            .compile()
            .unwrap(),
    );
    let parent = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", child)
        .set_entry_point("child")
        .set_finish_point("child")
        .with_checkpointer(saver)
        .compile()
        .unwrap();
    parent
        .run(
            &JsonState::new(),
            &RunConfig::new().with_thread("async-subgraph-states"),
        )
        .await
        .unwrap();
    let (root, subgraphs) = parent
        .get_state_subgraphs("async-subgraph-states")
        .unwrap()
        .expect("root state");
    assert_eq!(root.values.get("value"), Some(&json!(1)));
    assert_eq!(subgraphs.len(), 1);
}
