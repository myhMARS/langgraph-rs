#![cfg(feature = "async")]

//! Async parity APIs: update_state, history filters, checkpoint export/import.

use std::sync::Arc;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{InMemorySaver, JsonState, NodeOutput, RunConfig, StateSchema, json};

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

    graph
        .import_checkpoint("async-imported", &document)
        .unwrap();
    let snapshot = graph
        .get_state("async-imported")
        .unwrap()
        .expect("imported checkpoint");
    assert_eq!(snapshot.values.get("value"), Some(&json!(1)));
}
