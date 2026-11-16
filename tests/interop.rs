//! LangGraph checkpoint-tuple JSON export/import.

use std::sync::Arc;

use langgraph_rs::{CompileOptions, InMemorySaver, JsonState, NodeOutput, StateGraph, json};

fn graph() -> langgraph_rs::CompiledGraph<JsonState> {
    StateGraph::<JsonState>::with_schema(langgraph_rs::StateSchema::dynamic())
        .add_node("inc", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap()
}

#[test]
fn checkpoint_roundtrips_through_langgraph_json() {
    let graph = graph();
    let mut input = JsonState::new();
    input.insert("value".to_string(), json!(0));
    graph
        .run(&input, &langgraph_rs::RunConfig::new().with_thread("src"))
        .unwrap();

    let document = graph.export_checkpoint("src").unwrap().unwrap();
    assert_eq!(
        document["config"]["configurable"]["thread_id"],
        json!("src")
    );
    assert!(document["checkpoint"]["id"].as_str().is_some());
    assert_eq!(document["checkpoint"]["channel_values"]["value"], json!(1));
    assert!(document["metadata"]["langgraph_rs"]["next"].is_array());

    graph.import_checkpoint("dst", &document).unwrap();
    let snapshot = graph.get_state("dst").unwrap().unwrap();
    assert_eq!(snapshot.values.get("value"), Some(&json!(1)));
}

#[test]
fn imports_a_foreign_langgraph_checkpoint_document() {
    let graph = graph();
    let document = json!({
        "config": { "configurable": { "thread_id": "lg", "checkpoint_ns": "", "checkpoint_id": "cp-1" } },
        "checkpoint": {
            "v": 1,
            "id": "cp-1",
            "channel_values": { "value": 42 },
            "channel_versions": { "value": "1.0.1" },
            "versions_seen": {},
            "updated_channels": ["value"]
        },
        "metadata": { "source": "input", "step": 1 },
        "parent_config": null,
        "pending_writes": []
    });
    graph.import_checkpoint("foreign", &document).unwrap();
    let snapshot = graph.get_state("foreign").unwrap().unwrap();
    assert_eq!(snapshot.values.get("value"), Some(&json!(42)));
}

#[cfg(feature = "msgpack")]
#[test]
fn checkpoint_tuple_roundtrips_through_msgpack() {
    use langgraph_rs::{Checkpoint, json};
    use std::collections::BTreeMap;

    let mut values = serde_json::Map::new();
    values.insert("value".to_string(), json!(41));
    let mut checkpoint = Checkpoint::new("c-msgpack", "thread-msgpack", 2, values);
    checkpoint
        .metadata
        .insert("source".to_string(), json!("loop"));
    let document = langgraph_rs::interop::checkpoint_to_langgraph(&checkpoint);
    let bytes = langgraph_rs::interop::checkpoint_to_langgraph_msgpack(&checkpoint).unwrap();
    let decoded = langgraph_rs::interop::msgpack::decode(&bytes).unwrap();
    assert_eq!(decoded, document);
    let imported = langgraph_rs::interop::checkpoint_from_langgraph_msgpack(&bytes).unwrap();
    assert_eq!(imported.id, checkpoint.id);
    assert_eq!(imported.thread_id, checkpoint.thread_id);
    assert_eq!(imported.values, checkpoint.values);
    let _ = BTreeMap::<String, u8>::new();
}
