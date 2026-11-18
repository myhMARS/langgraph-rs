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
    let reexported = graph.export_checkpoint("dst").unwrap().unwrap();
    assert_eq!(
        reexported["checkpoint"]["channel_versions"],
        document["checkpoint"]["channel_versions"]
    );
    // One superstep: raw input (-1), post-input (0) and the terminal checkpoint (1).
    assert_eq!(reexported["metadata"]["step"], json!(1));
    let imported_step = reexported["metadata"]["step"].as_i64().unwrap();
    graph.import_checkpoint("dst-2", &reexported).unwrap();
    let again = graph.export_checkpoint("dst-2").unwrap().unwrap();
    assert_eq!(again["metadata"]["step"], json!(imported_step));
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
        "pending_writes": [
            { "task_id": "t1", "node": "n", "channel": "__interrupt__", "value": [], "replace": false },
            { "task_id": "t1", "node": "n", "channel": "__resume__", "value": 1, "replace": false },
            { "task_id": "t1", "node": "n", "channel": "extra", "value": 43, "replace": false }
        ]
    });
    graph.import_checkpoint("foreign", &document).unwrap();
    let snapshot = graph.get_state("foreign").unwrap().unwrap();
    assert_eq!(snapshot.values.get("value"), Some(&json!(42)));
    let reexported = graph.export_checkpoint("foreign").unwrap().unwrap();
    assert_eq!(
        reexported["checkpoint"]["channel_versions"]["value"],
        json!("1.0.1")
    );
    // Reserved `__` control writes are dropped on import; only real state survives.
    assert_eq!(
        reexported["pending_writes"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(reexported["pending_writes"][0]["channel"], json!("extra"));
}

#[test]
fn treats_plain_foreign_state_writes_as_finished_tasks() {
    let graph = graph();
    let document = json!({
        "config": { "configurable": { "thread_id": "lg-int", "checkpoint_ns": "", "checkpoint_id": "cp-i" } },
        "checkpoint": {
            "v": 1,
            "id": "cp-i",
            "channel_values": { "value": 5 },
            "channel_versions": { "value": "1.0.1" },
            "versions_seen": {},
            "updated_channels": ["value"]
        },
        "metadata": { "source": "loop", "step": 1 },
        "parent_config": null,
        "pending_writes": [
            { "task_id": "task-9", "node": "inc", "channel": "value", "value": 6, "replace": false }
        ],
        "metadata_extra": null
    });
    graph.import_checkpoint("lg-int", &document).unwrap();
    let snapshot = graph.get_state("lg-int").unwrap().unwrap();
    // Plain state writes belong to a task that already finished, so nothing is pending
    // and the checkpoint is not interrupted.
    assert!(!snapshot.interrupted);
    assert!(snapshot.next.is_empty());
}

#[test]
fn imports_interrupts_recorded_as_control_writes() {
    let graph = graph();
    // LangGraph persists interrupts as `__interrupt__` pending writes; no sidecar exists.
    let document = json!({
        "config": { "configurable": { "thread_id": "lg-ctrl", "checkpoint_ns": "", "checkpoint_id": "cp-c" } },
        "checkpoint": {
            "v": 1,
            "id": "cp-c",
            "channel_values": { "value": 5 },
            "channel_versions": { "value": "1.0.1" },
            "versions_seen": {},
            "updated_channels": ["value"]
        },
        "metadata": { "source": "loop", "step": 1 },
        "parent_config": null,
        "pending_writes": [
            {
                "task_id": "task-7",
                "node": "inc",
                "channel": "__interrupt__",
                "value": [{ "value": { "ask": "go?" }, "id": "int-7", "response_schema": null }],
                "replace": false
            }
        ],
        "metadata_extra": null
    });
    graph.import_checkpoint("lg-ctrl", &document).unwrap();
    let snapshot = graph.get_state("lg-ctrl").unwrap().unwrap();
    assert!(snapshot.interrupted);
    assert_eq!(snapshot.next, vec!["inc".to_string()]);
    assert_eq!(snapshot.interrupts.len(), 1);
    assert_eq!(snapshot.interrupts[0].id, "int-7");
    assert_eq!(snapshot.interrupts[0].value, json!({ "ask": "go?" }));
    assert_eq!(snapshot.interrupts[0].node, "inc");
}

#[test]
fn imports_failed_tasks_from_error_writes() {
    let graph = graph();
    let document = json!({
        "config": { "configurable": { "thread_id": "lg-err", "checkpoint_ns": "", "checkpoint_id": "cp-e" } },
        "checkpoint": {
            "v": 1,
            "id": "cp-e",
            "channel_values": { "value": 5 },
            "channel_versions": { "value": "1.0.1" },
            "versions_seen": {},
            "updated_channels": ["value"]
        },
        "metadata": { "source": "loop", "step": 1 },
        "parent_config": null,
        "pending_writes": [
            {
                "task_id": "task-3",
                "node": "inc",
                "channel": "__error__",
                "value": { "message": "boom" },
                "replace": false
            }
        ],
        "metadata_extra": null
    });
    graph.import_checkpoint("lg-err", &document).unwrap();
    let snapshot = graph.get_state("lg-err").unwrap().unwrap();
    assert_eq!(snapshot.next, vec!["inc".to_string()]);
    let task = &snapshot.tasks[0];
    let error = task.error.as_ref().expect("error rebuilt");
    assert_eq!(error.message, "boom");
}

#[cfg(feature = "msgpack")]
#[test]
fn drops_reserved_control_writes_in_tuple_form() {
    let graph = graph();
    let document = json!({
        "config": { "configurable": { "thread_id": "lg-tuple", "checkpoint_ns": "", "checkpoint_id": "cp-t" } },
        "checkpoint": {
            "v": 1,
            "id": "cp-t",
            "channel_values": { "value": 7 },
            "channel_versions": { "value": "1.0.1" },
            "versions_seen": {},
            "updated_channels": ["value"]
        },
        "metadata": { "source": "loop", "step": 1 },
        "parent_config": null,
        "pending_writes": [
            ["task-1", "__interrupt__", []],
            ["task-1", "__resume__", 3],
            ["task-1", "value", 8]
        ]
    });
    graph.import_checkpoint("tuple-form", &document).unwrap();
    let reexported = graph.export_checkpoint("tuple-form").unwrap().unwrap();
    let writes = reexported["pending_writes"].as_array().unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0]["channel"], json!("value"));
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
