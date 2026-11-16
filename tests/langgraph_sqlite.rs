#![cfg(feature = "sqlite")]

//! Reading a LangGraph SQLite checkpointer database.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, LangGraphSqliteReader, NodeOutput, StateGraph,
    StateSchema, interrupt, json,
};

fn fixture() -> String {
    format!(
        "{}/tests/fixtures/langgraph_sample.sqlite",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn reads_langgraph_checkpoints_and_normalizes_channels() {
    let reader = LangGraphSqliteReader::open(fixture()).unwrap();
    assert!(
        reader
            .threads()
            .unwrap()
            .iter()
            .any(|thread| thread == "t1")
    );

    let latest = reader.latest("t1", "").unwrap().expect("latest checkpoint");
    assert_eq!(latest.thread_id, "t1");
    assert_eq!(latest.values.get("value"), Some(&json!(2)));
    assert!(latest.values.get("__root__").is_none());
    assert!(latest.values.get("branch:to:inc").is_none());
    assert!(!latest.channel_versions.is_empty());
    assert_eq!(latest.metadata.get("source"), Some(&json!("loop")));

    let history = reader.history("t1", "").unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].id, latest.id);
    assert_eq!(history[2].values.get("value"), Some(&json!(1)));

    let first = reader
        .get("t1", "", &history[2].id)
        .unwrap()
        .expect("checkpoint by id");
    assert_eq!(first.id, history[2].id);

    let writes = reader.pending_writes("t1", "", &first.id).unwrap();
    assert!(
        writes
            .iter()
            .any(|write| write.channel == "__root__" && write.value.get("value").is_some())
    );
}

#[test]
fn reads_interrupts_and_resumes_from_an_imported_checkpoint() {
    let path = format!(
        "{}/tests/fixtures/langgraph_interrupt.sqlite",
        env!("CARGO_MANIFEST_DIR")
    );
    let reader = LangGraphSqliteReader::open(path).unwrap();
    let mut checkpoint = reader.latest("t1", "").unwrap().expect("latest");
    assert_eq!(checkpoint.values.get("value"), Some(&json!(1)));
    assert!(checkpoint.interrupted);
    assert_eq!(checkpoint.next.len(), 1);
    assert_eq!(checkpoint.next[0].node, "ask");
    assert_eq!(checkpoint.interrupts.len(), 1);
    assert_eq!(checkpoint.interrupts[0].value["q"], json!("?"));

    checkpoint.thread_id = "imported-interrupt".to_string();
    let saver = Arc::new(InMemorySaver::new());
    langgraph_rs::Checkpointer::put(saver.as_ref(), checkpoint).unwrap();

    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("ask", |_state: &JsonState| {
            let answer = interrupt(json!({"q": "?"}))?;
            Ok(NodeOutput::new().update("value", answer))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();
    let output = graph.resume("imported-interrupt", json!(5)).unwrap();
    assert_eq!(output.get("value"), Some(&json!(5)));
}
