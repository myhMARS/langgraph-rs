#![cfg(feature = "sqlite")]

//! Reading a LangGraph SQLite checkpointer database.

use langgraph_rs::{LangGraphSqliteReader, json};

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
