#![cfg(feature = "async")]

use std::sync::Arc;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{InMemorySaver, JsonState, NodeOutput, RunConfig, StateSchema, json};

#[tokio::test]
async fn async_nodes_can_use_the_free_helper_functions() {
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("ask", |_state: JsonState, _ctx| async move {
            langgraph_rs::custom(json!({"progress": 1}))?;
            langgraph_rs::record_tokens(2, 3)?;
            let answer = langgraph_rs::interrupt(json!({"q": "name"}))?;
            Ok(NodeOutput::new().update("answer", answer))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .with_checkpointer(Arc::new(InMemorySaver::new()))
        .compile()
        .unwrap();

    let chunks = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = chunks.clone();
    let error = graph
        .stream_with(
            &JsonState::new(),
            &RunConfig::new().with_thread("async-free-helpers"),
            move |chunk| sink.lock().unwrap().push(chunk),
        )
        .await
        .unwrap_err();
    assert!(error.is_interrupt());
    let _ = chunks;

    let output = graph
        .resume("async-free-helpers", json!("Ada"))
        .await
        .unwrap();
    assert_eq!(output.get("answer"), Some(&json!("Ada")));
}
