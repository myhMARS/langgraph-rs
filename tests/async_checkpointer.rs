#![cfg(feature = "async")]

//! Async checkpointer adapter (`BlockingSaver`) used by async graphs.

use std::sync::Arc;

use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{
    BlockingSaver, InMemorySaver, JsonState, NodeOutput, RunConfig, StateSchema, json,
};

#[tokio::test]
async fn async_graph_checkpoints_through_blocking_saver() {
    let saver = Arc::new(BlockingSaver::new(Arc::new(InMemorySaver::new())).unwrap());
    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("ask", |_state: JsonState, context| async move {
            let answer = context.interrupt(json!({ "question": "value?" }))?;
            Ok(NodeOutput::new().update("value", answer))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .with_async_checkpointer(saver)
        .compile()
        .unwrap();

    let error = graph
        .run(
            &JsonState::new(),
            &RunConfig::new().with_thread("async-saver"),
        )
        .await
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("async-saver").unwrap().unwrap();
    assert!(snapshot.interrupted);

    let output = graph.resume("async-saver", json!(7)).await.unwrap();
    assert_eq!(output.get("value"), Some(&json!(7)));
    assert!(!graph.get_state_history("async-saver").unwrap().is_empty());
    assert!(
        graph
            .threads()
            .unwrap()
            .contains(&"async-saver".to_string())
    );
}
