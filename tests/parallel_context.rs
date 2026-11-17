use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json,
};

#[test]
fn parallel_nodes_see_the_run_context() {
    let graph = StateGraph::<JsonState>::with_schema(
        StateSchema::dynamic().with_reducer("log", langgraph_rs::reducers::append()),
    )
    .add_node("a", |_state: &JsonState| {
        let thread = langgraph_rs::current_thread_id().unwrap_or_else(|| "none".into());
        Ok(NodeOutput::new().update("log", json!([format!("a:{thread}")])))
    })
    .add_node("b", |_state: &JsonState| {
        let thread = langgraph_rs::current_thread_id().unwrap_or_else(|| "none".into());
        Ok(NodeOutput::new().update("log", json!([format!("b:{thread}")])))
    })
    .add_edge(langgraph_rs::START, "a")
    .add_edge(langgraph_rs::START, "b")
    .set_finish_point("a")
    .set_finish_point("b")
    .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
    .unwrap();

    let output = graph
        .run(
            &JsonState::new(),
            &RunConfig::new().with_thread("parallel-ctx"),
        )
        .unwrap();
    let mut log: Vec<String> = output
        .state
        .get("log")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    log.sort();
    assert_eq!(log, vec!["a:parallel-ctx", "b:parallel-ctx"]);
}
