//! Graph name and serializable graph description.

use langgraph_rs::{
    CompileOptions, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json,
};

#[test]
fn compiled_graph_has_name_and_spec() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("work", |_state: &JsonState| Ok(NodeOutput::new()))
        .add_node("after", |_state: &JsonState| Ok(NodeOutput::new()))
        .set_entry_point("work")
        .add_edge("work", "after")
        .set_finish_point("after")
        .compile_with(CompileOptions::new().with_name("demo"))
        .unwrap();

    assert_eq!(graph.name(), Some("demo"));
    let spec = graph.graph_spec();
    let nodes = spec["nodes"].as_array().unwrap();
    assert!(nodes.iter().any(|node| node["id"] == json!("work")));
    assert!(nodes.iter().any(|node| node["id"] == json!("after")));
    let edges = spec["edges"].as_array().unwrap();
    assert!(
        edges
            .iter()
            .any(|edge| { edge["source"] == json!("work") && edge["target"] == json!("after") })
    );
    assert!(graph.to_mermaid().contains("work"));

    let output = graph.run(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(output.state.get("work"), None);
}

#[cfg(feature = "async")]
#[test]
fn async_saver_aliases_are_available() {
    use langgraph_rs::{AsyncInMemorySaver, InMemorySaver};
    let saver = AsyncInMemorySaver::new();
    let _: InMemorySaver = saver;
}
