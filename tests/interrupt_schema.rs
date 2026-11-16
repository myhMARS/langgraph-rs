//! `interrupt(response_schema=...)` validation.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt_with_schema, json,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct SchemaState {
    value: i64,
}

impl State for SchemaState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

#[test]
fn resume_values_are_validated_against_response_schema() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<SchemaState>::new()
        .add_node("ask", |_state: &SchemaState| {
            let answer = interrupt_with_schema(
                json!({ "question": "age?" }),
                Some(json!({ "type": "integer" })),
            )?;
            Ok(NodeOutput::new().set("value", answer.as_i64().unwrap_or_default()))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &SchemaState::default(),
            &RunConfig::new().with_thread("schema"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    let snapshot = graph.get_state("schema").unwrap().unwrap();
    assert!(snapshot.interrupts[0].response_schema.is_some());

    let error = graph.resume("schema", json!("nope")).unwrap_err();
    assert!(error.to_string().contains("response_schema"));

    let output = graph.resume("schema", json!(9)).unwrap();
    assert_eq!(output.value, 9);
}
