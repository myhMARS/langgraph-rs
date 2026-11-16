//! `DeltaChannel` and `LastValueAfterFinish` semantics.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct DeltaState {
    tokens: Vec<String>,
    final_answer: String,
    observed: Vec<String>,
}

impl State for DeltaState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .delta("tokens", reducers::append())
            .last_value_after_finish("final_answer")
            .with_reducer("observed", reducers::append())
    }
}

#[test]
fn delta_channel_folds_deltas_with_the_reducer() {
    let graph = StateGraph::<DeltaState>::new()
        .add_node("a", |_state: &DeltaState| {
            Ok(NodeOutput::new().set("tokens", vec!["a".to_string()]))
        })
        .add_node("b", |_state: &DeltaState| {
            Ok(NodeOutput::new().set("tokens", vec!["b".to_string()]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&DeltaState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.tokens, vec!["a", "b"]);
}

#[test]
fn last_value_after_finish_hides_writes_until_the_end() {
    let graph = StateGraph::<DeltaState>::new()
        .add_node("a", |state: &DeltaState| {
            Ok(NodeOutput::new()
                .set("final_answer", "draft")
                .set("observed", vec![format!("a:{}", state.final_answer)]))
        })
        .add_node("b", |state: &DeltaState| {
            Ok(NodeOutput::new()
                .set("final_answer", "done")
                .set("observed", vec![format!("b:{}", state.final_answer)]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&DeltaState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.final_answer, "done");
    assert_eq!(
        output.observed,
        vec!["a:".to_string(), "b:".to_string()],
        "intermediate writes must stay hidden from other nodes"
    );
}

#[test]
fn last_value_after_finish_is_absent_from_interrupt_checkpoints() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<DeltaState>::new()
        .add_node("a", |_state: &DeltaState| {
            Ok(NodeOutput::new().set("final_answer", "draft"))
        })
        .add_node("b", |_state: &DeltaState| {
            let answer = interrupt(json!({ "question": "answer" }))?;
            Ok(NodeOutput::new().set(
                "final_answer",
                answer.as_str().unwrap_or("done").to_string(),
            ))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &DeltaState::default(),
            &RunConfig::new().with_thread("finish-channel"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    let snapshot = graph.get_state("finish-channel").unwrap().unwrap();
    assert_eq!(
        snapshot.values.final_answer, "",
        "held writes must not leak into checkpoints"
    );

    let output = graph.resume("finish-channel", json!("resumed")).unwrap();
    assert_eq!(output.final_answer, "resumed");
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_last_value_after_finish_holds_writes() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<DeltaState>::new()
        .add_node("a", |_state: DeltaState, _ctx| async move {
            Ok(NodeOutput::new().set("final_answer", "draft"))
        })
        .add_node("b", |state: DeltaState, _ctx| async move {
            Ok(NodeOutput::new()
                .set("final_answer", "done")
                .set("observed", vec![format!("b:{}", state.final_answer)]))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let output = graph.invoke(&DeltaState::default()).await.unwrap();
    assert_eq!(output.final_answer, "done");
    assert_eq!(output.observed, vec!["b:".to_string()]);
}
