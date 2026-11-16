//! Multiple pending interrupts resumed through an `{interrupt id: value}` map.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json,
};
use serde::{Deserialize, Serialize};
use serde_json::Map;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct MultiState {
    a: i64,
    b: i64,
    done: bool,
}

impl State for MultiState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("a")
            .last_value("b")
            .last_value("done")
    }
}

fn interrupted_node(
    channel: &'static str,
) -> impl Fn(&MultiState) -> langgraph_rs::Result<NodeOutput> {
    move |_state: &MultiState| {
        let answer = interrupt(json!({ "question": channel }))?;
        Ok(NodeOutput::new().set(channel, answer.as_i64().unwrap_or_default()))
    }
}

fn two_interrupt_graph() -> langgraph_rs::CompiledGraph<MultiState> {
    let saver = Arc::new(InMemorySaver::new());
    StateGraph::<MultiState>::new()
        .add_node("a", interrupted_node("a"))
        .add_node("b", interrupted_node("b"))
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .set_finish_point("a")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap()
}

#[test]
fn multiple_interrupts_are_reported_and_resumed_by_id() {
    let graph = two_interrupt_graph();
    let error = graph
        .run(
            &MultiState::default(),
            &RunConfig::new().with_thread("multi-interrupt"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("multi-interrupt").unwrap().unwrap();
    assert_eq!(snapshot.interrupts.len(), 2);
    let mut ids: Vec<String> = snapshot
        .interrupts
        .iter()
        .map(|interrupt| interrupt.id.clone())
        .collect();
    assert!(ids.iter().all(|id| !id.is_empty()));
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 2);

    let mut resume = Map::new();
    for interrupt in &snapshot.interrupts {
        let value = if interrupt.node == "a" { 5 } else { 7 };
        resume.insert(interrupt.id.clone(), json!(value));
    }
    let output = graph
        .resume("multi-interrupt", serde_json::Value::Object(resume))
        .unwrap();
    assert_eq!(output.a, 5);
    assert_eq!(output.b, 7);
}

#[test]
fn successful_writes_are_committed_before_multiple_interrupts() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<MultiState>::new()
        .add_node("ok", |_state: &MultiState| {
            Ok(NodeOutput::new().set("done", true))
        })
        .add_node("a", interrupted_node("a"))
        .add_node("b", interrupted_node("b"))
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .set_finish_point("ok")
        .set_finish_point("a")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &MultiState::default(),
            &RunConfig::new().with_thread("multi-mixed"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());
    let snapshot = graph.get_state("multi-mixed").unwrap().unwrap();
    assert!(snapshot.values.done, "successful writes must be committed");
    assert_eq!(snapshot.interrupts.len(), 2);

    let mut resume = Map::new();
    for interrupt in &snapshot.interrupts {
        let value = if interrupt.node == "a" { 1 } else { 2 };
        resume.insert(interrupt.id.clone(), json!(value));
    }
    let output = graph
        .resume("multi-mixed", serde_json::Value::Object(resume))
        .unwrap();
    assert_eq!(output.a, 1);
    assert_eq!(output.b, 2);
    assert!(output.done);
}

#[test]
fn a_single_interrupt_still_accepts_object_payloads() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<MultiState>::new()
        .add_node("a", |_state: &MultiState| {
            let payload = interrupt(json!({ "question": "payload" }))?;
            Ok(NodeOutput::new().set(
                "a",
                payload
                    .get("answer")
                    .and_then(|value| value.as_i64())
                    .unwrap_or_default(),
            ))
        })
        .set_entry_point("a")
        .set_finish_point("a")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let _ = graph
        .run(
            &MultiState::default(),
            &RunConfig::new().with_thread("single-object"),
        )
        .unwrap_err();
    let output = graph
        .resume("single-object", json!({ "answer": 9 }))
        .unwrap();
    assert_eq!(output.a, 9);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_multiple_interrupts_resume_by_id() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let saver = Arc::new(InMemorySaver::new());
    let graph = AsyncStateGraph::<MultiState>::new()
        .add_node("a", |_state: MultiState, ctx| async move {
            let answer = ctx.interrupt(json!({ "question": "a" }))?;
            Ok(NodeOutput::new().set("a", answer.as_i64().unwrap_or_default()))
        })
        .add_node("b", |_state: MultiState, ctx| async move {
            let answer = ctx.interrupt(json!({ "question": "b" }))?;
            Ok(NodeOutput::new().set("b", answer.as_i64().unwrap_or_default()))
        })
        .add_edge(langgraph_rs::START, "a")
        .add_edge(langgraph_rs::START, "b")
        .set_finish_point("a")
        .set_finish_point("b")
        .with_checkpointer(saver)
        .compile()
        .unwrap();

    let error = graph
        .run(
            &MultiState::default(),
            &RunConfig::new().with_thread("async-multi"),
        )
        .await
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("async-multi").unwrap().unwrap();
    assert_eq!(snapshot.interrupts.len(), 2);
    let mut resume = Map::new();
    for interrupt in &snapshot.interrupts {
        let value = if interrupt.node == "a" { 11 } else { 13 };
        resume.insert(interrupt.id.clone(), json!(value));
    }
    let output = graph
        .resume("async-multi", serde_json::Value::Object(resume))
        .await
        .unwrap();
    assert_eq!(output.a, 11);
    assert_eq!(output.b, 13);
}
