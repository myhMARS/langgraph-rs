//! Successful siblings' routes must survive an interrupted superstep.

use std::sync::Arc;

use langgraph_rs::{
    CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph, StateSchema,
    interrupt, json, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct RouteState {
    log: Vec<String>,
}

impl State for RouteState {
    fn schema() -> StateSchema {
        StateSchema::new().with_reducer("log", reducers::append())
    }
}

#[test]
fn successful_sibling_routes_are_scheduled_after_a_single_interrupt() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask", |_state: &RouteState| {
            interrupt(json!({ "q": "ask" }))?;
            Ok(NodeOutput::new().update("log", json!(["ask"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route").unwrap().unwrap();
    let resume = json!({ snapshot.interrupts[0].id.clone(): "resumed" });
    let output = graph.resume("interrupt-route", resume).unwrap();
    assert_eq!(output.log, vec!["ok", "ask", "down"]);
}

#[test]
fn successful_sibling_routes_are_scheduled_after_multiple_interrupts() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<RouteState>::new()
        .add_node("ok", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["ok"])))
        })
        .add_node("ask_a", |_state: &RouteState| {
            interrupt(json!({ "q": "a" }))?;
            Ok(NodeOutput::new().update("log", json!(["a"])))
        })
        .add_node("ask_b", |_state: &RouteState| {
            interrupt(json!({ "q": "b" }))?;
            Ok(NodeOutput::new().update("log", json!(["b"])))
        })
        .add_node("down", |_state: &RouteState| {
            Ok(NodeOutput::new().update("log", json!(["down"])))
        })
        .add_edge(langgraph_rs::START, "ok")
        .add_edge(langgraph_rs::START, "ask_a")
        .add_edge(langgraph_rs::START, "ask_b")
        .add_edge("ok", "down")
        .set_finish_point("down")
        .set_finish_point("ask_a")
        .set_finish_point("ask_b")
        .compile_with(CompileOptions::new().with_checkpointer(saver))
        .unwrap();

    let error = graph
        .run(
            &RouteState::default(),
            &RunConfig::new().with_thread("interrupt-route-multi"),
        )
        .unwrap_err();
    assert!(error.is_interrupt());

    let snapshot = graph.get_state("interrupt-route-multi").unwrap().unwrap();
    let mut resume = serde_json::Map::new();
    for interrupt in &snapshot.interrupts {
        resume.insert(interrupt.id.clone(), json!("ok"));
    }
    let output = graph
        .resume("interrupt-route-multi", serde_json::Value::Object(resume))
        .unwrap();
    assert_eq!(output.log, vec!["ok", "a", "b", "down"]);
}
