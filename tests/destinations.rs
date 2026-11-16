//! Node `destinations` allow-lists and async batch execution.

use langgraph_rs::{NodeOutput, Route, RunConfig, Send, State, StateGraph, StateSchema, json};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct DestState {
    value: i64,
}

impl State for DestState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn node(value: i64) -> impl Fn(&DestState) -> langgraph_rs::Result<NodeOutput> {
    move |_state: &DestState| Ok(NodeOutput::new().set("value", value))
}

#[test]
fn declared_destinations_are_allowed() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], node(1))
        .add_node("b", node(2))
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    assert_eq!(
        graph
            .invoke(&DestState::default(), &RunConfig::new())
            .unwrap()
            .value,
        2
    );
}

#[test]
fn goto_outside_destinations_is_rejected() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], |_state: &DestState| {
            Ok(NodeOutput::new().goto("c"))
        })
        .add_node("b", node(2))
        .add_node("c", node(3))
        .set_entry_point("a")
        .set_finish_point("b")
        .set_finish_point("c")
        .compile()
        .unwrap();

    let error = graph
        .run(&DestState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("not a declared destination"),
        "unexpected error: {error}"
    );
}

#[test]
fn static_edge_outside_destinations_is_rejected() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], node(1))
        .add_node("b", node(2))
        .add_node("c", node(3))
        .set_entry_point("a")
        .add_edge("a", "c")
        .set_finish_point("b")
        .set_finish_point("c")
        .compile()
        .unwrap();

    let error = graph
        .run(&DestState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a declared destination"));
}

#[test]
fn conditional_route_outside_destinations_is_rejected() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], node(1))
        .add_node("b", node(2))
        .add_node("c", node(3))
        .set_entry_point("a")
        .add_conditional_edges("a", |_state: &DestState| Ok(Route::node("c")))
        .set_finish_point("b")
        .set_finish_point("c")
        .compile()
        .unwrap();

    let error = graph
        .run(&DestState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a declared destination"));
}

#[test]
fn send_outside_destinations_is_rejected() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], |_state: &DestState| {
            Ok(NodeOutput::new().send(Send::with_arg("c", json!({ "value": 3 }))))
        })
        .add_node("b", node(2))
        .add_node("c", node(3))
        .set_entry_point("a")
        .set_finish_point("b")
        .set_finish_point("c")
        .compile()
        .unwrap();

    let error = graph
        .run(&DestState::default(), &RunConfig::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a declared destination"));
}

#[test]
fn end_is_always_an_allowed_destination() {
    let graph = StateGraph::<DestState>::new()
        .add_node_with_destinations("a", ["b"], node(1))
        .add_node("b", node(2))
        .set_entry_point("a")
        .set_finish_point("a")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&DestState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 1);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_batch_preserves_order_and_reports_failures() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<DestState>::new()
        .add_node("work", |state: DestState, _ctx| async move {
            if state.value < 0 {
                return Err(langgraph_rs::Error::other("negative input"));
            }
            Ok(NodeOutput::new().set("value", state.value + 1))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap();

    let inputs = vec![0, -1, 5, 10]
        .into_iter()
        .map(|value| DestState { value });
    let results = graph
        .batch(inputs, &RunConfig::new().with_max_concurrency(2))
        .await;
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].as_ref().unwrap().value, 1);
    assert!(results[1].is_err());
    assert_eq!(results[2].as_ref().unwrap().value, 6);
    assert_eq!(results[3].as_ref().unwrap().value, 11);
}
