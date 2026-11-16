//! Nodes returning a LangGraph-style `Command`.

use langgraph_rs::{Command, JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json};

#[test]
fn sync_node_can_return_a_command() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_command_node("route", |_state: &JsonState| {
            Ok(Command::continue_thread()
                .goto("target")
                .with_update([("value", json!(5))]))
        })
        .add_node("target", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("route")
        .set_finish_point("target")
        .compile()
        .unwrap();

    let output = graph.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(output.get("value"), Some(&json!(6)));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_node_can_return_a_command() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_command_node("route", |_state: JsonState, _ctx| async move {
            Ok(Command::continue_thread()
                .goto("target")
                .with_update([("value", json!(5))]))
        })
        .add_node("target", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("route")
        .set_finish_point("target")
        .compile()
        .unwrap();

    let output = graph
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(6)));
}

#[test]
fn invoke_command_starts_at_goto_nodes() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("a", |_state: &JsonState| {
            Ok(NodeOutput::new().update("value", json!(100)))
        })
        .add_node("b", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("a")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let command = Command::continue_thread()
        .goto("b")
        .with_update([("value", json!(1))]);
    let output = graph.invoke_command(command, &RunConfig::new()).unwrap();
    assert_eq!(output.get("value"), Some(&json!(2)));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_invoke_command_starts_at_goto_nodes() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("a", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new().update("value", json!(100)))
        })
        .add_node("b", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("a")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let command = Command::continue_thread()
        .goto("b")
        .with_update([("value", json!(1))]);
    let output = graph
        .invoke_command(command, &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.get("value"), Some(&json!(2)));
}

#[test]
fn command_update_without_goto_starts_at_entry_points() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile()
        .unwrap();

    let command = Command::continue_thread().with_update([("value", json!(10))]);
    let output = graph.invoke_command(command, &RunConfig::new()).unwrap();
    assert_eq!(output.get("value"), Some(&json!(11)));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_command_update_without_goto_starts_at_entry_points() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile()
        .unwrap();

    let command = Command::continue_thread().with_update([("value", json!(10))]);
    let output = graph
        .invoke_command(command, &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.get("value"), Some(&json!(11)));
}
