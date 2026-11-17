//! `Command(graph=PARENT)` from a subgraph node (sync graphs).

use std::sync::Arc;

use langgraph_rs::{JsonState, NodeOutput, RunConfig, StateGraph, StateSchema, json};

#[test]
fn subgraph_node_can_route_to_the_parent_graph() {
    let child = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("jump", |_state: &JsonState| {
            Ok(NodeOutput::new()
                .update("value", json!(100))
                .goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let parent = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |state: &JsonState| {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(output.get("value"), Some(&json!(101)));
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_subgraph_can_route_to_the_parent_graph() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;
    use std::sync::Arc;

    let child = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("jump", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new()
                .update("value", json!(100))
                .goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let parent = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(101)));
}

#[test]
fn parent_command_updates_merge_through_reducers() {
    use langgraph_rs::{State, StateSchema, reducers};

    #[derive(Clone, Default, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct LogState {
        log: Vec<String>,
    }

    impl State for LogState {
        fn schema() -> StateSchema {
            StateSchema::new().with_reducer("log", reducers::append())
        }
    }

    let child = StateGraph::<LogState>::new()
        .add_node("jump", |_state: &LogState| {
            Ok(NodeOutput::new()
                .update("log", json!(["child"]))
                .goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let parent = StateGraph::<LogState>::new()
        .add_node("seed", |_state: &LogState| {
            Ok(NodeOutput::new().update("log", json!(["seed"])))
        })
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |_state: &LogState| Ok(NodeOutput::new()))
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent
        .invoke(&LogState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.log, vec!["seed", "child"]);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_parent_command_aborts_sibling_nodes() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SIBLING_RAN: AtomicUsize = AtomicUsize::new(0);

    let child = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("jump", |_state: JsonState, _ctx| async move {
            Ok(NodeOutput::new()
                .update("value", json!(100))
                .goto_parent("parent_after"))
        })
        .add_node("sibling", |_state: JsonState, _ctx| async move {
            SIBLING_RAN.fetch_add(1, Ordering::SeqCst);
            Ok(NodeOutput::new().update("value", json!(999)))
        })
        .set_entry_point("jump")
        .add_edge("jump", "sibling")
        .set_finish_point("sibling")
        .compile()
        .unwrap();

    let parent = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent
        .run(&JsonState::new(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.state.get("value"), Some(&json!(101)));
    assert_eq!(SIBLING_RAN.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_parent_command_merges_through_reducers() {
    use langgraph_rs::tokio_runtime::AsyncStateGraph;
    use langgraph_rs::{State, StateSchema, reducers};
    use std::sync::Arc;

    #[derive(Clone, Default, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct LogState {
        log: Vec<String>,
    }

    impl State for LogState {
        fn schema() -> StateSchema {
            StateSchema::new().with_reducer("log", reducers::append())
        }
    }

    let child = AsyncStateGraph::<LogState>::new()
        .add_node("jump", |_state: LogState, _ctx| async move {
            Ok(NodeOutput::new()
                .update("log", json!(["child"]))
                .goto_parent("parent_after"))
        })
        .set_entry_point("jump")
        .compile()
        .unwrap();

    let parent = AsyncStateGraph::<LogState>::new()
        .add_node("seed", |_state: LogState, _ctx| async move {
            Ok(NodeOutput::new().update("log", json!(["seed"])))
        })
        .add_subgraph("child", Arc::new(child))
        .add_node("parent_after", |_state: LogState, _ctx| async move {
            Ok(NodeOutput::new())
        })
        .set_entry_point("seed")
        .add_edge("seed", "child")
        .set_finish_point("parent_after")
        .compile()
        .unwrap();

    let output = parent
        .run(&LogState::default(), &RunConfig::new())
        .await
        .unwrap();
    assert_eq!(output.state.log, vec!["seed", "child"]);
}
