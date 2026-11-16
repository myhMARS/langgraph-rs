//! End-to-end tests for graph execution: edges, routing, parallelism and errors.

use std::sync::{Arc, Mutex};

use langgraph_rs::{
    CompileOptions, Error, InMemorySaver, JsonState, NodeOutput, PathMap, Route, RunConfig, Send,
    State, StateGraph, StateSchema, StreamChunk, custom, interrupt, reducers,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct WorkState {
    value: i64,
    branch: String,
    log: Vec<String>,
}

impl State for WorkState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .last_value("branch")
            .with_reducer("log", reducers::append())
    }
}

fn log(value: &str) -> NodeOutput {
    NodeOutput::new().set("log", vec![value.to_string()])
}

fn log_node(
    value: &'static str,
) -> impl Fn(&WorkState) -> langgraph_rs::Result<NodeOutput>
+ std::marker::Send
+ std::marker::Sync
+ 'static {
    move |_state: &WorkState| Ok(log(value))
}

#[test]
fn linear_graph_runs_every_node_in_order() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("first", |state: &WorkState| {
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["first".to_string()]))
        })
        .add_node("second", |state: &WorkState| {
            Ok(NodeOutput::new()
                .set("value", state.value * 2)
                .set("log", vec!["second".to_string()]))
        })
        .add_edge(langgraph_rs::START, "first")
        .add_edge("first", "second")
        .set_finish_point("second")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 2);
    assert_eq!(output.log, vec!["first", "second"]);
}

#[test]
fn conditional_edges_loop_until_the_router_returns_end() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("increment", |state: &WorkState| {
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("log", vec!["increment".to_string()]))
        })
        .set_entry_point("increment")
        .add_conditional_edges("increment", |state: &WorkState| {
            Ok(if state.value < 5 {
                Route::node("increment")
            } else {
                Route::end()
            })
        })
        .compile()
        .unwrap();

    let output = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.value, 5);
    assert_eq!(output.log.len(), 5);
}

#[test]
fn conditional_edges_resolve_symbolic_keys_through_a_path_map() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("decide", |_state: &WorkState| {
            Ok(NodeOutput::new().set("branch", "ok"))
        })
        .add_node("finish", log_node("finish"))
        .set_entry_point("decide")
        .add_conditional_edges_mapped(
            "decide",
            |state: &WorkState| Ok(Route::node(state.branch.as_str())),
            PathMap::from([("ok", "finish")]),
        )
        .compile()
        .unwrap();

    let output = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.log, vec!["finish"]);
}

#[test]
fn parallel_fan_out_runs_nodes_and_joins_once() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("start", |_state: &WorkState| Ok(log("start")))
        .add_node("left", |_state: &WorkState| Ok(log("left")))
        .add_node("right", |_state: &WorkState| Ok(log("right")))
        .add_node("join", |_state: &WorkState| Ok(log("join")))
        .set_entry_point("start")
        .add_edge("start", ["left", "right"])
        .add_edge("left", "join")
        .add_edge("right", "join")
        .set_finish_point("join")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert!(output.log.contains(&"left".to_string()));
    assert!(output.log.contains(&"right".to_string()));
    assert_eq!(output.log.first().map(String::as_str), Some("start"));
    assert_eq!(output.log.last().map(String::as_str), Some("join"));
    assert_eq!(
        output.log.iter().filter(|entry| *entry == "join").count(),
        1
    );
}

#[test]
fn conditional_entry_point_selects_the_first_node() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("positive", |_state: &WorkState| Ok(log("positive")))
        .add_node("negative", |_state: &WorkState| Ok(log("negative")))
        .set_conditional_entry_point(|state: &WorkState| {
            Ok(if state.value > 0 {
                Route::node("positive")
            } else {
                Route::node("negative")
            })
        })
        .set_finish_point("positive")
        .set_finish_point("negative")
        .compile()
        .unwrap();

    let output = graph
        .invoke(
            &WorkState {
                value: 3,
                ..WorkState::default()
            },
            &RunConfig::new(),
        )
        .unwrap();
    assert_eq!(output.log, vec!["positive"]);

    let output = graph
        .invoke(
            &WorkState {
                value: -1,
                ..WorkState::default()
            },
            &RunConfig::new(),
        )
        .unwrap();
    assert_eq!(output.log, vec!["negative"]);
}

#[test]
fn topic_channels_append_one_element_per_write() {
    let schema = StateSchema::dynamic().topic("events");
    let graph = StateGraph::<JsonState>::with_schema(schema)
        .add_node("a", |_state: &JsonState| {
            Ok(NodeOutput::new().set("events", "a"))
        })
        .add_node("b", |_state: &JsonState| {
            Ok(NodeOutput::new().set("events", "b"))
        })
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&JsonState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.get("events"), Some(&json!(["a", "b"])));
}

#[test]
fn explicit_goto_overrides_static_edges() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("start", |_state: &WorkState| {
            Ok(NodeOutput::new()
                .set("log", vec!["start".to_string()])
                .goto("target"))
        })
        .add_node("skipped", log_node("skipped"))
        .add_node("target", log_node("target"))
        .set_entry_point("start")
        .add_edge("start", "skipped")
        .set_finish_point("target")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.log, vec!["start", "target"]);
}

#[test]
fn send_fans_out_with_per_task_state() {
    let schema = StateSchema::dynamic()
        .last_value("done")
        .with_reducer("results", reducers::append());
    let graph = StateGraph::<JsonState>::with_schema(schema)
        .add_node("dispatch", |_state: &JsonState| {
            Ok(NodeOutput::new()
                .send(Send::with_arg("worker", json!({ "item": 1 })))
                .send(Send::with_arg("worker", json!({ "item": 2 })))
                .send(Send::with_arg("worker", json!({ "item": 3 }))))
        })
        .add_node("worker", |state: &JsonState| {
            let item = state
                .get("item")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or_default();
            Ok(NodeOutput::new().set("results", json!([item * 10])))
        })
        .add_node("finish", |_state: &JsonState| {
            Ok(NodeOutput::new().set("done", true))
        })
        .set_entry_point("dispatch")
        .add_edge("worker", "finish")
        .set_finish_point("finish")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&JsonState::default(), &RunConfig::new())
        .unwrap();
    let mut results: Vec<i64> = output.get_as("results").unwrap();
    results.sort_unstable();
    assert_eq!(results, vec![10, 20, 30]);
    assert_eq!(output.get("done"), Some(&json!(true)));
}

#[test]
fn custom_events_are_streamed() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("emit", |_state: &WorkState| {
            custom(json!({ "progress": 0.5 }))?;
            Ok(NodeOutput::new().set("value", 1))
        })
        .set_entry_point("emit")
        .set_finish_point("emit")
        .compile()
        .unwrap();

    let chunks = graph
        .stream(&WorkState::default(), &RunConfig::new())
        .unwrap();
    assert!(chunks.iter().any(|chunk| matches!(
        chunk,
        StreamChunk::Custom { node, value } if node == "emit" && value == &json!({ "progress": 0.5 })
    )));
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Values { .. }))
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| matches!(chunk, StreamChunk::Updates { .. }))
    );
}

#[test]
fn stream_with_keeps_chunks_when_the_graph_interrupts() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("ask", |_state: &WorkState| {
            let answer = interrupt(json!({ "question": "value?" }))?;
            Ok(NodeOutput::new().set("value", answer.as_i64().unwrap_or_default()))
        })
        .set_entry_point("ask")
        .set_finish_point("ask")
        .compile()
        .unwrap();

    let chunks = Arc::new(Mutex::new(Vec::new()));
    let sink_chunks = Arc::clone(&chunks);
    let error = graph
        .stream_with(&WorkState::default(), &RunConfig::new(), move |chunk| {
            sink_chunks.lock().unwrap().push(chunk);
        })
        .unwrap_err();
    assert!(error.is_interrupt());
    assert!(chunks.lock().unwrap().iter().any(|chunk| matches!(
        chunk,
        StreamChunk::Interrupt { node, .. } if node == "ask"
    )));
}

#[test]
fn recursion_limit_stops_runaway_loops() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("loop", |_state: &WorkState| Ok(NodeOutput::new()))
        .set_entry_point("loop")
        .add_edge("loop", "loop")
        .compile_with(CompileOptions::new().with_recursion_limit(3))
        .unwrap();

    let error = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap_err();
    assert!(matches!(error, Error::RecursionLimit(3)));
}

#[test]
fn per_run_recursion_limit_overrides_the_compiled_default() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("loop", |_state: &WorkState| Ok(NodeOutput::new()))
        .set_entry_point("loop")
        .add_edge("loop", "loop")
        .compile()
        .unwrap();

    let error = graph
        .invoke(
            &WorkState::default(),
            &RunConfig::new().with_recursion_limit(2),
        )
        .unwrap_err();
    assert!(matches!(error, Error::RecursionLimit(2)));
}

#[test]
fn concurrent_writes_to_a_last_value_channel_are_rejected() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| {
            Ok(NodeOutput::new().set("value", 1))
        })
        .add_node("b", |_state: &WorkState| {
            Ok(NodeOutput::new().set("value", 2))
        })
        .set_entry_point("a")
        .add_edge(langgraph_rs::START, ["a", "b"])
        .set_finish_point("a")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let error = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap_err();
    assert!(matches!(error, Error::ConcurrentUpdate { .. }));
}

#[test]
fn node_errors_are_wrapped_with_the_node_name() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("boom", |_state: &WorkState| {
            Err(Error::other("intentional failure"))
        })
        .set_entry_point("boom")
        .set_finish_point("boom")
        .compile()
        .unwrap();

    let error = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap_err();
    match error {
        Error::Node { node, message } => {
            assert_eq!(node, "boom");
            assert!(message.contains("intentional failure"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn validation_reports_missing_entry_points_and_unknown_targets() {
    let no_entry = StateGraph::<WorkState>::new()
        .add_node("lonely", |_state: &WorkState| Ok(NodeOutput::new()))
        .compile();
    assert!(matches!(no_entry, Err(Error::Validation(_))));

    let unknown_target = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| Ok(NodeOutput::new()))
        .set_entry_point("a")
        .add_edge("a", "ghost")
        .compile();
    assert!(matches!(unknown_target, Err(Error::Validation(_))));

    let duplicate = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| Ok(NodeOutput::new()))
        .add_node("a", |_state: &WorkState| Ok(NodeOutput::new()))
        .set_entry_point("a")
        .compile();
    assert!(matches!(duplicate, Err(Error::Validation(_))));
}

#[test]
fn writes_to_undeclared_channels_are_rejected() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| {
            Ok(NodeOutput::new().set("not_declared", 1))
        })
        .set_entry_point("a")
        .set_finish_point("a")
        .compile()
        .unwrap();

    let error = graph
        .invoke(&WorkState::default(), &RunConfig::new())
        .unwrap_err();
    assert!(matches!(error, Error::UnknownChannel(channel) if channel == "not_declared"));
}

#[test]
fn a_checkpointer_is_required_for_thread_bound_runs() {
    let graph = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| Ok(NodeOutput::new()))
        .set_entry_point("a")
        .set_finish_point("a")
        .compile()
        .unwrap();

    let error = graph
        .invoke(
            &WorkState::default(),
            &RunConfig::new().with_thread("thread"),
        )
        .unwrap_err();
    assert!(matches!(error, Error::MissingCheckpointer(_)));
}

#[test]
fn checkpointer_records_thread_history() {
    let saver = Arc::new(InMemorySaver::new());
    let graph = StateGraph::<WorkState>::new()
        .add_node("a", |_state: &WorkState| Ok(log("a")))
        .add_node("b", |_state: &WorkState| Ok(log("b")))
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile_with(CompileOptions::new().with_checkpointer(saver.clone()))
        .unwrap();

    let output = graph
        .invoke(
            &WorkState::default(),
            &RunConfig::new().with_thread("history-thread"),
        )
        .unwrap();
    assert_eq!(output.log, vec!["a", "b"]);

    let snapshot = graph.get_state("history-thread").unwrap().unwrap();
    assert_eq!(snapshot.values.log, vec!["a", "b"]);
    assert!(snapshot.next.is_empty());
    assert!(!snapshot.interrupted);

    let history = graph.get_state_history("history-thread").unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].checkpoint_id, snapshot.checkpoint_id);
    assert_eq!(history[2].step, 0);
}
