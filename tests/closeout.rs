//! Closeout checks: snapshot config, get_config, threads, retry predicates.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use langgraph_rs::{
    CompileOptions, InMemorySaver, JsonState, NodeOutput, RetryPolicy, RunConfig, RunControl,
    StateGraph, StateSchema, get_config, json,
};

#[test]
fn snapshot_exposes_config_and_threads() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("cfg", |_state: &JsonState| {
            Ok(NodeOutput::new().update("cfg", get_config()))
        })
        .set_entry_point("cfg")
        .set_finish_point("cfg")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    graph
        .run(&JsonState::new(), &RunConfig::new().with_thread("closeout"))
        .unwrap();
    let snapshot = graph.get_state("closeout").unwrap().unwrap();
    assert_eq!(
        snapshot.values.get("cfg").unwrap()["configurable"]["thread_id"],
        json!("closeout")
    );
    assert_eq!(
        snapshot.config["configurable"]["thread_id"],
        json!("closeout")
    );
    assert_eq!(
        snapshot.config["configurable"]["checkpoint_id"],
        json!(snapshot.checkpoint_id)
    );
    assert!(graph.threads().unwrap().contains(&"closeout".to_string()));
}

fn failing_graph(
    policy: RetryPolicy,
    calls: Arc<AtomicUsize>,
) -> langgraph_rs::CompiledGraph<JsonState> {
    StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node_with_retry("fail", policy, move |_state: &JsonState| {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(langgraph_rs::Error::other("boom"))
        })
        .set_entry_point("fail")
        .compile()
        .unwrap()
}

#[test]
fn retry_on_predicates_stop_unmatched_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let graph = failing_graph(
        RetryPolicy::new(3).with_retry_on(["connect"]),
        calls.clone(),
    );
    assert!(graph.run(&JsonState::new(), &RunConfig::new()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let graph = failing_graph(RetryPolicy::new(3).with_retry_on(["boom"]), calls.clone());
    assert!(graph.run(&JsonState::new(), &RunConfig::new()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn jitter_shortens_but_never_extends_backoff() {
    let base = RetryPolicy::new(3).with_backoff(std::time::Duration::from_millis(100));
    let jittered = base.clone().with_jitter(0.5);
    for attempt in 1..=3 {
        assert!(jittered.backoff_for(attempt) <= base.backoff_for(attempt));
    }
    assert!(jittered.backoff_for(1) > std::time::Duration::ZERO);
}

#[test]
fn topic_current_only_keeps_the_current_superstep() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::new().topic_current("events"))
        .add_node("first", |_state: &JsonState| {
            Ok(NodeOutput::new().update("events", json!(["a", "b"])))
        })
        .add_node("second", |_state: &JsonState| {
            Ok(NodeOutput::new().update("events", json!(["c"])))
        })
        .set_entry_point("first")
        .add_edge("first", "second")
        .set_finish_point("second")
        .compile()
        .unwrap();

    let output = graph.invoke(&JsonState::new(), &RunConfig::new()).unwrap();
    assert_eq!(output.get("events"), Some(&json!(["c"])));
}

#[test]
fn prune_keeps_the_newest_checkpoints() {
    let saver = Arc::new(InMemorySaver::new());
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
        .compile_with(CompileOptions::new().with_checkpointer(saver.clone()))
        .unwrap();

    for _ in 0..3 {
        graph
            .run(&JsonState::new(), &RunConfig::new().with_thread("prune"))
            .unwrap();
    }
    let before = graph.get_state_history("prune").unwrap();
    assert!(before.len() >= 3);
    let latest = graph.get_state("prune").unwrap().unwrap().checkpoint_id;
    graph.prune_thread("prune", 1).unwrap();
    let after = graph.get_state_history("prune").unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].checkpoint_id, latest);
}

#[test]
fn aborted_run_control_stops_the_graph() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |_state: &JsonState| {
            Ok(NodeOutput::new().update("value", json!(1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile()
        .unwrap();

    let control = RunControl::new();
    control.abort();
    let error = graph
        .run(&JsonState::new(), &RunConfig::new().with_control(control))
        .unwrap_err();
    assert!(error.to_string().contains("RunControl"));
}

#[test]
fn update_state_with_task_id_infers_the_pending_node() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("ask", |_state: &JsonState| {
            langgraph_rs::interrupt(json!({"question": "value?"}))?;
            Ok(NodeOutput::new())
        })
        .add_node("finish", |_state: &JsonState| Ok(NodeOutput::new()))
        .set_entry_point("ask")
        .add_edge("ask", "finish")
        .set_finish_point("finish")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    let _ = graph
        .run(&JsonState::new(), &RunConfig::new().with_thread("task-id"))
        .unwrap_err();
    let snapshot = graph.get_state("task-id").unwrap().unwrap();
    let task = snapshot
        .tasks
        .first()
        .expect("the interrupting node leaves a pending task")
        .clone();
    assert_eq!(task.node, "ask");
    assert!(!task.id.is_empty());

    let updated = graph
        .update_state_with_task("task-id", [("value", json!(5))], None, Some(&task.id))
        .unwrap();
    assert_eq!(updated.values.get("value"), Some(&json!(5)));
    assert!(updated.next.contains(&"finish".to_string()));
    assert!(
        graph
            .update_state_with_task("task-id", [("value", json!(6))], None, Some("missing-task"),)
            .is_err()
    );
}

#[test]
fn context_schema_is_validated_before_the_run() {
    let graph = StateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("noop", |_state: &JsonState| Ok(NodeOutput::new()))
        .set_entry_point("noop")
        .set_finish_point("noop")
        .compile_with(CompileOptions::new().with_context_schema(json!({
            "type": "object",
            "required": ["tenant"]
        })))
        .unwrap();

    graph
        .run(
            &JsonState::new(),
            &RunConfig::new().with_context(json!({"tenant": "acme"})),
        )
        .unwrap();
    let error = graph
        .run(
            &JsonState::new(),
            &RunConfig::new().with_context(json!({"other": 1})),
        )
        .unwrap_err();
    assert!(error.to_string().contains("context_schema"));
}

#[test]
fn checkpoints_carry_channel_versions() {
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
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    graph
        .run(&JsonState::new(), &RunConfig::new().with_thread("versions"))
        .unwrap();
    let document = graph.export_checkpoint("versions").unwrap().unwrap();
    let checkpoint = &document["checkpoint"];
    assert!(
        !checkpoint["channel_versions"]
            .as_object()
            .unwrap()
            .is_empty()
    );
    assert!(checkpoint["channel_versions"]["value"].is_string());
    assert!(
        checkpoint["updated_channels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|channel| channel == "value")
    );
    assert!(!checkpoint["versions_seen"].as_object().unwrap().is_empty());
}

#[test]
fn input_output_and_private_channels_are_filtered() {
    let schema = StateSchema::dynamic()
        .with_input_channels(["a"])
        .with_output_channels(["b"])
        .with_private_channels(["secret"]);
    let graph = StateGraph::<JsonState>::with_schema(schema)
        .add_node("inc", |state: &JsonState| {
            let value = state
                .get("a")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new()
                .update("b", json!(value + 1))
                .update("secret", json!("internal")))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
        .unwrap();

    let input = json!({"a": 1, "c": 9});
    let output = graph
        .run(
            &serde_json::from_value::<JsonState>(input).unwrap(),
            &RunConfig::new().with_thread("layers"),
        )
        .unwrap();
    assert_eq!(output.state.get("b"), Some(&json!(2)));
    assert_eq!(output.state.get("a"), None);
    assert_eq!(output.state.get("c"), None);
    assert_eq!(output.state.get("secret"), None);

    let snapshot = graph.get_state("layers").unwrap().unwrap();
    assert_eq!(snapshot.values.get("secret"), None);
    assert_eq!(snapshot.values.get("b"), Some(&json!(2)));
}
