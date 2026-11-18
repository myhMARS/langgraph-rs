//! Per-node consumption tracking tests.

use langgraph_rs::{
    NodeOutput, RunConfig, State, StateGraph, StateSchema, StreamChunk, Usage, record_tokens,
    record_usage, reducers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
struct CostState {
    value: i64,
    notes: Vec<String>,
}

impl State for CostState {
    fn schema() -> StateSchema {
        StateSchema::new()
            .last_value("value")
            .with_reducer("notes", reducers::append())
    }
}

fn cost_graph() -> langgraph_rs::CompiledGraph<CostState> {
    StateGraph::<CostState>::new()
        .add_node("planner", |state: &CostState| {
            // Usage can be reported piecewise; the engine sums it per execution.
            record_tokens(100, 20)?;
            record_usage(
                Usage::new()
                    .with_model("gpt-4o")
                    .with_cost_usd(0.01)
                    .with_metadata("tool", "search"),
            )?;
            Ok(NodeOutput::new()
                .set("value", state.value + 1)
                .set("notes", vec!["plan".to_string()]))
        })
        .add_node("writer", |_state: &CostState| {
            // Usage can also travel with the output.
            Ok(NodeOutput::new()
                .set("value", 2)
                .set("notes", vec!["write".to_string()])
                .with_usage(
                    Usage::new()
                        .with_model("gpt-4o-mini")
                        .with_input_tokens(10)
                        .with_output_tokens(5),
                ))
        })
        .set_entry_point("planner")
        .add_edge("planner", "writer")
        .set_finish_point("writer")
        .compile()
        .unwrap()
}

#[test]
fn run_report_tracks_duration_and_usage_per_node() {
    let graph = cost_graph();
    let outcome = graph.run(&CostState::default(), &RunConfig::new()).unwrap();

    assert_eq!(outcome.state.value, 2);
    assert_eq!(outcome.state.notes, vec!["plan", "write"]);
    assert_eq!(outcome.report.executions().len(), 2);

    let planner = &outcome.report.executions()[0];
    assert_eq!(planner.node, "planner");
    // Steps follow LangGraph's tick counter: the first superstep is 1.
    assert_eq!(planner.step, 1);
    let planner_usage = planner.usage.as_ref().expect("planner reported usage");
    assert_eq!(planner_usage.input_tokens, Some(100));
    assert_eq!(planner_usage.output_tokens, Some(20));
    assert_eq!(planner_usage.model.as_deref(), Some("gpt-4o"));
    assert_eq!(
        planner_usage.metadata.as_ref().unwrap()["tool"],
        serde_json::json!("search")
    );

    let writer = &outcome.report.executions()[1];
    assert_eq!(writer.node, "writer");
    assert_eq!(writer.step, 2);
    assert_eq!(writer.usage.as_ref().unwrap().total_tokens(), 15);

    let by_node = outcome.report.usage_by_node();
    assert_eq!(by_node["planner"].total_tokens(), 120);
    assert_eq!(by_node["planner"].cost_usd, Some(0.01));
    assert_eq!(by_node["writer"].model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(outcome.report.total_tokens(), 135);
    assert!((outcome.report.total_cost_usd() - 0.01).abs() < f64::EPSILON);
    assert!(outcome.report.total_duration_millis < 60_000);
}

#[test]
fn usage_is_streamed_per_node() {
    let graph = cost_graph();
    let chunks = graph
        .stream(&CostState::default(), &RunConfig::new())
        .unwrap();

    let usage_chunks: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk.execution())
        .collect();
    assert_eq!(usage_chunks.len(), 2);
    assert_eq!(usage_chunks[0].node, "planner");
    assert_eq!(usage_chunks[1].node, "writer");
    assert!(chunks.iter().any(|chunk| matches!(
        chunk,
        StreamChunk::Usage { execution } if execution.usage.as_ref().is_some_and(|usage| usage.total_tokens() == 120)
    )));
}

#[test]
fn usage_can_be_recorded_multiple_times_per_node() {
    let graph = StateGraph::<CostState>::new()
        .add_node("multi", |_state: &CostState| {
            record_tokens(1, 2)?;
            record_tokens(10, 20)?;
            Ok(NodeOutput::new())
        })
        .set_entry_point("multi")
        .set_finish_point("multi")
        .compile()
        .unwrap();

    let outcome = graph.run(&CostState::default(), &RunConfig::new()).unwrap();
    assert_eq!(outcome.report.total_tokens(), 33);
}

#[test]
fn recording_usage_outside_a_node_fails() {
    assert!(record_usage(Usage::new()).is_err());
    assert!(record_tokens(1, 1).is_err());
}
