//! Graphviz DOT/PNG rendering (PNG requires `dot` on PATH).

use langgraph_rs::{NodeOutput, State, StateGraph, StateSchema};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct DrawState {
    value: i64,
}

impl State for DrawState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

fn graph() -> langgraph_rs::CompiledGraph<DrawState> {
    StateGraph::<DrawState>::new()
        .add_node("work", |_state: &DrawState| {
            Ok(NodeOutput::new().set("value", 1))
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap()
}

#[test]
fn dot_contains_nodes_and_edges() {
    let dot = graph().to_dot();
    assert!(dot.contains("digraph langgraph_rs"));
    assert!(dot.contains("\"START\" -> \"work\""));
    assert!(dot.contains("\"work\" -> \"END\""));
}

#[test]
fn draw_png_uses_graphviz_when_available() {
    match graph().draw_png() {
        Ok(bytes) => {
            assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        }
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("Graphviz") || message.contains("graphviz"),
                "unexpected draw_png error: {message}"
            );
        }
    }
}
