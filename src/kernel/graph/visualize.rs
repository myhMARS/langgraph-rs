//! Mermaid rendering of a compiled graph.
//!
//! The renderer is layout-agnostic: the sync and async compiled graphs pass their node
//! names, edge tables and hooks, so both expose the same `to_mermaid()` output.

use std::collections::HashSet;
use std::fmt::Write;
use std::io::Write as _;
use std::process::{Command, Stdio};

use indexmap::IndexMap;

use super::Entry;
use crate::kernel::node::ConditionalEdge;
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::State;
use crate::{END, START};

/// Everything the renderer needs from a compiled graph.
pub(crate) struct GraphLayout<'a, S> {
    pub nodes: Vec<&'a str>,
    pub edges: &'a IndexMap<String, Vec<String>>,
    pub conditional: &'a IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub entry: &'a Entry<S>,
    pub retries: &'a IndexMap<String, RetryPolicy>,
    pub error_edges: &'a IndexMap<String, String>,
    pub interrupt_before: &'a HashSet<String>,
    pub interrupt_after: &'a HashSet<String>,
}

fn escape_label(label: &str) -> String {
    label.replace('"', "#quot;")
}

fn mermaid_id(name: &str) -> String {
    match name {
        START => "START".to_string(),
        END => "END".to_string(),
        other => {
            let mut id = String::from("n_");
            for character in other.chars() {
                if character.is_ascii_alphanumeric() {
                    id.push(character);
                } else {
                    id.push('_');
                }
            }
            id
        }
    }
}

/// Render a compiled graph as a Mermaid `graph TD` document.
pub(crate) fn render_mermaid<S: State>(layout: GraphLayout<'_, S>) -> String {
    let mut output = String::from("%% langgraph-rs graph\ngraph TD\n");
    for node in &layout.nodes {
        let mut label = escape_label(node);
        if let Some(policy) = layout.retries.get(*node) {
            if policy.is_retrying() {
                let _ = write!(label, "<br/>retry x{}", policy.max_attempts);
            }
        }
        if layout.interrupt_before.contains(*node) {
            label.push_str("<br/>pause before");
        }
        if layout.interrupt_after.contains(*node) {
            label.push_str("<br/>pause after");
        }
        let _ = writeln!(output, "    {}[\"{}\"]", mermaid_id(node), label);
    }
    output.push_str("    START([\"START\"])\n");
    output.push_str("    END([\"END\"])\n");

    match layout.entry {
        Entry::Static(entries) => {
            for entry in entries {
                let _ = writeln!(output, "    START --> {}", mermaid_id(entry));
            }
        }
        Entry::Conditional(edge) => {
            output.push_str("    START -.-> entry_router{\"entry router\"}\n");
            render_conditional_targets(&mut output, "entry_router", edge, &layout.nodes);
        }
    }

    for (source, targets) in layout.edges {
        for target in targets {
            let _ = writeln!(
                output,
                "    {} --> {}",
                mermaid_id(source),
                mermaid_id(target)
            );
        }
    }

    for (source, edges) in layout.conditional {
        for (index, edge) in edges.iter().enumerate() {
            let router = format!("{}_router_{index}", mermaid_id(source));
            let _ = writeln!(
                output,
                "    {} --> {router}{{\"router\"}}",
                mermaid_id(source)
            );
            render_conditional_targets(&mut output, &router, edge, &layout.nodes);
        }
    }

    for (source, handler) in layout.error_edges {
        let _ = writeln!(
            output,
            "    {} -.->|\"on error\"| {}",
            mermaid_id(source),
            mermaid_id(handler)
        );
    }

    output
}

fn render_conditional_targets<S>(
    output: &mut String,
    router: &str,
    edge: &ConditionalEdge<S>,
    nodes: &[&str],
) {
    match &edge.path_map {
        Some(path_map) => {
            for (key, targets) in path_map.entries() {
                for target in targets {
                    let _ = writeln!(
                        output,
                        "    {router} -.->|\"{}\"| {}",
                        escape_label(key),
                        mermaid_id(target)
                    );
                }
            }
        }
        None => {
            // Without a path map the router returns node names directly; render the
            // possible destinations as a comment instead of guessing edges.
            let names: Vec<String> = nodes
                .iter()
                .map(|node| mermaid_id(node))
                .filter(|id| id != "START" && id != "END")
                .collect();
            let _ = writeln!(
                output,
                "    %% router `{router}` may target any node: {} or END",
                names.join(", ")
            );
        }
    }
}

/// Render a compiled graph as a compact ASCII map (nodes plus outgoing edges).
pub(crate) fn render_ascii<S: State>(layout: GraphLayout<'_, S>) -> String {
    let mut output = String::new();
    output.push_str("START\n");
    match layout.entry {
        Entry::Static(entries) => {
            for entry in entries {
                let entry = if entry == START || entry == END {
                    entry.to_uppercase()
                } else {
                    entry.clone()
                };
                let _ = writeln!(output, "  -> {entry}");
            }
        }
        Entry::Conditional(_) => output.push_str("  -> (entry router)\n"),
    }
    for node in &layout.nodes {
        let mut targets: Vec<String> = layout
            .edges
            .get(*node)
            .map(|edges| {
                edges
                    .iter()
                    .map(|target| {
                        if target == END {
                            "END".to_string()
                        } else {
                            target.clone()
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        if layout.conditional.contains_key(*node) {
            targets.push("(router)".to_string());
        }
        if let Some(handler) = layout.error_edges.get(*node) {
            targets.push(format!("on_error:{handler}"));
        }
        targets.sort();
        targets.dedup();
        let _ = write!(output, "{node}");
        if layout.interrupt_before.contains(*node) {
            output.push_str(" [pause before]");
        }
        if layout.interrupt_after.contains(*node) {
            output.push_str(" [pause after]");
        }
        if targets.is_empty() {
            output.push_str(" -> END");
        } else {
            for target in targets {
                let _ = write!(output, " -> {target}");
            }
        }
        output.push('\n');
    }
    output
}

fn dot_id(name: &str) -> &str {
    match name {
        START => "START",
        END => "END",
        other => other,
    }
}

/// Render a compiled graph as a Graphviz DOT document.
pub(crate) fn render_dot<S: State>(layout: GraphLayout<'_, S>) -> String {
    let mut output = String::from("digraph langgraph_rs {\n    rankdir=TB;\n");
    output.push_str("    START [shape=oval];\n    END [shape=oval];\n");
    for node in &layout.nodes {
        let mut label = escape_label(dot_id(node));
        if let Some(policy) = layout.retries.get(*node) {
            if policy.is_retrying() {
                let _ = write!(label, "\\nretry x{}", policy.max_attempts);
            }
        }
        if layout.interrupt_before.contains(*node) {
            label.push_str("\\npause before");
        }
        if layout.interrupt_after.contains(*node) {
            label.push_str("\\npause after");
        }
        let _ = writeln!(
            output,
            "    \"{}\" [shape=box, style=rounded, label=\"{}\"];",
            escape_label(dot_id(node)),
            label
        );
    }
    match layout.entry {
        Entry::Static(entries) => {
            for entry in entries {
                let _ = writeln!(
                    output,
                    "    \"START\" -> \"{}\";",
                    escape_label(dot_id(entry))
                );
            }
        }
        Entry::Conditional(edge) => {
            if let Some(path_map) = &edge.path_map {
                for (key, targets) in path_map.entries() {
                    for target in targets {
                        let _ = writeln!(
                            output,
                            "    \"START\" -> \"{}\" [label=\"{}\", style=dashed];",
                            escape_label(dot_id(target)),
                            escape_label(key)
                        );
                    }
                }
            }
        }
    }
    for (source, targets) in layout.edges {
        for target in targets {
            let _ = writeln!(
                output,
                "    \"{}\" -> \"{}\";",
                escape_label(dot_id(source)),
                escape_label(dot_id(target))
            );
        }
    }
    for (source, edges) in layout.conditional {
        for edge in edges {
            if let Some(path_map) = &edge.path_map {
                for (key, targets) in path_map.entries() {
                    for target in targets {
                        let _ = writeln!(
                            output,
                            "    \"{}\" -> \"{}\" [label=\"{}\", style=dashed];",
                            escape_label(dot_id(source)),
                            escape_label(dot_id(target)),
                            escape_label(key)
                        );
                    }
                }
            }
        }
    }
    for (source, handler) in layout.error_edges {
        let _ = writeln!(
            output,
            "    \"{}\" -> \"{}\" [label=\"on error\", style=dotted];",
            escape_label(dot_id(source)),
            escape_label(dot_id(handler))
        );
    }
    output.push_str("}\n");
    output
}

/// Run Graphviz `dot -Tpng` over a DOT document.
pub(crate) fn dot_to_png(dot: &str) -> crate::kernel::error::Result<Vec<u8>> {
    let mut child = Command::new("dot")
        .arg("-Tpng")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            crate::kernel::error::Error::other(format!(
                "draw_png requires Graphviz `dot` on PATH: {error}"
            ))
        })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(dot.as_bytes()).map_err(|error| {
            crate::kernel::error::Error::other(format!("graphviz stdin: {error}"))
        })?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| crate::kernel::error::Error::other(format!("graphviz wait: {error}")))?;
    if !output.status.success() {
        return Err(crate::kernel::error::Error::other(format!(
            "graphviz `dot` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::graph::StateGraph;
    use crate::kernel::node::{NodeOutput, PathMap, Route};
    use crate::kernel::state::StateSchema;

    #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
    struct DemoState {
        value: i64,
    }

    impl State for DemoState {
        fn schema() -> StateSchema {
            StateSchema::new().last_value("value")
        }
    }

    #[test]
    fn mermaid_contains_nodes_edges_and_hooks() {
        let graph = StateGraph::<DemoState>::new()
            .add_node("work", |_state: &DemoState| Ok(NodeOutput::new()))
            .add_node("finish", |_state: &DemoState| Ok(NodeOutput::new()))
            .add_node("fallback", |_state: &DemoState| Ok(NodeOutput::new()))
            .set_entry_point("work")
            .add_conditional_edges_mapped(
                "work",
                |_state: &DemoState| Ok(Route::node("ok")),
                PathMap::from([("ok", "finish")]),
            )
            .add_error_edge("work", "fallback")
            .set_retry(
                "work",
                RetryPolicy::new(3).with_max_backoff(std::time::Duration::from_millis(1)),
            )
            .compile_with(crate::kernel::graph::CompileOptions::new().interrupt_before(["work"]))
            .unwrap();

        let mermaid = graph.to_mermaid();
        assert!(mermaid.starts_with("%% langgraph-rs graph\ngraph TD"));
        assert!(mermaid.contains("START --> n_work"));
        assert!(mermaid.contains("n_work[\"work<br/>retry x3<br/>pause before\"]"));
        assert!(mermaid.contains("n_work_router_0{\"router\"}"));
        assert!(mermaid.contains(".->|\"ok\"| n_finish"));
        assert!(mermaid.contains("-.->|\"on error\"| n_fallback"));
    }
}
