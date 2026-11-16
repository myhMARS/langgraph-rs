//! Scheduling of the superstep that follows a completed one.

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::graph::CompiledGraph;
use crate::state::State;
use crate::{END, START};

/// Collect the outgoing destinations of the nodes executed in a superstep.
///
/// A node that used an explicit `goto` contributes only its goto targets; every other
/// node contributes its static edges followed by its conditional edges, evaluated
/// against the already merged state.
pub(super) fn collect_next_nodes<S: State>(
    graph: &CompiledGraph<S>,
    state: &S,
    executed: &[String],
    goto_nodes: &HashSet<String>,
    goto_targets: &[(String, String)],
) -> Result<Vec<String>> {
    let mut next_nodes: Vec<String> = Vec::new();

    for node in executed {
        if goto_nodes.contains(node) {
            continue;
        }
        if let Some(targets) = graph.edges_of(node) {
            for target in targets {
                check_target(graph, node, target)?;
                next_nodes.push(target.clone());
            }
        }
        if let Some(edges) = graph.conditional_of(node) {
            for edge in edges {
                for target in edge.resolve(state, node)? {
                    check_target(graph, node, &target)?;
                    next_nodes.push(target);
                }
            }
        }
    }

    for (target, source) in goto_targets {
        if target == END {
            continue;
        }
        check_target(graph, source, target)?;
        next_nodes.push(target.clone());
    }

    Ok(next_nodes)
}

/// Drop `END`/`START` markers and de-duplicate node names, preserving first-seen order.
pub(super) fn dedupe(nodes: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut scheduled = Vec::new();
    for node in nodes {
        if node == END || node == START {
            continue;
        }
        if seen.insert(node.clone()) {
            scheduled.push(node);
        }
    }
    scheduled
}

fn check_target<S: State>(graph: &CompiledGraph<S>, source: &str, target: &str) -> Result<()> {
    if target == END {
        return Ok(());
    }
    if target == START {
        return Err(Error::routing(
            source,
            format!("`{START}` is not a valid destination"),
        ));
    }
    if graph.node(target).is_none() {
        return Err(Error::routing(
            source,
            format!("unknown destination node `{target}`"),
        ));
    }
    Ok(())
}
