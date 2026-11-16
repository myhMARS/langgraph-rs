//! Scheduling of the superstep that follows a completed one.
//!
//! The helpers are generic over closures instead of the compiled graph so both the
//! synchronous kernel executor and the async runtime adapter share one implementation.

use std::collections::HashSet;

use crate::kernel::error::{Error, Result};
use crate::kernel::node::ConditionalEdge;
use crate::kernel::state::State;
use crate::{END, START};

/// Collect the outgoing destinations of the nodes executed in a superstep.
///
/// A node that used an explicit `goto` contributes only its goto targets; every other
/// node contributes its static edges followed by its conditional edges, evaluated
/// against the already merged state.
pub(crate) fn collect_next_nodes<'a, S: State>(
    state: &S,
    executed: &[String],
    goto_nodes: &HashSet<String>,
    goto_targets: &[(String, String)],
    edges_of: impl Fn(&str) -> Option<&'a [String]>,
    conditional_of: impl Fn(&str) -> Option<&'a [ConditionalEdge<S>]>,
    node_exists: impl Fn(&str) -> bool,
) -> Result<Vec<String>> {
    let mut next_nodes: Vec<String> = Vec::new();

    for node in executed {
        if goto_nodes.contains(node) {
            continue;
        }
        if let Some(targets) = edges_of(node) {
            for target in targets {
                check_target(node, target, &node_exists)?;
                next_nodes.push(target.clone());
            }
        }
        if let Some(edges) = conditional_of(node) {
            for edge in edges {
                for target in edge.resolve(state, node)? {
                    check_target(node, &target, &node_exists)?;
                    next_nodes.push(target);
                }
            }
        }
    }

    for (target, source) in goto_targets {
        if target == END {
            continue;
        }
        check_target(source, target, &node_exists)?;
        next_nodes.push(target.clone());
    }

    Ok(next_nodes)
}

/// Drop `END`/`START` markers and de-duplicate node names, preserving first-seen order.
pub(crate) fn dedupe(nodes: Vec<String>) -> Vec<String> {
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

fn check_target(source: &str, target: &str, node_exists: impl Fn(&str) -> bool) -> Result<()> {
    if target == END {
        return Ok(());
    }
    if target == START {
        return Err(Error::routing(
            source,
            format!("`{START}` is not a valid destination"),
        ));
    }
    if !node_exists(target) {
        return Err(Error::routing(
            source,
            format!("unknown destination node `{target}`"),
        ));
    }
    Ok(())
}
