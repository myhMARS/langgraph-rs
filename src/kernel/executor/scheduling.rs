//! Scheduling of the superstep that follows a completed one.
//!
//! The helpers are generic over closures instead of the compiled graph so both the
//! synchronous kernel executor and the async runtime adapter share one implementation.

use std::collections::HashSet;

use crate::kernel::error::{Error, Result};
use crate::kernel::node::{ConditionalEdge, Send, Task};
use crate::kernel::state::State;
use crate::{END, START};

/// Destinations and dynamic Send tasks produced after a superstep.
pub(crate) type NextTargets = (Vec<String>, Vec<(String, Send)>);

/// Collect the outgoing destinations of the nodes executed in a superstep.
///
/// A node that used an explicit `goto` contributes only its goto targets; every other
/// node contributes its static edges followed by its conditional edges, evaluated
/// against the already merged state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_next_nodes<'a, S: State>(
    state: &S,
    executed: &[String],
    goto_nodes: &HashSet<String>,
    goto_targets: &[(String, String)],
    edges_of: impl Fn(&str) -> Option<&'a [String]>,
    conditional_of: impl Fn(&str) -> Option<&'a [ConditionalEdge<S>]>,
    node_exists: impl Fn(&str) -> bool,
    destinations_of: impl Fn(&str) -> Option<&'a [String]>,
) -> Result<NextTargets> {
    let mut next_nodes: Vec<String> = Vec::new();
    let mut router_sends: Vec<(String, Send)> = Vec::new();

    for node in executed {
        if goto_nodes.contains(node) {
            continue;
        }
        if let Some(targets) = edges_of(node) {
            for target in targets {
                check_target(node, target, &node_exists, &destinations_of)?;
                next_nodes.push(target.clone());
            }
        }
        if let Some(edges) = conditional_of(node) {
            for edge in edges {
                let (targets, sends) = edge.resolve(state, node)?;
                for target in targets {
                    check_target(node, &target, &node_exists, &destinations_of)?;
                    next_nodes.push(target);
                }
                router_sends.extend(sends.into_iter().map(|send| (node.clone(), send)));
            }
        }
    }

    for (target, source) in goto_targets {
        if target == END {
            continue;
        }
        check_target(source, target, &node_exists, &destinations_of)?;
        next_nodes.push(target.clone());
    }

    Ok((next_nodes, router_sends))
}

/// Convert a router [`Send`] into a task, validating its target and arg.
pub(crate) fn send_task(source: &str, send: Send) -> Result<Task> {
    let timeout_millis = send.timeout_millis;
    let input = match send.arg {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Object(map)) => Some(map),
        Some(other) => {
            return Err(Error::routing(
                source,
                format!(
                    "Send toward `{}` needs a JSON object arg, found `{other}`",
                    send.node
                ),
            ));
        }
    };
    let mut task = match input {
        Some(input) => Task::with_input(send.node, input),
        None => Task::new(send.node),
    };
    task.timeout_millis = timeout_millis;
    Ok(task)
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

fn check_target<'a>(
    source: &str,
    target: &str,
    node_exists: impl Fn(&str) -> bool,
    destinations_of: impl Fn(&str) -> Option<&'a [String]>,
) -> Result<()> {
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
    if let Some(allowed) = destinations_of(source) {
        if !allowed.iter().any(|node| node == target) {
            return Err(Error::routing(
                source,
                format!("`{target}` is not a declared destination"),
            ));
        }
    }
    Ok(())
}
