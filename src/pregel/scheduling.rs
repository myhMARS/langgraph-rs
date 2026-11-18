//! Scheduling of the superstep that follows a completed one.
//!
//! The helpers are generic over closures instead of the compiled graph so both the
//! synchronous kernel executor and the async runtime adapter share one implementation.

use std::collections::HashSet;

use crate::_internal::error::{Error, Result};
use crate::_internal::node::{ConditionalEdge, Send, Task};
use crate::_internal::state::State;
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
    task.push = true;
    task.timeout_millis = timeout_millis;
    Ok(task)
}

/// Fill each task's triggers for `stream_mode="tasks"` payloads.
///
/// Static edges give exact triggers; conditional/`Send` targets fall back to the last
/// executed node (LangGraph records the triggering task's channel writes there).
pub(crate) fn fill_triggers<'a>(
    tasks: &mut [Task],
    executed: &[String],
    edges_of: impl Fn(&str) -> Option<&'a [String]>,
) {
    for task in tasks.iter_mut() {
        if !task.triggers.is_empty() {
            continue;
        }
        if executed.is_empty() {
            task.triggers.push(START.to_string());
            continue;
        }
        for node in executed {
            if edges_of(node).is_some_and(|targets| targets.iter().any(|t| t == &task.node)) {
                task.triggers.push(node.clone());
            }
        }
        if task.triggers.is_empty()
            && let Some(last) = executed.last()
        {
            task.triggers.push(last.clone());
        }
    }
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

/// Turn resolved edge targets and router Sends into the next superstep's tasks.
///
/// `next_tasks` already contains error-handler tasks from the completed superstep;
/// those nodes are not scheduled again through a regular edge. Deferred targets are
/// held until a superstep has no regular work left.
#[allow(clippy::too_many_arguments)]
pub(crate) fn schedule_next_tasks<'a>(
    next_nodes: Vec<String>,
    router_sends: Vec<(String, Send)>,
    mut next_tasks: Vec<Task>,
    node_exists: impl Fn(&str) -> bool,
    destinations_of: impl Fn(&str) -> Option<&'a [String]>,
    is_deferred: impl Fn(&str) -> bool,
    pending_deferred: &mut Vec<Task>,
    promote_deferred: bool,
) -> Result<Vec<Task>> {
    let handler_nodes: HashSet<String> = next_tasks
        .iter()
        .filter(|task| task.error.is_some())
        .map(|task| task.node.clone())
        .collect();
    for node in dedupe(next_nodes) {
        if handler_nodes.contains(&node) {
            continue;
        }
        next_tasks.push(Task::new(node));
    }
    for (index, (source, send)) in router_sends.into_iter().enumerate() {
        let mut task = send_task(&source, send)?;
        task.push_index = Some(index);
        if !node_exists(&task.node) {
            return Err(Error::routing(
                &source,
                format!("unknown destination node `{}`", task.node),
            ));
        }
        if let Some(allowed) = destinations_of(&source) {
            if !allowed.iter().any(|node| node == &task.node) {
                return Err(Error::routing(
                    &source,
                    format!("`{}` is not a declared destination", task.node),
                ));
            }
        }
        next_tasks.push(task);
    }

    let (deferred_tasks, regular_tasks): (Vec<Task>, Vec<Task>) = next_tasks
        .into_iter()
        .partition(|task| is_deferred(&task.node));
    pending_deferred.extend(deferred_tasks);
    Ok(
        if promote_deferred && regular_tasks.is_empty() && !pending_deferred.is_empty() {
            std::mem::take(pending_deferred)
        } else {
            regular_tasks
        },
    )
}

/// Tasks created by a graph's entry point.
pub(crate) fn entry_tasks<S: State>(
    entry: &crate::graph::Entry<S>,
    state: &S,
) -> Result<Vec<Task>> {
    match entry {
        crate::graph::Entry::Static(nodes) => Ok(nodes
            .iter()
            .filter(|node| node.as_str() != END)
            .map(Task::new)
            .collect()),
        crate::graph::Entry::Conditional(edge) => {
            let (targets, sends) = edge.resolve(state, START)?;
            let mut tasks: Vec<Task> = targets
                .into_iter()
                .filter(|node| node != END)
                .map(Task::new)
                .collect();
            for (index, send) in sends.into_iter().enumerate() {
                let mut task = send_task(START, send)?;
                task.push_index = Some(index);
                tasks.push(task);
            }
            Ok(tasks)
        }
    }
}

/// Rewrite task ids with LangGraph's v4 scheme so cross-engine ids line up.
pub(crate) fn assign_v4_ids(
    tasks: &mut [crate::_internal::node::Task],
    parent_checkpoint_id: Option<&str>,
    namespace: &str,
    step: usize,
) {
    let Some(parent) = parent_checkpoint_id else {
        return;
    };
    // `Send` tasks are numbered in superstep order; Python hashes that index into the id.
    let mut push_counter = 0usize;
    for task in tasks.iter_mut() {
        let push_index = if task.push {
            let index = push_counter;
            push_counter += 1;
            task.push_index = Some(index);
            Some(index)
        } else {
            None
        };
        let node_ns = if namespace.is_empty() {
            task.node.clone()
        } else {
            format!("{namespace}|{}", task.node)
        };
        if let Some(id) =
            crate::checkpoint::v4_task_id(parent, &node_ns, &task.node, step, push_index)
        {
            task.id = id;
        }
    }
}
