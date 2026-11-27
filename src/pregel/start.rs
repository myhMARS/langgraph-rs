//! Shared run-start resolution for the sync and async Pregel loops.

use serde_json::{Map, Value};

use crate::_internal::error::{Error, Result};
use crate::_internal::state::{ChannelUpdate, State, StateSchema, apply_updates, values_to_state};
use crate::pregel::{StartInput, apply_pending_writes};

/// Resolved channel values, state and tasks for a run start.
pub(crate) type FreshStart<S> = Option<(Map<String, Value>, S, Vec<crate::_internal::node::Task>)>;

pub(crate) type ResumeStart<S> = (
    Map<String, Value>,
    S,
    Vec<crate::_internal::node::Task>,
    Vec<crate::checkpoint::PendingWrite>,
    // The `Command.update` writes, kept so a run with nothing left to do can store them as
    // pending writes (LangGraph does not create a checkpoint for them).
    Vec<crate::checkpoint::PendingWrite>,
    // Nodes whose writes were replayed from the checkpoint; they still contribute routes.
    Vec<String>,
);

/// Resolve the three non-resume start inputs into initial channel values, state and
/// entry tasks. Returns `None` for [`StartInput::Resume`].
pub(crate) fn resolve_fresh_start<S: State>(
    input: &StartInput,
    schema: &StateSchema,
    node_exists: impl Fn(&str) -> bool,
    entry_tasks: impl Fn(&S) -> Result<Vec<crate::_internal::node::Task>>,
) -> Result<FreshStart<S>> {
    match input {
        StartInput::Values(values) => {
            let values = schema.filter_input(values.clone());
            let state = values_to_state::<S>(&values)?;
            let tasks = entry_tasks(&state)?;
            Ok(Some((values, state, tasks)))
        }
        StartInput::Updates(updates) => {
            let updates = updates.clone();
            let mut values = Map::new();
            apply_updates(
                schema,
                &mut values,
                updates
                    .into_iter()
                    .map(|(channel, value)| ChannelUpdate {
                        channel,
                        value,
                        node: "<input>".to_string(),
                    })
                    .collect(),
            )?;
            let state = values_to_state::<S>(&values)?;
            let tasks = entry_tasks(&state)?;
            Ok(Some((values, state, tasks)))
        }
        StartInput::Gateways {
            values,
            nodes,
            sends,
        } => {
            let values = values.clone();
            let nodes = nodes.clone();
            let sends = sends.clone();
            // `Command(goto=...)` values are state updates, not run input.
            for node in &nodes {
                if node != crate::END && !node_exists(node) {
                    return Err(crate::_internal::error::Error::validation(format!(
                        "Command(goto=...): `{node}` is not a registered node"
                    )));
                }
            }
            let state = values_to_state::<S>(&values)?;
            // LangGraph schedules `Command(goto=...)` targets *in addition to* the entry
            // point, so a fresh run executes both in the first superstep and the entry
            // node's outgoing edges fire again afterwards.
            let mut tasks: Vec<crate::_internal::node::Task> = entry_tasks(&state)?;
            for node in crate::pregel::scheduling::dedupe(nodes) {
                if node == crate::END {
                    continue;
                }
                if !tasks.iter().any(|task| task.node == node) {
                    tasks.push(crate::_internal::node::Task::new(node));
                }
            }
            for send in sends {
                tasks.push(crate::pregel::scheduling::send_task("<command>", send)?);
            }
            Ok(Some((values, state, tasks)))
        }
        StartInput::Resume { .. } => Ok(None),
    }
}

/// Resolve a resume against an already-loaded checkpoint.
pub(crate) fn resolve_resume_start<S: State>(
    checkpoint: &crate::checkpoint::Checkpoint,
    resume: &Option<Value>,
    updates: Option<Map<String, Value>>,
    goto: &[String],
    sends: &[crate::_internal::node::Send],
    schema: &StateSchema,
) -> Result<ResumeStart<S>> {
    crate::runtime::context::load_task_results(&checkpoint.task_results);
    let mut values = checkpoint.values.clone();
    let mut finish_writes = Vec::new();
    let mut regular_writes = Vec::new();
    for write in &checkpoint.pending_writes {
        if schema.is_finish_value(&write.channel) {
            finish_writes.push(write.clone());
        } else {
            regular_writes.push(write.clone());
        }
    }
    apply_pending_writes(schema, &mut values, &regular_writes)?;
    // Nodes whose writes were replayed (including held finish values) still route.
    let mut replayed: Vec<String> = Vec::new();
    for write in &checkpoint.pending_writes {
        if !write.node.is_empty() && !write.node.starts_with('<') && !replayed.contains(&write.node)
        {
            replayed.push(write.node.clone());
        }
    }
    let mut update_writes: Vec<crate::checkpoint::PendingWrite> = Vec::new();
    if let Some(updates) = updates {
        for (channel, value) in &updates {
            update_writes.push(crate::checkpoint::PendingWrite {
                task_id: String::new(),
                node: "<resume>".to_string(),
                channel: channel.clone(),
                value: value.clone(),
                replace: false,
            });
        }
        apply_updates(
            schema,
            &mut values,
            updates
                .into_iter()
                .map(|(channel, value)| ChannelUpdate {
                    channel,
                    value,
                    node: "<resume>".to_string(),
                })
                .collect(),
        )?;
    }
    let state = values_to_state::<S>(&values)?;
    // LangGraph refuses an unkeyed resume while several interrupts are pending.
    if resume.is_some()
        && checkpoint.interrupts.len() > 1
        && !matches!(resume, Some(crate::Value::Object(_)))
    {
        // LangGraph raises `RuntimeError` with this wording.
        return Err(Error::other(
            "When there are multiple pending interrupts, you must specify the interrupt id when resuming. Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts.",
        ));
    }
    let mut tasks = crate::pregel::checkpointing::prepare_resume_tasks(checkpoint, resume)?;
    for node in goto {
        if node == crate::END {
            continue;
        }
        if !tasks.iter().any(|task| &task.node == node) {
            tasks.push(crate::_internal::node::Task::new(node.clone()));
        }
    }
    for send in sends {
        tasks.push(crate::pregel::scheduling::send_task(
            "<command>",
            send.clone(),
        )?);
    }
    Ok((values, state, tasks, finish_writes, update_writes, replayed))
}
