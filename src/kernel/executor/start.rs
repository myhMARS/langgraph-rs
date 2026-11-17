//! Shared run-start resolution for the sync and async Pregel loops.

use serde_json::{Map, Value};

use crate::kernel::error::Result;
use crate::kernel::executor::{StartInput, apply_pending_writes};
use crate::kernel::state::{ChannelUpdate, State, StateSchema, apply_updates, values_to_state};

/// Resolved channel values, state and tasks for a run start.
pub(crate) type FreshStart<S> = Option<(Map<String, Value>, S, Vec<crate::kernel::node::Task>)>;

/// Resolve the three non-resume start inputs into initial channel values, state and
/// entry tasks. Returns `None` for [`StartInput::Resume`].
pub(crate) fn resolve_fresh_start<S: State>(
    input: &StartInput,
    schema: &StateSchema,
    node_exists: impl Fn(&str) -> bool,
    entry_tasks: impl Fn(&S) -> Result<Vec<crate::kernel::node::Task>>,
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
        StartInput::Gateways { values, nodes } => {
            let values = values.clone();
            let nodes = nodes.clone();
            // `Command(goto=...)` values are state updates, not run input.
            for node in &nodes {
                if !node_exists(node) {
                    return Err(crate::kernel::error::Error::validation(format!(
                        "Command(goto=...): `{node}` is not a registered node"
                    )));
                }
            }
            let state = values_to_state::<S>(&values)?;
            let mut tasks = entry_tasks(&state)?;
            for node in crate::kernel::executor::scheduling::dedupe(nodes) {
                if !tasks.iter().any(|task| task.node == node) {
                    tasks.push(crate::kernel::node::Task::new(node));
                }
            }
            Ok(Some((values, state, tasks)))
        }
        StartInput::Resume { .. } => Ok(None),
    }
}

/// Resolve a resume against an already-loaded checkpoint.
pub(crate) fn resolve_resume_start<S: State>(
    checkpoint: &crate::kernel::checkpoint::Checkpoint,
    resume: &Option<Value>,
    updates: Option<Map<String, Value>>,
    schema: &StateSchema,
) -> Result<(Map<String, Value>, S, Vec<crate::kernel::node::Task>)> {
    crate::kernel::context::load_task_results(&checkpoint.task_results);
    let mut values = checkpoint.values.clone();
    apply_pending_writes(schema, &mut values, &checkpoint.pending_writes)?;
    if let Some(updates) = updates {
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
    let tasks = crate::kernel::executor::checkpointing::prepare_resume_tasks(checkpoint, resume);
    Ok((values, state, tasks))
}
