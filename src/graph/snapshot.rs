//! Typed views of checkpointed thread state.

use serde_json::{Map, Value};

use crate::_internal::error::{Interrupt, NodeFailure, Result};
use crate::_internal::state::{State, values_to_state};
use crate::checkpoint::Checkpoint;

/// A task that is scheduled after a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct StateTask {
    /// Stable task id.
    pub id: String,
    /// Node to run.
    pub node: String,
    /// Input passed to the task (`Send` argument), when it overrides state.
    pub state: Option<Value>,
    /// Failure routed to this task, when it is an error handler.
    pub error: Option<NodeFailure>,
    /// Resume value stamped onto the task (for interrupts).
    pub resume: Option<Value>,
    /// Channel writes produced by this task, when it already ran.
    pub result: Option<Value>,
    /// Interrupts attributed to this task.
    pub interrupts: Vec<Interrupt>,
}

/// A point-in-time view of a checkpointed thread.
#[derive(Clone, Debug)]
pub struct StateSnapshot<S: State> {
    /// Typed channel values.
    pub values: S,
    /// Nodes scheduled after this snapshot.
    pub next: Vec<String>,
    /// Checkpoint id.
    pub checkpoint_id: String,
    /// Thread id.
    pub thread_id: String,
    /// Thread-cumulative step (LangGraph metadata step).
    pub step: i64,
    /// `true` when the graph stopped because of an interrupt.
    pub interrupted: bool,
    /// Creation time in milliseconds since the Unix epoch.
    pub created_at_millis: u64,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// Pending tasks scheduled after this snapshot.
    pub tasks: Vec<StateTask>,
    /// Interrupts captured when this checkpoint was written.
    pub interrupts: Vec<Interrupt>,
    /// Parent checkpoint id (history chain).
    pub parent_checkpoint_id: Option<String>,
    /// Free-form checkpoint metadata.
    pub metadata: Map<String, Value>,
    /// Checkpoint namespace of this snapshot.
    pub checkpoint_ns: String,
    /// LangGraph-style config for this snapshot (`configurable` thread/ns/checkpoint id).
    pub config: Value,
    /// LangGraph-style config for the parent checkpoint, when present.
    pub parent_config: Option<Value>,
}

impl<S: State> StateSnapshot<S> {
    /// Build a live snapshot: pending sibling writes are applied to values and exposed
    /// as task results, matching LangGraph's `get_state` (`live=True`).
    pub(crate) fn from_checkpoint_live(
        checkpoint: Checkpoint,
        schema: &crate::_internal::state::StateSchema,
    ) -> Result<Self> {
        let mut values = checkpoint.values.clone();
        let mut regular = Vec::new();
        let mut task_results: std::collections::HashMap<String, Map<String, Value>> =
            std::collections::HashMap::new();
        for write in &checkpoint.pending_writes {
            if schema.is_finish_value(&write.channel) {
                continue;
            }
            task_results
                .entry(write.task_id.clone())
                .or_default()
                .insert(write.channel.clone(), write.value.clone());
            if write.replace {
                values.insert(write.channel.clone(), write.value.clone());
            } else {
                regular.push(crate::_internal::state::ChannelUpdate {
                    channel: write.channel.clone(),
                    value: write.value.clone(),
                    node: write.node.clone(),
                });
            }
        }
        crate::_internal::state::apply_updates(schema, &mut values, regular)?;
        let mut checkpoint = checkpoint;
        checkpoint.values = values;
        let mut snapshot = Self::from_checkpoint(checkpoint)?;
        for task in &mut snapshot.tasks {
            if task.result.is_none() {
                if let Some(writes) = task_results.remove(&task.id) {
                    task.result = Some(Value::Object(writes));
                }
            }
        }
        Ok(snapshot)
    }

    pub(crate) fn from_checkpoint(checkpoint: Checkpoint) -> Result<Self> {
        let tasks = checkpoint
            .next
            .iter()
            .map(|task| StateTask {
                id: task.id.clone(),
                node: task.node.clone(),
                // LangGraph only fills `state` for subgraph tasks; `embed_subgraph_states`
                // replaces it with the child snapshot when applicable.
                state: None,
                error: task.error.clone(),
                resume: task.resume.clone(),
                result: checkpoint.task_results.get(&task.id).cloned(),
                interrupts: checkpoint
                    .interrupts
                    .iter()
                    .filter(|interrupt| interrupt.task_id.as_deref() == Some(task.id.as_str()))
                    .cloned()
                    .collect(),
            })
            .collect();
        // Siblings that finished before the interrupt are reported as completed tasks.
        let mut tasks: Vec<StateTask> = tasks;
        for task in &checkpoint.completed_tasks {
            if tasks.iter().any(|known| known.id == task.id) {
                continue;
            }
            tasks.push(StateTask {
                id: task.id.clone(),
                node: task.node.clone(),
                state: None,
                error: None,
                resume: None,
                result: checkpoint.task_results.get(&task.id).cloned(),
                interrupts: Vec::new(),
            });
        }
        let config = serde_json::json!({
            "configurable": {
                "thread_id": checkpoint.thread_id,
                "checkpoint_ns": checkpoint.namespace,
                "checkpoint_id": checkpoint.id,
            }
        });
        let parent_config = checkpoint.parent_id.as_ref().map(|parent_id| {
            serde_json::json!({
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_ns": checkpoint.namespace,
                    "checkpoint_id": parent_id,
                }
            })
        });
        let created_at = crate::checkpoint::iso_timestamp(checkpoint.created_at_millis);
        Ok(Self {
            values: values_to_state::<S>(&checkpoint.values)?,
            next: checkpoint.next_nodes(),
            checkpoint_id: checkpoint.id,
            thread_id: checkpoint.thread_id,
            step: checkpoint.step as i64,
            interrupted: checkpoint.interrupted,
            created_at_millis: checkpoint.created_at_millis,
            created_at,
            tasks,
            // LangGraph does not expose static `interrupt_before`/`interrupt_after`
            // breakpoints as interrupts: they are a pause, not a question.
            interrupts: checkpoint
                .interrupts
                .into_iter()
                .filter(|interrupt| !interrupt.static_pause)
                .collect(),
            parent_checkpoint_id: checkpoint.parent_id,
            metadata: checkpoint.metadata,
            checkpoint_ns: checkpoint.namespace,
            config,
            parent_config,
        })
    }

    /// First interrupt captured by this snapshot, if any.
    pub fn interrupt(&self) -> Option<&Interrupt> {
        self.interrupts.first()
    }
}
