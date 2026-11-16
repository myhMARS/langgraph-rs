//! Typed views of checkpointed thread state.

use serde_json::{Map, Value};

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Interrupt, NodeFailure, Result};
use crate::kernel::state::{State, values_to_state};

/// A task that is scheduled after a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct StateTask {
    /// Node to run.
    pub node: String,
    /// Failure routed to this task, when it is an error handler.
    pub error: Option<NodeFailure>,
    /// Resume value stamped onto the task (for interrupts).
    pub resume: Option<Value>,
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
    /// Supersteps completed.
    pub step: usize,
    /// `true` when the graph stopped because of an interrupt.
    pub interrupted: bool,
    /// Creation time in milliseconds since the Unix epoch.
    pub created_at_millis: u64,
    /// Pending tasks scheduled after this snapshot.
    pub tasks: Vec<StateTask>,
    /// Interrupts captured when this checkpoint was written.
    pub interrupts: Vec<Interrupt>,
    /// Parent checkpoint id (history chain).
    pub parent_checkpoint_id: Option<String>,
    /// Free-form checkpoint metadata.
    pub metadata: Map<String, Value>,
}

impl<S: State> StateSnapshot<S> {
    pub(crate) fn from_checkpoint(checkpoint: Checkpoint) -> Result<Self> {
        let tasks = checkpoint
            .next
            .iter()
            .map(|task| StateTask {
                node: task.node.clone(),
                error: task.error.clone(),
                resume: task.resume.clone(),
            })
            .collect();
        Ok(Self {
            values: values_to_state::<S>(&checkpoint.values)?,
            next: checkpoint.next_nodes(),
            checkpoint_id: checkpoint.id,
            thread_id: checkpoint.thread_id,
            step: checkpoint.step,
            interrupted: checkpoint.interrupted,
            created_at_millis: checkpoint.created_at_millis,
            tasks,
            interrupts: checkpoint.interrupts,
            parent_checkpoint_id: checkpoint.parent_id,
            metadata: checkpoint.metadata,
        })
    }

    /// First interrupt captured by this snapshot, if any.
    pub fn interrupt(&self) -> Option<&Interrupt> {
        self.interrupts.first()
    }
}
