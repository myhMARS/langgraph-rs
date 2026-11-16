//! Typed views of checkpointed thread state.

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::Result;
use crate::kernel::state::{State, values_to_state};

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
}

impl<S: State> StateSnapshot<S> {
    pub(crate) fn from_checkpoint(checkpoint: Checkpoint) -> Result<Self> {
        Ok(Self {
            values: values_to_state::<S>(&checkpoint.values)?,
            next: checkpoint.next_nodes(),
            checkpoint_id: checkpoint.id,
            thread_id: checkpoint.thread_id,
            step: checkpoint.step,
            interrupted: checkpoint.interrupted,
            created_at_millis: checkpoint.created_at_millis,
        })
    }
}
