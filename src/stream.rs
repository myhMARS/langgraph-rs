//! Stream chunks emitted while a graph runs.

use serde_json::Value;

/// One event in a graph run.
///
/// Chunks are produced step by step and can be consumed with
/// [`CompiledGraph::stream_with`](crate::CompiledGraph::stream_with) for incremental
/// processing, or collected into a `Vec` with
/// [`CompiledGraph::stream`](crate::CompiledGraph::stream).
#[derive(Clone, Debug, PartialEq)]
pub enum StreamChunk {
    /// Full channel values after a superstep.
    Values {
        /// Superstep index.
        step: usize,
        /// Serialized state.
        values: Value,
    },
    /// Channel writes produced by a single node in a superstep.
    Updates {
        /// Superstep index.
        step: usize,
        /// Node that produced the writes.
        node: String,
        /// Map of channel -> write.
        update: Value,
    },
    /// A custom value pushed with [`custom`](crate::custom) from inside a node.
    Custom {
        /// Node that emitted the value.
        node: String,
        /// Payload.
        value: Value,
    },
    /// A node asked for human input.
    Interrupt {
        /// Interrupted node.
        node: String,
        /// Interrupt payload.
        value: Value,
    },
    /// A checkpoint was persisted.
    Checkpoint {
        /// Checkpoint id.
        checkpoint_id: String,
    },
}

impl StreamChunk {
    /// Checkpoint id when this chunk reports a persisted checkpoint.
    pub fn checkpoint_id(&self) -> Option<&str> {
        match self {
            StreamChunk::Checkpoint { checkpoint_id } => Some(checkpoint_id),
            _ => None,
        }
    }

    /// Serialized values when this chunk carries full state.
    pub fn values(&self) -> Option<&Value> {
        match self {
            StreamChunk::Values { values, .. } => Some(values),
            _ => None,
        }
    }

    /// Node name when this chunk is tied to a single node.
    pub fn node(&self) -> Option<&str> {
        match self {
            StreamChunk::Updates { node, .. }
            | StreamChunk::Custom { node, .. }
            | StreamChunk::Interrupt { node, .. } => Some(node),
            _ => None,
        }
    }
}
