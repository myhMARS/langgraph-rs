//! Stream chunks emitted while a graph runs.

use serde_json::Value;

use crate::kernel::error::NodeFailure;
use crate::kernel::usage::NodeExecution;

/// One event in a graph run.
///
/// Chunks are produced step by step and can be consumed with
/// [`CompiledGraph::stream_with`](crate::CompiledGraph::stream_with) for incremental
/// processing, or collected into a `Vec` with
/// [`CompiledGraph::stream`](crate::CompiledGraph::stream).
#[derive(Clone, Debug, PartialEq)]
pub enum StreamChunk {
    /// Per-node execution record: duration and consumption.
    Usage {
        /// Execution details.
        execution: NodeExecution,
    },
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
        /// Interrupt id.
        id: String,
        /// Interrupt payload.
        value: Value,
    },
    /// A checkpoint was persisted.
    Checkpoint {
        /// Checkpoint id.
        checkpoint_id: String,
        /// LangGraph v2 `CheckpointPayload` shape.
        payload: Value,
    },
    /// A streamed message emitted with [`message`](crate::message) inside a node
    /// (`stream_mode="messages"`).
    Message {
        /// Node that emitted the message.
        node: String,
        /// Message payload (typically a token or a message fragment).
        message: Value,
    },
    /// Task lifecycle record (`stream_mode="tasks"`).
    Task {
        /// Superstep index.
        step: usize,
        /// Stable task id.
        id: String,
        /// Node of the task.
        node: String,
        /// Input state override for `Send` tasks.
        input: Option<Value>,
        /// Channel writes produced by the task, when it succeeded.
        result: Option<Value>,
        /// Failure routed by the task, when it failed.
        error: Option<NodeFailure>,
    },
    /// Debug record for a completed superstep (`stream_mode="debug"`).
    Debug {
        /// Superstep index.
        step: usize,
        /// Structured debug payload.
        payload: Value,
    },
    /// A chunk produced inside a subgraph (`stream(subgraphs=True)`).
    Subgraph {
        /// Checkpoint namespace of the subgraph that emitted the chunk.
        namespace: String,
        /// Wrapped chunk.
        chunk: Box<StreamChunk>,
    },
}

/// LangGraph-compatible stream mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// Full channel values after each superstep.
    Values,
    /// Per-node channel writes.
    Updates,
    /// Values emitted with [`custom`](crate::custom).
    Custom,
    /// Messages emitted with [`message`](crate::message).
    Messages,
    /// Combined debug payloads.
    Debug,
    /// Task lifecycle records.
    Tasks,
    /// Persisted checkpoint ids.
    Checkpoints,
}

impl StreamMode {
    /// Parse a LangGraph mode name (`"values"`, `"updates"`, `"custom"`, `"messages"`,
    /// `"debug"`, `"tasks"`, `"checkpoints"`).
    pub fn parse(mode: &str) -> Option<Self> {
        Some(match mode {
            "values" => Self::Values,
            "updates" => Self::Updates,
            "custom" => Self::Custom,
            "messages" => Self::Messages,
            "debug" => Self::Debug,
            "tasks" => Self::Tasks,
            "checkpoints" => Self::Checkpoints,
            _ => return None,
        })
    }

    /// Canonical mode name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Values => "values",
            Self::Updates => "updates",
            Self::Custom => "custom",
            Self::Messages => "messages",
            Self::Debug => "debug",
            Self::Tasks => "tasks",
            Self::Checkpoints => "checkpoints",
        }
    }
}

impl StreamChunk {
    /// Checkpoint id when this chunk reports a persisted checkpoint.
    pub fn checkpoint_id(&self) -> Option<&str> {
        match self {
            StreamChunk::Checkpoint { checkpoint_id, .. } => Some(checkpoint_id),
            StreamChunk::Subgraph { chunk, .. } => chunk.checkpoint_id(),
            _ => None,
        }
    }

    /// Per-node execution record when this chunk reports one.
    pub fn execution(&self) -> Option<&NodeExecution> {
        match self {
            StreamChunk::Usage { execution } => Some(execution),
            StreamChunk::Subgraph { chunk, .. } => chunk.execution(),
            _ => None,
        }
    }

    /// Serialized values when this chunk carries full state.
    pub fn values(&self) -> Option<&Value> {
        match self {
            StreamChunk::Values { values, .. } => Some(values),
            StreamChunk::Subgraph { chunk, .. } => chunk.values(),
            _ => None,
        }
    }

    /// Node name when this chunk is tied to a single node.
    pub fn node(&self) -> Option<&str> {
        match self {
            StreamChunk::Usage { execution } => Some(execution.node.as_str()),
            StreamChunk::Updates { node, .. }
            | StreamChunk::Custom { node, .. }
            | StreamChunk::Interrupt { node, .. }
            | StreamChunk::Message { node, .. }
            | StreamChunk::Task { node, .. } => Some(node),
            StreamChunk::Subgraph { chunk, .. } => chunk.node(),
            _ => None,
        }
    }

    /// LangGraph stream mode this chunk belongs to.
    pub fn mode(&self) -> Option<StreamMode> {
        match self {
            StreamChunk::Values { .. } => Some(StreamMode::Values),
            StreamChunk::Updates { .. } => Some(StreamMode::Updates),
            StreamChunk::Custom { .. } => Some(StreamMode::Custom),
            StreamChunk::Message { .. } => Some(StreamMode::Messages),
            StreamChunk::Task { .. } => Some(StreamMode::Tasks),
            StreamChunk::Debug { .. } => Some(StreamMode::Debug),
            StreamChunk::Checkpoint { .. } => Some(StreamMode::Checkpoints),
            StreamChunk::Subgraph { chunk, .. } => chunk.mode(),
            // Usage and interrupt chunks are engine-level extras without a LangGraph mode.
            StreamChunk::Usage { .. } | StreamChunk::Interrupt { .. } => None,
        }
    }

    /// Checkpoint namespace of a subgraph chunk, when applicable.
    pub fn namespace(&self) -> Option<&str> {
        match self {
            StreamChunk::Subgraph { namespace, .. } => Some(namespace),
            _ => None,
        }
    }

    /// Like [StreamChunk::matches_modes] but drops engine-only chunks (usage).
    pub fn matches_modes_strict(&self, modes: &[StreamMode]) -> bool {
        self.mode().is_some_and(|_| self.matches_modes(modes))
    }

    /// Check whether this chunk is forwarded for the selected stream modes.
    ///
    /// Usage and interrupt chunks are always forwarded; the `debug` mode also includes
    /// values, updates, tasks and checkpoint records, matching LangGraph's behaviour.
    pub fn matches_modes(&self, modes: &[StreamMode]) -> bool {
        if modes.is_empty() {
            return true;
        }
        if let StreamChunk::Subgraph { chunk, .. } = self {
            return chunk.matches_modes(modes);
        }
        let Some(mode) = self.mode() else {
            return true;
        };
        if modes.contains(&mode) {
            return true;
        }
        if modes.contains(&StreamMode::Debug)
            && matches!(
                mode,
                StreamMode::Values
                    | StreamMode::Updates
                    | StreamMode::Tasks
                    | StreamMode::Checkpoints
            )
        {
            return true;
        }
        false
    }
}
