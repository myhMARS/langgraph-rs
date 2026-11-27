//! Error types produced while building or executing a graph.

use std::fmt;

use serde_json::Value;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// An interrupt raised by a node that is waiting for human input.
///
/// Interrupts are not failures: the graph state is checkpointed and execution
/// can be continued with [`CompiledGraph::resume`](crate::CompiledGraph::resume).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interrupt {
    /// Node that raised the interrupt.
    pub node: String,
    /// Payload the node attached to the interrupt (usually a question).
    pub value: Value,
    /// Unique id, used to resume several pending interrupts with a value map.
    #[serde(default)]
    pub id: String,
    /// Optional JSON schema the resume value must satisfy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    /// Task that raised this interrupt, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<Box<str>>,
    /// `true` for a static `interrupt_before`/`interrupt_after` breakpoint.
    ///
    /// LangGraph pauses on these without raising, so bindings return the state instead
    /// of surfacing an interrupt error.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub static_pause: bool,
    /// Channel values at the pause, so a binding can return LangGraph's paused payload even
    /// when the run has no checkpointer to read the state back from.
    ///
    /// Not serialized: the checkpoint already stores the same values.
    #[serde(skip)]
    pub values: Option<serde_json::Map<String, Value>>,
}

impl Interrupt {
    /// Create an interrupt with a fresh unique id.
    pub fn new(node: impl Into<String>, value: Value) -> Self {
        let node = node.into();
        Self {
            id: new_interrupt_id(&node),
            node,
            value,
            response_schema: None,
            task_id: None,
            static_pause: false,
            values: None,
        }
    }

    /// Create an interrupt with a deterministic id (LangGraph-compatible resume keys).
    pub(crate) fn with_schema_and_id(
        node: impl Into<String>,
        value: Value,
        response_schema: Option<Value>,
        id: String,
    ) -> Self {
        Self {
            id,
            node: node.into(),
            value,
            response_schema,
            task_id: None,
            static_pause: false,
            values: None,
        }
    }

    /// Create an interrupt with a JSON schema for the expected resume value.
    pub fn with_schema(
        node: impl Into<String>,
        value: Value,
        response_schema: Option<Value>,
    ) -> Self {
        let mut interrupt = Self::new(node, value);
        interrupt.response_schema = response_schema;
        interrupt
    }

    /// Resume value for this interrupt inside a multi-interrupt resume map.
    pub fn resume_from(&self, map: &serde_json::Map<String, Value>) -> Option<Value> {
        map.get(&self.id).cloned()
    }
}

/// Stable interrupt id derived from a run-scoped seed (xxh3-128, 32 hex chars).
pub(crate) fn deterministic_interrupt_id(seed: &str) -> String {
    format!("{:032x}", xxhash_rust::xxh3::xxh3_128(seed.as_bytes()))
}

fn new_interrupt_id(node: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("{node}-{millis}-{sequence}")
}

impl fmt::Display for Interrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node `{}` interrupted with {}", self.node, self.value)
    }
}

impl std::error::Error for Interrupt {}

/// A node failure captured by the engine.
///
/// Error-handler nodes can read the failure with [`last_error`](crate::last_error).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeFailure {
    /// Node that failed.
    pub node: String,
    /// Error message of the final attempt.
    pub message: String,
    /// Number of attempts that ran before giving up.
    pub attempts: u32,
}

impl NodeFailure {
    /// Create a failure record.
    pub fn new(node: impl Into<String>, message: impl Into<String>, attempts: u32) -> Self {
        Self {
            node: node.into(),
            message: message.into(),
            attempts,
        }
    }
}

impl std::fmt::Display for NodeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "node `{}` failed after {} attempt(s): {}",
            self.node, self.attempts, self.message
        )
    }
}

/// Errors returned while building or executing a graph.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The graph definition is inconsistent (missing entry point, unknown edge target, ...).
    #[error("graph validation failed: {0}")]
    Validation(String),
    /// The configured superstep budget was exhausted.
    #[error("recursion limit of {0} supersteps exceeded")]
    RecursionLimit(usize),
    /// A node returned an error.
    #[error("node `{node}` failed: {message}")]
    Node { node: String, message: String },
    /// A node exceeded its configured run timeout.
    #[error("node `{node}` timed out after {millis} ms")]
    Timeout { node: String, millis: u64 },
    /// A node stopped reporting progress for longer than its configured idle timeout.
    #[error("node `{node}` went idle for {millis} ms")]
    IdleTimeout { node: String, millis: u64 },
    /// A node wrote to a channel that is not part of the state schema.
    #[error("channel `{0}` is not declared in the state schema")]
    UnknownChannel(String),
    /// Two nodes wrote to a plain `LastValue` channel within the same superstep.
    #[error(
        "conflicting concurrent writes to channel `{channel}` (nodes: {nodes}); declare a reducer for this channel"
    )]
    ConcurrentUpdate { channel: String, nodes: String },
    /// A node requested human input.
    #[error(transparent)]
    Interrupt(#[from] Interrupt),
    /// A router returned a target that does not exist or is not a valid destination.
    #[error("invalid route from `{node}`: {message}")]
    Routing { node: String, message: String },
    /// The checkpointer failed.
    #[error("checkpoint failure: {0}")]
    Checkpoint(String),
    /// Serializing or deserializing state failed.
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    /// The two sides of a graph disagree about the state type.
    #[error("state must serialize to a JSON object, found {0}")]
    NonObjectState(String),
    /// An API that needs a checkpointer was called without one.
    #[error(
        "{0} requires a checkpointer; compile the graph with CompileOptions::with_checkpointer"
    )]
    MissingCheckpointer(&'static str),
    /// Generic error used for user supplied failures.
    #[error("{0}")]
    Other(String),
}

/// Drop the invisible `\u{1}pyerr:<id>\u{1}` marker some adapters append to a message.
pub fn strip_hidden_marker(message: &str) -> String {
    let marker = "\u{1}pyerr:";
    let Some(start) = message.rfind(marker) else {
        return message.to_string();
    };
    let rest = &message[start..];
    let Some(offset) = rest[1..].find('\u{1}') else {
        return message.to_string();
    };
    let end = start + 1 + offset + 1;
    format!("{}{}", &message[..start], &message[end..])
}

impl Error {
    /// Build a graph validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Error::Validation(message.into())
    }

    /// Build a generic error.
    pub fn other(message: impl Into<String>) -> Self {
        Error::Other(message.into())
    }

    /// Build a node error.
    pub fn node(node: impl Into<String>, message: impl fmt::Display) -> Self {
        Error::Node {
            node: node.into(),
            message: message.to_string(),
        }
    }

    /// Build a routing error.
    pub fn routing(node: impl Into<String>, message: impl fmt::Display) -> Self {
        Error::Routing {
            node: node.into(),
            message: message.to_string(),
        }
    }

    /// Returns `true` when the error is a human-in-the-loop interrupt.
    pub fn is_interrupt(&self) -> bool {
        matches!(self, Error::Interrupt(_))
    }

    /// Returns the interrupt payload when this error is an interrupt.
    pub fn interrupt(&self) -> Option<&Interrupt> {
        match self {
            Error::Interrupt(interrupt) => Some(interrupt),
            _ => None,
        }
    }
}
