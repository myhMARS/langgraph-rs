//! Commands for continuing an interrupted graph.

use indexmap::IndexMap;
use serde_json::Value;

/// Instruction for resuming a checkpointed graph.
///
/// Most callers use the convenience methods
/// [`CompiledGraph::resume`](crate::CompiledGraph::resume) or
/// [`CompiledGraph::continue_thread`](crate::CompiledGraph::continue_thread); this type
/// is the explicit form used when both a resume value and a state patch are needed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Command {
    /// Value handed back to the interrupted node's `interrupt()` call.
    pub resume: Option<Value>,
    /// Optional channel writes applied before the graph continues (e.g. human edits).
    pub update: Option<IndexMap<String, Value>>,
    /// Nodes to schedule after the command (LangGraph `Command(goto=...)`).
    pub goto: Vec<String>,
    /// Dynamic `Send` tasks scheduled by `Command(goto=Send(...))`.
    pub sends: Vec<crate::_internal::node::Send>,
    /// Route to the parent graph instead of this one (`Command(graph=PARENT)`).
    pub parent: bool,
}

impl Command {
    /// Resume with the given value.
    pub fn resume(value: Value) -> Self {
        Self {
            resume: Some(value),
            update: None,
            goto: Vec::new(),
            sends: Vec::new(),
            parent: false,
        }
    }

    /// Continue without a resume value (for `interrupt_before` / `interrupt_after`).
    pub fn continue_thread() -> Self {
        Self::default()
    }

    /// Jump to a node regardless of the static outgoing edges.
    pub fn goto(mut self, node: impl Into<String>) -> Self {
        self.goto.push(node.into());
        self
    }

    /// Route to a node of the parent graph (inside a subgraph node).
    pub fn goto_parent(mut self, node: impl Into<String>) -> Self {
        self.parent = true;
        self.goto.push(node.into());
        self
    }

    /// Schedule a dynamic `Send` task.
    pub fn send(mut self, send: crate::_internal::node::Send) -> Self {
        self.sends.push(send);
        self
    }

    /// Apply state writes before continuing.
    pub fn with_update<K, V>(mut self, updates: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<Value>,
    {
        let map = self.update.get_or_insert_with(IndexMap::new);
        for (channel, value) in updates {
            map.insert(channel.into(), value.into());
        }
        self
    }
}

impl From<Command> for crate::_internal::node::NodeOutput {
    fn from(command: Command) -> Self {
        let mut output = crate::_internal::node::NodeOutput::new();
        if let Some(updates) = command.update {
            for (channel, value) in updates {
                output = output.update(channel, value);
            }
        }
        for target in command.goto {
            output = if command.parent {
                output.goto_parent(target)
            } else {
                output.goto(target)
            };
        }
        for send in command.sends {
            output = output.send(send);
        }
        output
    }
}
