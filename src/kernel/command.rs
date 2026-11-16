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
}

impl Command {
    /// Resume with the given value.
    pub fn resume(value: Value) -> Self {
        Self {
            resume: Some(value),
            update: None,
        }
    }

    /// Continue without a resume value (for `interrupt_before` / `interrupt_after`).
    pub fn continue_thread() -> Self {
        Self::default()
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
