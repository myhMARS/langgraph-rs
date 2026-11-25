//! Graph definition, compilation and the public run API.
//!
//! The module is intentionally split by responsibility:
//!
//! - [`builder`] holds [`StateGraph`], the mutable builder and its validation rules.
//! - [`compiled`] holds [`CompiledGraph`], the immutable runnable form and the
//!   `invoke` / `stream` / `resume` facade.
//! - [`options`] holds [`CompileOptions`], [`RunConfig`], [`Targets`] and the
//!   default recursion limit.
//! - [`snapshot`] holds [`StateSnapshot`], the typed view of a checkpoint.

mod builder;
mod compiled;
mod options;
mod snapshot;
mod state_api;
mod threads;
pub(crate) mod visualize;

pub(crate) use options::default_recursion_limit;
pub(crate) use state_api::metadata_matches;

use crate::_internal::node::ConditionalEdge;

/// Convert a recorded parent command into a node output, folding repeated reducer
/// writes through the parent schema instead of collapsing them in an `IndexMap`.
pub(crate) fn parent_command_output(
    command: crate::runtime::context::ParentCommand,
    schema: &crate::_internal::state::StateSchema,
) -> crate::Result<crate::_internal::node::NodeOutput> {
    // Preserve ordered command writes. A channel written by several child nodes is
    // folded through a reducer against an empty map, producing a delta that still
    // folds with sibling writes when the parent applies the output.
    let mut grouped: indexmap::IndexMap<&String, Vec<serde_json::Value>> =
        indexmap::IndexMap::new();
    for (channel, value) in &command.updates {
        grouped.entry(channel).or_default().push(value.clone());
    }
    let mut result = crate::_internal::node::NodeOutput::new();
    for (channel, values) in grouped {
        if values.len() == 1 {
            result = result.update(channel.clone(), values.into_iter().next().unwrap());
            continue;
        }
        let updates: Vec<crate::_internal::state::ChannelUpdate> = values
            .into_iter()
            .map(|value| crate::_internal::state::ChannelUpdate {
                channel: channel.clone(),
                value,
                node: "<parent>".to_string(),
            })
            .collect();
        let mut folded = serde_json::Map::new();
        crate::_internal::state::apply_updates(schema, &mut folded, updates)?;
        if let Some(value) = folded.get(channel) {
            result = result.update(channel.clone(), value.clone());
        }
    }
    for (channel, value) in command.replaces {
        result = result.replace(channel, value);
    }
    for target in command.goto {
        result = result.goto(target);
    }
    Ok(result)
}

/// Checkpoint namespace for a child graph.
///
/// LangGraph nests child namespaces as `<parent>|<node>:<task_id>`. The task id is
/// optional so the async adapter (which does not carry a task id yet) can share the
/// same namespace construction while the sync Pregel loop passes the executing task.
/// Namespace for a chunk forwarded from a subgraph run.
///
/// Child chunks already carry an absolute namespace (`a:1|b:2`), so prefixing the parent again
/// would duplicate it (`a:1|a:1|b:2`). Only prepend when the child namespace is not already
/// anchored at the parent's.
pub(crate) fn subgraph_chunk_namespace(parent: &str, inner: &str) -> String {
    if parent.is_empty() {
        return inner.to_string();
    }
    if inner == parent || inner.starts_with(&format!("{parent}|")) {
        return inner.to_string();
    }
    format!("{parent}|{inner}")
}
pub fn subgraph_namespace(parent: &str, name: &str, task_id: Option<&str>) -> String {
    let base = if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}|{name}")
    };
    match task_id {
        Some(task_id) => format!("{base}:{task_id}"),
        None => base,
    }
}

pub use builder::StateGraph;
pub use compiled::CompiledGraph;
pub use options::{CompileOptions, DEFAULT_RECURSION_LIMIT, Durability, RunConfig, Targets};
pub use snapshot::StateSnapshot;

/// How the first superstep is chosen.
pub(crate) enum Entry<S> {
    /// One or more statically configured entry nodes.
    Static(Vec<String>),
    /// A router evaluated against the initial state.
    Conditional(ConditionalEdge<S>),
}

impl<S> Clone for Entry<S> {
    fn clone(&self) -> Self {
        match self {
            Entry::Static(nodes) => Entry::Static(nodes.clone()),
            Entry::Conditional(edge) => Entry::Conditional(edge.clone()),
        }
    }
}
