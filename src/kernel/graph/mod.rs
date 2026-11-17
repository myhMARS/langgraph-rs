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
pub(crate) mod visualize;

use crate::kernel::node::ConditionalEdge;

/// Convert a recorded parent command into a node output, folding repeated reducer
/// writes through the parent schema instead of collapsing them in an `IndexMap`.
pub(crate) fn parent_command_output<S: crate::kernel::state::State>(
    command: crate::kernel::context::ParentCommand,
    state: &S,
) -> crate::Result<crate::kernel::node::NodeOutput> {
    let mut parent_values = crate::kernel::state::state_to_values(state)?;
    let updates: Vec<crate::kernel::state::ChannelUpdate> = command
        .updates
        .iter()
        .map(|(channel, value)| crate::kernel::state::ChannelUpdate {
            channel: channel.clone(),
            value: value.clone(),
            node: "<parent>".to_string(),
        })
        .collect();
    crate::kernel::state::apply_updates(&S::schema(), &mut parent_values, updates)?;

    let mut result = crate::kernel::node::NodeOutput::new();
    let mut channels: Vec<&String> = Vec::new();
    for (channel, _) in &command.updates {
        if !channels.contains(&channel) {
            channels.push(channel);
            if let Some(value) = parent_values.get(channel) {
                result = result.replace(channel.clone(), value.clone());
            }
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
