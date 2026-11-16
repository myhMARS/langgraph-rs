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

pub use builder::StateGraph;
pub use compiled::CompiledGraph;
pub use options::{CompileOptions, DEFAULT_RECURSION_LIMIT, RunConfig, Targets};
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
