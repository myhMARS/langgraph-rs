//! Execution backend abstraction for the shared Pregel loop.
//!
//! The loop owns all superstep/checkpoint decisions; a backend provides graph metadata
//! and the two runtime-specific operations: executing a node batch and persisting a
//! checkpoint.

use serde_json::Value;

use crate::_internal::error::Result;
use crate::_internal::node::{ConditionalEdge, Task};
use crate::_internal::state::{State, StateSchema};
use crate::checkpoint::Checkpoint;
use crate::graph::CompiledGraph;
use crate::pregel::outcome::TaskOutcome;

/// Runtime-specific half of the shared Pregel loop.
pub(crate) trait PregelBackend<S: State>: super::outcome::OutcomeGraph {
    fn schema(&self) -> &StateSchema;
    fn recursion_limit(&self) -> usize;
    fn entry_tasks(&self, state: &S) -> Result<Vec<Task>>;
    fn node_exists(&self, node: &str) -> bool;
    fn edges_of(&self, node: &str) -> Option<&[String]>;
    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]>;
    fn destinations_of(&self, node: &str) -> Option<&[String]>;
    fn interrupts_before(&self, node: &str) -> bool;
    fn interrupts_after(&self, node: &str) -> bool;
    fn is_deferred(&self, node: &str) -> bool;
    fn context_schema(&self) -> Option<&Value>;
    fn store(&self) -> Option<std::sync::Arc<dyn crate::store::BaseStore>>;
    fn has_checkpointer(&self) -> bool;
    fn clear_ephemeral(&self, values: &mut serde_json::Map<String, Value>) -> bool;
    fn collector(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::stream::StreamChunk>>>>;
    fn run_superstep(
        &self,
        state: &S,
        tasks: &[Task],
        step: usize,
        control: Option<&crate::RunControl>,
    ) -> impl std::future::Future<Output = Result<Vec<TaskOutcome>>>;
    fn load_checkpoint(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &str,
    ) -> impl std::future::Future<Output = Result<Option<Checkpoint>>>;
    fn persist_checkpoint(
        &self,
        checkpoint: Checkpoint,
    ) -> impl std::future::Future<Output = Result<()>>;
}

/// Sync backend: scoped OS threads and a blocking checkpointer.
pub(crate) struct SyncBackend<'a, S: State> {
    pub(crate) graph: &'a CompiledGraph<S>,
    /// `Arc` view of `graph`, shared with pool workers that need an owned handle.
    pub(crate) graph_arc: std::sync::Arc<CompiledGraph<S>>,
    pub(crate) writer: Option<&'a crate::pregel::checkpointing::CheckpointWriter>,
    pub(crate) max_concurrency: Option<usize>,
}

impl<'a, S: State> PregelBackend<S> for SyncBackend<'a, S> {
    fn schema(&self) -> &StateSchema {
        self.graph.schema()
    }

    fn recursion_limit(&self) -> usize {
        self.graph.recursion_limit()
    }

    fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        self.graph.entry_tasks(state)
    }

    fn node_exists(&self, node: &str) -> bool {
        self.graph.node(node).is_some()
    }

    fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.graph.edges_of(node)
    }

    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.graph.conditional_of(node)
    }

    fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.graph.destinations_of(node)
    }

    fn interrupts_before(&self, node: &str) -> bool {
        self.graph.interrupts_before(node)
    }

    fn interrupts_after(&self, node: &str) -> bool {
        self.graph.interrupts_after(node)
    }

    fn is_deferred(&self, node: &str) -> bool {
        self.graph.is_deferred(node)
    }

    fn context_schema(&self) -> Option<&Value> {
        self.graph.context_schema()
    }

    fn store(&self) -> Option<std::sync::Arc<dyn crate::store::BaseStore>> {
        self.graph.store().cloned()
    }

    fn has_checkpointer(&self) -> bool {
        self.graph.checkpointer().is_some()
    }

    fn clear_ephemeral(&self, values: &mut serde_json::Map<String, Value>) -> bool {
        self.graph.schema().clear_ephemeral(values)
    }

    fn collector(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::stream::StreamChunk>>>> {
        None
    }

    async fn run_superstep(
        &self,
        state: &S,
        tasks: &[Task],
        _step: usize,
        _control: Option<&crate::RunControl>,
    ) -> Result<Vec<TaskOutcome>> {
        Ok(super::superstep::run_superstep(
            &self.graph_arc,
            state,
            tasks,
            self.max_concurrency,
        ))
    }

    async fn load_checkpoint(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &str,
    ) -> Result<Option<Checkpoint>> {
        let Some(checkpointer) = self.graph.checkpointer() else {
            return Ok(None);
        };
        match checkpoint_id {
            Some(id) => checkpointer.get(thread_id, id),
            None => checkpointer.latest_in(thread_id, namespace),
        }
    }

    async fn persist_checkpoint(&self, checkpoint: Checkpoint) -> Result<()> {
        let Some(checkpointer) = self.graph.checkpointer() else {
            return Ok(());
        };
        crate::pregel::checkpointing::persist_checkpoint(
            checkpointer.as_ref(),
            checkpoint,
            self.writer,
        )
    }
}

impl<'a, S: State> super::outcome::OutcomeGraph for SyncBackend<'a, S> {
    fn trace_sinks(&self) -> &[std::sync::Arc<dyn crate::_internal::ports::TraceSink>] {
        self.graph.trace_sinks()
    }

    fn error_handler(&self, node: &str) -> Option<&str> {
        self.graph.error_handler(node).map(String::as_str)
    }

    fn has_node(&self, node: &str) -> bool {
        self.graph.node(node).is_some()
    }

    fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.graph.destinations_of(node)
    }

    fn record_parent_command(
        &self,
        updates: Vec<(String, Value)>,
        replaces: Vec<(String, Value)>,
        goto: Vec<String>,
    ) {
        crate::runtime::context::record_parent_command(updates, replaces, goto);
    }

    fn has_parent_command(&self) -> bool {
        crate::runtime::context::has_parent_command()
    }
}
