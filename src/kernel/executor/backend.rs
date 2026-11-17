//! Execution backend abstraction for the shared Pregel loop.
//!
//! The loop owns all superstep/checkpoint decisions; a backend provides graph metadata
//! and the two runtime-specific operations: executing a node batch and persisting a
//! checkpoint.

use serde_json::Value;

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::Result;
use crate::kernel::executor::outcome::TaskOutcome;
use crate::kernel::executor::scheduling;
use crate::kernel::graph::CompiledGraph;
use crate::kernel::node::{ConditionalEdge, Task};
use crate::kernel::state::{State, StateSchema};

/// Runtime-specific half of the shared Pregel loop.
#[allow(dead_code)]
pub(crate) trait PregelBackend<S: State> {
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
    fn run_superstep(
        &self,
        state: &S,
        tasks: &[Task],
    ) -> impl std::future::Future<Output = Vec<TaskOutcome>>;
    fn load_checkpoint(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &str,
    ) -> impl std::future::Future<Output = Result<Option<Checkpoint>>>;
    fn save_checkpoint(
        &self,
        checkpoint: Checkpoint,
    ) -> impl std::future::Future<Output = Result<()>>;
}

/// Sync backend: scoped OS threads and a blocking checkpointer.
#[allow(dead_code)]
pub(crate) struct SyncBackend<'a, S: State> {
    pub(crate) graph: &'a CompiledGraph<S>,
    pub(crate) writer: Option<&'a crate::kernel::executor::checkpointing::CheckpointWriter>,
}

#[allow(dead_code)]
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

    async fn run_superstep(&self, state: &S, tasks: &[Task]) -> Vec<TaskOutcome> {
        super::superstep::run_superstep(self.graph, state, tasks)
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

    async fn save_checkpoint(&self, checkpoint: Checkpoint) -> Result<()> {
        let Some(checkpointer) = self.graph.checkpointer() else {
            return Ok(());
        };
        crate::kernel::executor::checkpointing::persist_checkpoint(
            checkpointer.as_ref(),
            checkpoint,
            self.writer,
        )
    }
}

impl<'a, S: State> SyncBackend<'a, S> {
    #[allow(dead_code)]
    pub(crate) fn schedule(
        &self,
        next_nodes: Vec<String>,
        router_sends: Vec<(String, crate::kernel::node::Send)>,
        next_tasks: Vec<Task>,
        pending_deferred: &mut Vec<Task>,
    ) -> Result<Vec<Task>> {
        scheduling::schedule_next_tasks(
            next_nodes,
            router_sends,
            next_tasks,
            |node| self.node_exists(node),
            |node| self.destinations_of(node),
            |node| self.is_deferred(node),
            pending_deferred,
        )
    }
}
