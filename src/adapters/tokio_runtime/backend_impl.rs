//! `PregelBackend` / `OutcomeGraph` implementations for the async backend.
#![allow(clippy::too_many_arguments)]

use super::compiled::AsyncBackend;
use crate::application::pregel::outcome::TaskOutcome;
use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::node::{ConditionalEdge, Task};
use crate::kernel::state::State;
use crate::kernel::stream::StreamChunk;
use futures::future::join_all;
use serde_json::Value;

impl<'a, S: State> crate::application::pregel::backend::PregelBackend<S> for AsyncBackend<'a, S> {
    fn schema(&self) -> &crate::kernel::state::StateSchema {
        &self.graph.schema
    }

    fn recursion_limit(&self) -> usize {
        self.graph.recursion_limit
    }

    fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        self.graph.entry_tasks(state)
    }

    fn node_exists(&self, node: &str) -> bool {
        self.graph.nodes.contains_key(node)
    }

    fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.graph.edges.get(node).map(Vec::as_slice)
    }

    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.graph.conditional.get(node).map(Vec::as_slice)
    }

    fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.graph.destinations_of(node)
    }

    fn interrupts_before(&self, node: &str) -> bool {
        self.graph.interrupt_before.contains(node)
    }

    fn interrupts_after(&self, node: &str) -> bool {
        self.graph.interrupt_after.contains(node)
    }

    fn is_deferred(&self, node: &str) -> bool {
        self.graph.deferred.contains(node)
    }

    fn context_schema(&self) -> Option<&Value> {
        self.graph.context_schema()
    }

    fn store(&self) -> Option<std::sync::Arc<dyn crate::kernel::store::BaseStore>> {
        self.graph.store.clone()
    }

    fn has_checkpointer(&self) -> bool {
        self.graph.has_checkpointer()
    }

    fn clear_ephemeral(&self, values: &mut serde_json::Map<String, Value>) -> bool {
        self.graph.schema.clear_ephemeral(values)
    }

    fn collector(&self) -> Option<std::sync::Arc<std::sync::Mutex<Vec<StreamChunk>>>> {
        self.run.collector.clone()
    }

    async fn run_superstep(
        &self,
        state: &S,
        tasks: &[Task],
        step: usize,
        control: Option<&crate::RunControl>,
    ) -> Result<Vec<TaskOutcome>> {
        let run = async {
            match self.run.max_concurrency.filter(|limit| *limit > 0) {
                Some(limit) if tasks.len() > limit => {
                    let mut outcomes = Vec::with_capacity(tasks.len());
                    for chunk in tasks.chunks(limit) {
                        outcomes.extend(
                            join_all(
                                chunk
                                    .iter()
                                    .map(|task| self.graph.run_task(task, state, step, self.run)),
                            )
                            .await,
                        );
                    }
                    outcomes
                }
                _ => {
                    join_all(
                        tasks
                            .iter()
                            .map(|task| self.graph.run_task(task, state, step, self.run)),
                    )
                    .await
                }
            }
        };
        match control {
            Some(control) => tokio::select! {
                outcomes = run => Ok(outcomes),
                _ = control.wait_aborted() => {
                    Err(Error::other("run aborted by RunControl"))
                }
            },
            None => Ok(run.await),
        }
    }

    async fn load_checkpoint(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &str,
    ) -> Result<Option<Checkpoint>> {
        match checkpoint_id {
            Some(id) => self.graph.saver_get(thread_id, id).await,
            None => self.graph.saver_latest_in(thread_id, namespace).await,
        }
    }

    async fn persist_checkpoint(&self, checkpoint: Checkpoint) -> Result<()> {
        if let Some(writer) = self.writer {
            writer
                .send(checkpoint)
                .map_err(|_| Error::Checkpoint("checkpoint writer stopped".to_string()))?;
            return Ok(());
        }
        if let Some(async_saver) = &self.graph.async_checkpointer {
            async_saver.put(checkpoint).await?;
        } else if let Some(saver) = &self.graph.checkpointer {
            crate::application::pregel::checkpointing::persist_checkpoint(
                saver.as_ref(),
                checkpoint,
                None,
            )?;
        } else {
            return Err(Error::MissingCheckpointer("persist_checkpoint"));
        }
        Ok(())
    }
}

impl<'a, S: State> crate::application::pregel::outcome::OutcomeGraph for AsyncBackend<'a, S> {
    fn trace_sinks(&self) -> &[std::sync::Arc<dyn crate::kernel::ports::TraceSink>] {
        &self.graph.trace_sinks
    }

    fn error_handler(&self, node: &str) -> Option<&str> {
        self.graph.error_edges.get(node).map(String::as_str)
    }

    fn has_node(&self, node: &str) -> bool {
        self.graph.nodes.contains_key(node)
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
        super::context::record_parent_command(updates, replaces, goto);
    }

    fn has_parent_command(&self) -> bool {
        super::context::has_parent_command()
    }
}
