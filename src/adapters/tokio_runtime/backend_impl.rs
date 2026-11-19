//! `PregelBackend` / `OutcomeGraph` implementations for the async backend.
#![allow(clippy::too_many_arguments)]

use super::compiled::AsyncBackend;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::{ConditionalEdge, Task};
use crate::_internal::state::State;
use crate::_internal::stream::StreamChunk;
use crate::checkpoint::Checkpoint;
use crate::pregel::outcome::TaskOutcome;
use futures::stream::StreamExt;
use serde_json::Value;

async fn with_index<F: std::future::Future<Output = TaskOutcome>>(
    index: usize,
    future: F,
) -> (usize, TaskOutcome) {
    (index, future.await)
}

impl<'a, S: State> crate::pregel::backend::PregelBackend<S> for AsyncBackend<'a, S> {
    fn schema(&self) -> &crate::_internal::state::StateSchema {
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

    fn store(&self) -> Option<std::sync::Arc<dyn crate::store::BaseStore>> {
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
        let limit = self
            .run
            .max_concurrency
            .filter(|limit| *limit > 0)
            .unwrap_or_else(|| tasks.len().max(1));
        let run = async {
            let mut pending = futures::stream::FuturesUnordered::new();
            let mut remaining = tasks.iter().enumerate();
            let mut outcomes: Vec<Option<TaskOutcome>> = (0..tasks.len()).map(|_| None).collect();
            for _ in 0..limit {
                match remaining.next() {
                    Some((index, task)) => pending.push(with_index(
                        index,
                        self.graph.run_task(task, state, step, self.run),
                    )),
                    None => break,
                }
            }
            while let Some((index, outcome)) = pending.next().await {
                outcomes[index] = Some(outcome);
                if let Some((index, task)) = remaining.next() {
                    pending.push(with_index(
                        index,
                        self.graph.run_task(task, state, step, self.run),
                    ));
                }
            }
            outcomes
                .into_iter()
                .map(|outcome| outcome.expect("every task has an outcome"))
                .collect()
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

    async fn persist_checkpoint(&self, checkpoint: std::sync::Arc<Checkpoint>) -> Result<()> {
        if let Some(writer) = self.writer {
            writer
                .send(checkpoint)
                .map_err(|_| Error::Checkpoint("checkpoint writer stopped".to_string()))?;
            return Ok(());
        }
        if let Some(async_saver) = &self.graph.async_checkpointer {
            async_saver.put_shared(checkpoint).await?;
        } else if let Some(saver) = &self.graph.checkpointer {
            crate::pregel::checkpointing::persist_checkpoint(saver.as_ref(), checkpoint, None)?;
        } else {
            return Err(Error::MissingCheckpointer("persist_checkpoint"));
        }
        Ok(())
    }
}

impl<'a, S: State> crate::pregel::outcome::OutcomeGraph for AsyncBackend<'a, S> {
    fn trace_sinks(&self) -> &[std::sync::Arc<dyn crate::_internal::ports::TraceSink>] {
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
