//! Async state, history, update-state and thread management APIs.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::AsyncCompiledGraph;
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::{Error, Result};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe, send_task};
use crate::kernel::graph::StateSnapshot;
use crate::kernel::node::Task;
use crate::kernel::state::{ChannelUpdate, State, apply_updates, values_to_state};

impl<S: State> AsyncCompiledGraph<S> {
    pub fn get_state(&self, thread_id: &str) -> Result<Option<StateSnapshot<S>>> {
        self.get_state_in(thread_id, "")
    }

    /// Read the latest checkpoint of a thread inside a checkpoint namespace.
    pub fn get_state_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<StateSnapshot<S>>> {
        match self.saver_latest_in_sync(thread_id, namespace)? {
            Some(checkpoint) => {
                let history = self.saver_history_in_sync(thread_id, namespace)?;
                let total = history.len() as i64;
                let step = history
                    .iter()
                    .position(|item| item.id == checkpoint.id)
                    .map(|index| total - 2 - index as i64)
                    .unwrap_or(checkpoint.step as i64);
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = step;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    /// Resume a thread from a specific checkpoint id.
    pub async fn resume_from(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
        value: Value,
    ) -> Result<S> {
        self.resume_inner(thread_id, Some(checkpoint_id), Some(value), None, None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume a thread from a checkpoint id inside a namespace.
    pub async fn resume_from_in(
        &self,
        thread_id: &str,
        namespace: &str,
        checkpoint_id: &str,
        value: Value,
    ) -> Result<S> {
        self.resume_inner(
            thread_id,
            Some(checkpoint_id),
            Some(value),
            None,
            Some(namespace),
        )
        .await
        .map(|outcome| outcome.state)
    }

    /// Export the latest checkpoint as a LangGraph-shaped JSON document.
    pub fn export_checkpoint(&self, thread_id: &str) -> Result<Option<Value>> {
        let checkpoint = futures::executor::block_on(self.saver_latest(thread_id))?;
        Ok(checkpoint.map(|checkpoint| crate::interop::checkpoint_to_langgraph(&checkpoint)))
    }

    /// Import a LangGraph-shaped checkpoint document into a thread.
    pub fn import_checkpoint(&self, thread_id: &str, document: &Value) -> Result<()> {
        let mut checkpoint = crate::interop::checkpoint_from_langgraph(document)?;
        checkpoint.thread_id = thread_id.to_string();
        futures::executor::block_on(self.saver_put(checkpoint))
    }

    /// History with `before` / `limit` filters.
    pub fn get_state_history_filtered(
        &self,
        thread_id: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        self.get_state_history_filtered_in(thread_id, "", before, limit)
    }

    /// History inside a namespace with `before` / `limit` filters.
    pub fn get_state_history_filtered_in(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        let all = futures::executor::block_on(self.saver_history_in(thread_id, namespace))?;
        let total = all.len() as i64;
        let step_by_id: std::collections::HashMap<String, usize> = all
            .iter()
            .enumerate()
            .map(|(index, checkpoint)| (checkpoint.id.clone(), index))
            .collect();
        let mut checkpoints = all;
        if let Some(before) = before {
            match checkpoints
                .iter()
                .position(|checkpoint| checkpoint.id == before)
            {
                Some(index) => {
                    checkpoints.drain(0..=index);
                }
                None => checkpoints.clear(),
            }
        }
        if let Some(limit) = limit {
            checkpoints.truncate(limit);
        }
        checkpoints
            .into_iter()
            .map(|checkpoint| {
                let step = step_by_id
                    .get(&checkpoint.id)
                    .map(|index| total - 2 - *index as i64)
                    .unwrap_or(checkpoint.step as i64);
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = step;
                Ok(snapshot)
            })
            .collect()
    }

    /// Apply state updates without running nodes.
    pub async fn update_state<I, K>(
        &self,
        thread_id: &str,
        updates: I,
        as_node: Option<&str>,
    ) -> Result<StateSnapshot<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        self.update_state_in(thread_id, "", updates, as_node).await
    }

    /// Apply state updates and infer `as_node` from a pending task id.
    pub async fn update_state_with_task<I, K>(
        &self,
        thread_id: &str,
        updates: I,
        as_node: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<StateSnapshot<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        if task_id.is_some() {
            // Validate the task id but keep the checkpoint's pending `next`, matching
            // the sync implementation's preserve-next semantics.
            self.resolve_task_node(thread_id, "", task_id).await?;
            return self.update_state_in(thread_id, "", updates, None).await;
        }
        let as_node = as_node.map(str::to_owned);
        self.update_state_in(thread_id, "", updates, as_node.as_deref())
            .await
    }

    async fn resolve_task_node(
        &self,
        thread_id: &str,
        namespace: &str,
        task_id: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(task_id) = task_id else {
            return Ok(None);
        };
        let checkpoint = self
            .saver_latest_in(thread_id, namespace)
            .await?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{thread_id}`"))
            })?;
        checkpoint
            .next
            .iter()
            .find(|task| task.id == task_id)
            .map(|task| Some(task.node.clone()))
            .ok_or_else(|| {
                Error::validation(format!(
                    "update_state: task id `{task_id}` is not pending on thread `{thread_id}`"
                ))
            })
    }

    /// Apply state updates inside a checkpoint namespace.
    pub async fn update_state_in<I, K>(
        &self,
        thread_id: &str,
        namespace: &str,
        updates: I,
        as_node: Option<&str>,
    ) -> Result<StateSnapshot<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let checkpoint = self
            .saver_latest_in(thread_id, namespace)
            .await?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{thread_id}`"))
            })?;
        let mut values = checkpoint.values.clone();
        let updates: Vec<ChannelUpdate> = updates
            .into_iter()
            .map(|(channel, value)| ChannelUpdate {
                channel: channel.into(),
                value,
                node: "<update>".to_string(),
            })
            .collect();
        let updated_channels: Vec<String> = updates
            .iter()
            .map(|update| update.channel.clone())
            .collect();
        apply_updates(&self.schema, &mut values, updates)?;
        let values = self.schema.persisted_values(&values);
        let state = values_to_state::<S>(&values)?;
        let next = match as_node {
            Some(node) => {
                if !self.nodes.contains_key(node) {
                    return Err(Error::validation(format!(
                        "update_state: `{node}` is not a registered node"
                    )));
                }
                let executed = vec![node.to_string()];
                let (targets, router_sends) = collect_next_nodes(
                    &state,
                    &executed,
                    &HashSet::new(),
                    &[],
                    |candidate| self.edges_of(candidate),
                    |candidate| self.conditional_of(candidate),
                    |candidate| self.nodes.contains_key(candidate),
                    |candidate| self.destinations_of(candidate),
                )?;
                let mut tasks: Vec<Task> = dedupe(targets).into_iter().map(Task::new).collect();
                for (source, send) in router_sends {
                    tasks.push(send_task(&source, send)?);
                }
                tasks
            }
            None => checkpoint.next.clone(),
        };
        let id = new_checkpoint_id(thread_id, checkpoint.step);
        let updated = Checkpoint {
            id,
            thread_id: thread_id.to_string(),
            step: checkpoint.step,
            values,
            next,
            interrupted: checkpoint.interrupted,
            paused_before: checkpoint.paused_before,
            parent_id: Some(checkpoint.id),
            created_at_millis: now_millis(),
            interrupts: checkpoint.interrupts.clone(),
            metadata: Map::from_iter([(
                String::from("source"),
                Value::String(String::from("update")),
            )]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions: checkpoint.channel_versions.clone(),
            versions_seen: checkpoint.versions_seen.clone(),
            updated_channels,
            pending_writes: checkpoint.pending_writes.clone(),
            deferred: checkpoint.deferred.clone(),
        };
        let snapshot = StateSnapshot::from_checkpoint(updated.clone())?;
        self.saver_put(updated).await?;
        Ok(snapshot)
    }

    /// List threads that have at least one checkpoint.
    pub fn threads(&self) -> Result<Vec<String>> {
        self.saver_threads_sync()
    }

    /// Keep only the newest `keep_last` checkpoints of a thread.
    pub fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        if let Some(saver) = &self.async_checkpointer {
            futures::executor::block_on(saver.prune_thread(thread_id, keep_last))
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.prune_thread(thread_id, keep_last)
        } else {
            Err(Error::MissingCheckpointer("prune_thread"))
        }
    }

    /// Delete every checkpoint of a thread.
    pub fn delete_thread(&self, thread_id: &str) -> Result<()> {
        if let Some(saver) = &self.async_checkpointer {
            futures::executor::block_on(saver.delete_thread(thread_id))
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.delete_thread(thread_id)
        } else {
            Err(Error::MissingCheckpointer("delete_thread"))
        }
    }

    /// Copy every checkpoint of one thread to another thread id.
    pub fn copy_thread(&self, source: &str, target: &str) -> Result<()> {
        if let Some(saver) = &self.async_checkpointer {
            futures::executor::block_on(saver.copy_thread(source, target))
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.copy_thread(source, target)
        } else {
            Err(Error::MissingCheckpointer("copy_thread"))
        }
    }

    /// Read the full checkpoint history of a thread, newest first.
    pub fn get_state_history(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        let all = self.saver_history_sync(thread_id)?;
        let total = all.len() as i64;
        all.into_iter()
            .enumerate()
            .map(|(index, checkpoint)| {
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = total - 2 - index as i64;
                Ok(snapshot)
            })
            .collect()
    }

    /// Read the full checkpoint history of a thread inside a checkpoint namespace.
    pub fn get_state_history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Vec<StateSnapshot<S>>> {
        let all = self.saver_history_in_sync(thread_id, namespace)?;
        let total = all.len() as i64;
        all.into_iter()
            .enumerate()
            .map(|(index, checkpoint)| {
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = total - 2 - index as i64;
                Ok(snapshot)
            })
            .collect()
    }
}
