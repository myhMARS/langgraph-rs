//! Async state, history, update-state and thread management APIs.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::AsyncCompiledGraph;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::Task;
use crate::_internal::state::{ChannelUpdate, State, apply_updates, values_to_state};
use crate::checkpoint::{Checkpoint, infer_as_node, new_checkpoint_id, now_millis, versions_for};
use crate::graph::StateSnapshot;
use crate::graph::metadata_matches;
use crate::pregel::scheduling::{collect_next_nodes, dedupe, send_task};

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
                let mut snapshot = StateSnapshot::from_checkpoint_live(checkpoint, &self.schema)?;
                snapshot.step = snapshot.step.saturating_sub(1);
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }

    /// Read a specific checkpoint of a thread.
    pub fn get_state_at(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<StateSnapshot<S>>> {
        self.get_state_in_at(thread_id, "", checkpoint_id)
    }

    /// Read a specific checkpoint of a thread inside a checkpoint namespace.
    pub fn get_state_in_at(
        &self,
        thread_id: &str,
        _namespace: &str,
        checkpoint_id: &str,
    ) -> Result<Option<StateSnapshot<S>>> {
        let Some(checkpoint) =
            futures::executor::block_on(self.saver_get(thread_id, checkpoint_id))?
        else {
            return Ok(None);
        };
        let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
        snapshot.step = snapshot.step.saturating_sub(1);
        Ok(Some(snapshot))
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

    /// History filtered by metadata values (LangGraph `get_state_history(filter=...)`).
    pub fn get_state_history_where(
        &self,
        thread_id: &str,
        filter: Option<&serde_json::Map<String, serde_json::Value>>,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        self.get_state_history_where_in(thread_id, "", filter, before, limit)
    }

    /// History inside a namespace, filtered by metadata values.
    pub fn get_state_history_where_in(
        &self,
        thread_id: &str,
        namespace: &str,
        filter: Option<&serde_json::Map<String, serde_json::Value>>,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        let history = self.get_state_history_filtered_in(thread_id, namespace, before, None)?;
        let mut matches: Vec<StateSnapshot<S>> = match filter {
            Some(filter) => history
                .into_iter()
                .filter(|snapshot| metadata_matches(snapshot, filter))
                .collect(),
            None => history,
        };
        if let Some(limit) = limit {
            matches.truncate(limit);
        }
        Ok(matches)
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
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = snapshot.step.saturating_sub(1);
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
            return self
                .update_state_inner(thread_id, "", updates, None, true)
                .await;
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
        self.update_state_inner(thread_id, namespace, updates, as_node, false)
            .await
    }

    async fn update_state_inner<I, K>(
        &self,
        thread_id: &str,
        namespace: &str,
        updates: I,
        as_node: Option<&str>,
        preserve_next: bool,
    ) -> Result<StateSnapshot<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let mut checkpoint = self
            .saver_latest_in(thread_id, namespace)
            .await?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{thread_id}`"))
            })?;
        let mut values = checkpoint.values.clone();
        // Match the sync runtime and `get_state`: pending sibling writes are part of the
        // visible state, so they must survive an update.
        crate::pregel::apply_pending_writes(&self.schema, &mut values, &checkpoint.pending_writes)?;
        let updates: Vec<ChannelUpdate> = updates
            .into_iter()
            .map(|(channel, value)| ChannelUpdate {
                channel: channel.into(),
                value,
                node: "<update>".to_string(),
            })
            .collect();
        apply_updates(&self.schema, &mut values, updates)?;
        let values = self.schema.persisted_values(&values);
        let state = values_to_state::<S>(&values)?;
        let inferred = if preserve_next || as_node.is_some() || !checkpoint.next.is_empty() {
            None
        } else {
            let single = if self.nodes.len() == 1 {
                self.nodes.keys().next().map(String::as_str)
            } else {
                None
            };
            infer_as_node(&checkpoint, single, |node| self.nodes.contains_key(node))?
        };
        // A paused thread keeps its pending tasks unless `as_node` overrides them.
        let next = if preserve_next || (as_node.is_none() && !checkpoint.next.is_empty()) {
            checkpoint.next.clone()
        } else if as_node == Some(crate::END) {
            Vec::new()
        } else if matches!(as_node, Some(crate::START) | Some("__input__")) {
            self.entry_tasks(&state)?
        } else {
            match as_node.or(inferred.as_deref()) {
                None => {
                    return Err(Error::validation("Ambiguous update, specify as_node"));
                }
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
            }
        };
        let step = checkpoint.step + 1;
        let (channel_versions, versions_seen, updated_channels) = versions_for(
            Some(&mut checkpoint),
            &values,
            step,
            &[],
            &std::collections::HashMap::new(),
        );
        let id = new_checkpoint_id(thread_id, step);
        let mut next = next;
        crate::pregel::scheduling::assign_v4_ids(&mut next, Some(&id), namespace, step);
        let updated = Checkpoint {
            id,
            thread_id: thread_id.to_string(),
            step,
            values,
            next,
            deferred_routes: Vec::new(),
            resume_tasks: Vec::new(),
            goto_sources: Vec::new(),
            completed_tasks: Vec::new(),
            // Fresh `source="update"` checkpoint, matching the sync runtime.
            interrupted: false,
            // Keep the pause marker so resuming after an edit does not re-run the
            // `interrupt_before` hook that already fired.
            paused_before: checkpoint.paused_before,
            parent_id: Some(checkpoint.id),
            created_at_millis: now_millis(),
            interrupts: Vec::new(),
            metadata: Map::from_iter([(
                String::from("source"),
                Value::String(String::from("update")),
            )]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions,
            versions_seen,
            updated_channels,
            pending_writes: Vec::new(),
            deferred: checkpoint.deferred.clone(),
        };
        let mut snapshot = StateSnapshot::from_checkpoint(updated.clone())?;
        snapshot.step = snapshot.step.saturating_sub(1);
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
        all.into_iter()
            .map(|checkpoint| {
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = snapshot.step.saturating_sub(1);
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
        all.into_iter()
            .map(|checkpoint| {
                let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
                snapshot.step = snapshot.step.saturating_sub(1);
                Ok(snapshot)
            })
            .collect()
    }
}
