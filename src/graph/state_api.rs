//! State snapshot, history and `update_state` APIs.
#![allow(clippy::too_many_arguments)]

use super::compiled::CompiledGraph;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::Task;
use crate::_internal::state::ChannelUpdate;
use crate::_internal::state::apply_updates;
use crate::_internal::state::{State, values_to_state};
use crate::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::graph::StateSnapshot;
use crate::pregel::scheduling::{collect_next_nodes, dedupe, send_task};
use serde_json::{Map, Value};
use std::collections::HashSet;

impl<S: State> CompiledGraph<S> {
    pub fn get_state(&self, thread_id: &str) -> Result<Option<StateSnapshot<S>>> {
        self.get_state_in(thread_id, "")
    }

    /// Read the latest checkpoint of a thread inside a checkpoint namespace.
    pub fn get_state_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state"))?;
        match checkpointer.latest_in(thread_id, namespace)? {
            Some(checkpoint) => {
                let history = checkpointer.history_in(thread_id, namespace)?;
                let total = history.len() as i64;
                let step = history
                    .iter()
                    .position(|item| item.id == checkpoint.id)
                    .map(|index| total - 2 - index as i64)
                    .unwrap_or(checkpoint.step as i64);
                let mut snapshot = StateSnapshot::from_checkpoint_live(checkpoint, &self.schema)?;
                snapshot.step = step;
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
        namespace: &str,
        checkpoint_id: &str,
    ) -> Result<Option<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state_at"))?;
        let Some(checkpoint) = checkpointer.get(thread_id, checkpoint_id)? else {
            return Ok(None);
        };
        let history = checkpointer.history_in(thread_id, namespace)?;
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

    /// Latest state of every subgraph namespace of a thread (LangGraph `subgraphs=True`).
    pub fn subgraph_states(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("subgraph_states"))?;
        let mut namespaces = std::collections::BTreeSet::new();
        for checkpoint in checkpointer.history(thread_id)? {
            if !checkpoint.namespace.is_empty() {
                namespaces.insert(checkpoint.namespace);
            }
        }
        let mut states = Vec::new();
        for namespace in namespaces {
            if let Some(checkpoint) = checkpointer.latest_in(thread_id, &namespace)? {
                states.push(StateSnapshot::from_checkpoint(checkpoint)?);
            }
        }
        Ok(states)
    }

    /// Root state plus the latest state of each subgraph namespace.
    #[allow(clippy::type_complexity)]
    pub fn get_state_subgraphs(
        &self,
        thread_id: &str,
    ) -> Result<Option<(StateSnapshot<S>, Vec<StateSnapshot<S>>)>> {
        let Some(root) = self.get_state(thread_id)? else {
            return Ok(None);
        };
        Ok(Some((root, self.subgraph_states(thread_id)?)))
    }

    /// Read the full checkpoint history of a thread, newest first.
    pub fn get_state_history(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        self.get_state_history_filtered(thread_id, None, None)
    }

    /// Read history inside a checkpoint namespace.
    pub fn get_state_history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Vec<StateSnapshot<S>>> {
        self.get_state_history_filtered_in(thread_id, namespace, None, None)
    }

    /// Read history with an optional `before` checkpoint id and a result limit.
    ///
    /// `before` is exclusive, matching LangGraph's `get_state_history`.
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
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state_history"))?;
        let all = checkpointer.history_in(thread_id, namespace)?;
        let total = all.len() as i64;
        let step_by_id: std::collections::HashMap<String, usize> = all
            .iter()
            .enumerate()
            .map(|(index, checkpoint)| (checkpoint.id.clone(), index))
            .collect();
        let mut checkpoints = all;
        if let Some(before) = before {
            match checkpoints.iter().position(|cp| cp.id == before) {
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

    /// Write state updates into the latest checkpoint without running nodes.
    ///
    /// With `as_node`, the next tasks are derived from that node's outgoing edges
    /// (LangGraph `update_state(as_node=...)` semantics); without it the pending tasks
    /// of the latest checkpoint are preserved.
    pub fn update_state<I, K>(
        &self,
        thread_id: &str,
        updates: I,
        as_node: Option<&str>,
    ) -> Result<StateSnapshot<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        self.update_state_in(thread_id, "", updates, as_node)
    }

    /// Update state and, when `task_id` is given, infer `as_node` from that pending task.
    ///
    /// LangGraph uses `task_id` to attribute updates to a specific task; this engine stores
    /// pending tasks inside the checkpoint, so a matching task supplies the node whose
    /// outgoing edges recompute the next tasks.
    pub fn update_state_with_task<I, K>(
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
        let resolved = self.resolve_task_node(thread_id, "", task_id)?;
        let as_node = as_node.map(str::to_owned).or(resolved);
        self.update_state_inner(
            thread_id,
            "",
            updates,
            as_node.as_deref(),
            task_id.is_some(),
        )
    }

    fn resolve_task_node(
        &self,
        thread_id: &str,
        namespace: &str,
        task_id: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(task_id) = task_id else {
            return Ok(None);
        };
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("update_state(task_id)"))?;
        let checkpoint = checkpointer
            .latest_in(thread_id, namespace)?
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
    pub fn update_state_in<I, K>(
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
    }

    fn update_state_inner<I, K>(
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
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("update_state"))?;
        let checkpoint = checkpointer
            .latest_in(thread_id, namespace)?
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

        let next = if preserve_next {
            checkpoint.next.clone()
        } else {
            match as_node {
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
            }
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
            metadata: Map::from_iter([("source".to_string(), Value::String("update".to_string()))]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions: checkpoint.channel_versions.clone(),
            versions_seen: checkpoint.versions_seen.clone(),
            updated_channels,
            pending_writes: checkpoint.pending_writes.clone(),
            deferred: checkpoint.deferred.clone(),
        };
        let snapshot = StateSnapshot::from_checkpoint(updated.clone())?;
        checkpointer.put(updated)?;
        Ok(snapshot)
    }
}
