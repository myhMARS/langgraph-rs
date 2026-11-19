//! State snapshot, history and `update_state` APIs.
#![allow(clippy::too_many_arguments)]

use super::compiled::CompiledGraph;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::Task;
use crate::_internal::state::ChannelUpdate;
use crate::_internal::state::apply_updates;
use crate::_internal::state::{State, values_to_state};
use crate::checkpoint::{Checkpoint, infer_as_node, new_checkpoint_id, now_millis, versions_for};
use crate::graph::StateSnapshot;
use crate::pregel::scheduling::{collect_next_nodes, dedupe, send_task};
use serde_json::{Map, Value};
use std::collections::HashSet;

impl<S: State> CompiledGraph<S> {
    /// `true` when the thread has at least one checkpoint.
    ///
    /// Bindings use this instead of building a full snapshot just to branch on existence.
    pub fn has_thread(&self, thread_id: &str) -> Result<bool> {
        match self.checkpointer() {
            Some(checkpointer) => checkpointer.has_thread(thread_id),
            None => Ok(false),
        }
    }

    /// Namespace-scoped variant of [`CompiledGraph::has_thread`].
    pub fn has_thread_in(&self, thread_id: &str, namespace: &str) -> Result<bool> {
        match self.checkpointer() {
            Some(checkpointer) => Ok(checkpointer.latest_in(thread_id, namespace)?.is_some()),
            None => Ok(false),
        }
    }

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
        match checkpointer.latest_in_shared(thread_id, namespace)? {
            Some(checkpoint) => {
                let mut snapshot =
                    StateSnapshot::from_checkpoint_live_borrowed(&checkpoint, &self.schema)?;
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
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state_at"))?;
        let Some(checkpoint) = checkpointer.get(thread_id, checkpoint_id)? else {
            return Ok(None);
        };
        let mut snapshot = StateSnapshot::from_checkpoint(checkpoint)?;
        snapshot.step = snapshot.step.saturating_sub(1);
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
                // Child snapshots are live in LangGraph: pending sibling writes are applied.
                let mut snapshot = StateSnapshot::from_checkpoint_live(checkpoint, &self.schema)?;
                snapshot.step = snapshot.step.saturating_sub(1);
                states.push(snapshot);
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

    /// History filtered by metadata values (LangGraph `get_state_history(filter=...)`).
    pub fn get_state_history_where_in(
        &self,
        thread_id: &str,
        namespace: &str,
        filter: Option<&Map<String, Value>>,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        // Without a filter the `limit` can be pushed into the page read: materializing the
        // whole history first made `get_state_history(limit=5)` scale with the thread.
        let Some(filter) = filter else {
            return self.get_state_history_filtered_in(thread_id, namespace, before, limit);
        };
        let history = self.get_state_history_filtered_in(thread_id, namespace, before, None)?;
        let mut matches: Vec<StateSnapshot<S>> = history
            .into_iter()
            .filter(|snapshot| metadata_matches(snapshot, filter))
            .collect();
        if let Some(limit) = limit {
            matches.truncate(limit);
        }
        Ok(matches)
    }

    /// History filtered by metadata values.
    pub fn get_state_history_where(
        &self,
        thread_id: &str,
        filter: Option<&Map<String, Value>>,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<StateSnapshot<S>>> {
        self.get_state_history_where_in(thread_id, "", filter, before, limit)
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
        let checkpoints =
            checkpointer.history_page_in_shared(thread_id, namespace, before, limit)?;
        checkpoints
            .into_iter()
            .map(|checkpoint| {
                let mut snapshot = StateSnapshot::from_checkpoint_borrowed(&checkpoint)?;
                snapshot.step = snapshot.step.saturating_sub(1);
                Ok(snapshot)
            })
            .collect()
    }

    /// History checkpoint handles without building snapshots.
    ///
    /// The Python binding exposes history as a lazy sequence: this is the cheap first step
    /// (pointer clones for in-memory savers), and each entry is converted on demand.
    pub(crate) fn history_handles(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<std::sync::Arc<Checkpoint>>> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state_history"))?;
        checkpointer.history_page_in_shared(thread_id, namespace, before, limit)
    }

    /// Build one snapshot from a history handle (step is reported one lower, like `get_state`).
    pub(crate) fn snapshot_from_handle(
        &self,
        handle: &std::sync::Arc<Checkpoint>,
    ) -> Result<StateSnapshot<S>> {
        let mut snapshot = StateSnapshot::from_checkpoint_borrowed(handle)?;
        snapshot.step = snapshot.step.saturating_sub(1);
        Ok(snapshot)
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
        // Share the stored checkpoint instead of deep-copying it: only the value map is
        // mutated by the update, everything else is carried over by reference/clone-on-write.
        let checkpoint = checkpointer
            .latest_in_shared(thread_id, namespace)?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{thread_id}`"))
            })?;
        let checkpoint = checkpoint.as_ref();
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
        // Share the stored checkpoint instead of deep-copying it: only the value map is
        // mutated by the update, everything else is carried over by reference/clone-on-write.
        let checkpoint = checkpointer
            .latest_in_shared(thread_id, namespace)?
            .ok_or_else(|| {
                Error::Checkpoint(format!("no checkpoint found for thread `{thread_id}`"))
            })?;
        let checkpoint = checkpoint.as_ref();

        let mut values = checkpoint.values.clone();
        // `get_state` shows sibling writes from the paused superstep (they live in
        // `pending_writes`), so the update checkpoint must start from the same view.
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
        // A dynamic schema persists every channel, so the copy `persisted_values` would make
        // is redundant here: the merged map is owned and can move into the checkpoint.
        let values = if self.schema.persists_all(&values) {
            values
        } else {
            self.schema.persisted_values(&values)
        };
        let state = values_to_state::<S>(&values)?;

        let inferred = if preserve_next || as_node.is_some() || !checkpoint.next.is_empty() {
            None
        } else {
            let single = if self.nodes.len() == 1 {
                self.nodes.keys().next().map(String::as_str)
            } else {
                None
            };
            infer_as_node(checkpoint, single, |node| self.nodes.contains_key(node))?
        };
        // A paused thread keeps its pending tasks unless `as_node` overrides them,
        // matching LangGraph's `update_state` on an interrupted checkpoint.
        let next = if preserve_next || (as_node.is_none() && !checkpoint.next.is_empty()) {
            checkpoint.next.clone()
        } else if as_node == Some(crate::END) {
            // LangGraph: `as_node=END` records the update as coming from the end, so no
            // follow-up tasks are scheduled.
            Vec::new()
        } else if matches!(as_node, Some(crate::START) | Some("__input__")) {
            self.entry_tasks(&state)?
        } else {
            match as_node.or(inferred.as_deref()) {
                None => {
                    // LangGraph refuses an update whose author cannot be inferred.
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

        // LangGraph advances the step for the update checkpoint and bumps the versions of
        // the channels it changed, so later inference can see the update.
        let step = checkpoint.step + 1;
        // LangGraph's synthetic update task only has the INTERRUPT trigger, which is not a
        // channel version, so `versions_seen` is left untouched here.
        let (channel_versions, versions_seen, updated_channels) = versions_for(
            Some(checkpoint),
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
            // LangGraph writes a fresh `source="update"` checkpoint: earlier pending writes
            // and interrupts belong to the previous tuple and are not carried over.
            interrupted: false,
            // Keep the pause marker so resuming after an edit does not re-run the
            // `interrupt_before` hook that already fired.
            paused_before: checkpoint.paused_before,
            parent_id: Some(checkpoint.id.clone()),
            created_at_millis: now_millis(),
            interrupts: Vec::new(),
            metadata: Map::from_iter([("source".to_string(), Value::String("update".to_string()))]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions,
            versions_seen,
            updated_channels,
            pending_writes: Vec::new(),
            deferred: checkpoint.deferred.clone(),
        };
        // The merged `state` is already materialised, so build the snapshot from it instead
        // of deserialising the checkpoint values a second time.
        let mut snapshot = StateSnapshot::from_parts(&updated, state)?;
        // Readers report `checkpoint.step - 1`; keep this snapshot consistent with them.
        snapshot.step = snapshot.step.saturating_sub(1);
        checkpointer.put(updated)?;
        Ok(snapshot)
    }
}

/// Compare a snapshot's metadata (and step) against a LangGraph `filter` map.
pub(crate) fn metadata_matches<S: crate::_internal::state::State>(
    snapshot: &StateSnapshot<S>,
    filter: &Map<String, Value>,
) -> bool {
    filter.iter().all(|(key, expected)| {
        if key == "step" {
            return expected.as_i64() == Some(snapshot.step);
        }
        snapshot
            .metadata
            .get(key)
            .is_some_and(|actual| actual == expected)
    })
}
