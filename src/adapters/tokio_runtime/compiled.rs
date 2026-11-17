//! Compiled async graph and its superstep loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::context::AsyncContext;
use super::{AsyncNodeFn, BoxNodeFuture};
use crate::kernel::cache::{CachePolicy, CachedOutput, NodeCache};
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::executor::StartInput;
use crate::kernel::executor::outcome::{TaskFailure, TaskOutcome, TaskResult};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe, send_task};
use crate::kernel::graph::{Entry, RunConfig, StateSnapshot};
use crate::kernel::node::{ConditionalEdge, Task};
use crate::kernel::ports::{AsyncCheckpointer, Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{
    ChannelUpdate, State, StateSchema, apply_updates, state_to_values, values_to_state,
};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::{StreamChunk, StreamMode};
use crate::kernel::timeout::TimeoutPolicy;
use crate::kernel::usage::{RunOutcome, Usage};

pub struct AsyncCompiledGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, AsyncNodeFn<S>>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) entry: Entry<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) async_checkpointer: Option<Arc<dyn AsyncCheckpointer>>,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) timeouts: IndexMap<String, TimeoutPolicy>,
    pub(crate) cache: Option<Arc<dyn NodeCache>>,
    pub(crate) cache_policies: IndexMap<String, CachePolicy>,
    pub(crate) deferred: HashSet<String>,
    pub(crate) destinations: IndexMap<String, Vec<String>>,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) trace_sinks: Vec<Arc<dyn TraceSink>>,
    pub(crate) recursion_limit: usize,
    pub(crate) context_schema: Option<Value>,
    pub(crate) name: Option<String>,
}

/// Where a run starts from.
struct AsyncRunInfo {
    thread_id: Option<String>,
    namespace: String,
    store: Option<Arc<dyn BaseStore>>,
    context: Option<Value>,
    collector: Option<Arc<std::sync::Mutex<Vec<StreamChunk>>>>,
}

impl<S: State> AsyncCompiledGraph<S> {
    /// Timeout policy of one node, when configured.
    /// JSON schema configured for `RunConfig::context`, if any.
    pub fn context_schema(&self) -> Option<&Value> {
        self.context_schema.as_ref()
    }

    pub fn timeout_of(&self, node: &str) -> Option<&TimeoutPolicy> {
        self.timeouts.get(node)
    }

    /// Allowed destinations of a node, when an allow-list is configured.
    pub fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.destinations.get(node).map(Vec::as_slice)
    }

    /// Whether the node is deferred until the graph has no regular work left.
    pub fn is_deferred(&self, node: &str) -> bool {
        self.deferred.contains(node)
    }

    /// Node cache backend, when caching is enabled.
    pub fn cache(&self) -> Option<&Arc<dyn NodeCache>> {
        self.cache.as_ref()
    }

    /// Cache policy of one node, when configured.
    pub fn cache_policy(&self, node: &str) -> Option<&CachePolicy> {
        self.cache_policies.get(node)
    }

    /// Channel schema of the graph.
    pub fn schema(&self) -> &StateSchema {
        &self.schema
    }

    /// Default superstep budget.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Render the graph as a Mermaid `graph TD` diagram.
    pub fn to_mermaid(&self) -> String {
        crate::kernel::graph::visualize::render_mermaid(
            crate::kernel::graph::visualize::GraphLayout {
                nodes: self.nodes.keys().map(String::as_str).collect(),
                edges: &self.edges,
                conditional: &self.conditional,
                entry: &self.entry,
                retries: &self.retries,
                error_edges: &self.error_edges,
                interrupt_before: &self.interrupt_before,
                interrupt_after: &self.interrupt_after,
            },
        )
    }

    /// Render the graph as a Graphviz DOT document.
    pub fn to_dot(&self) -> String {
        crate::kernel::graph::visualize::render_dot(crate::kernel::graph::visualize::GraphLayout {
            nodes: self.nodes.keys().map(String::as_str).collect(),
            edges: &self.edges,
            conditional: &self.conditional,
            entry: &self.entry,
            retries: &self.retries,
            error_edges: &self.error_edges,
            interrupt_before: &self.interrupt_before,
            interrupt_after: &self.interrupt_after,
        })
    }

    /// Render the graph as a PNG image using Graphviz `dot`.
    pub fn draw_png(&self) -> Result<Vec<u8>> {
        crate::kernel::graph::visualize::dot_to_png(&self.to_dot())
    }

    /// Graph name configured with `with_name`.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Latest state of every subgraph namespace of a thread.
    pub fn subgraph_states(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        let mut namespaces = std::collections::BTreeSet::new();
        for checkpoint in futures::executor::block_on(self.saver_history(thread_id))? {
            if !checkpoint.namespace.is_empty() {
                namespaces.insert(checkpoint.namespace);
            }
        }
        let mut states = Vec::new();
        for namespace in namespaces {
            if let Some(checkpoint) =
                futures::executor::block_on(self.saver_latest_in(thread_id, &namespace))?
            {
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

    /// Returns `true` when checkpointing is enabled.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some() || self.async_checkpointer.is_some()
    }

    /// Run the graph over several inputs concurrently (bounded by `max_concurrency`).
    ///
    /// Results keep the input order; a failed invocation does not abort the batch.
    pub async fn batch<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<S>>
    where
        I: IntoIterator<Item = S>,
    {
        use futures::StreamExt;

        let inputs: Vec<S> = inputs.into_iter().collect();
        let max_concurrency = config.max_concurrency.unwrap_or(1).max(1);
        if max_concurrency <= 1 || inputs.len() <= 1 {
            let mut results = Vec::with_capacity(inputs.len());
            for input in inputs {
                results.push(self.run(&input, config).await.map(|outcome| outcome.state));
            }
            return results;
        }
        let mut indexed: Vec<(usize, Result<S>)> =
            futures::stream::iter(inputs.into_iter().enumerate().map(
                |(index, input)| async move {
                    (
                        index,
                        self.run(&input, config).await.map(|outcome| outcome.state),
                    )
                },
            ))
            .buffer_unordered(max_concurrency)
            .collect()
            .await;
        indexed.sort_by_key(|(index, _)| *index);
        indexed.into_iter().map(|(_, result)| result).collect()
    }

    /// Batch variant of [`Self::run`] that also returns run reports.
    pub async fn batch_run<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<RunOutcome<S>>>
    where
        I: IntoIterator<Item = S>,
    {
        use futures::StreamExt;

        let inputs: Vec<S> = inputs.into_iter().collect();
        let max_concurrency = config.max_concurrency.unwrap_or(1).max(1);
        if max_concurrency <= 1 || inputs.len() <= 1 {
            let mut results = Vec::with_capacity(inputs.len());
            for input in inputs {
                results.push(self.run(&input, config).await);
            }
            return results;
        }
        let mut indexed: Vec<(usize, Result<RunOutcome<S>>)> = futures::stream::iter(
            inputs
                .into_iter()
                .enumerate()
                .map(|(index, input)| async move { (index, self.run(&input, config).await) }),
        )
        .buffer_unordered(max_concurrency)
        .collect()
        .await;
        indexed.sort_by_key(|(index, _)| *index);
        indexed.into_iter().map(|(_, result)| result).collect()
    }

    /// Run a graph starting at the `Command`'s `goto` nodes.
    pub async fn run_command(
        &self,
        command: crate::Command,
        config: &RunConfig,
    ) -> Result<RunOutcome<S>> {
        if command.resume.is_some() {
            return Err(Error::validation(
                "Command(goto=...) cannot also carry a resume value",
            ));
        }
        if command.parent {
            return Err(Error::validation(
                "Command(graph=PARENT) is only supported for subgraphs",
            ));
        }
        let mut sink = |_chunk: StreamChunk| {};
        if command.goto.is_empty() {
            let Some(updates) = command.update else {
                return Err(Error::validation(
                    "Command requires either `goto` or `update`",
                ));
            };
            return self
                .execute(
                    StartInput::Updates(updates.into_iter().collect()),
                    config,
                    &mut sink,
                )
                .await;
        }
        let mut values = Map::new();
        if let Some(updates) = command.update {
            apply_updates(
                &self.schema,
                &mut values,
                updates
                    .into_iter()
                    .map(|(channel, value)| ChannelUpdate {
                        channel,
                        value,
                        node: "<input>".to_string(),
                    })
                    .collect(),
            )?;
        }
        self.execute(
            StartInput::Gateways {
                values,
                nodes: command.goto,
            },
            config,
            &mut sink,
        )
        .await
    }

    /// Convenience state-only form of [`Self::run_command`].
    pub async fn invoke_command(&self, command: crate::Command, config: &RunConfig) -> Result<S> {
        self.run_command(command, config)
            .await
            .map(|outcome| outcome.state)
    }

    /// Run the graph to completion.
    pub async fn invoke(&self, input: &S) -> Result<S> {
        self.run(input, &RunConfig::new())
            .await
            .map(|outcome| outcome.state)
    }

    /// Run the graph and return state plus the consumption report.
    pub async fn run(&self, input: &S, config: &RunConfig) -> Result<RunOutcome<S>> {
        let values = state_to_values(input)?;
        let mut sink = |_chunk: StreamChunk| {};
        self.execute(StartInput::Values(values), config, &mut sink)
            .await
    }

    /// Run the graph from a partial state patch.
    pub async fn run_updates<I, K>(&self, updates: I, config: &RunConfig) -> Result<RunOutcome<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let values: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut sink = |_chunk: StreamChunk| {};
        self.execute(StartInput::Updates(values), config, &mut sink)
            .await
    }

    /// Run the graph and collect every stream chunk.
    pub async fn stream(&self, input: &S, config: &RunConfig) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        self.stream_with(input, config, |chunk| chunks.push(chunk))
            .await?;
        Ok(chunks)
    }

    /// Run the graph while pushing every chunk to `sink`.
    pub async fn stream_with<F>(&self, input: &S, config: &RunConfig, mut sink: F) -> Result<S>
    where
        F: FnMut(StreamChunk) + Send,
    {
        let values = state_to_values(input)?;
        self.execute(StartInput::Values(values), config, &mut sink)
            .await
            .map(|outcome| outcome.state)
    }

    /// Stream only the selected LangGraph stream modes.
    pub async fn stream_modes(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
    ) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        self.stream_modes_with(input, config, modes, |chunk| chunks.push(chunk))
            .await?;
        Ok(chunks)
    }

    /// Stream selected modes while forwarding matching chunks to `sink`.
    pub async fn stream_modes_with<F>(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
        mut sink: F,
    ) -> Result<S>
    where
        F: FnMut(StreamChunk) + Send,
    {
        let modes = modes.to_vec();
        let values = state_to_values(input)?;
        let mut filtered = |chunk: StreamChunk| {
            if chunk.matches_modes(&modes) {
                sink(chunk);
            }
        };
        self.execute(StartInput::Values(values), config, &mut filtered)
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume an interrupted thread.
    pub async fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume an interrupted thread inside a checkpoint namespace.
    /// Resume a namespace-bound thread while forwarding chunks to `sink`.
    pub(crate) async fn resume_in_streaming(
        &self,
        thread_id: &str,
        namespace: &str,
        value: Value,
        overrides: Option<&RunConfig>,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<S> {
        let mut config = RunConfig::new()
            .with_thread(thread_id)
            .with_checkpoint_ns(namespace);
        if let Some(overrides) = overrides {
            config.store = overrides.store.clone();
            config.context = overrides.context.clone();
            config.stream_subgraphs = true;
        }
        self.execute(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: None,
                resume: Some(value),
                updates: None,
            },
            &config,
            sink,
        )
        .await
        .map(|outcome| outcome.state)
    }

    pub async fn resume_in(&self, thread_id: &str, namespace: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, Some(namespace))
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume an interrupted thread after applying state patches.
    pub async fn resume_with_updates<I, K>(
        &self,
        thread_id: &str,
        value: Value,
        updates: I,
    ) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        self.resume_inner(thread_id, None, Some(value), Some(updates), None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Continue a thread paused by an `interrupt_before` / `interrupt_after` hook.
    pub async fn continue_thread(&self, thread_id: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Continue a namespace-bound thread paused by an interrupt hook.
    pub async fn continue_thread_in(&self, thread_id: &str, namespace: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, Some(namespace))
            .await
            .map(|outcome| outcome.state)
    }

    /// Read the latest checkpoint of a thread.
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
        let mut checkpoints = all.clone();
        if let Some(before) = before {
            if let Some(index) = checkpoints
                .iter()
                .position(|checkpoint| checkpoint.id == before)
            {
                checkpoints.drain(0..=index);
            }
        }
        if let Some(limit) = limit {
            checkpoints.truncate(limit);
        }
        checkpoints
            .into_iter()
            .map(|checkpoint| {
                let step = all
                    .iter()
                    .position(|item| item.id == checkpoint.id)
                    .map(|index| total - 2 - index as i64)
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
        let resolved = self.resolve_task_node(thread_id, "", task_id).await?;
        let as_node = as_node.map(str::to_owned).or(resolved);
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
            interrupted: false,
            paused_before: false,
            parent_id: Some(checkpoint.id),
            created_at_millis: now_millis(),
            interrupts: Vec::new(),
            metadata: Map::from_iter([(
                String::from("source"),
                Value::String(String::from("update")),
            )]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions: checkpoint.channel_versions.clone(),
            versions_seen: checkpoint.versions_seen.clone(),
            updated_channels,
            pending_writes: Vec::new(),
            deferred: Vec::new(),
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

    async fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        namespace: Option<&str>,
    ) -> Result<RunOutcome<S>> {
        let mut config = RunConfig::new().with_thread(thread_id);
        if let Some(namespace) = namespace {
            config = config.with_checkpoint_ns(namespace);
        }
        let mut sink = |_chunk: StreamChunk| {};
        self.execute(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: checkpoint_id.map(str::to_string),
                resume,
                updates,
            },
            &config,
            &mut sink,
        )
        .await
    }

    async fn execute(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<RunOutcome<S>> {
        if config.durability != crate::Durability::Async
            || !self.has_checkpointer()
            || config.thread_id.is_none()
        {
            return self.execute_inner(start, config, sink, None).await;
        }
        let (writer, mut receiver) = tokio::sync::mpsc::unbounded_channel::<Checkpoint>();
        let checkpointer = self.checkpointer.clone();
        let async_saver = self.async_checkpointer.clone();
        let handle = tokio::spawn(async move {
            while let Some(checkpoint) = receiver.recv().await {
                if let Some(saver) = &async_saver {
                    let _ = saver.put(checkpoint).await;
                } else if let Some(saver) = &checkpointer {
                    let _ = saver.put(checkpoint);
                }
            }
        });
        let result = self.execute_inner(start, config, sink, Some(&writer)).await;
        drop(writer);
        let _ = handle.await;
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_inner(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
        writer: Option<&tokio::sync::mpsc::UnboundedSender<Checkpoint>>,
    ) -> Result<RunOutcome<S>> {
        let thread_id = config.thread_id.clone();
        let namespace = config.checkpoint_ns.clone().unwrap_or_default();
        let collector = config
            .stream_subgraphs
            .then(|| Arc::new(std::sync::Mutex::new(Vec::new())));
        let run = AsyncRunInfo {
            thread_id,
            namespace,
            store: config.store.clone().or_else(|| self.store.clone()),
            context: config.context.clone(),
            collector,
        };
        let backend = AsyncBackend {
            graph: self,
            writer,
            run: &run,
        };
        let run_context = crate::kernel::context::run_context(
            run.thread_id.clone(),
            run.namespace.clone(),
            run.store.clone(),
            run.context.clone(),
        );
        let mut adapt = |chunk| sink(chunk);
        let future = crate::kernel::executor::pregel::run_loop(&backend, start, config, &mut adapt);
        crate::kernel::context::with_async_run(run_context, future).await
    }

    fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        crate::kernel::executor::scheduling::entry_tasks(&self.entry, state)
    }

    async fn run_task(
        &self,
        task: &Task,
        state: &S,
        step: usize,
        run: &AsyncRunInfo,
    ) -> TaskOutcome {
        let Some(node) = self.nodes.get(&task.node) else {
            return TaskOutcome::Failed(Box::new(TaskFailure {
                failure: NodeFailure::new(
                    &task.node,
                    format!("node `{}` is not registered in this graph", task.node),
                    1,
                ),
                usage: Vec::new(),
                duration_millis: 0,
                input: task.input.clone().map(Value::Object),
            }));
        };

        let input = match &task.input {
            Some(input) => match values_to_state::<S>(input) {
                Ok(state) => state,
                Err(error) => {
                    return TaskOutcome::Failed(Box::new(TaskFailure {
                        failure: NodeFailure::new(&task.node, error.to_string(), 1),
                        usage: Vec::new(),
                        duration_millis: 0,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
            },
            None => state.clone(),
        };

        let cache = self.cache.clone();
        let cache_policy = self.cache_policy(&task.node).cloned();
        let cache_key = match (&cache, &cache_policy) {
            (Some(_), Some(policy)) => state_to_values(&input)
                .ok()
                .map(|values| policy.cache_key(&task.node, &values)),
            _ => None,
        };
        if let (Some(cache), Some(key)) = (&cache, &cache_key) {
            match cache.get(key) {
                Ok(Some(cached)) => {
                    let usage = cached
                        .usage
                        .clone()
                        .map(|usage| vec![usage])
                        .unwrap_or_default();
                    return TaskOutcome::Ok(Box::new(TaskResult {
                        node: task.node.clone(),
                        output: crate::kernel::node::NodeOutput::from_cached(cached),
                        custom: Vec::new(),
                        messages: Vec::new(),
                        usage,
                        duration_millis: 0,
                        attempts: 0,
                        cached: true,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
                Ok(None) => {}
                Err(error) => {
                    return TaskOutcome::Failed(Box::new(TaskFailure {
                        failure: NodeFailure::new(
                            &task.node,
                            format!("node cache lookup failed: {error}"),
                            1,
                        ),
                        usage: Vec::new(),
                        duration_millis: 0,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
            }
        }

        let policy = self.retries.get(&task.node).cloned().unwrap_or_default();
        let max_attempts = policy.max_attempts.max(1);
        let started = Instant::now();
        let mut usage: Vec<Usage> = Vec::new();
        let mut last_error: Option<String> = None;
        let mut attempts_used = 0u32;

        for attempt in 1..=max_attempts {
            attempts_used = attempt as u32;
            let timeout_policy = self.timeout_of(&task.node).cloned().unwrap_or_default();
            let run_timeout = task
                .timeout_millis
                .map(Duration::from_millis)
                .or_else(|| timeout_policy.run_timeout());
            let context = AsyncContext::new(
                task.node.clone(),
                step,
                task.error.clone(),
                task.resume_history.clone(),
            )
            .with_run_timeout(run_timeout)
            .with_run_info(
                run.thread_id.clone(),
                run.namespace.clone(),
                run.store.clone(),
                run.context.clone(),
                run.collector.clone(),
            );
            let future: BoxNodeFuture = Box::pin(crate::kernel::context::with_async_node(
                task.node.clone(),
                Some(task.id.clone()),
                (node.as_ref())(input.clone(), context.clone()),
            ));
            let idle_timeout = timeout_policy.idle_timeout();
            let result = if let Some(timeout) = run_timeout {
                match tokio::time::timeout(timeout, future).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout {
                        node: task.node.clone(),
                        millis: timeout.as_millis() as u64,
                    }),
                }
            } else if let Some(idle) = idle_timeout {
                run_with_idle_timeout(future, idle, context.progress_counter(), task.node.clone())
                    .await
            } else {
                future.await
            };
            let (mut reported, custom, messages) = context.take();
            usage.append(&mut reported);
            match result {
                Ok(output) => {
                    usage.extend(output.usage().cloned());
                    if let (Some(cache), Some(cache_policy), Some(key)) =
                        (&cache, &cache_policy, &cache_key)
                    {
                        let mut combined: Option<Usage> = None;
                        for item in &usage {
                            combined.get_or_insert_with(Usage::new).merge(item);
                        }
                        let cached_output = CachedOutput {
                            updates: output.updates_map().clone(),
                            replaces: output.replaces_map().clone(),
                            goto: output.goto_targets().to_vec(),
                            sends: output.sends().to_vec(),
                            usage: combined,
                        };
                        if let Err(error) = cache.put(key, cached_output, cache_policy.ttl()) {
                            return TaskOutcome::Failed(Box::new(TaskFailure {
                                failure: NodeFailure::new(
                                    &task.node,
                                    format!("node cache store failed: {error}"),
                                    attempt as u32,
                                ),
                                usage,
                                duration_millis: started.elapsed().as_millis() as u64,
                                input: task.input.clone().map(Value::Object),
                            }));
                        }
                    }
                    return TaskOutcome::Ok(Box::new(TaskResult {
                        node: task.node.clone(),
                        output,
                        custom,
                        messages,
                        usage,
                        duration_millis: started.elapsed().as_millis() as u64,
                        attempts: attempt as u32,
                        cached: false,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
                Err(Error::Interrupt(interrupt)) => {
                    return TaskOutcome::Interrupted(Interrupt {
                        id: interrupt.id,
                        node: task.node.clone(),
                        value: interrupt.value,
                        response_schema: interrupt.response_schema,
                        task_id: Some(task.id.clone().into_boxed_str()),
                    });
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if !policy.should_retry(&error.to_string()) {
                        break;
                    }
                    if attempt < max_attempts {
                        let backoff = policy.backoff_for(attempt);
                        if !backoff.is_zero() {
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
        }

        TaskOutcome::Failed(Box::new(TaskFailure {
            failure: NodeFailure::new(
                &task.node,
                last_error.unwrap_or_else(|| "node failed without an error message".to_string()),
                attempts_used,
            ),
            usage,
            duration_millis: started.elapsed().as_millis() as u64,
            input: task.input.clone().map(Value::Object),
        }))
    }

    fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.conditional.get(node).map(Vec::as_slice)
    }
}

/// Await a node future, resetting the idle timer whenever the node reports progress.
async fn run_with_idle_timeout(
    mut future: BoxNodeFuture,
    idle: Duration,
    progress: Arc<std::sync::atomic::AtomicU64>,
    node: String,
) -> Result<crate::kernel::node::NodeOutput> {
    loop {
        let seen = progress.load(std::sync::atomic::Ordering::Relaxed);
        tokio::select! {
            result = &mut future => return result,
            _ = tokio::time::sleep(idle) => {
                if progress.load(std::sync::atomic::Ordering::Relaxed) == seen {
                    return Err(Error::Timeout {
                        node,
                        millis: idle.as_millis() as u64,
                    });
                }
            }
        }
    }
}

impl<S: State> AsyncCompiledGraph<S> {
    async fn saver_put(&self, checkpoint: Checkpoint) -> Result<()> {
        if let Some(saver) = &self.async_checkpointer {
            saver.put(checkpoint).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.put(checkpoint)
        } else {
            Err(Error::MissingCheckpointer("save_checkpoint"))
        }
    }

    async fn saver_get(&self, thread_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.get(thread_id, checkpoint_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.get(thread_id, checkpoint_id)
        } else {
            Err(Error::MissingCheckpointer("get"))
        }
    }

    async fn saver_latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.latest(thread_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.latest(thread_id)
        } else {
            Err(Error::MissingCheckpointer("latest"))
        }
    }

    async fn saver_latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> Result<Option<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.latest_in(thread_id, namespace).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.latest_in(thread_id, namespace)
        } else {
            Err(Error::MissingCheckpointer("latest_in"))
        }
    }

    async fn saver_history(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.history(thread_id).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.history(thread_id)
        } else {
            Err(Error::MissingCheckpointer("history"))
        }
    }

    async fn saver_history_in(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        if let Some(saver) = &self.async_checkpointer {
            saver.history_in(thread_id, namespace).await
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.history_in(thread_id, namespace)
        } else {
            Err(Error::MissingCheckpointer("history_in"))
        }
    }

    fn saver_history_in_sync(&self, thread_id: &str, namespace: &str) -> Result<Vec<Checkpoint>> {
        futures::executor::block_on(self.saver_history_in(thread_id, namespace))
    }

    fn saver_threads_sync(&self) -> Result<Vec<String>> {
        if let Some(saver) = &self.async_checkpointer {
            futures::executor::block_on(saver.threads())
        } else if let Some(checkpointer) = &self.checkpointer {
            checkpointer.threads()
        } else {
            Err(Error::MissingCheckpointer("threads"))
        }
    }

    fn saver_latest_in_sync(&self, thread_id: &str, namespace: &str) -> Result<Option<Checkpoint>> {
        futures::executor::block_on(self.saver_latest_in(thread_id, namespace))
    }

    fn saver_history_sync(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        futures::executor::block_on(self.saver_history(thread_id))
    }
}

/// Async backend: future joins and awaitable checkpoint I/O.
pub(crate) struct AsyncBackend<'a, S: State> {
    graph: &'a AsyncCompiledGraph<S>,
    writer: Option<&'a tokio::sync::mpsc::UnboundedSender<Checkpoint>>,
    run: &'a AsyncRunInfo,
}

impl<'a, S: State> crate::kernel::executor::backend::PregelBackend<S> for AsyncBackend<'a, S> {
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

    fn clear_ephemeral(&self, values: &mut serde_json::Map<String, Value>) {
        self.graph.schema.clear_ephemeral(values);
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
        let run = join_all(
            tasks
                .iter()
                .map(|task| self.graph.run_task(task, state, step, self.run)),
        );
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
            crate::kernel::executor::checkpointing::persist_checkpoint(
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

impl<'a, S: State> crate::kernel::executor::outcome::OutcomeGraph for AsyncBackend<'a, S> {
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
