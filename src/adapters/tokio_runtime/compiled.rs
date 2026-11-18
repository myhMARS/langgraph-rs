//! Compiled async graph and its superstep loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::{AsyncNodeFn, BoxNodeFuture};
use crate::application::graph::{Entry, RunConfig, StateSnapshot};
use crate::application::pregel::StartInput;
use crate::kernel::cache::{CachePolicy, NodeCache};
use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::node::ConditionalEdge;
use crate::kernel::ports::{AsyncCheckpointer, Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{ChannelUpdate, State, StateSchema, apply_updates, state_to_values};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::{StreamChunk, StreamMode};
use crate::kernel::timeout::TimeoutPolicy;
use crate::kernel::usage::RunOutcome;

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
pub(crate) struct AsyncRunInfo {
    pub(crate) max_concurrency: Option<usize>,
    pub(crate) thread_id: Option<String>,
    pub(crate) namespace: String,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) context: Option<Value>,
    pub(crate) collector: Option<Arc<std::sync::Mutex<Vec<StreamChunk>>>>,
}

/// Async backend: future joins and awaitable checkpoint I/O.
pub(crate) struct AsyncBackend<'a, S: State> {
    pub(crate) graph: &'a AsyncCompiledGraph<S>,
    pub(crate) writer: Option<&'a tokio::sync::mpsc::UnboundedSender<Checkpoint>>,
    pub(crate) run: &'a AsyncRunInfo,
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
        crate::application::graph::visualize::render_mermaid(
            crate::application::graph::visualize::GraphLayout {
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
        crate::application::graph::visualize::render_dot(
            crate::application::graph::visualize::GraphLayout {
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

    /// Render the graph as a PNG image using Graphviz `dot`.
    pub fn draw_png(&self) -> Result<Vec<u8>> {
        crate::application::graph::visualize::dot_to_png(&self.to_dot())
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
        self.execute_quiet(
            StartInput::Gateways {
                values,
                nodes: command.goto,
                sends: command.sends,
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
        self.execute_quiet(StartInput::Values(values), config, &mut sink)
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
        self.execute_quiet(StartInput::Updates(values), config, &mut sink)
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

    /// Stream a resume/continue run for an existing thread.
    #[allow(clippy::too_many_arguments)]
    pub async fn stream_thread_with<F>(
        &self,
        thread_id: &str,
        resume: Option<Value>,
        updates: Option<serde_json::Map<String, Value>>,
        goto: Vec<String>,
        config: &RunConfig,
        modes: &[StreamMode],
        mut sink: F,
    ) -> Result<S>
    where
        F: FnMut(StreamChunk) + Send,
    {
        let requested = modes.to_vec();
        let mut config = config.clone();
        config.thread_id = Some(thread_id.to_string());
        let mut filtered = |chunk: StreamChunk| {
            if chunk.matches_modes(&requested) {
                sink(chunk);
            }
        };
        self.execute(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume,
                updates,
                goto,
                sends: Vec::new(),
            },
            &config,
            &mut filtered,
        )
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
        self.execute_quiet(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: None,
                resume: Some(value),
                updates: None,
                goto: Vec::new(),
                sends: Vec::new(),
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
    pub(crate) fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    pub(crate) fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
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

#[path = "execute.rs"]
mod execute;

#[path = "state_api.rs"]
mod state_api;
