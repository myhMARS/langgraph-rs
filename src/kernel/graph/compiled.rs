//! The immutable, runnable graph.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::Entry;
use super::options::RunConfig;
use super::snapshot::StateSnapshot;
use crate::kernel::cache::{CachePolicy, NodeCache};
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::command::Command;
use crate::kernel::error::{Error, Result};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe, send_task};
use crate::kernel::executor::{self, StartInput};
use crate::kernel::node::{ConditionalEdge, NodeFn, Task};
use crate::kernel::ports::{Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{
    ChannelUpdate, State, StateSchema, apply_updates, state_to_values, values_to_state,
};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::{StreamChunk, StreamMode};
use crate::kernel::timeout::TimeoutPolicy;
use crate::kernel::usage::RunOutcome;
use crate::{END, START};

/// A graph that is ready to run.
///
/// Build one with [`StateGraph::compile`](crate::StateGraph::compile). The graph is
/// immutable afterwards and can be shared between threads.
pub struct CompiledGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, NodeFn<S>>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) entry: Entry<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) trace_sinks: Vec<Arc<dyn TraceSink>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) node_metadata: indexmap::IndexMap<String, serde_json::Map<String, Value>>,
    pub(crate) timeouts: indexmap::IndexMap<String, TimeoutPolicy>,
    pub(crate) deferred: std::collections::HashSet<String>,
    pub(crate) destinations: indexmap::IndexMap<String, Vec<String>>,
    pub(crate) cache: Option<Arc<dyn NodeCache>>,
    pub(crate) cache_policies: IndexMap<String, CachePolicy>,
    pub(crate) recursion_limit: usize,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) name: Option<String>,
    pub(crate) context_schema: Option<Value>,
}

impl<S: State> Clone for CompiledGraph<S> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            conditional: self.conditional.clone(),
            entry: self.entry.clone(),
            checkpointer: self.checkpointer.clone(),
            trace_sinks: self.trace_sinks.clone(),
            retries: self.retries.clone(),
            error_edges: self.error_edges.clone(),
            store: self.store.clone(),
            node_metadata: self.node_metadata.clone(),
            timeouts: self.timeouts.clone(),
            deferred: self.deferred.clone(),
            destinations: self.destinations.clone(),
            cache: self.cache.clone(),
            cache_policies: self.cache_policies.clone(),
            recursion_limit: self.recursion_limit,
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
            name: self.name.clone(),
            context_schema: self.context_schema.clone(),
        }
    }
}

impl<S: State> CompiledGraph<S> {
    /// Channel schema of the graph.
    pub fn schema(&self) -> &StateSchema {
        &self.schema
    }

    /// Registered node names in insertion order.
    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(String::as_str)
    }

    /// Name given at compile time, when any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// JSON schema configured for `RunConfig::context`, if any.
    pub fn context_schema(&self) -> Option<&Value> {
        self.context_schema.as_ref()
    }

    /// Serializable graph description (nodes, static edges, conditional edges).
    pub fn graph_spec(&self) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .keys()
            .map(|node| {
                let mut spec = serde_json::Map::new();
                spec.insert("id".to_string(), Value::String(node.clone()));
                if self.deferred.contains(node) {
                    spec.insert("deferred".to_string(), Value::Bool(true));
                }
                if let Some(metadata) = self.node_metadata.get(node)
                    && !metadata.is_empty()
                {
                    spec.insert("metadata".to_string(), Value::Object(metadata.clone()));
                }
                Value::Object(spec)
            })
            .collect();
        let edges: Vec<Value> = self
            .edges
            .iter()
            .flat_map(|(source, targets)| {
                targets.iter().map(move |target| {
                    serde_json::json!({
                        "source": source,
                        "target": target,
                        "conditional": false,
                    })
                })
            })
            .collect();
        let conditional: Vec<Value> = self
            .conditional
            .keys()
            .map(|source| serde_json::json!({ "source": source, "conditional": true }))
            .collect();
        serde_json::json!({
            "name": self.name,
            "nodes": nodes,
            "edges": edges,
            "conditional_edges": conditional,
        })
    }

    /// Returns `true` when checkpointing is enabled.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some()
    }

    /// The configured checkpointer, if any.
    pub fn checkpointer(&self) -> Option<&Arc<dyn Checkpointer>> {
        self.checkpointer.as_ref()
    }

    /// Export the latest checkpoint of a thread as LangGraph-shaped JSON.
    pub fn export_checkpoint(&self, thread_id: &str) -> Result<Option<Value>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("export_checkpoint"))?;
        Ok(checkpointer
            .latest(thread_id)?
            .map(|checkpoint| crate::interop::checkpoint_to_langgraph(&checkpoint)))
    }

    /// Import a LangGraph-shaped checkpoint document as a checkpoint of `thread_id`.
    pub fn import_checkpoint(&self, thread_id: &str, document: &Value) -> Result<()> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("import_checkpoint"))?;
        let mut checkpoint = crate::interop::checkpoint_from_langgraph(document)?;
        checkpoint.thread_id = thread_id.to_string();
        checkpointer.put(checkpoint)
    }

    /// Default superstep budget.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Render the graph as a Mermaid `graph TD` diagram.
    ///
    /// Static edges are solid arrows, conditional edges are dashed arrows through a
    /// router node (with path-map labels), error edges are labelled `on error`, and node
    /// labels include retry policies and interrupt hooks.
    ///
    /// ```
    /// use langgraph_rs::{NodeOutput, State, StateGraph, StateSchema};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Default, Serialize, Deserialize)]
    /// struct Demo {
    ///     value: i64,
    /// }
    ///
    /// impl State for Demo {
    ///     fn schema() -> StateSchema {
    ///         StateSchema::new().last_value("value")
    ///     }
    /// }
    ///
    /// let graph = StateGraph::<Demo>::new()
    ///     .add_node("work", |_state: &Demo| Ok(NodeOutput::new()))
    ///     .set_entry_point("work")
    ///     .set_finish_point("work")
    ///     .compile()
    ///     .unwrap();
    /// let mermaid = graph.to_mermaid();
    /// assert!(mermaid.contains("graph TD"));
    /// assert!(mermaid.contains("START --> n_work"));
    /// ```
    /// Render the graph as a compact ASCII map.
    pub fn to_ascii(&self) -> String {
        crate::kernel::graph::visualize::render_ascii(
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

    /// Run the graph to completion and return the final typed state.
    ///
    /// `input` is the initial state of a fresh run. Use
    /// [`CompiledGraph::invoke_updates`] when only a state patch is available.
    pub fn invoke(&self, input: &S, config: &RunConfig) -> Result<S> {
        let values = state_to_values(input)?;
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(self, StartInput::Values(values), config, &mut sink)
            .map(|outcome| outcome.state)
    }

    /// Run the graph from a partial state patch applied over the default state.
    ///
    /// This is the closest equivalent of Python's `graph.invoke({"key": value})`.
    pub fn invoke_updates<I, K>(&self, updates: I, config: &RunConfig) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let values: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(self, StartInput::Updates(values), config, &mut sink)
            .map(|outcome| outcome.state)
    }

    /// Run the graph and return the final state together with a consumption report.
    ///
    /// The report contains one [`NodeExecution`](crate::NodeExecution) per node run
    /// (wall-clock duration plus any usage the node reported through
    /// [`record_usage`](crate::record_usage) or
    /// [`NodeOutput::with_usage`](crate::NodeOutput::with_usage)).
    pub fn run(&self, input: &S, config: &RunConfig) -> Result<RunOutcome<S>> {
        let values = state_to_values(input)?;
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(self, StartInput::Values(values), config, &mut sink)
    }

    /// Run the graph from a partial state patch and return state plus report.
    pub fn run_updates<I, K>(&self, updates: I, config: &RunConfig) -> Result<RunOutcome<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let values: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(self, StartInput::Updates(values), config, &mut sink)
    }

    /// Run the graph and return its final state while streaming every chunk.
    pub fn stream(&self, input: &S, config: &RunConfig) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        self.stream_with(input, config, |chunk| chunks.push(chunk))?;
        Ok(chunks)
    }

    /// Run the graph while handing every [`StreamChunk`] to `sink`.
    pub fn stream_with<F>(&self, input: &S, config: &RunConfig, mut sink: F) -> Result<S>
    where
        F: FnMut(StreamChunk),
    {
        let values = state_to_values(input)?;
        executor::execute(self, StartInput::Values(values), config, &mut sink)
            .map(|outcome| outcome.state)
    }

    /// Invoke the graph over several inputs, respecting `RunConfig::max_concurrency`.
    ///
    /// Results are returned in input order; a failed invocation does not abort the batch.
    pub fn batch<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<S>>
    where
        I: IntoIterator<Item = S>,
    {
        let inputs: Vec<S> = inputs.into_iter().collect();
        let max_concurrency = config.max_concurrency.unwrap_or(1).max(1);
        if max_concurrency <= 1 || inputs.len() <= 1 {
            return inputs
                .into_iter()
                .map(|input| self.invoke(&input, config))
                .collect();
        }
        let mut results: Vec<Option<Result<S>>> = (0..inputs.len()).map(|_| None).collect();
        let chunks: Vec<Vec<(usize, S)>> = inputs
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .chunks(max_concurrency)
            .map(|chunk| chunk.to_vec())
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(|| {
                        chunk
                            .into_iter()
                            .map(|(index, input)| (index, self.invoke(&input, config)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for handle in handles {
                for (index, result) in handle.join().expect("batch worker panicked") {
                    results[index] = Some(result);
                }
            }
        });
        results
            .into_iter()
            .map(|result| result.expect("every batch entry produced a result"))
            .collect()
    }

    /// Batch variant of [`CompiledGraph::run`] that also returns run reports.
    pub fn batch_run<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<RunOutcome<S>>>
    where
        I: IntoIterator<Item = S>,
    {
        inputs
            .into_iter()
            .map(|input| self.run(&input, config))
            .collect()
    }

    /// Stream only the selected LangGraph stream modes.
    pub fn stream_modes(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
    ) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        self.stream_modes_with(input, config, modes, |chunk| chunks.push(chunk))?;
        Ok(chunks)
    }

    /// Stream selected modes while forwarding matching chunks to `sink`.
    ///
    /// `debug` includes values, updates, tasks and checkpoint records; usage and
    /// interrupt chunks are always forwarded.
    pub fn stream_modes_with<F>(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
        mut sink: F,
    ) -> Result<S>
    where
        F: FnMut(StreamChunk),
    {
        let modes = modes.to_vec();
        let values = state_to_values(input)?;
        let mut filtered = |chunk: StreamChunk| {
            if chunk.matches_modes(&modes) {
                sink(chunk);
            }
        };
        executor::execute(self, StartInput::Values(values), config, &mut filtered)
            .map(|outcome| outcome.state)
    }

    /// Continue an interrupted thread by handing `value` to the pending `interrupt()`.
    pub fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, None, None)
    }

    /// Resume an interrupted thread bound to a checkpoint namespace.
    pub fn resume_in(&self, thread_id: &str, namespace: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, Some(namespace), None)
    }

    /// Resume an interrupted thread and apply state writes (e.g. human edits) first.
    pub fn resume_with_updates<I, K>(&self, thread_id: &str, value: Value, updates: I) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        self.resume_inner(thread_id, None, Some(value), Some(updates), None, None)
    }

    /// Resume an interrupted namespace-bound thread after applying state writes.
    pub fn resume_with_updates_in<I, K>(
        &self,
        thread_id: &str,
        namespace: &str,
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
        self.resume_inner(
            thread_id,
            None,
            Some(value),
            Some(updates),
            Some(namespace),
            None,
        )
    }

    /// Resume from a specific checkpoint (time travel).
    pub fn resume_from(&self, thread_id: &str, checkpoint_id: &str, value: Value) -> Result<S> {
        self.resume_inner(
            thread_id,
            Some(checkpoint_id),
            Some(value),
            None,
            None,
            None,
        )
    }

    /// Continue a thread that stopped on an `interrupt_before` / `interrupt_after` hook.
    pub fn continue_thread(&self, thread_id: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, None, None)
    }

    /// Continue a namespace-bound thread paused by an interrupt hook.
    pub fn continue_thread_in(&self, thread_id: &str, namespace: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, Some(namespace), None)
    }

    /// Resume a namespace-bound thread while forwarding chunks to `sink`.
    pub(crate) fn resume_in_streaming(
        &self,
        thread_id: &str,
        namespace: &str,
        value: Value,
        overrides: Option<&RunConfig>,
        sink: &mut dyn FnMut(StreamChunk),
    ) -> Result<S> {
        let mut config = RunConfig::new()
            .with_thread(thread_id)
            .with_checkpoint_ns(namespace);
        if let Some(overrides) = overrides {
            config.store = overrides.store.clone();
            config.context = overrides.context.clone();
            config.stream_subgraphs = true;
        }
        executor::execute(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: None,
                resume: Some(value),
                updates: None,
            },
            &config,
            sink,
        )
        .map(|outcome| outcome.state)
    }

    /// Resume a namespace-bound thread, inheriting store/context from `overrides`.
    pub fn resume_in_with_run_config(
        &self,
        thread_id: &str,
        namespace: &str,
        value: Value,
        overrides: &RunConfig,
    ) -> Result<S> {
        self.resume_inner(
            thread_id,
            None,
            Some(value),
            None,
            Some(namespace),
            Some(overrides),
        )
    }

    /// Resume a checkpointed thread with an explicit [`Command`].
    ///
    /// This is the builder-friendly form of
    /// [`CompiledGraph::resume_with_updates`] and
    /// [`CompiledGraph::continue_thread`]:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use langgraph_rs::{
    /// #     Command, CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph,
    /// #     StateSchema,
    /// # };
    /// # use serde::{Deserialize, Serialize};
    /// #
    /// # #[derive(Clone, Default, Debug, Serialize, Deserialize)]
    /// # struct Demo {
    /// #     value: i64,
    /// # }
    /// #
    /// # impl State for Demo {
    /// #     fn schema() -> StateSchema {
    /// #         StateSchema::new().last_value("value")
    /// #     }
    /// # }
    /// let graph = StateGraph::<Demo>::new()
    ///     .add_node("ask", |_state: &Demo| {
    ///         let answer = langgraph_rs::interrupt(serde_json::json!({ "question": "?" }))?;
    ///         Ok(NodeOutput::new().set("value", answer.as_i64().unwrap_or_default()))
    ///     })
    ///     .set_entry_point("ask")
    ///     .set_finish_point("ask")
    ///     .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
    ///     .unwrap();
    ///
    /// let config = RunConfig::new().with_thread("demo");
    /// let _ = graph.invoke(&Demo::default(), &config).unwrap_err();
    ///
    /// let command = Command::resume(serde_json::json!(41));
    /// let state = graph.resume_command("demo", command).unwrap();
    /// assert_eq!(state.value, 41);
    /// ```
    /// Run a graph starting at the `Command`'s `goto` nodes (LangGraph
    /// `invoke(Command(goto=...))`), applying `Command.update` first.
    pub fn run_command(
        &self,
        command: Command,
        config: &RunConfig,
    ) -> Result<crate::RunOutcome<S>> {
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
        if command.goto.is_empty() {
            return Err(Error::validation(
                "Command(goto=...) requires at least one target node",
            ));
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
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(
            self,
            executor::StartInput::Gateways {
                values,
                nodes: command.goto,
            },
            config,
            &mut sink,
        )
    }

    /// Convenience state-only form of [`Self::run_command`].
    pub fn invoke_command(&self, command: Command, config: &RunConfig) -> Result<S> {
        self.run_command(command, config)
            .map(|outcome| outcome.state)
    }

    pub fn resume_command(&self, thread_id: &str, command: Command) -> Result<S> {
        let updates = command
            .update
            .map(|updates| updates.into_iter().collect::<Map<String, Value>>());
        self.resume_inner(thread_id, None, command.resume, updates, None, None)
    }

    fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        namespace: Option<&str>,
        overrides: Option<&RunConfig>,
    ) -> Result<S> {
        let mut config = RunConfig::new().with_thread(thread_id);
        if let Some(namespace) = namespace {
            config = config.with_checkpoint_ns(namespace);
        }
        if let Some(overrides) = overrides {
            config.store = overrides.store.clone();
            config.context = overrides.context.clone();
        }
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: checkpoint_id.map(str::to_string),
                resume,
                updates,
            },
            &config,
            &mut sink,
        )
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
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state"))?;
        match checkpointer.latest_in(thread_id, namespace)? {
            Some(checkpoint) => Ok(Some(StateSnapshot::from_checkpoint(checkpoint)?)),
            None => Ok(None),
        }
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
        let mut checkpoints = checkpointer.history_in(thread_id, namespace)?;
        if let Some(before) = before {
            if let Some(index) = checkpoints.iter().position(|cp| cp.id == before) {
                checkpoints.drain(0..=index);
            }
        }
        if let Some(limit) = limit {
            checkpoints.truncate(limit);
        }
        checkpoints
            .into_iter()
            .map(StateSnapshot::from_checkpoint)
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
        self.update_state_in(thread_id, "", updates, as_node.as_deref())
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
            metadata: Map::from_iter([("source".to_string(), Value::String("update".to_string()))]),
            namespace: namespace.to_string(),
            task_results: checkpoint.task_results.clone(),
            channel_versions: checkpoint.channel_versions.clone(),
            versions_seen: checkpoint.versions_seen.clone(),
            updated_channels,
        };
        let snapshot = StateSnapshot::from_checkpoint(updated.clone())?;
        checkpointer.put(updated)?;
        Ok(snapshot)
    }

    pub(crate) fn node(&self, name: &str) -> Option<&NodeFn<S>> {
        self.nodes.get(name)
    }

    pub(crate) fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    pub(crate) fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.conditional.get(node).map(Vec::as_slice)
    }

    /// Trace sink adapters installed through the kernel.
    pub(crate) fn trace_sinks(&self) -> &[Arc<dyn TraceSink>] {
        &self.trace_sinks
    }

    /// Retry policy of a node, if one was configured.
    pub fn retry_policy(&self, node: &str) -> Option<&RetryPolicy> {
        self.retries.get(node)
    }

    /// Error-handler node for a node, if one was configured.
    pub fn error_handler(&self, node: &str) -> Option<&String> {
        self.error_edges.get(node)
    }

    /// List threads that have at least one checkpoint.
    pub fn threads(&self) -> Result<Vec<String>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("threads"))?;
        checkpointer.threads()
    }

    /// Keep only the newest `keep_last` checkpoints of a thread.
    pub fn prune_thread(&self, thread_id: &str, keep_last: usize) -> Result<()> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(crate::Error::MissingCheckpointer("prune_thread"))?;
        checkpointer.prune_thread(thread_id, keep_last)
    }

    /// Delete every checkpoint of a thread.
    pub fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("delete_thread"))?;
        checkpointer.delete_thread(thread_id)
    }

    /// Copy every checkpoint of one thread to another thread id.
    pub fn copy_thread(&self, source_thread: &str, target_thread: &str) -> Result<()> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("copy_thread"))?;
        checkpointer.copy_thread(source_thread, target_thread)
    }

    /// Allowed destinations of a node, when an allow-list is configured.
    pub fn destinations_of(&self, node: &str) -> Option<&[String]> {
        self.destinations.get(node).map(Vec::as_slice)
    }

    /// Whether the node is deferred until the graph has no regular work left.
    pub fn is_deferred(&self, node: &str) -> bool {
        self.deferred.contains(node)
    }

    /// Timeout policy of one node, when configured.
    pub fn timeout_of(&self, node: &str) -> Option<&TimeoutPolicy> {
        self.timeouts.get(node)
    }

    /// Metadata attached to nodes (owner, tags, ...).
    pub fn node_metadata(&self) -> &indexmap::IndexMap<String, serde_json::Map<String, Value>> {
        &self.node_metadata
    }

    /// JSON input schema attached to a node, when present.
    pub fn node_input_schema(&self, node: &str) -> Option<&Value> {
        self.node_metadata
            .get(node)
            .and_then(|metadata| metadata.get("input_schema"))
    }

    /// Metadata of one node.
    pub fn node_metadata_for(&self, node: &str) -> Option<&serde_json::Map<String, Value>> {
        self.node_metadata.get(node)
    }

    /// Long-term memory store, when the graph was compiled with one.
    pub(crate) fn store(&self) -> Option<&Arc<dyn BaseStore>> {
        self.store.as_ref()
    }

    /// Cache backend, when node caching is enabled.
    pub(crate) fn cache(&self) -> Option<&Arc<dyn NodeCache>> {
        self.cache.as_ref()
    }

    /// Cache policy of a node, if caching is enabled for it.
    pub(crate) fn cache_policy(&self, node: &str) -> Option<&CachePolicy> {
        self.cache_policies.get(node)
    }

    pub(crate) fn interrupts_before(&self, node: &str) -> bool {
        self.interrupt_before.contains(node)
    }

    pub(crate) fn interrupts_after(&self, node: &str) -> bool {
        self.interrupt_after.contains(node)
    }

    pub(crate) fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        match &self.entry {
            Entry::Static(nodes) => Ok(nodes
                .iter()
                .filter(|node| node.as_str() != END)
                .map(Task::new)
                .collect()),
            Entry::Conditional(edge) => {
                let (targets, sends) = edge.resolve(state, START)?;
                let mut tasks: Vec<Task> = targets
                    .into_iter()
                    .filter(|node| node != END)
                    .map(Task::new)
                    .collect();
                for send in sends {
                    tasks.push(send_task(START, send)?);
                }
                Ok(tasks)
            }
        }
    }
}
