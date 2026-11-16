//! The immutable, runnable graph.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::Entry;
use super::options::RunConfig;
use super::snapshot::StateSnapshot;
use crate::kernel::command::Command;
use crate::kernel::error::{Error, Result};
use crate::kernel::executor::{self, StartInput};
use crate::kernel::node::{ConditionalEdge, NodeFn, Task};
use crate::kernel::ports::{Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{State, StateSchema, state_to_values};
use crate::kernel::stream::StreamChunk;
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
    pub(crate) recursion_limit: usize,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
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
            recursion_limit: self.recursion_limit,
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
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

    /// Returns `true` when checkpointing is enabled.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some()
    }

    /// The configured checkpointer, if any.
    pub fn checkpointer(&self) -> Option<&Arc<dyn Checkpointer>> {
        self.checkpointer.as_ref()
    }

    /// Default superstep budget.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
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

    /// Continue an interrupted thread by handing `value` to the pending `interrupt()`.
    pub fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None)
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
        self.resume_inner(thread_id, None, Some(value), Some(updates))
    }

    /// Resume from a specific checkpoint (time travel).
    pub fn resume_from(&self, thread_id: &str, checkpoint_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, Some(checkpoint_id), Some(value), None)
    }

    /// Continue a thread that stopped on an `interrupt_before` / `interrupt_after` hook.
    pub fn continue_thread(&self, thread_id: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None)
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
    pub fn resume_command(&self, thread_id: &str, command: Command) -> Result<S> {
        let updates = command
            .update
            .map(|updates| updates.into_iter().collect::<Map<String, Value>>());
        self.resume_inner(thread_id, None, command.resume, updates)
    }

    fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
    ) -> Result<S> {
        let config = RunConfig::new().with_thread(thread_id);
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
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state"))?;
        match checkpointer.latest(thread_id)? {
            Some(checkpoint) => Ok(Some(StateSnapshot::from_checkpoint(checkpoint)?)),
            None => Ok(None),
        }
    }

    /// Read the full checkpoint history of a thread, newest first.
    pub fn get_state_history(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer()
            .ok_or(Error::MissingCheckpointer("get_state_history"))?;
        checkpointer
            .history(thread_id)?
            .into_iter()
            .map(StateSnapshot::from_checkpoint)
            .collect()
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
            Entry::Conditional(edge) => Ok(edge
                .resolve(state, START)?
                .into_iter()
                .filter(|node| node != END)
                .map(Task::new)
                .collect()),
        }
    }
}
