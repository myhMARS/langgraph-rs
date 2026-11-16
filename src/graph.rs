//! Graph definition, compilation and the public run API.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::checkpoint::{Checkpoint, Checkpointer};
use crate::error::{Error, Result};
use crate::executor::{self, StartInput};
use crate::node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route, Task};
use crate::state::{State, StateSchema, state_to_values, values_to_state};
use crate::stream::StreamChunk;
use crate::{END, START};

/// Default number of supersteps a single graph run may execute.
pub const DEFAULT_RECURSION_LIMIT: usize = 25;

/// One or more edge targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Targets(Vec<String>);

impl Targets {
    /// Build a target list.
    pub fn new<I, T>(targets: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self(targets.into_iter().map(Into::into).collect())
    }

    /// Consume the target list.
    pub fn into_vec(self) -> Vec<String> {
        self.0
    }

    /// Targets as a slice.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }
}

impl From<&str> for Targets {
    fn from(target: &str) -> Self {
        Targets(vec![target.to_string()])
    }
}

impl From<String> for Targets {
    fn from(target: String) -> Self {
        Targets(vec![target])
    }
}

impl From<Vec<String>> for Targets {
    fn from(targets: Vec<String>) -> Self {
        Targets(targets)
    }
}

impl From<Vec<&str>> for Targets {
    fn from(targets: Vec<&str>) -> Self {
        Targets(targets.into_iter().map(str::to_string).collect())
    }
}

impl<const N: usize> From<[&str; N]> for Targets {
    fn from(targets: [&str; N]) -> Self {
        Targets(targets.into_iter().map(str::to_string).collect())
    }
}

/// Options used when compiling a graph.
///
/// ```
/// use std::sync::Arc;
/// use langgraph_rs::{CompileOptions, InMemorySaver};
///
/// let options = CompileOptions::new()
///     .with_checkpointer(Arc::new(InMemorySaver::new()))
///     .with_recursion_limit(10)
///     .interrupt_before(["human_review"]);
/// assert_eq!(options.recursion_limit, Some(10));
/// ```
#[derive(Clone, Default)]
pub struct CompileOptions {
    /// Enables checkpointing, thread history and human-in-the-loop resuming.
    pub checkpointer: Option<Arc<dyn Checkpointer>>,
    /// Overrides [`DEFAULT_RECURSION_LIMIT`].
    pub recursion_limit: Option<usize>,
    /// Nodes that must be interrupted *before* they run.
    pub interrupt_before: Vec<String>,
    /// Nodes that must be interrupted *after* they run.
    pub interrupt_after: Vec<String>,
}

impl fmt::Debug for CompileOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompileOptions")
            .field("checkpointer", &self.checkpointer.is_some())
            .field("recursion_limit", &self.recursion_limit)
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .finish()
    }
}

impl CompileOptions {
    /// Create default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a checkpointer.
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Override the superstep budget.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    /// Interrupt before the given nodes run.
    pub fn interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before = nodes.into_iter().map(Into::into).collect();
        self
    }

    /// Interrupt after the given nodes ran.
    pub fn interrupt_after<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_after = nodes.into_iter().map(Into::into).collect();
        self
    }
}

/// Per-run configuration.
#[derive(Clone, Debug, Default)]
pub struct RunConfig {
    /// Thread id used for checkpointing. Required when the graph has a checkpointer.
    pub thread_id: Option<String>,
    /// Overrides the compiled recursion limit for this run.
    pub recursion_limit: Option<usize>,
}

impl RunConfig {
    /// Create a config without a thread.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind the run to a checkpoint thread.
    pub fn with_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Override the superstep budget for this run.
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }
}

/// A graph that is ready to run.
///
/// Build one with [`StateGraph::compile`]. The graph is immutable afterwards and can be
/// shared between threads.
pub struct CompiledGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, NodeFn<S>>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) entry: Entry<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
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
            recursion_limit: self.recursion_limit,
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
        }
    }
}

pub(crate) enum Entry<S> {
    Static(Vec<String>),
    Conditional(ConditionalEdge<S>),
}

impl<S> Clone for Entry<S> {
    fn clone(&self) -> Self {
        match self {
            Entry::Static(nodes) => Entry::Static(nodes.clone()),
            Entry::Conditional(edge) => Entry::Conditional(edge.clone()),
        }
    }
}

/// Builder for a stateful graph.
///
/// ```
/// use langgraph_rs::{NodeOutput, State, StateGraph, StateSchema};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Clone, Default, Serialize, Deserialize)]
/// struct State1 {
///     value: i64,
/// }
///
/// impl State for State1 {
///     fn schema() -> StateSchema {
///         StateSchema::new().last_value("value")
///     }
/// }
///
/// let graph = StateGraph::<State1>::new()
///     .add_node("step", |state: &State1| Ok(NodeOutput::new().set("value", state.value + 1)))
///     .set_entry_point("step")
///     .set_finish_point("step")
///     .compile()
///     .unwrap();
/// let output = graph.invoke(&State1::default(), &Default::default()).unwrap();
/// assert_eq!(output.value, 1);
/// ```
pub struct StateGraph<S: State> {
    schema: StateSchema,
    nodes: IndexMap<String, NodeFn<S>>,
    duplicate_nodes: Vec<String>,
    edges: IndexMap<String, Vec<String>>,
    conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    entry: Vec<String>,
    conditional_entry: Option<ConditionalEdge<S>>,
}

impl<S: State> StateGraph<S> {
    /// Create a builder using the state's [`State::schema`].
    pub fn new() -> Self {
        Self::with_schema(S::schema())
    }

    /// Create a builder with an explicit channel schema.
    pub fn with_schema(schema: StateSchema) -> Self {
        Self {
            schema,
            nodes: IndexMap::new(),
            duplicate_nodes: Vec::new(),
            edges: IndexMap::new(),
            conditional: IndexMap::new(),
            entry: Vec::new(),
            conditional_entry: None,
        }
    }

    /// Register a node.
    pub fn add_node<F>(mut self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        if self.nodes.insert(name.clone(), Arc::new(node)).is_some() {
            self.duplicate_nodes.push(name);
        }
        self
    }

    /// Add a static edge.
    ///
    /// Use [`START`] as the source to set the entry point and [`END`] as a target to
    /// finish the graph.
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<Targets>) -> Self {
        let from = from.into();
        let targets = to.into().into_vec();
        if from == START {
            for target in targets {
                if !self.entry.contains(&target) {
                    self.entry.push(target);
                }
            }
        } else {
            self.edges.entry(from).or_default().extend(targets);
        }
        self
    }

    /// Add a conditional edge whose router returns node names directly.
    pub fn add_conditional_edges<F>(self, from: impl Into<String>, router: F) -> Self
    where
        F: Fn(&S) -> Result<Route> + Send + Sync + 'static,
    {
        self.add_conditional_edge(from, ConditionalEdge::new(router))
    }

    /// Add a conditional edge whose router returns symbolic keys resolved by `path_map`.
    ///
    /// ```
    /// use langgraph_rs::{PathMap, Route, State, StateGraph, StateSchema};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Default, Serialize, Deserialize)]
    /// struct S1 {
    ///     done: bool,
    /// }
    /// impl State for S1 {
    ///     fn schema() -> StateSchema {
    ///         StateSchema::new().last_value("done")
    ///     }
    /// }
    ///
    /// let graph = StateGraph::<S1>::new()
    ///     .add_node("work", |_state: &S1| Ok(Default::default()))
    ///     .add_node("finish", |_state: &S1| Ok(Default::default()))
    ///     .set_entry_point("work")
    ///     .add_conditional_edges_mapped(
    ///         "work",
    ///         |state: &S1| Ok(if state.done { Route::node("done") } else { Route::node("again") }),
    ///         [("done", "finish"), ("again", "work")],
    ///     )
    ///     .compile()
    ///     .unwrap();
    /// ```
    pub fn add_conditional_edges_mapped<F, M>(
        self,
        from: impl Into<String>,
        router: F,
        path_map: M,
    ) -> Self
    where
        F: Fn(&S) -> Result<Route> + Send + Sync + 'static,
        M: Into<PathMap>,
    {
        self.add_conditional_edge(from, ConditionalEdge::new(router).with_path_map(path_map))
    }

    fn add_conditional_edge(mut self, from: impl Into<String>, edge: ConditionalEdge<S>) -> Self {
        self.conditional.entry(from.into()).or_default().push(edge);
        self
    }

    /// Set the static entry point.
    pub fn set_entry_point(mut self, node: impl Into<String>) -> Self {
        let node = node.into();
        if !self.entry.contains(&node) {
            self.entry.push(node);
        }
        self
    }

    /// Use a router to decide the entry point at run time.
    pub fn set_conditional_entry_point<F>(mut self, router: F) -> Self
    where
        F: Fn(&S) -> Result<Route> + Send + Sync + 'static,
    {
        self.conditional_entry = Some(ConditionalEdge::new(router));
        self
    }

    /// Add an edge from `node` to [`END`].
    pub fn set_finish_point(self, node: impl Into<String>) -> Self {
        self.add_edge(node, END)
    }

    /// Validate the definition and compile it into a runnable graph.
    pub fn compile(self) -> Result<CompiledGraph<S>> {
        self.compile_with(CompileOptions::default())
    }

    /// Validate the definition and compile it with checkpointing / interrupt options.
    pub fn compile_with(self, options: CompileOptions) -> Result<CompiledGraph<S>> {
        self.validate()?;

        for node in options
            .interrupt_before
            .iter()
            .chain(options.interrupt_after.iter())
        {
            if !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "interrupt hook references unknown node `{node}`"
                )));
            }
        }

        let entry = match self.conditional_entry {
            Some(edge) => Entry::Conditional(edge),
            None => Entry::Static(self.entry.clone()),
        };

        Ok(CompiledGraph {
            schema: self.schema,
            nodes: self.nodes,
            edges: self.edges,
            conditional: self.conditional,
            entry,
            checkpointer: options.checkpointer,
            recursion_limit: options.recursion_limit.unwrap_or(DEFAULT_RECURSION_LIMIT),
            interrupt_before: options.interrupt_before.into_iter().collect(),
            interrupt_after: options.interrupt_after.into_iter().collect(),
        })
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("the graph has no nodes"));
        }
        if !self.duplicate_nodes.is_empty() {
            return Err(Error::validation(format!(
                "nodes registered more than once: {}",
                self.duplicate_nodes.join(", ")
            )));
        }
        for reserved in [START, END] {
            if self.nodes.contains_key(reserved) {
                return Err(Error::validation(format!(
                    "`{reserved}` is a reserved node name"
                )));
            }
        }

        if self.conditional_entry.is_some() && !self.entry.is_empty() {
            return Err(Error::validation(
                "a graph cannot combine a static entry point with a conditional entry point",
            ));
        }
        if self.conditional_entry.is_none() && self.entry.is_empty() {
            return Err(Error::validation(
                "no entry point: call set_entry_point(...) or add_edge(START, ...)",
            ));
        }

        for name in &self.entry {
            self.require_node("entry point", name)?;
        }

        for (source, targets) in &self.edges {
            self.require_node("edge source", source)?;
            for target in targets {
                self.require_target(source, target)?;
            }
        }

        for (source, edges) in &self.conditional {
            self.require_node("conditional edge source", source)?;
            for edge in edges {
                if let Some(path_map) = &edge.path_map {
                    for (key, targets) in path_map.entries() {
                        for target in targets {
                            if target == START {
                                return Err(Error::validation(format!(
                                    "conditional edge `{source}` maps key `{key}` to `{START}`, which is not a valid destination"
                                )));
                            }
                            self.require_target(source, target)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn require_node(&self, what: &str, name: &str) -> Result<()> {
        if name == START || name == END {
            return Err(Error::validation(format!(
                "`{name}` cannot be used as a {what}"
            )));
        }
        if !self.nodes.contains_key(name) {
            return Err(Error::validation(format!(
                "{what} `{name}` is not a registered node"
            )));
        }
        Ok(())
    }

    fn require_target(&self, source: &str, target: &str) -> Result<()> {
        if target == END {
            return Ok(());
        }
        if target == START {
            return Err(Error::validation(format!(
                "edge `{source}` targets `{START}`, which is not a valid destination"
            )));
        }
        if !self.nodes.contains_key(target) {
            return Err(Error::validation(format!(
                "edge `{source}` targets unknown node `{target}`"
            )));
        }
        Ok(())
    }
}

impl<S: State> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time view of a checkpointed thread.
#[derive(Clone, Debug)]
pub struct StateSnapshot<S: State> {
    /// Typed channel values.
    pub values: S,
    /// Nodes scheduled after this snapshot.
    pub next: Vec<String>,
    /// Checkpoint id.
    pub checkpoint_id: String,
    /// Thread id.
    pub thread_id: String,
    /// Supersteps completed.
    pub step: usize,
    /// `true` when the graph stopped because of an interrupt.
    pub interrupted: bool,
    /// Creation time in milliseconds since the Unix epoch.
    pub created_at_millis: u64,
}

impl<S: State> StateSnapshot<S> {
    fn from_checkpoint(checkpoint: Checkpoint) -> Result<Self> {
        Ok(Self {
            values: values_to_state::<S>(&checkpoint.values)?,
            next: checkpoint.next_nodes(),
            checkpoint_id: checkpoint.id,
            thread_id: checkpoint.thread_id,
            step: checkpoint.step,
            interrupted: checkpoint.interrupted,
            created_at_millis: checkpoint.created_at_millis,
        })
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
