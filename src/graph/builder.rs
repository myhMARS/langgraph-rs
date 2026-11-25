//! The mutable graph builder.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;

use super::Entry;
use super::compiled::CompiledGraph;
use super::options::{CompileOptions, Targets};
use crate::_internal::cache::{CachePolicy, NodeCache};
use crate::_internal::command::Command;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route};
use crate::_internal::retry::RetryPolicy;
use crate::_internal::state::state_to_values;
use crate::_internal::state::{State, StateSchema};
use crate::_internal::stream::StreamChunk;
use crate::_internal::timeout::TimeoutPolicy;
use crate::store::BaseStore;
use crate::{END, START};
use serde_json::{Map, Value};

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
#[derive(Clone)]

pub struct StateGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, NodeFn<S>>,
    pub(crate) duplicate_nodes: Vec<String>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) cache: Option<Arc<dyn NodeCache>>,
    pub(crate) cache_policies: IndexMap<String, CachePolicy>,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) node_metadata: IndexMap<String, Map<String, Value>>,
    pub(crate) timeouts: IndexMap<String, TimeoutPolicy>,
    pub(crate) deferred: HashSet<String>,
    pub(crate) destinations: IndexMap<String, Vec<String>>,
    pub(crate) entry: Vec<String>,
    pub(crate) conditional_entry: Option<ConditionalEdge<S>>,
    /// Hooks that propagate the parent's saver into subgraph children at compile time.
    pub(crate) subgraph_hooks: crate::_internal::ports::SaverHooks,
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
            retries: IndexMap::new(),
            error_edges: IndexMap::new(),
            cache: None,
            cache_policies: IndexMap::new(),
            store: None,
            node_metadata: IndexMap::new(),
            timeouts: IndexMap::new(),
            deferred: HashSet::new(),
            destinations: IndexMap::new(),
            entry: Vec::new(),
            conditional_entry: None,
            subgraph_hooks: Vec::new(),
        }
    }

    /// Register a node that returns a LangGraph-style [`Command`].
    pub fn add_command_node<F>(self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(&S) -> Result<Command> + Send + Sync + 'static,
    {
        self.add_node(name, move |state: &S| Ok(node(state)?.into()))
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

    /// Register a node together with arbitrary metadata (owner, tags, ...).
    pub fn add_node_with_metadata<F>(
        self,
        name: impl Into<String>,
        metadata: Map<String, Value>,
        node: F,
    ) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        self.set_node_metadata(name.clone(), metadata)
            .add_node(name, node)
    }

    /// Attach a timeout policy to a node name (`TimeoutPolicy`).
    pub fn set_timeout(mut self, name: impl Into<String>, policy: TimeoutPolicy) -> Self {
        self.timeouts.insert(name.into(), policy);
        self
    }

    /// Restrict the nodes an edge from `name` may target (LangGraph node `destinations`).
    pub fn set_destinations<I, T>(mut self, name: impl Into<String>, destinations: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.destinations.insert(
            name.into(),
            destinations.into_iter().map(Into::into).collect(),
        );
        self
    }

    /// Register a node together with an explicit destination allow-list.
    pub fn add_node_with_destinations<F, I, T>(
        self,
        name: impl Into<String>,
        destinations: I,
        node: F,
    ) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let name = name.into();
        self.set_destinations(name.clone(), destinations)
            .add_node(name, node)
    }

    /// Register a node that runs after every regular node finished (LangGraph `deferred=True`).
    pub fn add_node_deferred<F>(self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        self.set_deferred(name.clone()).add_node(name, node)
    }

    /// Mark an existing node name as deferred.
    pub fn set_deferred(mut self, name: impl Into<String>) -> Self {
        self.deferred.insert(name.into());
        self
    }

    /// Register a node together with a timeout policy.
    pub fn add_node_with_timeout<F>(
        self,
        name: impl Into<String>,
        policy: TimeoutPolicy,
        node: F,
    ) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        self.set_timeout(name.clone(), policy).add_node(name, node)
    }

    /// Attach a JSON input schema to a node (stored as metadata, not enforced).
    pub fn set_node_input_schema(mut self, name: impl Into<String>, schema: Value) -> Self {
        self.node_metadata
            .entry(name.into())
            .or_default()
            .insert("input_schema".to_string(), schema);
        self
    }

    /// Register a node together with a JSON input schema (stored as metadata).
    pub fn add_node_with_input_schema<F>(
        self,
        name: impl Into<String>,
        schema: Value,
        node: F,
    ) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        self.set_node_input_schema(name.clone(), schema)
            .add_node(name, node)
    }

    /// Attach metadata to a node name (LangGraph node `metadata`).
    pub fn set_node_metadata(
        mut self,
        name: impl Into<String>,
        metadata: Map<String, Value>,
    ) -> Self {
        self.node_metadata.insert(name.into(), metadata);
        self
    }

    /// Register a node with a retry policy.
    pub fn add_node_with_retry<F>(
        mut self,
        name: impl Into<String>,
        policy: RetryPolicy,
        node: F,
    ) -> Self
    where
        F: Fn(&S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        self.retries.insert(name.clone(), policy);
        self.add_node(name, node)
    }

    /// Attach a retry policy to an already registered node.
    pub fn set_retry(mut self, node: impl Into<String>, policy: RetryPolicy) -> Self {
        self.retries.insert(node.into(), policy);
        self
    }

    /// Attach a long-term memory store (LangGraph `BaseStore`).
    pub fn with_store(mut self, store: Arc<dyn BaseStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach a node cache backend.
    pub fn with_cache(mut self, cache: Arc<dyn NodeCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Enable caching for one node.
    pub fn set_cache(mut self, node: impl Into<String>, policy: CachePolicy) -> Self {
        self.cache_policies.insert(node.into(), policy);
        self
    }

    /// Route permanent failures of `from` to the `handler` node.
    ///
    /// The handler runs in the next superstep and can inspect the failure with
    /// [`last_error`](crate::last_error); without an error edge a node failure aborts
    /// the run.
    pub fn add_error_edge(mut self, from: impl Into<String>, handler: impl Into<String>) -> Self {
        self.error_edges.insert(from.into(), handler.into());
        self
    }

    /// Embed another compiled graph as a node.
    ///
    /// The subgraph shares this graph's state type and runs without its own
    /// checkpointer: its final values are published as direct channel replacements for
    /// every channel it changed. Nested interrupts are not supported; an interrupt
    /// raised inside a subgraph surfaces as a node failure.
    pub fn add_subgraph(
        mut self,
        name: impl Into<String>,
        subgraph: Arc<CompiledGraph<S>>,
    ) -> Self {
        let name = name.into();
        let schema = self.schema.clone();
        // The parent hands its saver to the child at compile time (LangGraph inheritance).
        let child = subgraph.clone();
        self.subgraph_hooks
            .push(Arc::new(move |saver| child.set_inherited_saver(saver.clone())));
        self.add_node(name.clone(), move |state: &S| {
            let before = state_to_values(state)?;
            crate::runtime::context::clear_parent_command();
            let output = run_subgraph(&subgraph, state, &name)?;
            if let Some(command) = crate::runtime::context::take_parent_command() {
                return super::parent_command_output(command, &schema);
            }
            let after = state_to_values(&output)?;
            let mut result = NodeOutput::new();
            for (channel, value) in after {
                if before.get(&channel) != Some(&value) {
                    result = result.replace(channel, value);
                }
            }
            Ok(result)
        })
    }

    /// Embed a subgraph with a **different state type**, mapping the parent state into
    /// the subgraph input and the subgraph output back into parent writes.
    ///
    /// The subgraph shares the parent thread id and runs in a child checkpoint namespace
    /// (`<parent_ns>|<name>`), so an `interrupt()` inside the subgraph pauses the parent
    /// run and a parent resume continues the subgraph from its checkpoint.
    pub fn add_subgraph_mapped<Sub, Input, Output>(
        mut self,
        name: impl Into<String>,
        subgraph: Arc<CompiledGraph<Sub>>,
        input: Input,
        output: Output,
    ) -> Self
    where
        Sub: State,
        Input: Fn(&S) -> Result<Sub> + Send + Sync + 'static,
        Output: Fn(&Sub, &S) -> Result<NodeOutput> + Send + Sync + 'static,
    {
        let name = name.into();
        let schema = self.schema.clone();
        let child = subgraph.clone();
        self.subgraph_hooks
            .push(Arc::new(move |saver| child.set_inherited_saver(saver.clone())));
        self.add_node(name.clone(), move |state: &S| {
            let sub_state = input(state)?;
            crate::runtime::context::clear_parent_command();
            let sub_output = run_subgraph(&subgraph, &sub_state, &name)?;
            if let Some(command) = crate::runtime::context::take_parent_command() {
                return super::parent_command_output(command, &schema);
            }
            output(&sub_output, state)
        })
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

    /// Add nodes as a sequence, chaining static edges and setting the entry/finish points.
    ///
    /// Nodes are passed as `(name, node_fn)` pairs; use `Arc::new(closure)` for the node
    /// function when the sequence contains different closures (`NodeFn` is a type alias).
    pub fn add_sequence<I, N>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = (N, crate::_internal::node::NodeFn<S>)>,
        N: Into<String>,
    {
        let mut previous: Option<String> = None;
        for (name, node) in nodes {
            let name = name.into();
            let node = node.clone();
            self = self.add_node(name.clone(), move |state: &S| node(state));
            match previous.take() {
                Some(previous) => self = self.add_edge(previous, name.clone()),
                None => self = self.set_entry_point(name.clone()),
            }
            previous = Some(name);
        }
        if let Some(last) = previous {
            self = self.set_finish_point(last);
        }
        self
    }

    /// Add static edges from several sources to one target.
    pub fn add_edges_from<I, N>(mut self, sources: I, to: impl Into<String>) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        let target = to.into();
        for source in sources {
            self = self.add_edge(source.into(), target.clone());
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
        let from = from.into();
        if from == START {
            self.conditional_entry = Some(edge);
        } else {
            self.conditional.entry(from).or_default().push(edge);
        }
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
}

impl<S: State> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Child checkpoint namespace of a subgraph node.
fn child_namespace(parent: &str, name: &str) -> String {
    let task_id = crate::runtime::context::current_task_id();
    super::subgraph_namespace(parent, name, task_id.as_deref())
}

/// Run a subgraph with parent thread/namespace propagation and resume support.
fn run_subgraph<Sub: State>(subgraph: &CompiledGraph<Sub>, input: &Sub, name: &str) -> Result<Sub> {
    let thread = crate::runtime::context::current_thread_id();
    let namespace = child_namespace(&crate::runtime::context::current_namespace(), name);
    // Only resume the child when it actually has a paused checkpoint in its namespace;
    // otherwise a parent resume value would make a sibling subgraph reload stale state.
    // The child namespace carries a task id (`sub:<task id>`) that changes on every parent
    // resume, so look the paused child up under the task-id-free prefix instead.
    let base = namespace
        .rsplit_once(':')
        .map(|(head, _)| head)
        .unwrap_or(&namespace);
    let paused_in = thread
        .as_deref()
        .and_then(|thread| subgraph.paused_namespace_in(thread, base));
    let resume_namespace = paused_in.clone().unwrap_or_else(|| namespace.clone());
    let resume = match thread.as_deref() {
        Some(_) if paused_in.is_some() => crate::runtime::context::current_resume_last()
            .or_else(crate::runtime::context::current_pending_resume),
        _ => None,
    };

    let mut config = super::options::RunConfig::new().with_checkpoint_ns(namespace.clone());
    if let Some(store) = crate::runtime::context::get_store() {
        config = config.with_store(store);
    }
    if let Some(context) = crate::runtime::context::current_context() {
        config = config.with_context(context);
    }
    let collector = crate::runtime::context::current_subgraph_collector();
    if let Some(collector) = collector {
        config = config.with_stream_subgraphs(true);
        let chunk_namespace = namespace.clone();
        let mut sink = move |chunk: StreamChunk| {
            let wrapped = match chunk {
                StreamChunk::Subgraph {
                    namespace: inner,
                    chunk,
                } => StreamChunk::Subgraph {
                    namespace: format!("{chunk_namespace}|{inner}"),
                    chunk,
                },
                other => StreamChunk::Subgraph {
                    namespace: chunk_namespace.clone(),
                    chunk: Box::new(other),
                },
            };
            collector
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(wrapped);
        };
        if !subgraph.has_checkpointer() {
            return subgraph.stream_with(input, &config, sink);
        }
        return match (thread, resume) {
            (Some(thread), Some(value)) => {
                subgraph.resume_in_streaming(&thread, &resume_namespace, value, Some(&config), &mut sink)
            }
            (Some(thread), None) => subgraph.stream_with(input, &config.with_thread(thread), sink),
            (None, _) => subgraph.stream_with(input, &config, sink),
        };
    }
    if !subgraph.has_checkpointer() {
        return subgraph.run(input, &config).map(|outcome| outcome.state);
    }
    match (thread, resume) {
        (Some(thread), Some(value)) => {
            subgraph.resume_in_with_run_config(&thread, &resume_namespace, value, &config)
        }
        (Some(thread), None) => subgraph
            .run(input, &config.with_thread(thread))
            .map(|outcome| outcome.state),
        (None, _) => subgraph.run(input, &config).map(|outcome| outcome.state),
    }
}

#[path = "builder_compile.rs"]
mod builder_compile;
