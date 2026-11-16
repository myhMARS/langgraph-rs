//! The mutable graph builder.

use std::sync::Arc;

use indexmap::IndexMap;

use super::Entry;
use super::compiled::CompiledGraph;
use super::options::{CompileOptions, DEFAULT_RECURSION_LIMIT, Targets};
use crate::kernel::cache::{CachePolicy, NodeCache};
use crate::kernel::error::{Error, Result};
use crate::kernel::node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::state_to_values;
use crate::kernel::state::{State, StateSchema};
use crate::kernel::store::BaseStore;
use crate::{END, START};

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
    pub(crate) entry: Vec<String>,
    pub(crate) conditional_entry: Option<ConditionalEdge<S>>,
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
    pub fn add_subgraph(self, name: impl Into<String>, subgraph: Arc<CompiledGraph<S>>) -> Self {
        let name = name.into();
        self.add_node(name.clone(), move |state: &S| {
            let before = state_to_values(state)?;
            let output = run_subgraph(&subgraph, state, &name)?;
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
        self,
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
        self.add_node(name.clone(), move |state: &S| {
            let sub_state = input(state)?;
            let sub_output = run_subgraph(&subgraph, &sub_state, &name)?;
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

        for node in self.retries.keys() {
            if !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "retry policy references unknown node `{node}`"
                )));
            }
        }
        for (source, handler) in &self.error_edges {
            self.require_node("error edge source", source)?;
            self.require_node("error handler", handler)?;
        }
        for node in self.cache_policies.keys() {
            self.require_node("cache policy", node)?;
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
            trace_sinks: options.kernel.trace_sinks().to_vec(),
            retries: self.retries,
            error_edges: self.error_edges,
            cache: self.cache,
            cache_policies: self.cache_policies,
            store: options.store.or(self.store),
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

/// Child checkpoint namespace of a subgraph node.
fn child_namespace(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}|{name}")
    }
}

/// Run a subgraph with parent thread/namespace propagation and resume support.
fn run_subgraph<Sub: State>(subgraph: &CompiledGraph<Sub>, input: &Sub, name: &str) -> Result<Sub> {
    let thread = crate::kernel::context::current_thread_id();
    let namespace = child_namespace(&crate::kernel::context::current_namespace(), name);
    let resume = crate::kernel::context::current_resume();
    let mut config = super::options::RunConfig::new().with_checkpoint_ns(namespace.clone());
    if let Some(store) = crate::kernel::context::get_store() {
        config = config.with_store(store);
    }
    if let Some(context) = crate::kernel::context::current_context() {
        config = config.with_context(context);
    }
    if !subgraph.has_checkpointer() {
        return subgraph.run(input, &config).map(|outcome| outcome.state);
    }
    match (thread, resume) {
        (Some(thread), Some(value)) => {
            subgraph.resume_in_with_run_config(&thread, &namespace, value, &config)
        }
        (Some(thread), None) => subgraph
            .run(input, &config.with_thread(thread))
            .map(|outcome| outcome.state),
        (None, _) => subgraph.run(input, &config).map(|outcome| outcome.state),
    }
}
