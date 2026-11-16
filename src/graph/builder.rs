//! The mutable graph builder.

use std::sync::Arc;

use indexmap::IndexMap;

use super::Entry;
use super::compiled::CompiledGraph;
use super::options::{CompileOptions, DEFAULT_RECURSION_LIMIT, Targets};
use crate::error::{Error, Result};
use crate::node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route};
use crate::state::{State, StateSchema};
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
