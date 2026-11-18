//! Async graph builder.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use indexmap::IndexMap;

use super::compiled::AsyncCompiledGraph;
use super::{AsyncContext, AsyncNodeFn, BoxNodeFuture};
use crate::_internal::cache::{CachePolicy, NodeCache};
use crate::_internal::command::Command;
use crate::_internal::error::{Error, Result};
use crate::_internal::node::{ConditionalEdge, NodeOutput, PathMap, Route};
use crate::_internal::plugin::Kernel;
use crate::_internal::ports::{AsyncCheckpointer, Checkpointer};
use crate::_internal::retry::RetryPolicy;
use crate::_internal::state::{State, StateSchema, state_to_values};
use crate::_internal::stream::StreamChunk;
use crate::_internal::timeout::TimeoutPolicy;
use crate::graph::Entry;
use crate::graph::RunConfig;
use crate::store::BaseStore;
use crate::{END, START};

/// Builder for an async graph.
///
/// Mirrors the synchronous [`StateGraph`](crate::StateGraph) surface for async node
/// closures and adds checkpointing, interrupts, retries and error routing. See the
/// [module docs](crate::tokio_runtime) for an example.
#[derive(Clone)]
pub struct AsyncStateGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, AsyncNodeFn<S>>,
    pub(crate) duplicate_nodes: Vec<String>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) entry: Vec<String>,
    pub(crate) conditional_entry: Option<ConditionalEdge<S>>,
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
    pub(crate) kernel: Kernel,
    pub(crate) recursion_limit: Option<usize>,
    pub(crate) context_schema: Option<serde_json::Value>,
    pub(crate) name: Option<String>,
}

impl<S: State> Default for AsyncStateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> AsyncStateGraph<S> {
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
            entry: Vec::new(),
            conditional_entry: None,
            checkpointer: None,
            async_checkpointer: None,
            store: None,
            timeouts: IndexMap::new(),
            cache: None,
            cache_policies: IndexMap::new(),
            deferred: HashSet::new(),
            destinations: IndexMap::new(),
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
            kernel: Kernel::builtin(),
            recursion_limit: None,
            context_schema: None,
            name: None,
        }
    }

    /// Register an async node.
    ///
    /// The node receives a clone of the current state and an [`AsyncContext`]; it must
    /// return a `Send` future resolving to [`NodeOutput`].
    pub fn add_node<F, Fut>(mut self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + std::marker::Send + std::marker::Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + std::marker::Send + 'static,
    {
        let name = name.into();
        let node: AsyncNodeFn<S> = Arc::new(move |state, context| {
            let future = node(state, context);
            Box::pin(future) as BoxNodeFuture
        });
        if self.nodes.insert(name.clone(), node).is_some() {
            self.duplicate_nodes.push(name);
        }
        self
    }

    /// Register an async node that returns a LangGraph-style [`Command`].
    pub fn add_command_node<F, Fut>(self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + std::marker::Send + std::marker::Sync + 'static,
        Fut: Future<Output = Result<Command>> + std::marker::Send + 'static,
    {
        self.add_node(name, move |state, context| {
            let future = node(state, context);
            async move { Ok::<NodeOutput, Error>(future.await?.into()) }
        })
    }

    /// Register a node with a retry policy.
    pub fn add_node_with_retry<F, Fut>(
        mut self,
        name: impl Into<String>,
        policy: RetryPolicy,
        node: F,
    ) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + std::marker::Send + std::marker::Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + std::marker::Send + 'static,
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

    /// Route permanent failures of `from` to the `handler` node.
    pub fn add_error_edge(mut self, from: impl Into<String>, handler: impl Into<String>) -> Self {
        self.error_edges.insert(from.into(), handler.into());
        self
    }

    /// Enable checkpointing (required for resume and state queries).
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Enable checkpointing through an async checkpointer (non-blocking I/O).
    pub fn with_async_checkpointer(mut self, checkpointer: Arc<dyn AsyncCheckpointer>) -> Self {
        self.async_checkpointer = Some(checkpointer);
        self
    }

    /// Restrict the nodes an edge from `name` may target.
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

    /// Mark a node as deferred until the graph has no regular work left.
    pub fn set_deferred(mut self, name: impl Into<String>) -> Self {
        self.deferred.insert(name.into());
        self
    }

    /// Attach a node cache backend (LangGraph `BaseCache`).
    pub fn with_cache(mut self, cache: Arc<dyn NodeCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Attach a cache policy to a node.
    pub fn set_cache(mut self, node: impl Into<String>, policy: CachePolicy) -> Self {
        self.cache_policies.insert(node.into(), policy);
        self
    }

    /// Embed an async subgraph as a node (child checkpoint namespace, nested interrupts).
    pub fn add_subgraph(
        self,
        name: impl Into<String>,
        subgraph: Arc<AsyncCompiledGraph<S>>,
    ) -> Self {
        let name = name.into();
        let node_name = name.clone();
        let child = subgraph;
        let schema = self.schema.clone();
        self.add_node(name, move |state: S, context: AsyncContext| {
            let child = child.clone();
            let node_name = node_name.clone();
            let schema = schema.clone();
            async move { run_async_subgraph(child, state, node_name, context, &schema).await }
        })
    }

    /// Attach a timeout policy to a node.
    pub fn set_timeout(mut self, name: impl Into<String>, policy: TimeoutPolicy) -> Self {
        self.timeouts.insert(name.into(), policy);
        self
    }

    /// Attach a long-term memory store (`BaseStore`).
    /// Validate `RunConfig::context` against a JSON schema before each run.
    pub fn with_context_schema(mut self, schema: serde_json::Value) -> Self {
        self.context_schema = Some(schema);
        self
    }

    pub fn with_store(mut self, store: Arc<dyn BaseStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Pause before the given nodes run.
    pub fn interrupt_before<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_before = nodes.into_iter().map(Into::into).collect();
        self
    }

    /// Pause after the given nodes ran.
    pub fn interrupt_after<I, T>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.interrupt_after = nodes.into_iter().map(Into::into).collect();
        self
    }

    /// Attach a name to the compiled graph.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Register a node that runs after every regular node finished.
    pub fn add_node_deferred<F, Fut>(self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + Send + 'static,
    {
        let name = name.into();
        self.set_deferred(name.clone()).add_node(name, node)
    }

    /// Register a node together with a timeout policy.
    pub fn add_node_with_timeout<F, Fut>(
        self,
        name: impl Into<String>,
        policy: TimeoutPolicy,
        node: F,
    ) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + Send + 'static,
    {
        let name = name.into();
        self.set_timeout(name.clone(), policy).add_node(name, node)
    }

    /// Register a node together with an allow-list of downstream destinations.
    pub fn add_node_with_destinations<F, Fut, I, T>(
        self,
        name: impl Into<String>,
        destinations: I,
        node: F,
    ) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + Send + 'static,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        let name = name.into();
        self.set_destinations(name.clone(), destinations)
            .add_node(name, node)
    }

    /// Register a chain of async nodes, each followed by an edge.
    pub fn add_sequence<I, N>(mut self, nodes: I) -> Self
    where
        I: IntoIterator<Item = (N, AsyncNodeFn<S>)>,
        N: Into<String>,
    {
        let mut previous: Option<String> = None;
        for (name, node) in nodes {
            let name = name.into();
            self = self.add_node(name.clone(), move |state, context| node(state, context));
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

    /// Add a static edge (`START` / `END` supported).
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let from = from.into();
        let target = to.into();
        if from == START {
            if !self.entry.contains(&target) {
                self.entry.push(target);
            }
        } else {
            self.edges.entry(from).or_default().push(target);
        }
        self
    }

    /// Add a conditional edge returning node names directly.
    pub fn add_conditional_edges<F>(self, from: impl Into<String>, router: F) -> Self
    where
        F: Fn(&S) -> Result<Route> + std::marker::Send + std::marker::Sync + 'static,
    {
        self.add_conditional_edge(from, ConditionalEdge::new(router))
    }

    /// Add a conditional edge with a path map.
    pub fn add_conditional_edges_mapped<F, M>(
        self,
        from: impl Into<String>,
        router: F,
        path_map: M,
    ) -> Self
    where
        F: Fn(&S) -> Result<Route> + std::marker::Send + std::marker::Sync + 'static,
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
        F: Fn(&S) -> Result<Route> + std::marker::Send + std::marker::Sync + 'static,
    {
        self.conditional_entry = Some(ConditionalEdge::new(router));
        self
    }

    /// Add an edge from `node` to `END`.
    pub fn set_finish_point(self, node: impl Into<String>) -> Self {
        self.add_edge(node, END)
    }

    /// Install a microkernel registry (trace sinks and channel plugins).
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Override the default superstep budget (25).
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    /// Validate and compile the async graph.
    pub fn compile(self) -> Result<AsyncCompiledGraph<S>> {
        self.validate()?;

        for node in self.cache_policies.keys() {
            self.require_node("cache policy", node)?;
        }
        for node in self.timeouts.keys() {
            self.require_node("timeout policy", node)?;
        }
        for (node, destinations) in &self.destinations {
            self.require_node("destination list", node)?;
            for target in destinations {
                if target == END {
                    continue;
                }
                if target == START {
                    return Err(Error::validation(format!(
                        "`{START}` is not a valid destination of `{node}`"
                    )));
                }
                self.require_node("destination", target)?;
            }
        }
        for node in &self.deferred {
            self.require_node("deferred node", node)?;
            if self.entry.contains(node) {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot be an entry point"
                )));
            }
            if self
                .edges
                .get(node)
                .is_some_and(|targets| !targets.is_empty())
            {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot have outgoing edges"
                )));
            }
            if self
                .conditional
                .get(node)
                .is_some_and(|edges| !edges.is_empty())
            {
                return Err(Error::validation(format!(
                    "deferred node `{node}` cannot have conditional edges"
                )));
            }
        }

        let entry = match self.conditional_entry {
            Some(edge) => Entry::Conditional(edge),
            None => Entry::Static(self.entry.clone()),
        };

        Ok(AsyncCompiledGraph {
            schema: self.schema,
            nodes: self.nodes,
            edges: self.edges,
            conditional: self.conditional,
            retries: self.retries,
            error_edges: self.error_edges,
            entry,
            checkpointer: self.checkpointer,
            async_checkpointer: self.async_checkpointer,
            store: self.store,
            timeouts: self.timeouts,
            cache: self.cache,
            cache_policies: self.cache_policies,
            deferred: self.deferred,
            destinations: self.destinations,
            interrupt_before: self.interrupt_before,
            interrupt_after: self.interrupt_after,
            trace_sinks: self.kernel.trace_sinks().to_vec(),
            recursion_limit: self
                .recursion_limit
                .unwrap_or(crate::DEFAULT_RECURSION_LIMIT),
            context_schema: self.context_schema,
            name: self.name,
        })
    }
}

#[path = "builder_subgraph.rs"]
mod builder_subgraph;
use builder_subgraph::run_async_subgraph;
