//! Async graph builder.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use indexmap::IndexMap;

use super::compiled::AsyncCompiledGraph;
use super::{AsyncContext, AsyncNodeFn, BoxNodeFuture};
use crate::kernel::cache::{CachePolicy, NodeCache};
use crate::kernel::error::{Error, Result};
use crate::kernel::graph::Entry;
use crate::kernel::graph::RunConfig;
use crate::kernel::node::{ConditionalEdge, NodeOutput, PathMap, Route};
use crate::kernel::plugin::Kernel;
use crate::kernel::ports::{AsyncCheckpointer, Checkpointer};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{State, StateSchema, state_to_values};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::StreamChunk;
use crate::kernel::timeout::TimeoutPolicy;
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
        self.add_node(name, move |state: S, context: AsyncContext| {
            let child = child.clone();
            let node_name = node_name.clone();
            async move { run_async_subgraph(child, state, node_name, context).await }
        })
    }

    /// Attach a timeout policy to a node.
    pub fn set_timeout(mut self, name: impl Into<String>, policy: TimeoutPolicy) -> Self {
        self.timeouts.insert(name.into(), policy);
        self
    }

    /// Attach a long-term memory store (`BaseStore`).
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
        })
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::validation("the async graph has no nodes"));
        }
        if !self.duplicate_nodes.is_empty() {
            return Err(Error::validation(format!(
                "async nodes registered more than once: {}",
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
                "an async graph cannot combine a static entry point with a conditional one",
            ));
        }
        if self.conditional_entry.is_none() && self.entry.is_empty() {
            return Err(Error::validation(
                "no entry point: call set_entry_point(...) or add_edge(START, ...)",
            ));
        }
        for node in &self.entry {
            self.require_node("entry point", node)?;
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
                                    "conditional edge `{source}` maps key `{key}` to `{START}`"
                                )));
                            }
                            self.require_target(source, target)?;
                        }
                    }
                }
            }
        }
        for node in self.retries.keys() {
            self.require_node("retry policy", node)?;
        }
        for (source, handler) in &self.error_edges {
            self.require_node("error edge source", source)?;
            self.require_node("error handler", handler)?;
        }
        for node in self
            .interrupt_before
            .iter()
            .chain(self.interrupt_after.iter())
        {
            self.require_node("interrupt hook", node)?;
        }
        Ok(())
    }

    fn require_node(&self, what: &str, name: &str) -> Result<()> {
        if name == START || name == END {
            return Err(Error::validation(format!("`{name}` cannot be a {what}")));
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

fn child_namespace(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}|{name}")
    }
}

async fn run_async_subgraph<S: State>(
    child: Arc<AsyncCompiledGraph<S>>,
    state: S,
    name: String,
    context: AsyncContext,
) -> Result<NodeOutput> {
    let before = state_to_values(&state)?;
    let thread = context.thread_id().map(str::to_string);
    let parent_namespace = context.namespace();
    let namespace = child_namespace(parent_namespace, &name);
    let resume = context.take_resume().into_iter().next_back();
    let mut config = RunConfig::new().with_checkpoint_ns(namespace.clone());
    if let Some(store) = context.store() {
        config = config.with_store(store);
    }
    if let Some(user_context) = context.context() {
        config = config.with_context(user_context.clone());
    }
    let collector = context.collector();
    let (output, parent_command) = super::context::run_with_parent_scope(async {
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
            if !child.has_checkpointer() {
                child.stream_with(&state, &config, sink).await
            } else if let (Some(thread), Some(resume)) = (thread.clone(), resume) {
                child
                    .resume_in_streaming(&thread, &namespace, resume, Some(&config), &mut sink)
                    .await
            } else if let Some(thread) = thread {
                child
                    .stream_with(&state, &config.with_thread(thread), sink)
                    .await
            } else {
                child.stream_with(&state, &config, sink).await
            }
        } else if !child.has_checkpointer() {
            child
                .run(&state, &config)
                .await
                .map(|outcome| outcome.state)
        } else if let (Some(thread), Some(resume)) = (thread.clone(), resume) {
            child.resume_in(&thread, &namespace, resume).await
        } else if let Some(thread) = thread {
            child
                .run(&state, &config.with_thread(thread))
                .await
                .map(|outcome| outcome.state)
        } else {
            child
                .run(&state, &config)
                .await
                .map(|outcome| outcome.state)
        }
    })
    .await;
    if let Some(command) = parent_command {
        let mut result = NodeOutput::new();
        for (channel, value) in command.updates {
            result = result.replace(channel, value);
        }
        for target in command.goto {
            result = result.goto(target);
        }
        return Ok(result);
    }
    let output = output?;
    let after = state_to_values(&output)?;
    let mut result = NodeOutput::new();
    for (channel, value) in after {
        if before.get(&channel) != Some(&value) {
            result = result.replace(channel, value);
        }
    }
    Ok(result)
}
