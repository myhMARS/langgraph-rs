//! Tokio async runtime adapter (feature `async`).
//!
//! Async graphs mirror the sync kernel: nodes are async callables, a superstep awaits
//! all currently scheduled nodes concurrently, writes merge through the same channels,
//! and scheduling reuses the kernel's edge evaluation. The adapter is deliberately
//! separate from the kernel so the sync engine has no runtime dependency:
//!
//! - **No blocking.** Every node returns a future and supersteps use
//!   [`futures::future::join_all`], so pending I/O overlaps.
//! - **Explicit context.** Because futures interleave on one task, the thread-local
//!   context of the sync API is not available; nodes receive an [`AsyncContext`](crate::tokio_runtime::AsyncContext) with
//!   `record_usage`, `record_tokens` and `custom`.
//! - **Scope.** Async graphs support nodes, static/conditional edges, goto, sends,
//!   usage/trace reporting and streaming. Checkpointing, interrupts and error routing
//!   stay on the sync path for now (see `docs/architecture.md`).
//!
//! ```
//! use langgraph_rs::tokio_runtime::AsyncStateGraph;
//! use langgraph_rs::{JsonState, NodeOutput};
//!
//! # async fn run() -> langgraph_rs::Result<()> {
//! let graph = AsyncStateGraph::<JsonState>::new()
//!     .add_node("increment", |state: JsonState, _ctx| async move {
//!         let value = state.get("value").and_then(|value| value.as_i64()).unwrap_or(0);
//!         Ok(NodeOutput::new().set("value", value + 1))
//!     })
//!     .set_entry_point("increment")
//!     .set_finish_point("increment")
//!     .compile()?;
//!
//! let input = JsonState::default().with("value", serde_json::json!(0));
//! let output = graph.invoke(&input).await?;
//! assert_eq!(output.get("value"), Some(&serde_json::json!(1)));
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::join_all;
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe};
use crate::kernel::graph::{Entry, RunConfig};
use crate::kernel::node::{ConditionalEdge, NodeOutput, PathMap, Route, Task};
use crate::kernel::plugin::Kernel;
use crate::kernel::ports::TraceSink;
use crate::kernel::state::{
    ChannelUpdate, State, StateSchema, apply_updates, state_to_values, values_to_state,
};
use crate::kernel::stream::StreamChunk;
use crate::kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};
use crate::{END, START};

/// Boxed future returned by an async node.
pub type BoxNodeFuture = Pin<Box<dyn Future<Output = Result<NodeOutput>>>>;

/// Async execution function of a graph node.
pub type AsyncNodeFn<S> =
    Arc<dyn Fn(S, AsyncContext) -> BoxNodeFuture + std::marker::Send + std::marker::Sync>;

/// Per-node execution context for async nodes.
///
/// The context is cheap to clone; clones share the same capture buffer. Nodes use it to
/// report consumption (`record_usage`, `record_tokens`), emit custom stream values
/// (`custom`) and inspect a routed failure (`last_error`).
#[derive(Clone)]
pub struct AsyncContext {
    node: String,
    step: usize,
    error: Option<NodeFailure>,
    capture: Arc<Mutex<Capture>>,
}

#[derive(Default)]
struct Capture {
    usage: Vec<Usage>,
    custom: Vec<Value>,
}

impl AsyncContext {
    fn new(node: String, step: usize, error: Option<NodeFailure>) -> Self {
        Self {
            node,
            step,
            error,
            capture: Arc::new(Mutex::new(Capture::default())),
        }
    }

    /// Node currently executing.
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Superstep the node runs in.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Failure that caused this node to be scheduled, when it is an error handler.
    pub fn last_error(&self) -> Option<&NodeFailure> {
        self.error.as_ref()
    }

    /// Record consumption for this execution (multiple records are summed).
    pub fn record_usage(&self, usage: Usage) {
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .usage
            .push(usage);
    }

    /// Convenience shorthand for token usage.
    pub fn record_tokens(&self, input_tokens: u64, output_tokens: u64) {
        self.record_usage(
            Usage::new()
                .with_input_tokens(input_tokens)
                .with_output_tokens(output_tokens),
        );
    }

    /// Emit a custom value into the stream.
    pub fn custom(&self, value: impl Serialize) -> Result<()> {
        let value = serde_json::to_value(value)?;
        self.capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .custom
            .push(value);
        Ok(())
    }

    fn take(&self) -> (Vec<Usage>, Vec<Value>) {
        let mut capture = self
            .capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            std::mem::take(&mut capture.usage),
            std::mem::take(&mut capture.custom),
        )
    }
}

/// Builder for an async graph.
pub struct AsyncStateGraph<S: State> {
    schema: StateSchema,
    nodes: IndexMap<String, AsyncNodeFn<S>>,
    duplicate_nodes: Vec<String>,
    edges: IndexMap<String, Vec<String>>,
    conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    entry: Vec<String>,
    conditional_entry: Option<ConditionalEdge<S>>,
    kernel: Kernel,
    recursion_limit: Option<usize>,
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
            entry: Vec::new(),
            conditional_entry: None,
            kernel: Kernel::builtin(),
            recursion_limit: None,
        }
    }

    /// Register an async node.
    ///
    /// The node receives a clone of the current state and the [`AsyncContext`]; it
    /// returns a future resolving to [`NodeOutput`].
    pub fn add_node<F, Fut>(mut self, name: impl Into<String>, node: F) -> Self
    where
        F: Fn(S, AsyncContext) -> Fut + std::marker::Send + std::marker::Sync + 'static,
        Fut: Future<Output = Result<NodeOutput>> + 'static,
    {
        let name = name.into();
        let node: AsyncNodeFn<S> = Arc::new(move |state, context| Box::pin(node(state, context)));
        if self.nodes.insert(name.clone(), node).is_some() {
            self.duplicate_nodes.push(name);
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
            entry,
            trace_sinks: self.kernel.trace_sinks().to_vec(),
            recursion_limit: self
                .recursion_limit
                .unwrap_or(crate::kernel::graph::DEFAULT_RECURSION_LIMIT),
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

/// A compiled async graph.
#[derive(Clone)]
pub struct AsyncCompiledGraph<S: State> {
    schema: StateSchema,
    nodes: IndexMap<String, AsyncNodeFn<S>>,
    edges: IndexMap<String, Vec<String>>,
    conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    entry: Entry<S>,
    trace_sinks: Vec<Arc<dyn TraceSink>>,
    recursion_limit: usize,
}

struct AsyncTaskResult {
    node: String,
    output: NodeOutput,
    custom: Vec<Value>,
    usage: Vec<Usage>,
    duration_millis: u64,
}

enum AsyncOutcome {
    Ok(Box<AsyncTaskResult>),
    Interrupted(Interrupt),
    Failed(Box<NodeFailure>),
}

impl<S: State> AsyncCompiledGraph<S> {
    /// Channel schema of the graph.
    pub fn schema(&self) -> &StateSchema {
        &self.schema
    }

    /// Default superstep budget.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Run the graph to completion.
    pub async fn invoke(&self, input: &S) -> Result<S> {
        self.run(input, &RunConfig::new())
            .await
            .map(|outcome| outcome.state)
    }

    /// Run the graph and return state plus the consumption report.
    pub async fn run(&self, input: &S, config: &RunConfig) -> Result<RunOutcome<S>> {
        let mut sink = |_chunk: StreamChunk| {};
        self.execute(input, config, &mut sink).await
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
        F: FnMut(StreamChunk),
    {
        self.execute(input, config, &mut sink)
            .await
            .map(|outcome| outcome.state)
    }

    async fn execute(
        &self,
        input: &S,
        config: &RunConfig,
        sink: &mut dyn FnMut(StreamChunk),
    ) -> Result<RunOutcome<S>> {
        let started = Instant::now();
        let recursion_limit = config.recursion_limit.unwrap_or(self.recursion_limit);

        let mut values = state_to_values(input)?;
        let mut state = input.clone();
        let mut tasks: Vec<Task> = match &self.entry {
            Entry::Static(nodes) => nodes
                .iter()
                .filter(|node| node.as_str() != END)
                .map(Task::new)
                .collect(),
            Entry::Conditional(edge) => edge
                .resolve(&state, START)?
                .into_iter()
                .filter(|node| node != END)
                .map(Task::new)
                .collect(),
        };

        let mut report = RunReport::new();
        let mut step = 0usize;

        while !tasks.is_empty() {
            if step >= recursion_limit {
                return Err(Error::RecursionLimit(recursion_limit));
            }

            let outcomes =
                join_all(tasks.iter().map(|task| self.run_task(task, &state, step))).await;

            if let Some(interrupted) = outcomes.iter().find_map(|outcome| match outcome {
                AsyncOutcome::Interrupted(interrupt) => Some(interrupt.clone()),
                _ => None,
            }) {
                return Err(Error::Interrupt(interrupted));
            }

            let mut channel_updates: Vec<ChannelUpdate> = Vec::new();
            let mut replacements: Vec<(String, Value)> = Vec::new();
            let mut next_tasks: Vec<Task> = Vec::new();
            let mut goto_targets: Vec<(String, String)> = Vec::new();
            let mut executed: Vec<String> = Vec::new();
            let mut goto_nodes: HashSet<String> = HashSet::new();

            for outcome in outcomes {
                match outcome {
                    AsyncOutcome::Interrupted(_) => unreachable!("handled above"),
                    AsyncOutcome::Failed(failure) => {
                        let execution = NodeExecution {
                            node: failure.node.clone(),
                            step,
                            duration_millis: 0,
                            attempts: failure.attempts,
                            error: Some(failure.message.clone()),
                            usage: None,
                        };
                        sink(StreamChunk::Usage {
                            execution: execution.clone(),
                        });
                        for trace_sink in &self.trace_sinks {
                            trace_sink.record(&execution);
                        }
                        report.record(execution);
                        return Err(Error::Node {
                            node: failure.node.clone(),
                            message: failure.message.clone(),
                        });
                    }
                    AsyncOutcome::Ok(result) => {
                        let result = *result;
                        executed.push(result.node.clone());
                        for value in result.custom {
                            sink(StreamChunk::Custom {
                                node: result.node.clone(),
                                value,
                            });
                        }
                        let mut combined: Option<Usage> = None;
                        for item in result.usage {
                            combined.get_or_insert_with(Usage::new).merge(&item);
                        }
                        let execution = NodeExecution {
                            node: result.node.clone(),
                            step,
                            duration_millis: result.duration_millis,
                            attempts: 1,
                            error: None,
                            usage: combined,
                        };
                        sink(StreamChunk::Usage {
                            execution: execution.clone(),
                        });
                        for trace_sink in &self.trace_sinks {
                            trace_sink.record(&execution);
                        }
                        report.record(execution);

                        let (updates, replaces, goto, sends) = result.output.into_parts();
                        if !updates.is_empty() {
                            let update = Value::Object(
                                updates
                                    .iter()
                                    .map(|(channel, value)| (channel.clone(), value.clone()))
                                    .collect(),
                            );
                            sink(StreamChunk::Updates {
                                step,
                                node: result.node.clone(),
                                update,
                            });
                            for (channel, value) in updates {
                                channel_updates.push(ChannelUpdate {
                                    channel,
                                    value,
                                    node: result.node.clone(),
                                });
                            }
                        }
                        for (channel, value) in replaces {
                            replacements.push((channel, value));
                        }
                        for target in goto {
                            goto_nodes.insert(result.node.clone());
                            goto_targets.push((target, result.node.clone()));
                        }
                        for send in sends {
                            let input = match send.arg {
                                None | Some(Value::Null) => None,
                                Some(Value::Object(map)) => Some(map),
                                Some(other) => {
                                    return Err(Error::routing(
                                        &result.node,
                                        format!(
                                            "Send toward `{}` needs a JSON object, found `{other}`",
                                            send.node
                                        ),
                                    ));
                                }
                            };
                            if !self.nodes.contains_key(&send.node) {
                                return Err(Error::routing(
                                    &result.node,
                                    format!("Send targets unknown node `{}`", send.node),
                                ));
                            }
                            next_tasks.push(Task {
                                node: send.node,
                                input,
                                resume: None,
                                error: None,
                            });
                        }
                    }
                }
            }

            apply_updates(&self.schema, &mut values, channel_updates)?;
            for (channel, value) in replacements {
                self.schema.resolve(&channel)?;
                values.insert(channel, value);
            }
            state = values_to_state::<S>(&values)?;
            sink(StreamChunk::Values {
                step,
                values: Value::Object(values.clone()),
            });

            let next_nodes = collect_next_nodes(
                &state,
                &executed,
                &goto_nodes,
                &goto_targets,
                |node| self.edges_of(node),
                |node| self.conditional_of(node),
                |node| self.nodes.contains_key(node),
            )?;
            for node in dedupe(next_nodes) {
                next_tasks.push(Task::new(node));
            }

            tasks = next_tasks;
            step += 1;
        }

        report.total_duration_millis = started.elapsed().as_millis() as u64;
        Ok(RunOutcome { state, report })
    }

    async fn run_task(&self, task: &Task, state: &S, step: usize) -> AsyncOutcome {
        let Some(node) = self.nodes.get(&task.node) else {
            return AsyncOutcome::Failed(Box::new(NodeFailure::new(
                &task.node,
                format!("node `{}` is not registered in this graph", task.node),
                1,
            )));
        };
        let input = match &task.input {
            Some(input) => match values_to_state::<S>(input) {
                Ok(state) => state,
                Err(error) => {
                    return AsyncOutcome::Failed(Box::new(NodeFailure::new(
                        &task.node,
                        error.to_string(),
                        1,
                    )));
                }
            },
            None => state.clone(),
        };

        let context = AsyncContext::new(task.node.clone(), step, task.error.clone());
        let started = Instant::now();
        let result = (node.as_ref())(input, context.clone()).await;
        let duration_millis = started.elapsed().as_millis() as u64;
        let (usage, custom) = context.take();

        match result {
            Ok(output) => AsyncOutcome::Ok(Box::new(AsyncTaskResult {
                node: task.node.clone(),
                output,
                custom,
                usage,
                duration_millis,
            })),
            Err(Error::Interrupt(interrupt)) => AsyncOutcome::Interrupted(Interrupt {
                node: task.node.clone(),
                value: interrupt.value,
            }),
            Err(error) => {
                AsyncOutcome::Failed(Box::new(NodeFailure::new(&task.node, error.to_string(), 1)))
            }
        }
    }

    fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.conditional.get(node).map(Vec::as_slice)
    }
}
