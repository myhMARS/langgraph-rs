//! Compiled async graph and its superstep loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::context::AsyncContext;
use super::{AsyncNodeFn, BoxNodeFuture};
use crate::kernel::cache::{CachePolicy, CachedOutput, NodeCache};
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe, send_task};
use crate::kernel::graph::{Entry, RunConfig, StateSnapshot};
use crate::kernel::node::{ConditionalEdge, Task};
use crate::kernel::ports::{Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{
    ChannelUpdate, State, StateSchema, apply_updates, state_to_values, values_to_state,
};
use crate::kernel::store::BaseStore;
use crate::kernel::stream::{StreamChunk, StreamMode};
use crate::kernel::timeout::TimeoutPolicy;
use crate::kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};
use crate::{END, START};

/// Where a run starts from.
enum StartInput {
    Values(Map<String, Value>),
    Updates(Map<String, Value>),
    Resume {
        thread_id: String,
        checkpoint_id: Option<String>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
    },
}

struct AsyncRunInfo {
    thread_id: Option<String>,
    namespace: String,
    store: Option<Arc<dyn BaseStore>>,
    context: Option<Value>,
}

struct AsyncTaskResult {
    node: String,
    output: crate::kernel::node::NodeOutput,
    custom: Vec<Value>,
    messages: Vec<Value>,
    usage: Vec<Usage>,
    duration_millis: u64,
    attempts: u32,
    cached: bool,
    input: Option<Value>,
}

struct AsyncTaskFailure {
    failure: NodeFailure,
    usage: Vec<Usage>,
    duration_millis: u64,
    input: Option<Value>,
}

enum TaskOutcome {
    Ok(Box<AsyncTaskResult>),
    Interrupted(Interrupt),
    Failed(Box<AsyncTaskFailure>),
}

struct ProcessedResults {
    channel_updates: Vec<ChannelUpdate>,
    replacements: Vec<(String, Value)>,
    next_tasks: Vec<Task>,
    goto_targets: Vec<(String, String)>,
    executed: Vec<String>,
    goto_nodes: HashSet<String>,
}

/// A compiled async graph.
///
/// Build one with [`AsyncStateGraph::compile`](crate::tokio_runtime::AsyncStateGraph::compile).
/// The graph is immutable afterwards and can be shared between tasks.
#[derive(Clone)]
pub struct AsyncCompiledGraph<S: State> {
    pub(crate) schema: StateSchema,
    pub(crate) nodes: IndexMap<String, AsyncNodeFn<S>>,
    pub(crate) edges: IndexMap<String, Vec<String>>,
    pub(crate) conditional: IndexMap<String, Vec<ConditionalEdge<S>>>,
    pub(crate) retries: IndexMap<String, RetryPolicy>,
    pub(crate) error_edges: IndexMap<String, String>,
    pub(crate) entry: Entry<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer>>,
    pub(crate) store: Option<Arc<dyn BaseStore>>,
    pub(crate) timeouts: IndexMap<String, TimeoutPolicy>,
    pub(crate) cache: Option<Arc<dyn NodeCache>>,
    pub(crate) cache_policies: IndexMap<String, CachePolicy>,
    pub(crate) deferred: HashSet<String>,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) trace_sinks: Vec<Arc<dyn TraceSink>>,
    pub(crate) recursion_limit: usize,
}

impl<S: State> AsyncCompiledGraph<S> {
    /// Timeout policy of one node, when configured.
    pub fn timeout_of(&self, node: &str) -> Option<&TimeoutPolicy> {
        self.timeouts.get(node)
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

    /// Returns `true` when checkpointing is enabled.
    pub fn has_checkpointer(&self) -> bool {
        self.checkpointer.is_some()
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
        self.execute(StartInput::Values(values), config, &mut sink)
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
        self.execute(StartInput::Updates(values), config, &mut sink)
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

    /// Resume an interrupted thread.
    pub async fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume an interrupted thread inside a checkpoint namespace.
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
            .checkpointer
            .as_ref()
            .ok_or(Error::MissingCheckpointer("get_state"))?;
        match checkpointer.latest_in(thread_id, namespace)? {
            Some(checkpoint) => Ok(Some(StateSnapshot::from_checkpoint(checkpoint)?)),
            None => Ok(None),
        }
    }

    /// Read the full checkpoint history of a thread, newest first.
    pub fn get_state_history(&self, thread_id: &str) -> Result<Vec<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(Error::MissingCheckpointer("get_state_history"))?;
        checkpointer
            .history(thread_id)?
            .into_iter()
            .map(StateSnapshot::from_checkpoint)
            .collect()
    }

    async fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        namespace: Option<&str>,
    ) -> Result<RunOutcome<S>> {
        let mut config = RunConfig::new().with_thread(thread_id);
        if let Some(namespace) = namespace {
            config = config.with_checkpoint_ns(namespace);
        }
        let mut sink = |_chunk: StreamChunk| {};
        self.execute(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: checkpoint_id.map(str::to_string),
                resume,
                updates,
            },
            &config,
            &mut sink,
        )
        .await
    }

    async fn execute(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<RunOutcome<S>> {
        let started = Instant::now();
        let mut report = RunReport::new();
        let recursion_limit = config.recursion_limit.unwrap_or(self.recursion_limit);
        let thread_id = config.thread_id.clone();
        let namespace = config.checkpoint_ns.clone().unwrap_or_default();
        let store = config.store.clone().or_else(|| self.store.clone());
        let run_info = AsyncRunInfo {
            thread_id: thread_id.clone(),
            namespace: namespace.clone(),
            store: store.clone(),
            context: config.context.clone(),
        };
        let _run_guard = crate::kernel::context::install_run(
            thread_id.clone(),
            namespace.clone(),
            store,
            config.context.clone(),
        );

        if thread_id.is_some() && self.checkpointer.is_none() {
            return Err(Error::MissingCheckpointer("a run bound to a thread_id"));
        }

        let fresh = matches!(&start, StartInput::Values(_) | StartInput::Updates(_));
        let mut parent: Option<String> = None;
        let mut skip_interrupt_before = false;

        let (mut values, mut state, mut tasks, mut step) = match start {
            StartInput::Values(values) => {
                let state = values_to_state::<S>(&values)?;
                let tasks = self.entry_tasks(&state)?;
                (values, state, tasks, 0usize)
            }
            StartInput::Updates(updates) => {
                let mut values = Map::new();
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
                let state = values_to_state::<S>(&values)?;
                let tasks = self.entry_tasks(&state)?;
                (values, state, tasks, 0usize)
            }
            StartInput::Resume {
                thread_id: resume_thread,
                checkpoint_id,
                resume,
                updates,
            } => {
                let checkpointer = self
                    .checkpointer
                    .as_ref()
                    .ok_or(Error::MissingCheckpointer("resume"))?;
                let checkpoint = match checkpoint_id {
                    Some(id) => checkpointer.get(&resume_thread, &id)?,
                    None => checkpointer.latest_in(&resume_thread, &namespace)?,
                }
                .ok_or_else(|| {
                    Error::Checkpoint(format!("no checkpoint found for thread `{resume_thread}`"))
                })?;

                parent = Some(checkpoint.id.clone());
                skip_interrupt_before = checkpoint.paused_before;
                let mut values = checkpoint.values.clone();
                if let Some(updates) = updates {
                    apply_updates(
                        &self.schema,
                        &mut values,
                        updates
                            .into_iter()
                            .map(|(channel, value)| ChannelUpdate {
                                channel,
                                value,
                                node: "<resume>".to_string(),
                            })
                            .collect(),
                    )?;
                }
                let state = values_to_state::<S>(&values)?;
                let mut tasks = checkpoint.next.clone();
                for task in &mut tasks {
                    task.resume = resume.clone();
                }
                (values, state, tasks, checkpoint.step)
            }
        };

        if tasks.is_empty() {
            return Ok(finish(state, report, started));
        }

        if fresh {
            if let (Some(checkpointer), Some(tid)) = (&self.checkpointer, thread_id.as_deref()) {
                parent = checkpointer.latest(tid)?.map(|checkpoint| checkpoint.id);
            }
            self.save_checkpoint(
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &tasks,
                step,
                false,
                false,
                Vec::new(),
                sink,
            )?;
        }

        let mut pending_deferred: Vec<Task> = Vec::new();
        let mut iteration = 0usize;
        loop {
            if iteration >= recursion_limit {
                return Err(Error::RecursionLimit(recursion_limit));
            }
            iteration += 1;
            let _step_guard = crate::kernel::context::install_step(step, recursion_limit);

            let paused = if skip_interrupt_before {
                None
            } else {
                tasks
                    .iter()
                    .find(|task| self.interrupt_before.contains(&task.node))
                    .cloned()
            };
            skip_interrupt_before = false;
            if let Some(interrupted) = paused {
                let interrupted_interrupt = Interrupt {
                    node: interrupted.node.clone(),
                    value: Value::Null,
                };
                self.save_checkpoint(
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &tasks,
                    step,
                    true,
                    true,
                    vec![interrupted_interrupt],
                    sink,
                )?;
                sink(StreamChunk::Interrupt {
                    node: interrupted.node.clone(),
                    value: Value::Null,
                });
                return Err(Error::Interrupt(Interrupt {
                    node: interrupted.node,
                    value: Value::Null,
                }));
            }

            let outcomes = join_all(
                tasks
                    .iter()
                    .map(|task| self.run_task(task, &state, step, &run_info)),
            )
            .await;

            if let Some(interrupted) = outcomes.iter().find_map(|outcome| match outcome {
                TaskOutcome::Interrupted(interrupt) => Some(interrupt.clone()),
                _ => None,
            }) {
                self.save_checkpoint(
                    thread_id.as_deref(),
                    &namespace,
                    &mut parent,
                    &values,
                    &tasks,
                    step,
                    true,
                    false,
                    vec![interrupted.clone()],
                    sink,
                )?;
                sink(StreamChunk::Interrupt {
                    node: interrupted.node.clone(),
                    value: interrupted.value.clone(),
                });
                return Err(Error::Interrupt(interrupted));
            }

            let processed = self.process_outcomes(outcomes, step, &mut report, sink)?;
            let channel_updates = processed.channel_updates;
            let replacements = processed.replacements;
            let next_tasks = processed.next_tasks;
            let goto_targets = processed.goto_targets;
            let executed = processed.executed;
            let goto_nodes = processed.goto_nodes;

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

            let (next_nodes, router_sends) = collect_next_nodes(
                &state,
                &executed,
                &goto_nodes,
                &goto_targets,
                |node| self.edges_of(node),
                |node| self.conditional_of(node),
                |node| self.nodes.contains_key(node),
            )?;
            let mut next_tasks = next_tasks;
            let handler_nodes: HashSet<String> = next_tasks
                .iter()
                .filter(|task| task.error.is_some())
                .map(|task| task.node.clone())
                .collect();
            for node in dedupe(next_nodes) {
                if handler_nodes.contains(&node) {
                    continue;
                }
                next_tasks.push(Task::new(node));
            }
            for (source, send) in router_sends {
                let task = send_task(&source, send)?;
                if !self.nodes.contains_key(&task.node) {
                    return Err(Error::routing(
                        &source,
                        format!("unknown destination node `{}`", task.node),
                    ));
                }
                next_tasks.push(task);
            }

            let (deferred_tasks, regular_tasks): (Vec<Task>, Vec<Task>) = next_tasks
                .into_iter()
                .partition(|task| self.is_deferred(&task.node));
            pending_deferred.extend(deferred_tasks);
            let next_tasks = if regular_tasks.is_empty() && !pending_deferred.is_empty() {
                std::mem::take(&mut pending_deferred)
            } else {
                regular_tasks
            };

            step += 1;
            let interrupt_after = executed
                .iter()
                .find(|node| self.interrupt_after.contains(*node))
                .cloned();
            self.save_checkpoint(
                thread_id.as_deref(),
                &namespace,
                &mut parent,
                &values,
                &next_tasks,
                step,
                interrupt_after.is_some(),
                false,
                interrupt_after
                    .iter()
                    .map(|node| Interrupt {
                        node: node.clone(),
                        value: Value::Null,
                    })
                    .collect(),
                sink,
            )?;
            if let Some(node) = interrupt_after {
                sink(StreamChunk::Interrupt {
                    node: node.clone(),
                    value: Value::Null,
                });
                return Err(Error::Interrupt(Interrupt {
                    node,
                    value: Value::Null,
                }));
            }

            if next_tasks.is_empty() {
                return Ok(finish(state, report, started));
            }
            self.schema.clear_ephemeral(&mut values);
            state = values_to_state::<S>(&values)?;
            tasks = next_tasks;
        }
    }

    fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
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

    async fn run_task(
        &self,
        task: &Task,
        state: &S,
        step: usize,
        run: &AsyncRunInfo,
    ) -> TaskOutcome {
        let Some(node) = self.nodes.get(&task.node) else {
            return TaskOutcome::Failed(Box::new(AsyncTaskFailure {
                failure: NodeFailure::new(
                    &task.node,
                    format!("node `{}` is not registered in this graph", task.node),
                    1,
                ),
                usage: Vec::new(),
                duration_millis: 0,
                input: task.input.clone().map(Value::Object),
            }));
        };

        let input = match &task.input {
            Some(input) => match values_to_state::<S>(input) {
                Ok(state) => state,
                Err(error) => {
                    return TaskOutcome::Failed(Box::new(AsyncTaskFailure {
                        failure: NodeFailure::new(&task.node, error.to_string(), 1),
                        usage: Vec::new(),
                        duration_millis: 0,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
            },
            None => state.clone(),
        };

        let cache = self.cache.clone();
        let cache_policy = self.cache_policy(&task.node).cloned();
        let cache_key = match (&cache, &cache_policy) {
            (Some(_), Some(policy)) => state_to_values(&input)
                .ok()
                .map(|values| policy.cache_key(&task.node, &values)),
            _ => None,
        };
        if let (Some(cache), Some(key)) = (&cache, &cache_key) {
            if let Ok(Some(cached)) = cache.get(key) {
                let usage = cached
                    .usage
                    .clone()
                    .map(|usage| vec![usage])
                    .unwrap_or_default();
                return TaskOutcome::Ok(Box::new(AsyncTaskResult {
                    node: task.node.clone(),
                    output: crate::kernel::node::NodeOutput::from_cached(cached),
                    custom: Vec::new(),
                    messages: Vec::new(),
                    usage,
                    duration_millis: 0,
                    attempts: 0,
                    cached: true,
                    input: task.input.clone().map(Value::Object),
                }));
            }
        }

        let policy = self.retries.get(&task.node).cloned().unwrap_or_default();
        let max_attempts = policy.max_attempts.max(1);
        let started = Instant::now();
        let mut usage: Vec<Usage> = Vec::new();
        let mut last_error: Option<String> = None;

        for attempt in 1..=max_attempts {
            let context = AsyncContext::new(
                task.node.clone(),
                step,
                task.error.clone(),
                task.resume.clone(),
            )
            .with_run_info(
                run.thread_id.clone(),
                run.namespace.clone(),
                run.store.clone(),
                run.context.clone(),
            );
            let future: BoxNodeFuture = (node.as_ref())(input.clone(), context.clone());
            let timeout = task.timeout_millis.map(Duration::from_millis).or_else(|| {
                self.timeout_of(&task.node)
                    .and_then(|policy| policy.run_timeout())
            });
            let result = match timeout {
                Some(timeout) => match tokio::time::timeout(timeout, future).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout {
                        node: task.node.clone(),
                        millis: timeout.as_millis() as u64,
                    }),
                },
                None => future.await,
            };
            let (mut reported, custom, messages) = context.take();
            usage.append(&mut reported);
            match result {
                Ok(output) => {
                    usage.extend(output.usage().cloned());
                    if let (Some(cache), Some(cache_policy), Some(key)) =
                        (&cache, &cache_policy, &cache_key)
                    {
                        let mut combined: Option<Usage> = None;
                        for item in &usage {
                            combined.get_or_insert_with(Usage::new).merge(item);
                        }
                        let cached_output = CachedOutput {
                            updates: output.updates_map().clone(),
                            replaces: output.replaces_map().clone(),
                            goto: output.goto_targets().to_vec(),
                            sends: output.sends().to_vec(),
                            usage: combined,
                        };
                        if let Err(error) = cache.put(key, cached_output, cache_policy.ttl()) {
                            return TaskOutcome::Failed(Box::new(AsyncTaskFailure {
                                failure: NodeFailure::new(
                                    &task.node,
                                    format!("node cache store failed: {error}"),
                                    attempt as u32,
                                ),
                                usage,
                                duration_millis: started.elapsed().as_millis() as u64,
                                input: task.input.clone().map(Value::Object),
                            }));
                        }
                    }
                    return TaskOutcome::Ok(Box::new(AsyncTaskResult {
                        node: task.node.clone(),
                        output,
                        custom,
                        messages,
                        usage,
                        duration_millis: started.elapsed().as_millis() as u64,
                        attempts: attempt as u32,
                        cached: false,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
                Err(Error::Interrupt(interrupt)) => {
                    return TaskOutcome::Interrupted(Interrupt {
                        node: task.node.clone(),
                        value: interrupt.value,
                    });
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if attempt < max_attempts {
                        let backoff = policy.backoff_for(attempt);
                        if !backoff.is_zero() {
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
        }

        TaskOutcome::Failed(Box::new(AsyncTaskFailure {
            failure: NodeFailure::new(
                &task.node,
                last_error.unwrap_or_else(|| "node failed without an error message".to_string()),
                max_attempts as u32,
            ),
            usage,
            duration_millis: started.elapsed().as_millis() as u64,
            input: task.input.clone().map(Value::Object),
        }))
    }

    fn process_outcomes(
        &self,
        outcomes: Vec<TaskOutcome>,
        step: usize,
        report: &mut RunReport,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<ProcessedResults> {
        let mut channel_updates: Vec<ChannelUpdate> = Vec::new();
        let mut replacements: Vec<(String, Value)> = Vec::new();
        let mut next_tasks: Vec<Task> = Vec::new();
        let mut goto_targets: Vec<(String, String)> = Vec::new();
        let mut executed: Vec<String> = Vec::new();
        let mut goto_nodes: HashSet<String> = HashSet::new();
        let mut routed_handlers: HashSet<String> = HashSet::new();

        for outcome in outcomes {
            match outcome {
                TaskOutcome::Interrupted(_) => unreachable!("handled before processing"),
                TaskOutcome::Failed(failure) => {
                    let failure = *failure;
                    let execution = NodeExecution {
                        node: failure.failure.node.clone(),
                        step,
                        duration_millis: failure.duration_millis,
                        attempts: failure.failure.attempts,
                        cached: false,
                        error: Some(failure.failure.message.clone()),
                        usage: merge_usage(failure.usage),
                    };
                    sink(StreamChunk::Usage {
                        execution: execution.clone(),
                    });
                    for trace_sink in &self.trace_sinks {
                        trace_sink.record(&execution);
                    }
                    report.record(execution);

                    sink(StreamChunk::Task {
                        step,
                        node: failure.failure.node.clone(),
                        input: failure.input,
                        result: None,
                        error: Some(failure.failure.clone()),
                    });

                    match self.error_edges.get(&failure.failure.node) {
                        Some(handler) if routed_handlers.insert(handler.clone()) => {
                            next_tasks.push(Task {
                                node: handler.clone(),
                                input: None,
                                resume: None,
                                error: Some(failure.failure),
                                timeout_millis: None,
                            });
                        }
                        Some(_) => {}
                        None => {
                            return Err(Error::Node {
                                node: failure.failure.node.clone(),
                                message: format!(
                                    "{} (failed after {} attempt(s))",
                                    failure.failure.message, failure.failure.attempts
                                ),
                            });
                        }
                    }
                }
                TaskOutcome::Ok(result) => {
                    let result = *result;
                    executed.push(result.node.clone());
                    for value in result.custom {
                        sink(StreamChunk::Custom {
                            node: result.node.clone(),
                            value,
                        });
                    }
                    for message in result.messages {
                        sink(StreamChunk::Message {
                            node: result.node.clone(),
                            message,
                        });
                    }
                    let execution = NodeExecution {
                        node: result.node.clone(),
                        step,
                        duration_millis: result.duration_millis,
                        attempts: result.attempts,
                        cached: result.cached,
                        error: None,
                        usage: merge_usage(result.usage),
                    };
                    sink(StreamChunk::Usage {
                        execution: execution.clone(),
                    });
                    for trace_sink in &self.trace_sinks {
                        trace_sink.record(&execution);
                    }
                    report.record(execution);

                    let task_result = {
                        let updates = result.output.updates_map();
                        if updates.is_empty() {
                            None
                        } else {
                            Some(Value::Object(
                                updates
                                    .iter()
                                    .map(|(channel, value)| (channel.clone(), value.clone()))
                                    .collect(),
                            ))
                        }
                    };
                    sink(StreamChunk::Task {
                        step,
                        node: result.node.clone(),
                        input: result.input,
                        result: task_result,
                        error: None,
                    });
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
                            timeout_millis: send.timeout_millis,
                        });
                    }
                }
            }
        }

        Ok(ProcessedResults {
            channel_updates,
            replacements,
            next_tasks,
            goto_targets,
            executed,
            goto_nodes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn save_checkpoint(
        &self,
        thread_id: Option<&str>,
        namespace: &str,
        parent: &mut Option<String>,
        values: &Map<String, Value>,
        next: &[Task],
        step: usize,
        interrupted: bool,
        paused_before: bool,
        interrupts: Vec<Interrupt>,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<()> {
        let (Some(thread_id), Some(checkpointer)) = (thread_id, self.checkpointer.as_ref()) else {
            return Ok(());
        };
        let id = new_checkpoint_id(thread_id, step);
        let checkpoint = Checkpoint {
            id: id.clone(),
            thread_id: thread_id.to_string(),
            step,
            values: self.schema.persisted_values(values),
            next: next.to_vec(),
            interrupted,
            paused_before,
            parent_id: parent.clone(),
            created_at_millis: now_millis(),
            interrupts,
            metadata: Map::from_iter([("source".to_string(), Value::String("loop".to_string()))]),
            namespace: namespace.to_string(),
        };
        checkpointer.put(checkpoint)?;
        *parent = Some(id.clone());
        sink(StreamChunk::Checkpoint { checkpoint_id: id });
        Ok(())
    }

    fn edges_of(&self, node: &str) -> Option<&[String]> {
        self.edges.get(node).map(Vec::as_slice)
    }

    fn conditional_of(&self, node: &str) -> Option<&[ConditionalEdge<S>]> {
        self.conditional.get(node).map(Vec::as_slice)
    }
}

fn merge_usage(usage: Vec<Usage>) -> Option<Usage> {
    let mut combined: Option<Usage> = None;
    for item in usage {
        combined.get_or_insert_with(Usage::new).merge(&item);
    }
    combined
}

fn finish<S>(state: S, mut report: RunReport, started: Instant) -> RunOutcome<S> {
    report.total_duration_millis = started.elapsed().as_millis() as u64;
    RunOutcome { state, report }
}
