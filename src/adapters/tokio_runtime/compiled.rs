//! Compiled async graph and its superstep loop.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::context::AsyncContext;
use super::{AsyncNodeFn, BoxNodeFuture};
use crate::kernel::checkpoint::{Checkpoint, new_checkpoint_id, now_millis};
use crate::kernel::error::{Error, Interrupt, NodeFailure, Result};
use crate::kernel::executor::scheduling::{collect_next_nodes, dedupe};
use crate::kernel::graph::{Entry, RunConfig, StateSnapshot};
use crate::kernel::node::{ConditionalEdge, Task};
use crate::kernel::ports::{Checkpointer, TraceSink};
use crate::kernel::retry::RetryPolicy;
use crate::kernel::state::{
    ChannelUpdate, State, StateSchema, apply_updates, state_to_values, values_to_state,
};
use crate::kernel::stream::StreamChunk;
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

struct AsyncTaskResult {
    node: String,
    output: crate::kernel::node::NodeOutput,
    custom: Vec<Value>,
    usage: Vec<Usage>,
    duration_millis: u64,
    attempts: u32,
}

struct AsyncTaskFailure {
    failure: NodeFailure,
    usage: Vec<Usage>,
    duration_millis: u64,
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
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) trace_sinks: Vec<Arc<dyn TraceSink>>,
    pub(crate) recursion_limit: usize,
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
        F: FnMut(StreamChunk),
    {
        let values = state_to_values(input)?;
        self.execute(StartInput::Values(values), config, &mut sink)
            .await
            .map(|outcome| outcome.state)
    }

    /// Resume an interrupted thread.
    pub async fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None)
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
        self.resume_inner(thread_id, None, Some(value), Some(updates))
            .await
            .map(|outcome| outcome.state)
    }

    /// Continue a thread paused by an `interrupt_before` / `interrupt_after` hook.
    pub async fn continue_thread(&self, thread_id: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None)
            .await
            .map(|outcome| outcome.state)
    }

    /// Read the latest checkpoint of a thread.
    pub fn get_state(&self, thread_id: &str) -> Result<Option<StateSnapshot<S>>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or(Error::MissingCheckpointer("get_state"))?;
        match checkpointer.latest(thread_id)? {
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
    ) -> Result<RunOutcome<S>> {
        let config = RunConfig::new().with_thread(thread_id);
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
        sink: &mut dyn FnMut(StreamChunk),
    ) -> Result<RunOutcome<S>> {
        let started = Instant::now();
        let mut report = RunReport::new();
        let recursion_limit = config.recursion_limit.unwrap_or(self.recursion_limit);
        let thread_id = config.thread_id.clone();

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
                    None => checkpointer.latest(&resume_thread)?,
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
                &mut parent,
                &values,
                &tasks,
                step,
                false,
                false,
                sink,
            )?;
        }

        let mut iteration = 0usize;
        loop {
            if iteration >= recursion_limit {
                return Err(Error::RecursionLimit(recursion_limit));
            }
            iteration += 1;

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
                self.save_checkpoint(
                    thread_id.as_deref(),
                    &mut parent,
                    &values,
                    &tasks,
                    step,
                    true,
                    true,
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

            let outcomes =
                join_all(tasks.iter().map(|task| self.run_task(task, &state, step))).await;

            if let Some(interrupted) = outcomes.iter().find_map(|outcome| match outcome {
                TaskOutcome::Interrupted(interrupt) => Some(interrupt.clone()),
                _ => None,
            }) {
                self.save_checkpoint(
                    thread_id.as_deref(),
                    &mut parent,
                    &values,
                    &tasks,
                    step,
                    true,
                    false,
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

            let next_nodes = collect_next_nodes(
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

            step += 1;
            let interrupt_after = executed
                .iter()
                .find(|node| self.interrupt_after.contains(*node))
                .cloned();
            self.save_checkpoint(
                thread_id.as_deref(),
                &mut parent,
                &values,
                &next_tasks,
                step,
                interrupt_after.is_some(),
                false,
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
            Entry::Conditional(edge) => Ok(edge
                .resolve(state, START)?
                .into_iter()
                .filter(|node| node != END)
                .map(Task::new)
                .collect()),
        }
    }

    async fn run_task(&self, task: &Task, state: &S, step: usize) -> TaskOutcome {
        let Some(node) = self.nodes.get(&task.node) else {
            return TaskOutcome::Failed(Box::new(AsyncTaskFailure {
                failure: NodeFailure::new(
                    &task.node,
                    format!("node `{}` is not registered in this graph", task.node),
                    1,
                ),
                usage: Vec::new(),
                duration_millis: 0,
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
                    }));
                }
            },
            None => state.clone(),
        };

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
            );
            let future: BoxNodeFuture = (node.as_ref())(input.clone(), context.clone());
            let result = future.await;
            let (mut reported, custom) = context.take();
            usage.append(&mut reported);
            match result {
                Ok(output) => {
                    usage.extend(output.usage().cloned());
                    return TaskOutcome::Ok(Box::new(AsyncTaskResult {
                        node: task.node.clone(),
                        output,
                        custom,
                        usage,
                        duration_millis: started.elapsed().as_millis() as u64,
                        attempts: attempt as u32,
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
        }))
    }

    fn process_outcomes(
        &self,
        outcomes: Vec<TaskOutcome>,
        step: usize,
        report: &mut RunReport,
        sink: &mut dyn FnMut(StreamChunk),
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

                    match self.error_edges.get(&failure.failure.node) {
                        Some(handler) if routed_handlers.insert(handler.clone()) => {
                            next_tasks.push(Task {
                                node: handler.clone(),
                                input: None,
                                resume: None,
                                error: Some(failure.failure),
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
                    let execution = NodeExecution {
                        node: result.node.clone(),
                        step,
                        duration_millis: result.duration_millis,
                        attempts: result.attempts,
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
        parent: &mut Option<String>,
        values: &Map<String, Value>,
        next: &[Task],
        step: usize,
        interrupted: bool,
        paused_before: bool,
        sink: &mut dyn FnMut(StreamChunk),
    ) -> Result<()> {
        let (Some(thread_id), Some(checkpointer)) = (thread_id, self.checkpointer.as_ref()) else {
            return Ok(());
        };
        let id = new_checkpoint_id(thread_id, step);
        let checkpoint = Checkpoint {
            id: id.clone(),
            thread_id: thread_id.to_string(),
            step,
            values: values.clone(),
            next: next.to_vec(),
            interrupted,
            paused_before,
            parent_id: parent.clone(),
            created_at_millis: now_millis(),
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
