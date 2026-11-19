//! Async execution: superstep task runner, tracing and run-loop adapter.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use super::super::BoxNodeFuture;
use super::super::context::AsyncContext;
use super::{AsyncBackend, AsyncCompiledGraph, AsyncRunInfo, run_with_idle_timeout};
use crate::_internal::cache::CachedOutput;
use crate::_internal::error::{Error, Interrupt, NodeFailure, Result};
use crate::_internal::node::Task;
use crate::_internal::state::{State, state_to_values, values_to_state};
use crate::_internal::stream::StreamChunk;
use crate::_internal::usage::{RunOutcome, Usage};
use crate::RunConfig;
use crate::checkpoint::Checkpoint;
use crate::pregel::StartInput;
use crate::pregel::outcome::{TaskFailure, TaskOutcome, TaskResult};

impl<S: State> AsyncCompiledGraph<S> {
    pub(crate) async fn resume_inner(
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
        self.execute_quiet(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: checkpoint_id.map(str::to_string),
                resume,
                updates,
                goto: Vec::new(),
                sends: Vec::new(),
            },
            &config,
            &mut sink,
        )
        .await
    }

    /// Continue a thread with an explicit per-run config.
    pub async fn continue_thread_with_config(
        &self,
        thread_id: &str,
        config: &RunConfig,
    ) -> Result<S> {
        let mut config = config.clone();
        config.thread_id = Some(thread_id.to_string());
        let mut sink = |_chunk: StreamChunk| {};
        self.execute_quiet(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume: None,
                updates: None,
                goto: Vec::new(),
                sends: Vec::new(),
            },
            &config,
            &mut sink,
        )
        .await
        .map(|outcome| outcome.state)
    }

    /// Resume with updates using an explicit per-run config.
    pub async fn resume_with_updates_config<I, K>(
        &self,
        thread_id: &str,
        value: Value,
        updates: I,
        sends: Vec<crate::_internal::node::Send>,
        config: &RunConfig,
    ) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        self.resume_command_start(
            thread_id,
            Some(value),
            Some(updates),
            Vec::new(),
            sends,
            config,
        )
        .await
    }

    /// Continue with goto targets and no resume value using a per-run config.
    pub async fn resume_continue_with_goto_config<I, K>(
        &self,
        thread_id: &str,
        updates: I,
        goto: Vec<String>,
        sends: Vec<crate::_internal::node::Send>,
        config: &RunConfig,
    ) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        self.resume_command_start(thread_id, None, Some(updates), goto, sends, config)
            .await
    }

    /// Resume with updates and goto targets using a per-run config.
    pub async fn resume_with_updates_and_goto_config<I, K>(
        &self,
        thread_id: &str,
        value: Value,
        updates: I,
        goto: Vec<String>,
        sends: Vec<crate::_internal::node::Send>,
        config: &RunConfig,
    ) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        self.resume_command_start(thread_id, Some(value), Some(updates), goto, sends, config)
            .await
    }

    async fn resume_command_start(
        &self,
        thread_id: &str,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        goto: Vec<String>,
        sends: Vec<crate::_internal::node::Send>,
        config: &RunConfig,
    ) -> Result<S> {
        for node in &goto {
            if node != crate::END && !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "Command(goto=...): `{node}` is not a registered node"
                )));
            }
        }
        let mut config = config.clone();
        config.thread_id = Some(thread_id.to_string());
        let mut sink = |_chunk: StreamChunk| {};
        self.execute_quiet(
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume,
                updates,
                goto,
                sends,
            },
            &config,
            &mut sink,
        )
        .await
        .map(|outcome| outcome.state)
    }

    pub(crate) async fn execute(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<RunOutcome<S>> {
        self.execute_with_emit(start, config, sink, true).await
    }

    pub(crate) async fn execute_quiet(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<RunOutcome<S>> {
        self.execute_with_emit(start, config, sink, false).await
    }

    pub(crate) async fn execute_with_emit(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
        emit: bool,
    ) -> Result<RunOutcome<S>> {
        // An in-memory saver cannot block, so the writer task (and its channel clone per
        // checkpoint) is pure overhead - the same fast path the sync backend takes.
        let in_memory_only = self.async_checkpointer.is_none()
            && self
                .checkpointer
                .as_ref()
                .is_some_and(|saver| saver.is_in_memory());
        if config.durability != crate::Durability::Async
            || !self.has_checkpointer()
            || config.thread_id.is_none()
            || in_memory_only
        {
            return self.execute_inner(start, config, sink, None, emit).await;
        }
        let (writer, mut receiver) = tokio::sync::mpsc::unbounded_channel::<std::sync::Arc<Checkpoint>>();
        let checkpointer = self.checkpointer.clone();
        let async_saver = self.async_checkpointer.clone();
        let handle = tokio::spawn(async move {
            while let Some(checkpoint) = receiver.recv().await {
                if let Some(saver) = &async_saver {
                    let _ = saver.put_shared(checkpoint).await;
                } else if let Some(saver) = &checkpointer {
                    let _ = saver.put_shared(checkpoint);
                }
            }
        });
        let result = self
            .execute_inner(start, config, sink, Some(&writer), emit)
            .await;
        drop(writer);
        let _ = handle.await;
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_inner(
        &self,
        start: StartInput,
        config: &RunConfig,
        sink: &mut (dyn FnMut(StreamChunk) + Send),
        writer: Option<&tokio::sync::mpsc::UnboundedSender<std::sync::Arc<Checkpoint>>>,
        emit: bool,
    ) -> Result<RunOutcome<S>> {
        let thread_id = config.thread_id.clone();
        let namespace = config.checkpoint_ns.clone().unwrap_or_default();
        let collector = config
            .stream_subgraphs
            .then(|| Arc::new(std::sync::Mutex::new(Vec::new())));
        let run = AsyncRunInfo {
            max_concurrency: config.max_concurrency,
            thread_id,
            namespace,
            store: config.store.clone().or_else(|| self.store.clone()),
            context: config.context.clone(),
            collector,
        };
        let backend = AsyncBackend {
            graph: self,
            writer,
            run: &run,
        };
        let run_context = crate::runtime::context::run_context(
            run.thread_id.clone(),
            run.namespace.clone(),
            run.store.clone(),
            run.context.clone(),
        );
        let mut adapt = |chunk| sink(chunk);
        let future = crate::pregel::run_loop::run_loop(&backend, start, config, &mut adapt, emit);
        crate::runtime::context::with_async_run(run_context, future).await
    }

    pub(crate) fn entry_tasks(&self, state: &S) -> Result<Vec<Task>> {
        crate::pregel::scheduling::entry_tasks(&self.entry, state)
    }

    pub(crate) async fn run_task(
        &self,
        task: &Task,
        state: &S,
        step: usize,
        run: &AsyncRunInfo,
    ) -> TaskOutcome {
        let Some(node) = self.nodes.get(&task.node) else {
            return TaskOutcome::Failed(Box::new(TaskFailure {
                id: task.id.clone(),
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
                    return TaskOutcome::Failed(Box::new(TaskFailure {
                        id: task.id.clone(),
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
            match cache.get(key) {
                Ok(Some(cached)) => {
                    let usage = cached
                        .usage
                        .clone()
                        .map(|usage| vec![usage])
                        .unwrap_or_default();
                    return TaskOutcome::Ok(Box::new(TaskResult {
                        id: task.id.clone(),
                        node: task.node.clone(),
                        output: crate::_internal::node::NodeOutput::from_cached(cached),
                        custom: Vec::new(),
                        messages: Vec::new(),
                        usage,
                        duration_millis: 0,
                        attempts: 0,
                        cached: true,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
                Ok(None) => {}
                Err(error) => {
                    return TaskOutcome::Failed(Box::new(TaskFailure {
                        id: task.id.clone(),
                        failure: NodeFailure::new(
                            &task.node,
                            format!("node cache lookup failed: {error}"),
                            1,
                        ),
                        usage: Vec::new(),
                        duration_millis: 0,
                        input: task.input.clone().map(Value::Object),
                    }));
                }
            }
        }

        let policy = self.retries.get(&task.node).cloned().unwrap_or_default();
        let max_attempts = policy.max_attempts.max(1);
        let started = Instant::now();
        let mut usage: Vec<Usage> = Vec::new();
        let mut last_error: Option<String> = None;
        let mut attempts_used = 0u32;

        for attempt in 1..=max_attempts {
            attempts_used = attempt as u32;
            let timeout_policy = self.timeout_of(&task.node).cloned().unwrap_or_default();
            let run_timeout = task
                .timeout_millis
                .map(Duration::from_millis)
                .or_else(|| timeout_policy.run_timeout());
            let context = AsyncContext::new(
                task.node.clone(),
                step,
                task.error.clone(),
                task.resume_history.clone(),
            )
            .with_run_timeout(run_timeout)
            .with_run_info(
                run.thread_id.clone(),
                run.namespace.clone(),
                run.store.clone(),
                run.context.clone(),
                run.collector.clone(),
            );
            let task_ops: std::sync::Arc<dyn crate::runtime::context::AsyncTaskOps> =
                std::sync::Arc::new(context.clone());
            let future: BoxNodeFuture = Box::pin(crate::runtime::context::with_async_task(
                task_ops,
                crate::runtime::context::with_async_node(
                    task.node.clone(),
                    Some(task.id.clone()),
                    (node.as_ref())(input.clone(), context.clone()),
                ),
            ));
            let idle_timeout = timeout_policy.idle_timeout();
            let result = if let Some(timeout) = run_timeout {
                match tokio::time::timeout(timeout, future).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout {
                        node: task.node.clone(),
                        millis: timeout.as_millis() as u64,
                    }),
                }
            } else if let Some(idle) = idle_timeout {
                run_with_idle_timeout(future, idle, context.progress_counter(), task.node.clone())
                    .await
            } else {
                future.await
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
                            return TaskOutcome::Failed(Box::new(TaskFailure {
                                id: task.id.clone(),
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
                    return TaskOutcome::Ok(Box::new(TaskResult {
                        id: task.id.clone(),
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
                        id: interrupt.id,
                        node: task.node.clone(),
                        value: interrupt.value,
                        response_schema: interrupt.response_schema,
                        task_id: Some(task.id.clone().into_boxed_str()),
                        static_pause: interrupt.static_pause,
                    });
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    if !policy.should_retry(&error.to_string()) {
                        break;
                    }
                    if attempt < max_attempts {
                        let backoff = policy.backoff_for(attempt);
                        if !backoff.is_zero() {
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
        }

        TaskOutcome::Failed(Box::new(TaskFailure {
            id: task.id.clone(),
            failure: NodeFailure::new(
                &task.node,
                last_error.unwrap_or_else(|| "node failed without an error message".to_string()),
                attempts_used,
            ),
            usage,
            duration_millis: started.elapsed().as_millis() as u64,
            input: task.input.clone().map(Value::Object),
        }))
    }
}
