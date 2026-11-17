//! Public run, stream, resume and command APIs.

use serde_json::Value;

use super::*;
use crate::kernel::command::Command;
use crate::kernel::error::{Error, Result};
use crate::kernel::executor::{self, StartInput};
use crate::kernel::graph::options::RunConfig;
use crate::kernel::state::{State, state_to_values};
use crate::kernel::stream::{StreamChunk, StreamMode};
use crate::kernel::usage::RunOutcome;

impl<S: State> CompiledGraph<S> {
    pub fn invoke(&self, input: &S, config: &RunConfig) -> Result<S> {
        let values = state_to_values(input)?;
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute_quiet(self, StartInput::Values(values), config, &mut sink)
            .map(|outcome| outcome.state)
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
        executor::execute_quiet(self, StartInput::Updates(values), config, &mut sink)
            .map(|outcome| outcome.state)
    }

    /// Run the graph and return the final state together with a consumption report.
    ///
    /// The report contains one [`NodeExecution`](crate::NodeExecution) per node run
    /// (wall-clock duration plus any usage the node reported through
    /// [`record_usage`](crate::record_usage) or
    /// [`NodeOutput::with_usage`](crate::NodeOutput::with_usage)).
    pub fn run(&self, input: &S, config: &RunConfig) -> Result<RunOutcome<S>> {
        let values = state_to_values(input)?;
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute_quiet(self, StartInput::Values(values), config, &mut sink)
    }

    /// Run the graph from a partial state patch and return state plus report.
    pub fn run_updates<I, K>(&self, updates: I, config: &RunConfig) -> Result<RunOutcome<S>>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let values: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute_quiet(self, StartInput::Updates(values), config, &mut sink)
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
            .map(|outcome| outcome.state)
    }

    /// Invoke the graph over several inputs, respecting `RunConfig::max_concurrency`.
    ///
    /// Results are returned in input order; a failed invocation does not abort the batch.
    pub fn batch<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<S>>
    where
        I: IntoIterator<Item = S>,
    {
        let inputs: Vec<S> = inputs.into_iter().collect();
        let max_concurrency = config.max_concurrency.unwrap_or(1).max(1);
        if max_concurrency <= 1 || inputs.len() <= 1 {
            return inputs
                .into_iter()
                .map(|input| self.invoke(&input, config))
                .collect();
        }
        let mut results: Vec<Option<Result<S>>> = (0..inputs.len()).map(|_| None).collect();
        let chunks: Vec<Vec<(usize, S)>> = inputs
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .chunks(max_concurrency)
            .map(|chunk| chunk.to_vec())
            .collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    scope.spawn(|| {
                        chunk
                            .into_iter()
                            .map(|(index, input)| (index, self.invoke(&input, config)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for handle in handles {
                for (index, result) in handle.join().expect("batch worker panicked") {
                    results[index] = Some(result);
                }
            }
        });
        results
            .into_iter()
            .map(|result| result.expect("every batch entry produced a result"))
            .collect()
    }

    /// Batch variant of [`CompiledGraph::run`] that also returns run reports.
    pub fn batch_run<I>(&self, inputs: I, config: &RunConfig) -> Vec<Result<RunOutcome<S>>>
    where
        I: IntoIterator<Item = S>,
    {
        inputs
            .into_iter()
            .map(|input| self.run(&input, config))
            .collect()
    }

    /// Stream only the selected LangGraph stream modes.
    pub fn stream_modes(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
    ) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        self.stream_modes_with(input, config, modes, |chunk| chunks.push(chunk))?;
        Ok(chunks)
    }

    /// Stream selected modes while forwarding matching chunks to `sink`.
    ///
    /// `debug` includes values, updates, tasks and checkpoint records; usage and
    /// interrupt chunks are always forwarded.
    pub fn stream_modes_with<F>(
        &self,
        input: &S,
        config: &RunConfig,
        modes: &[StreamMode],
        mut sink: F,
    ) -> Result<S>
    where
        F: FnMut(StreamChunk),
    {
        let modes = modes.to_vec();
        let values = state_to_values(input)?;
        let mut filtered = |chunk: StreamChunk| {
            if chunk.matches_modes(&modes) {
                sink(chunk);
            }
        };
        executor::execute(self, StartInput::Values(values), config, &mut filtered)
            .map(|outcome| outcome.state)
    }

    /// Continue an interrupted thread by handing `value` to the pending `interrupt()`.
    pub fn resume(&self, thread_id: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, None, None)
    }

    /// Resume an interrupted thread bound to a checkpoint namespace.
    pub fn resume_in(&self, thread_id: &str, namespace: &str, value: Value) -> Result<S> {
        self.resume_inner(thread_id, None, Some(value), None, Some(namespace), None)
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
        self.resume_inner(thread_id, None, Some(value), Some(updates), None, None)
    }

    /// Resume an interrupted namespace-bound thread after applying state writes.
    pub fn resume_with_updates_in<I, K>(
        &self,
        thread_id: &str,
        namespace: &str,
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
        self.resume_inner(
            thread_id,
            None,
            Some(value),
            Some(updates),
            Some(namespace),
            None,
        )
    }

    /// Resume from a specific checkpoint (time travel).
    pub fn resume_from(&self, thread_id: &str, checkpoint_id: &str, value: Value) -> Result<S> {
        self.resume_inner(
            thread_id,
            Some(checkpoint_id),
            Some(value),
            None,
            None,
            None,
        )
    }

    /// Continue a thread that stopped on an `interrupt_before` / `interrupt_after` hook.
    pub fn continue_thread(&self, thread_id: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, None, None)
    }

    /// Continue a namespace-bound thread paused by an interrupt hook.
    pub fn continue_thread_in(&self, thread_id: &str, namespace: &str) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, Some(namespace), None)
    }

    /// Resume a namespace-bound thread while forwarding chunks to `sink`.
    pub(crate) fn resume_in_streaming(
        &self,
        thread_id: &str,
        namespace: &str,
        value: Value,
        overrides: Option<&RunConfig>,
        sink: &mut dyn FnMut(StreamChunk),
    ) -> Result<S> {
        let mut config = RunConfig::new()
            .with_thread(thread_id)
            .with_checkpoint_ns(namespace);
        if let Some(overrides) = overrides {
            config.store = overrides.store.clone();
            config.context = overrides.context.clone();
            config.stream_subgraphs = true;
        }
        executor::execute_quiet(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: None,
                resume: Some(value),
                updates: None,
            },
            &config,
            sink,
        )
        .map(|outcome| outcome.state)
    }

    /// Resume a namespace-bound thread, inheriting store/context from `overrides`.
    pub fn resume_in_with_run_config(
        &self,
        thread_id: &str,
        namespace: &str,
        value: Value,
        overrides: &RunConfig,
    ) -> Result<S> {
        self.resume_inner(
            thread_id,
            None,
            Some(value),
            None,
            Some(namespace),
            Some(overrides),
        )
    }

    /// Resume a checkpointed thread with an explicit [`Command`].
    ///
    /// This is the builder-friendly form of
    /// [`CompiledGraph::resume_with_updates`] and
    /// [`CompiledGraph::continue_thread`]:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use langgraph_rs::{
    /// #     Command, CompileOptions, InMemorySaver, NodeOutput, RunConfig, State, StateGraph,
    /// #     StateSchema,
    /// # };
    /// # use serde::{Deserialize, Serialize};
    /// #
    /// # #[derive(Clone, Default, Debug, Serialize, Deserialize)]
    /// # struct Demo {
    /// #     value: i64,
    /// # }
    /// #
    /// # impl State for Demo {
    /// #     fn schema() -> StateSchema {
    /// #         StateSchema::new().last_value("value")
    /// #     }
    /// # }
    /// let graph = StateGraph::<Demo>::new()
    ///     .add_node("ask", |_state: &Demo| {
    ///         let answer = langgraph_rs::interrupt(serde_json::json!({ "question": "?" }))?;
    ///         Ok(NodeOutput::new().set("value", answer.as_i64().unwrap_or_default()))
    ///     })
    ///     .set_entry_point("ask")
    ///     .set_finish_point("ask")
    ///     .compile_with(CompileOptions::new().with_checkpointer(Arc::new(InMemorySaver::new())))
    ///     .unwrap();
    ///
    /// let config = RunConfig::new().with_thread("demo");
    /// let _ = graph.invoke(&Demo::default(), &config).unwrap_err();
    ///
    /// let command = Command::resume(serde_json::json!(41));
    /// let state = graph.resume_command("demo", command).unwrap();
    /// assert_eq!(state.value, 41);
    /// ```
    /// Run a graph starting at the `Command`'s `goto` nodes (LangGraph
    /// `invoke(Command(goto=...))`), applying `Command.update` first.
    pub fn run_command(
        &self,
        command: Command,
        config: &RunConfig,
    ) -> Result<crate::RunOutcome<S>> {
        if command.resume.is_some() {
            return Err(Error::validation(
                "Command(goto=...) cannot also carry a resume value",
            ));
        }
        if command.parent {
            return Err(Error::validation(
                "Command(graph=PARENT) is only supported for subgraphs",
            ));
        }
        let mut sink = |_chunk: StreamChunk| {};
        if command.goto.is_empty() {
            // LangGraph starts a Command without `goto` at the normal entry points,
            // applying `update` first.
            let Some(updates) = command.update else {
                return Err(Error::validation(
                    "Command requires either `goto` or `update`",
                ));
            };
            return executor::execute_quiet(
                self,
                executor::StartInput::Updates(updates.into_iter().collect()),
                config,
                &mut sink,
            );
        }
        let mut values = Map::new();
        if let Some(updates) = command.update {
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
        }
        executor::execute_quiet(
            self,
            executor::StartInput::Gateways {
                values,
                nodes: command.goto,
            },
            config,
            &mut sink,
        )
    }

    /// Convenience state-only form of [`Self::run_command`].
    pub fn invoke_command(&self, command: Command, config: &RunConfig) -> Result<S> {
        self.run_command(command, config)
            .map(|outcome| outcome.state)
    }

    pub fn resume_command(&self, thread_id: &str, command: Command) -> Result<S> {
        let updates = command
            .update
            .map(|updates| updates.into_iter().collect::<Map<String, Value>>());
        self.resume_inner(thread_id, None, command.resume, updates, None, None)
    }

    fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        namespace: Option<&str>,
        overrides: Option<&RunConfig>,
    ) -> Result<S> {
        let mut config = RunConfig::new().with_thread(thread_id);
        if let Some(namespace) = namespace {
            config = config.with_checkpoint_ns(namespace);
        }
        if let Some(overrides) = overrides {
            config.store = overrides.store.clone();
            config.context = overrides.context.clone();
        }
        let mut sink = |_chunk: StreamChunk| {};
        executor::execute_quiet(
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
        .map(|outcome| outcome.state)
    }
}
