//! Public run, stream, resume and command APIs.

use serde_json::Value;

use super::*;
use crate::_internal::command::Command;
use crate::_internal::error::{Error, Result};
use crate::_internal::state::{ChannelUpdate, State, apply_updates};
use crate::_internal::stream::StreamChunk;
use crate::graph::options::RunConfig;
use crate::pregel::{self, StartInput};

impl<S: State> CompiledGraph<S> {
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
        if command.goto.is_empty() && command.sends.is_empty() {
            // LangGraph starts a Command without `goto` at the normal entry points,
            // applying `update` first (over the latest checkpoint when thread-bound).
            let Some(updates) = command.update else {
                return Err(Error::validation(
                    "Command requires either `goto` or `update`",
                ));
            };
            if let Some(thread) = config.thread_id.as_deref() {
                return self.run_on_thread(thread, updates, config);
            }
            return pregel::execute_quiet(
                self,
                pregel::StartInput::Updates(updates.into_iter().collect()),
                config,
                &mut sink,
            );
        }
        // Seed from the latest checkpoint when the Command targets an existing
        // thread, matching LangGraph's `invoke(Command(goto=...), config)`.
        let mut values = match (config.thread_id.as_deref(), self.checkpointer()) {
            (Some(thread), Some(checkpointer)) => {
                let checkpoint = match &config.checkpoint_id {
                    Some(id) => checkpointer.get(thread, id),
                    None => checkpointer
                        .latest_in(thread, config.checkpoint_ns.as_deref().unwrap_or("")),
                };
                checkpoint
                    .map(|checkpoint| {
                        checkpoint
                            .map(|checkpoint| checkpoint.values)
                            .unwrap_or_default()
                    })
                    .unwrap_or_default()
            }
            _ => Map::new(),
        };
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
        pregel::execute_quiet(
            self,
            pregel::StartInput::Gateways {
                values,
                nodes: command.goto,
                sends: command.sends,
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

    /// Resume a thread and additionally schedule explicit `goto` targets.
    pub fn resume_with_updates_and_goto<I, K>(
        &self,
        thread_id: &str,
        value: Value,
        updates: I,
        goto: Vec<String>,
    ) -> Result<S>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        for node in &goto {
            if node != crate::END && !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "Command(goto=...): `{node}` is not a registered node"
                )));
            }
        }
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let config = RunConfig::new().with_thread(thread_id);
        let mut sink = |_chunk: StreamChunk| {};
        pregel::execute_quiet(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume: Some(value),
                updates: Some(updates),
                goto,
                sends: Vec::new(),
            },
            &config,
            &mut sink,
        )
        .map(|outcome| outcome.state)
    }

    /// Continue a thread using an explicit per-run config.
    pub fn continue_thread_with_config(&self, thread_id: &str, config: &RunConfig) -> Result<S> {
        self.resume_inner(thread_id, None, None, None, None, Some(config))
    }

    /// Resume with updates using an explicit per-run config.
    pub fn resume_with_updates_config<I, K>(
        &self,
        thread_id: &str,
        value: Value,
        updates: I,
        _sends: Vec<crate::_internal::node::Send>,
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
        self.resume_inner(
            thread_id,
            None,
            Some(value),
            Some(updates),
            config.checkpoint_ns.as_deref(),
            Some(config),
        )
    }

    /// Resume with updates and explicit goto targets using a per-run config.
    pub fn resume_with_updates_and_goto_config<I, K>(
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
        for node in &goto {
            if node != crate::END && !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "Command(goto=...): `{node}` is not a registered node"
                )));
            }
        }
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut config = config.clone();
        config.thread_id = Some(thread_id.to_string());
        let mut sink = |_chunk: StreamChunk| {};
        pregel::execute_quiet(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume: Some(value),
                updates: Some(updates),
                goto,
                sends,
            },
            &config,
            &mut sink,
        )
        .map(|outcome| outcome.state)
    }

    /// Continue with goto targets and no resume value using a per-run config.
    pub fn resume_continue_with_goto_config<I, K>(
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
        for node in &goto {
            if node != crate::END && !self.nodes.contains_key(node) {
                return Err(Error::validation(format!(
                    "Command(goto=...): `{node}` is not a registered node"
                )));
            }
        }
        let updates: Map<String, Value> = updates
            .into_iter()
            .map(|(channel, value)| (channel.into(), value))
            .collect();
        let mut config = config.clone();
        config.thread_id = Some(thread_id.to_string());
        let mut sink = |_chunk: StreamChunk| {};
        pregel::execute_quiet(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id: config.checkpoint_id.clone(),
                resume: None,
                updates: Some(updates),
                goto,
                sends,
            },
            &config,
            &mut sink,
        )
        .map(|outcome| outcome.state)
    }

    /// Resume a checkpointed thread with an explicit [`Command`].
    ///
    /// Runs `Command(goto=...)` targets, applying `Command.update` first; a
    /// `Command(resume=...)` value is handed to the paused task.
    pub fn resume_command(&self, thread_id: &str, command: Command) -> Result<S> {
        let updates = command
            .update
            .map(|updates| updates.into_iter().collect::<Map<String, Value>>());
        match (command.resume, command.goto.is_empty()) {
            (Some(value), false) => self.resume_with_updates_and_goto(
                thread_id,
                value,
                updates.unwrap_or_default(),
                command.goto,
            ),
            (None, false) => {
                self.resume_continue_with_goto(thread_id, updates.unwrap_or_default(), command.goto)
            }
            (resume, _) => self.resume_inner(thread_id, None, resume, updates, None, None),
        }
    }

    /// Shared resume implementation for the public resume helpers.
    pub(crate) fn resume_inner(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        resume: Option<Value>,
        updates: Option<Map<String, Value>>,
        namespace: Option<&str>,
        overrides: Option<&RunConfig>,
    ) -> Result<S> {
        let mut config = overrides.cloned().unwrap_or_default();
        config.thread_id = Some(thread_id.to_string());
        if let Some(namespace) = namespace {
            config = config.with_checkpoint_ns(namespace);
        }
        let mut sink = |_chunk: StreamChunk| {};
        let checkpoint_id = checkpoint_id
            .map(str::to_string)
            .or_else(|| config.checkpoint_id.clone());
        pregel::execute_quiet(
            self,
            StartInput::Resume {
                thread_id: thread_id.to_string(),
                checkpoint_id,
                resume,
                updates,
                goto: Vec::new(),
                sends: Vec::new(),
            },
            &config,
            &mut sink,
        )
        .map(|outcome| outcome.state)
    }
}
