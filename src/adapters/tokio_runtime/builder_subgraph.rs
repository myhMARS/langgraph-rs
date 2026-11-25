//! Async subgraph execution and build-time validation.

use super::*;
impl<S: State> AsyncStateGraph<S> {
    pub(crate) fn validate(&self) -> Result<()> {
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
        if let Some(edge) = &self.conditional_entry
            && let Some(path_map) = &edge.path_map
        {
            for (key, targets) in path_map.entries() {
                for target in targets {
                    if target == START {
                        return Err(Error::validation(format!(
                            "conditional entry point maps key `{key}` to `{START}`, which is not a valid destination"
                        )));
                    }
                    self.require_target(START, target)?;
                }
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

    pub(crate) fn require_node(&self, what: &str, name: &str) -> Result<()> {
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

    pub(crate) fn require_target(&self, source: &str, target: &str) -> Result<()> {
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

pub(crate) fn child_namespace(parent: &str, name: &str) -> String {
    crate::graph::subgraph_namespace(
        parent,
        name,
        crate::runtime::context::current_task_id().as_deref(),
    )
}

pub(crate) async fn run_async_subgraph<S: State>(
    child: Arc<AsyncCompiledGraph<S>>,
    state: S,
    name: String,
    context: AsyncContext,
    schema: &crate::_internal::state::StateSchema,
) -> Result<NodeOutput> {
    let before = state_to_values(&state)?;
    let thread = context.thread_id().map(str::to_string);
    let parent_namespace = context.namespace();
    let namespace = child_namespace(parent_namespace, &name);
    // Resume only a child that actually paused in this namespace; otherwise a parent
    // resume value would reload a sibling subgraph's stale checkpoint.
    let base = namespace
        .rsplit_once(':')
        .map(|(head, _)| head)
        .unwrap_or(&namespace);
    let paused_in = match thread.as_deref() {
        Some(thread) => child.paused_namespace_in(thread, base).await,
        None => None,
    };
    let resume_namespace = paused_in.clone().unwrap_or_else(|| namespace.clone());

    let resume = match thread.as_deref() {
        Some(_) if paused_in.is_some() => context.take_resume().into_iter().next_back(),
        _ => None,
    };
    let mut config = RunConfig::new().with_checkpoint_ns(namespace.clone());
    if let Some(store) = context.store() {
        config = config.with_store(store);
    }
    if let Some(user_context) = context.context() {
        config = config.with_context(user_context.clone());
    }
    let collector = context.collector();
    let (output, parent_command) =
        crate::adapters::tokio_runtime::context::run_with_parent_scope(async {
            if let Some(collector) = collector {
                config = config.with_stream_subgraphs(true);
                let chunk_namespace = namespace.clone();
                let mut sink = move |chunk: StreamChunk| {
                    let wrapped = match chunk {
                        StreamChunk::Subgraph {
                            namespace: inner,
                            chunk,
                        } => StreamChunk::Subgraph {
                            namespace: crate::graph::subgraph_chunk_namespace(
                                &chunk_namespace,
                                &inner,
                            ),
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
                        .resume_in_streaming(&thread, &resume_namespace, resume, Some(&config), &mut sink)
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
                child.resume_in(&thread, &resume_namespace, resume).await
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
    // Propagate a child failure even if another task recorded a parent command.
    let output = output?;
    if let Some(command) = parent_command {
        return crate::graph::parent_command_output(command, schema);
    }
    let after = state_to_values(&output)?;
    let mut result = NodeOutput::new();
    for (channel, value) in after {
        if before.get(&channel) != Some(&value) {
            result = result.replace(channel, value);
        }
    }
    Ok(result)
}
