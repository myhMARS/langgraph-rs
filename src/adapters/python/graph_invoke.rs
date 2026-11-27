//! `CompiledGraph.invoke` implementation, kept out of the PyO3 method block.

use super::*;

/// Run the graph and return the final state dict.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invoke(
    py: Python<'_>,
    graph: &crate::graph::CompiledGraph<crate::JsonState>,
    input: Option<&Bound<'_, PyAny>>,
    thread_id: Option<&Bound<'_, PyAny>>,
    context: Option<&Bound<'_, PyAny>>,
    control: Option<PyRef<'_, crate::adapters::python::types::PyRunControl>>,
    checkpoint_ns: Option<String>,
    config: Option<&Bound<'_, PyAny>>,
    lifecycle: Option<&crate::_internal::lifecycle::LifecycleSink>,
) -> PyResult<Py<PyAny>> {
    let (thread_id, base_config) =
        crate::adapters::python::convert::config_from_py(py, config.or(thread_id))?;
    // Install the LangChain callback context once for the whole run: nodes on this thread
    // then skip their per-node set/reset pair. A plain `invoke` does not stream model tokens.
    crate::adapters::python::convert::set_stream_tokens(false);
    let _callback_scope = crate::adapters::python::convert::begin_callback_scope(py);
    let mut invoke_config = base_config.clone();
    if let Some(namespace) = checkpoint_ns.clone() {
        invoke_config = invoke_config.with_checkpoint_ns(namespace);
    }
    // Captured after the explicit `checkpoint_ns` override is applied.
    let static_ns = invoke_config.checkpoint_ns.clone();
    if let Some(control) = control.as_ref() {
        invoke_config = invoke_config.with_control(control.inner.clone());
    }
    if let Some(context) = context {
        invoke_config = invoke_config.with_context(py_to_value(context)?);
    }
    if let Some(lifecycle) = lifecycle {
        invoke_config = invoke_config.with_lifecycle_sink(lifecycle.clone());
    }
    if let Some(obj) = input {
        if let Ok(command) = obj.cast::<PyCommand>().map(|cell| cell.borrow()) {
            let (targets, sends) = command.goto_parts(py)?;
            if (!targets.is_empty() || !sends.is_empty()) && command.resume.is_none() {
                let updates = command.to_update_map(py)?;
                let config = invoke_config.clone();
                let thread = thread_id.clone();
                let command = crate::Command {
                    resume: None,
                    update: Some(updates.clone()),
                    goto: targets.clone(),
                    sends: sends.clone(),
                    parent: false,
                };
                let exists = thread
                    .as_deref()
                    .map(|thread| {
                        match config.checkpoint_ns.as_deref() {
                            Some(namespace) => graph.has_thread_in(thread, namespace),
                            None => graph.has_thread(thread),
                        }
                        .unwrap_or(false)
                    })
                    .unwrap_or(false);
                let result = py.detach(|| {
                    if let (true, Some(thread)) = (exists && sends.is_empty(), thread.as_deref()) {
                        graph.resume_continue_with_goto_config(
                            thread,
                            updates.clone(),
                            targets.clone(),
                            Vec::new(),
                            &config,
                        )
                    } else {
                        graph
                            .run_command(command, &config)
                            .map(|outcome| outcome.state)
                    }
                });
                return match result {
                    Ok(state) => json_state_to_py(py, &state),
                    Err(error) => match crate::adapters::python::graph_pause::pause_values(
                        py,
                        graph,
                        thread_id.as_deref(),
                        static_ns.as_deref(),
                        &error,
                        config.lifecycle.as_ref(),
                )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    },
                };
            }
            if thread_id.is_none() && command.resume.is_none() {
                let updates = command.to_update_map(py)?;
                if updates.is_empty() {
                    return Err(PyTypeError::new_err(
                        "Command requires either `goto` or `update`",
                    ));
                }
                let updates = updates.into_iter().collect();
                let command = crate::Command {
                    resume: None,
                    update: Some(updates),
                    goto: Vec::new(),
                    sends: Vec::new(),
                    parent: false,
                };
                let config = invoke_config.clone();
                let result = py.detach(|| graph.run_command(command, &config));
                return match result {
                    Ok(outcome) => json_state_to_py(py, &outcome.state),
                    Err(error) => match crate::adapters::python::graph_pause::pause_values(
                        py,
                        graph,
                        thread_id.as_deref(),
                        static_ns.as_deref(),
                        &error,
                        config.lifecycle.as_ref(),
                )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    },
                };
            }
            let thread = thread_id
                .as_deref()
                .ok_or_else(|| PyTypeError::new_err("invoke(Command(...)) requires thread_id"))?;
            let (resume, updates) = command.to_resume_command(py)?;
            let (goto, sends) = command.goto_parts(py)?;
            let resume_config = invoke_config.clone();
            let resume_checkpoint = thread_id
                .as_deref()
                .and_then(|thread| match static_ns.as_deref() {
                    Some(namespace) => graph.get_state_in(thread, namespace).ok().flatten(),
                    None => graph.get_state(thread).ok().flatten(),
                })
                .map(|snapshot| snapshot.checkpoint_id)
                .unwrap_or_default();
            crate::adapters::python::graph_pause::record_resume(
                resume_config.lifecycle.as_ref(),
                &resume_checkpoint,
                static_ns.as_deref().unwrap_or(""),
            );
            // LangGraph starts a fresh thread when a `Command` arrives for one: `resume` is then
            // ignored and `update` becomes the input.
            let has_checkpoint = match static_ns.as_deref() {
                Some(namespace) => graph.get_state_in(thread, namespace).ok().flatten(),
                None => graph.get_state(thread).ok().flatten(),
            }
            .is_some();
            if !has_checkpoint {
                // `update` becomes the initial input, as LangGraph does.
                let input_state = py.detach(|| {
                    let mut state = crate::JsonState::new();
                    for (channel, value) in updates.clone() {
                        state = state.with(channel, value);
                    }
                    graph.invoke(&state, &resume_config)
                });
                return match input_state {
                    Ok(state) => json_state_to_py(py, &state),
                    Err(error) => match crate::adapters::python::graph_pause::pause_values(
                        py,
                        graph,
                        thread_id.as_deref(),
                        static_ns.as_deref(),
                        &error,
                        resume_config.lifecycle.as_ref(),
                    )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    },
                };
            }
            let result = py.detach(|| match resume {
                Some(value) if goto.is_empty() && sends.is_empty() => graph
                    .resume_with_updates_config(thread, value, updates, Vec::new(), &resume_config),
                Some(value) => graph.resume_with_updates_and_goto_config(
                    thread,
                    value,
                    updates,
                    goto,
                    sends,
                    &resume_config,
                ),
                None => graph.resume_continue_with_goto_config(
                    thread,
                    updates,
                    goto,
                    sends,
                    &resume_config,
                ),
            });
            return match result {
                Ok(state) => json_state_to_py(py, &state),
                Err(error) => match crate::adapters::python::graph_pause::pause_values(
                    py,
                    graph,
                    thread_id.as_deref(),
                    static_ns.as_deref(),
                    &error,
                    resume_config.lifecycle.as_ref(),
                )? {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                },
            };
        }
    }
    let config = invoke_config;
    let input_is_none = input.map(|value| value.is_none()).unwrap_or(true);
    if input_is_none {
        if let Some(thread) = thread_id.as_deref() {
            if let Some(snapshot) = match static_ns.as_deref() {
                Some(namespace) => graph.get_state_in(thread, namespace),
                None => graph.get_state(thread),
            }
            .ok()
            .flatten()
            {
                crate::adapters::python::graph_pause::record_resume(
                    config.lifecycle.as_ref(),
                    &snapshot.checkpoint_id,
                    static_ns.as_deref().unwrap_or(""),
                );
            }
            let result = py.detach(|| graph.continue_thread_with_config(thread, &config));
            return match result {
                Ok(state) => json_state_to_py(py, &state),
                Err(error) => match crate::adapters::python::graph_pause::pause_values(
                    py,
                    graph,
                    thread_id.as_deref(),
                    static_ns.as_deref(),
                    &error,
                    config.lifecycle.as_ref(),
                )? {
                    Some(value) => Ok(value),
                    None => Err(to_pyerr(py, error)),
                },
            };
        }
    }
    if let (Some(obj), Some(thread)) = (input, thread_id.as_deref()) {
        let exists = match config.checkpoint_ns.as_deref() {
            Some(namespace) => graph.has_thread_in(thread, namespace),
            None => graph.has_thread(thread),
        }
        .unwrap_or(false);
        if exists {
            let updates = py_to_value(obj)?;
            if let Some(map) = updates.as_object() {
                let updates: Vec<(String, crate::Value)> = map
                    .iter()
                    .map(|(channel, value)| (channel.clone(), value.clone()))
                    .collect();
                let result = py.detach(|| graph.run_on_thread(thread, updates, &config));
                return match result {
                    Ok(outcome) => json_state_to_py(py, &outcome.state),
                    Err(error) => match crate::adapters::python::graph_pause::pause_values(
                        py,
                        graph,
                        thread_id.as_deref(),
                        static_ns.as_deref(),
                        &error,
                        config.lifecycle.as_ref(),
                )? {
                        Some(value) => Ok(value),
                        None => Err(to_pyerr(py, error)),
                    },
                };
            }
        }
    }
    let state = state_from_py(input)?;
    let result = py.detach(|| graph.invoke(&state, &config));
    match result {
        Ok(state) => json_state_to_py(py, &state),
        Err(error) => match crate::adapters::python::graph_pause::pause_values(
            py,
            graph,
            thread_id.as_deref(),
            static_ns.as_deref(),
            &error,
            config.lifecycle.as_ref(),
        )? {
            Some(value) => Ok(value),
            None => Err(to_pyerr(py, error)),
        },
    }
}
