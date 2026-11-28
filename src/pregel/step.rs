//! Shared prologue for one Pregel superstep.

use serde_json::{Map, Value};

use crate::_internal::error::{Error, Result};
use crate::_internal::node::Task;
use crate::_internal::state::{State, StateSchema};
use crate::RunControl;

/// Named barriers that are currently open (all writers seen).
///
/// Captured before a superstep so the barrier a superstep opens stays open for the
/// node that reads it; only a barrier that was already open is consumed afterwards.
pub(crate) fn open_named_barriers(
    schema: &StateSchema,
    values: &Map<String, Value>,
) -> Vec<String> {
    schema
        .names()
        .filter(|name| {
            schema
                .channel(name)
                .is_some_and(crate::channels::Channel::is_named_barrier)
                && values.get(*name) == Some(&Value::Bool(true))
        })
        .map(str::to_owned)
        .collect()
}

/// Consume named barriers that this superstep read (LangGraph `NamedBarrierValue.consume`).
///
/// Only barriers in `armed` are cleared, so a barrier that opened during this superstep
/// remains visible to the nodes it schedules. Returns `true` when a channel changed.
pub(crate) fn consume_named_barriers(
    values: &mut Map<String, Value>,
    armed: &[String],
    written: &std::collections::HashSet<String>,
) -> bool {
    let mut changed = false;
    for name in armed {
        // A barrier re-armed by this superstep's own writes must stay open, matching
        // LangGraph, which consumes before applying the new writes.
        if written.contains(name) {
            continue;
        }
        if values.get(name) == Some(&Value::Bool(true)) {
            // LangGraph raises `EmptyChannelError` after consume; dropping the key makes
            // typed state fall back to its default, which is the closest analogue.
            values.remove(name);
            changed = true;
        }
    }
    changed
}

/// State produced by the shared superstep prologue.
pub(crate) struct StepStart {
    pub(crate) iteration: usize,
    pub(crate) paused: Option<Task>,
    pub(crate) guard: crate::runtime::context::StepGuard,
}

/// Apply abort/recursion/step-context/managed-value handling and find an
/// `interrupt_before` pause. The caller owns the guard for the step's lifetime.
#[allow(clippy::too_many_arguments)]
pub(crate) fn begin_step<S: State>(
    control: Option<&RunControl>,
    schema: &StateSchema,
    values: &mut Map<String, Value>,
    state: &mut S,
    step: usize,
    recursion_limit: usize,
    iteration: usize,
    tasks: &[Task],
    skip_interrupt_before: bool,
    is_before: impl Fn(&str) -> bool,
) -> Result<StepStart> {
    if let Some(reason) = control.and_then(RunControl::drain_reason) {
        return Err(crate::_internal::error::Error::Drained { reason });
    }
    if control.is_some_and(RunControl::is_aborted) {
        return Err(Error::other("run aborted by RunControl"));
    }
    crate::pregel::apply_managed_values(schema, values, state, iteration, recursion_limit)?;
    if iteration >= recursion_limit {
        return Err(Error::RecursionLimit(recursion_limit));
    }
    // `current_step` mirrors LangGraph's `langgraph_step` metadata, which is one below
    // our internal checkpoint counter.
    let guard =
        crate::runtime::context::install_step(step.saturating_sub(1), iteration, recursion_limit);
    let paused = if skip_interrupt_before {
        None
    } else {
        tasks.iter().find(|task| is_before(&task.node)).cloned()
    };
    Ok(StepStart {
        iteration: iteration + 1,
        paused,
        guard,
    })
}

/// Guards and shared buffers that scope one run.
pub(crate) struct RunSetup {
    _run: crate::runtime::context::RunGuard,
    collector: Option<std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::stream::StreamChunk>>>>,
    _collector: Option<crate::runtime::context::CollectorGuard>,
}

impl RunSetup {
    pub(crate) fn collector(
        &self,
    ) -> Option<&std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::stream::StreamChunk>>>> {
        self.collector.as_ref()
    }
}

/// Validate context, install the run scope and optional subgraph chunk collector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn begin_run(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn crate::store::BaseStore>>,
    context: Option<Value>,
    context_schema: Option<&Value>,
    stream_subgraphs: bool,
    collector: Option<std::sync::Arc<std::sync::Mutex<Vec<crate::_internal::stream::StreamChunk>>>>,
    live_sink: Option<crate::runtime::context::LiveSink>,
) -> Result<RunSetup> {
    // A new run must not inherit a parent command left by an unconsumed goto_parent.
    crate::runtime::context::clear_parent_command();
    // LangGraph validates the Python object itself; a context the engine carries by reference (no
    // JSON form) has already been checked by Python, so there is nothing left to validate here.
    let stashed = context
        .as_ref()
        .map(crate::interop::is_stashed_object)
        .unwrap_or(false);
    if let (Some(schema), Some(context)) = (context_schema, context.as_ref())
        && !stashed
    {
        crate::runtime::context::validate_response(context, Some(schema)).map_err(|error| {
            Error::validation(format!("context does not match context_schema: {error}"))
        })?;
    }
    let run = crate::runtime::context::install_run(thread_id, namespace, store, context, live_sink);
    let collector = collector.or_else(|| {
        stream_subgraphs.then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
    });
    let collector_guard = collector
        .as_ref()
        .map(|collector| crate::runtime::context::install_subgraph_collector(collector.clone()));
    Ok(RunSetup {
        _run: run,
        collector,
        _collector: collector_guard,
    })
}
