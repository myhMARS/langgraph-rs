//! Shared prologue for one Pregel superstep.

use serde_json::{Map, Value};

use crate::RunControl;
use crate::kernel::error::{Error, Result};
use crate::kernel::node::Task;
use crate::kernel::state::{State, StateSchema};

/// State produced by the shared superstep prologue.
pub(crate) struct StepStart {
    pub(crate) iteration: usize,
    pub(crate) paused: Option<Task>,
    pub(crate) guard: crate::application::context::StepGuard,
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
    if control.is_some_and(RunControl::is_aborted) {
        return Err(Error::other("run aborted by RunControl"));
    }
    crate::application::executor::apply_managed_values(
        schema,
        values,
        state,
        step,
        recursion_limit,
    )?;
    if iteration >= recursion_limit {
        return Err(Error::RecursionLimit(recursion_limit));
    }
    let guard = crate::application::context::install_step(step, recursion_limit);
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
    _run: crate::application::context::RunGuard,
    collector: Option<std::sync::Arc<std::sync::Mutex<Vec<crate::kernel::stream::StreamChunk>>>>,
    _collector: Option<crate::application::context::CollectorGuard>,
}

impl RunSetup {
    pub(crate) fn collector(
        &self,
    ) -> Option<&std::sync::Arc<std::sync::Mutex<Vec<crate::kernel::stream::StreamChunk>>>> {
        self.collector.as_ref()
    }
}

/// Validate context, install the run scope and optional subgraph chunk collector.
pub(crate) fn begin_run(
    thread_id: Option<String>,
    namespace: String,
    store: Option<std::sync::Arc<dyn crate::kernel::store::BaseStore>>,
    context: Option<Value>,
    context_schema: Option<&Value>,
    stream_subgraphs: bool,
    collector: Option<std::sync::Arc<std::sync::Mutex<Vec<crate::kernel::stream::StreamChunk>>>>,
) -> Result<RunSetup> {
    // A new run must not inherit a parent command left by an unconsumed goto_parent.
    crate::application::context::clear_parent_command();
    if let (Some(schema), Some(context)) = (context_schema, context.as_ref()) {
        crate::application::context::validate_response(context, Some(schema)).map_err(|error| {
            Error::validation(format!("context does not match context_schema: {error}"))
        })?;
    }
    let run = crate::application::context::install_run(thread_id, namespace, store, context);
    let collector = collector.or_else(|| {
        stream_subgraphs.then(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
    });
    let collector_guard = collector.as_ref().map(|collector| {
        crate::application::context::install_subgraph_collector(collector.clone())
    });
    Ok(RunSetup {
        _run: run,
        collector,
        _collector: collector_guard,
    })
}
