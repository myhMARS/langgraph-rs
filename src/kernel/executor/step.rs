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
    pub(crate) guard: crate::kernel::context::StepGuard,
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
    crate::kernel::executor::apply_managed_values(schema, values, state, step, recursion_limit)?;
    if iteration >= recursion_limit {
        return Err(Error::RecursionLimit(recursion_limit));
    }
    let guard = crate::kernel::context::install_step(step, recursion_limit);
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
