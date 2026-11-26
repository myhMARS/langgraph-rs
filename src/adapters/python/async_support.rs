//! Async Python binding helpers (timeouts, snapshot steps, subgraph embedding).

use pyo3::types::PyAny;

use super::*;

pub(crate) fn wrap_async_timeout(
    py: Python<'_>,
    function: Py<PyAny>,
    seconds: f64,
) -> PyResult<Py<PyAny>> {
    let locals = pyo3::types::PyDict::new(py);
    locals.set_item("fn", function.bind(py))?;
    locals.set_item("seconds", seconds.max(0.0))?;
    py.run(
        c"async def _timeout_wrapper(state, fn=fn, seconds=seconds):
    import asyncio
    return await asyncio.wait_for(fn(state), seconds)
wrapper = _timeout_wrapper",
        None,
        Some(&locals),
    )?;
    locals
        .get_item("wrapper")?
        .map(|value| value.unbind())
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("timeout wrapper missing"))
}

thread_local! {
    /// One asyncio loop per OS thread. Creating a loop costs ~1 ms, far more than
    /// running a node coroutine, so the loop outlives individual node calls.
    static NODE_LOOP: std::cell::RefCell<Option<Py<PyAny>>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn run_timed_awaitable<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    NODE_LOOP.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let asyncio = py.import("asyncio")?;
            let loop_object = asyncio.getattr("new_event_loop")?.call0()?;
            asyncio.getattr("set_event_loop")?.call1((&loop_object,))?;
            *slot = Some(loop_object.unbind());
        }
        let loop_object = slot
            .as_ref()
            .expect("loop installed above")
            .bind(py);
        loop_object.call_method1("run_until_complete", (value,))
    })
}

/// Attach child subgraph snapshots to the root snapshot (task-id aware).
pub(crate) fn embed_subgraph_states(
    py: Python<'_>,
    root: &crate::StateSnapshot<JsonState>,
    subgraphs: &[crate::StateSnapshot<JsonState>],
) -> PyResult<Py<PyAny>> {
    crate::adapters::python::graph::embed_subgraph_states(py, root, subgraphs)
}

/// Async counterpart of [`super::graph_support::static_pause_values`]: a static
/// `interrupt_before`/`interrupt_after` pause returns the current state instead of raising.
pub(crate) fn static_pause_values(
    py: Python<'_>,
    graph: &crate::adapters::tokio_runtime::AsyncCompiledGraph<crate::JsonState>,
    thread: Option<&str>,
    namespace: Option<&str>,
    error: &crate::Error,
    py_config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Py<PyAny>>> {
    let crate::Error::Interrupt(interrupt) = error else {
        return Ok(None);
    };
    let snapshot = match thread {
        Some(thread) => match namespace {
            Some(namespace) => graph.get_state_in(thread, namespace),
            None => graph.get_state(thread),
        }
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?,
        None => None,
    };
    match snapshot {
        Some(snapshot) => {
            let values =
                crate::adapters::python::convert::json_state_to_py(py, &snapshot.values)?;
            if !interrupt.static_pause {
                crate::adapters::python::graph_pause::attach_interrupts(
                    py,
                    values.bind(py),
                    std::slice::from_ref(interrupt),
                )?;
            }
            crate::adapters::python::graph_pause::dispatch_interrupt(
                py,
                py_config,
                &snapshot.checkpoint_id,
                &snapshot.interrupts,
            )?;
            Ok(Some(values))
        }
        None => match &interrupt.values {
            Some(values) => {
                let values = crate::adapters::python::convert::json_state_to_py(
                    py,
                    &crate::JsonState(values.clone()),
                )?;
                if !interrupt.static_pause {
                    crate::adapters::python::graph_pause::attach_interrupts(
                        py,
                        values.bind(py),
                        std::slice::from_ref(interrupt),
                    )?;
                }
                Ok(Some(values))
            }
            None => Ok(Some(py.None())),
        },
    }
}

/// Wrap a **synchronous** result as an awaitable.
///
/// `pyo3_async_runtimes::tokio::future_into_py` schedules a Tokio task and wakes the asyncio
/// loop for every call (~0.3 ms), which dwarfs reading a checkpoint out of memory. State
/// reads are synchronous in this engine, so evaluate them eagerly and hand back an
/// already-resolved ``asyncio.Future``; fall back to a Tokio future when no loop is running.
pub(crate) fn immediate_awaitable<'py, F>(py: Python<'py>, work: F) -> PyResult<Bound<'py, PyAny>>
where
    F: FnOnce() -> PyResult<Option<Py<PyAny>>>,
{
    let value = match work()? {
        Some(value) => value,
        None => py.None(),
    };
    match py
        .import("asyncio")
        .and_then(|asyncio| asyncio.call_method0("get_running_loop"))
    {
        Ok(loop_object) => {
            let future = loop_object.call_method0("create_future")?;
            future.call_method1("set_result", (value,))?;
            Ok(future)
        }
        Err(_) => pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(value) }),
    }
}
