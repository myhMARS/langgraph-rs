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

pub(crate) fn run_timed_awaitable<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Timed nodes reuse a per-thread asyncio loop so the wait_for overhead is not
    // paid on every attempt; ordinary nodes keep the cheaper asyncio.run path.
    let asyncio = py.import("asyncio")?;
    let loop_object = match asyncio
        .getattr("get_event_loop")
        .and_then(|get| get.call0())
    {
        Ok(loop_object) => loop_object,
        Err(_) => {
            let loop_object = asyncio.getattr("new_event_loop")?.call0()?;
            asyncio.getattr("set_event_loop")?.call1((&loop_object,))?;
            loop_object
        }
    };
    loop_object.call_method1("run_until_complete", (value,))
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
) -> PyResult<Option<Py<PyAny>>> {
    let crate::Error::Interrupt(interrupt) = error else {
        return Ok(None);
    };
    if !interrupt.static_pause {
        return Ok(None);
    }
    let Some(thread) = thread else {
        return Ok(None);
    };
    let snapshot = match namespace {
        Some(namespace) => graph.get_state_in(thread, namespace),
        None => graph.get_state(thread),
    }
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    match snapshot {
        Some(snapshot) => {
            crate::adapters::python::convert::json_state_to_py(py, &snapshot.values).map(Some)
        }
        None => Ok(Some(py.None())),
    }
}
