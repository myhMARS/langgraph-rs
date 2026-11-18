//! Async Python binding helpers (timeouts, snapshot steps, subgraph embedding).

use pyo3::types::{PyAny, PyDict, PyList};

use super::*;
use crate::adapters::python::convert::snapshot_to_py;

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

pub(crate) fn global_step_for(
    history: &[crate::StateSnapshot<crate::JsonState>],
    checkpoint_id: &str,
) -> Option<i64> {
    let total = history.len() as i64;
    let index = history
        .iter()
        .position(|snapshot| snapshot.checkpoint_id == checkpoint_id)? as i64;
    Some(total - 2 - index)
}

pub(crate) fn snapshot_to_py_with_step(
    py: Python<'_>,
    snapshot: &crate::StateSnapshot<crate::JsonState>,
    step: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let value = snapshot_to_py(py, snapshot)?;
    let Some(step) = step else {
        return Ok(value);
    };
    if let Ok(snapshot) = value
        .bind(py)
        .extract::<pyo3::PyRef<'_, crate::adapters::python::types::PyStateSnapshot>>()
    {
        let dict = snapshot.dict(py);
        dict.set_item("step", step)?;
        if let Some(metadata) = dict.get_item("metadata")? {
            if let Ok(metadata) = metadata.cast::<pyo3::types::PyDict>() {
                metadata.set_item("step", step)?;
            }
        }
        return Ok(value);
    }
    let dict = value.bind(py).cast::<pyo3::types::PyDict>()?;
    dict.set_item("step", step)?;
    if let Some(metadata) = dict.get_item("metadata")? {
        if let Ok(metadata) = metadata.cast::<pyo3::types::PyDict>() {
            metadata.set_item("step", step)?;
        }
    }
    Ok(value)
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

pub(crate) fn embed_subgraph_states(
    py: Python<'_>,
    root: &crate::StateSnapshot<JsonState>,
    subgraphs: &[crate::StateSnapshot<JsonState>],
) -> PyResult<Py<PyAny>> {
    let root_py = snapshot_to_py(py, root)?;
    if let Ok(root_snapshot) = root_py
        .bind(py)
        .extract::<PyRef<crate::adapters::python::types::PyStateSnapshot>>()
        && let Ok(Some(tasks)) = root_snapshot.dict(py).get_item("tasks")
        && let Ok(tasks) = tasks.cast::<PyList>()
    {
        for task in tasks {
            let Ok(task_dict) = task.cast::<PyDict>() else {
                continue;
            };
            let Ok(Some(node)) = task_dict.get_item("node") else {
                continue;
            };
            let Ok(node) = node.extract::<String>() else {
                continue;
            };
            let child = subgraphs.iter().find(|snapshot| {
                snapshot
                    .checkpoint_ns
                    .split('|')
                    .next_back()
                    .is_some_and(|segment| {
                        segment == node
                            || segment
                                .strip_prefix(&node)
                                .is_some_and(|rest| rest.starts_with(':'))
                    })
            });
            if let Some(child) = child {
                task_dict.set_item("state", snapshot_to_py(py, child)?)?;
            }
        }
    }
    Ok(root_py)
}
