//! Paused-run payloads shared by the sync and async bindings.
//!
//! LangGraph does not raise for a paused run: `invoke`/`resume`/`batch` return the current
//! state, and a dynamic `interrupt()` adds an `__interrupt__` list to it. These helpers build
//! that shape from the engine's interrupt error.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::adapters::python::types::PyInterrupt;
use crate::adapters::python::{json_state_to_py, value_to_py};

/// Add LangGraph's `__interrupt__` entry to a state dict.
pub(crate) fn attach_interrupts(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    interrupts: &[crate::Interrupt],
) -> PyResult<()> {
    let dict = values.cast::<PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("interrupt payload expects a state dict")
    })?;
    let mut items: Vec<Py<PyAny>> = Vec::with_capacity(interrupts.len());
    for interrupt in interrupts {
        let item = Py::new(
            py,
            PyInterrupt {
                id: interrupt.id.clone(),
                value: value_to_py(py, &interrupt.value)?,
                node: Some(interrupt.node.clone()),
                resumable: true,
                response_schema: match &interrupt.response_schema {
                    Some(schema) => Some(value_to_py(py, schema)?),
                    None => None,
                },
            },
        )?;
        items.push(item.into_any());
    }
    dict.set_item("__interrupt__", PyList::new(py, items)?)?;
    Ok(())
}

/// Build the paused result LangGraph returns from `invoke`/`resume`/`batch`.
///
/// A static `interrupt_before`/`interrupt_after` breakpoint returns the current state
/// unchanged; a dynamic `interrupt()` attaches the `__interrupt__` list. Returns `None` for
/// errors that are not pauses, or when the paused state cannot be read (no checkpointer).
pub(crate) fn pause_values(
    py: Python<'_>,
    graph: &crate::graph::CompiledGraph<crate::JsonState>,
    thread: Option<&str>,
    namespace: Option<&str>,
    error: &crate::Error,
) -> PyResult<Option<Py<PyAny>>> {
    let crate::Error::Interrupt(interrupt) = error else {
        return Ok(None);
    };
    let Some(thread) = thread else {
        return Ok(None);
    };
    let snapshot = match namespace {
        Some(namespace) => graph.get_state_in(thread, namespace),
        None => graph.get_state(thread),
    }
    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    let Some(snapshot) = snapshot else {
        return Ok(None);
    };
    let values = json_state_to_py(py, &snapshot.values)?;
    if !interrupt.static_pause {
        attach_interrupts(py, values.bind(py), std::slice::from_ref(interrupt))?;
    }
    Ok(Some(values))
}

