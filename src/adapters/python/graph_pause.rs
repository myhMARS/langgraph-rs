//! Paused-run payloads shared by the sync and async bindings.
//!
//! LangGraph does not raise for a paused run: `invoke`/`resume`/`batch` return the current
//! state, and a dynamic `interrupt()` adds an `__interrupt__` list to it. These helpers build
//! that shape from the engine's interrupt error.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::adapters::python::types::PyInterrupt;
use crate::adapters::python::{json_state_to_py, value_to_py};


/// Record a pause for the callback layer.
///
/// The engine does not call into Python here: it appends an event to the run's
/// [`LifecycleSink`](crate::_internal::lifecycle::LifecycleSink), and the Python facade drains it
/// once the call returns — the same shape as LangGraph's loop queue.
pub(crate) fn record_interrupt(
    lifecycle: Option<&crate::_internal::lifecycle::LifecycleSink>,
    status: &str,
    checkpoint_id: &str,
    namespace: &str,
    interrupts: &[crate::Interrupt],
) {
    crate::_internal::lifecycle::record(
        lifecycle,
        crate::_internal::lifecycle::LifecycleEvent::interrupt(
            status,
            checkpoint_id,
            namespace,
            interrupts.to_vec(),
        ),
    );
}

/// Record a resume for the callback layer.
pub(crate) fn record_resume(
    lifecycle: Option<&crate::_internal::lifecycle::LifecycleSink>,
    checkpoint_id: &str,
    namespace: &str,
) {
    crate::_internal::lifecycle::record(
        lifecycle,
        crate::_internal::lifecycle::LifecycleEvent::resume("pending", checkpoint_id, namespace),
    );
}

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
    lifecycle: Option<&crate::_internal::lifecycle::LifecycleSink>,
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
    // With a checkpointer the paused snapshot is authoritative; without one the engine
    // attaches the paused values to the interrupt so LangGraph's shape still works.
    let checkpoint_id = snapshot
        .as_ref()
        .map(|snapshot| snapshot.checkpoint_id.clone())
        .unwrap_or_default();
    // A superstep can pause several tasks at once; LangGraph surfaces every interrupt, and the
    // paused checkpoint already carries them, so it is authoritative over the error payload (which
    // holds only the first one).
    let interrupts: Vec<crate::Interrupt> = match snapshot.as_ref() {
        Some(snapshot) if !snapshot.interrupts.is_empty() => snapshot.interrupts.clone(),
        _ => vec![interrupt.clone()],
    };
    let values = match snapshot {
        Some(snapshot) => json_state_to_py(py, &snapshot.values)?,
        None => match &interrupt.values {
            Some(values) => json_state_to_py(py, &crate::JsonState(values.clone()))?,
            None => return Ok(None),
        },
    };
    if !interrupt.static_pause {
        attach_interrupts(py, values.bind(py), &interrupts)?;
    }
    record_interrupt(
        lifecycle,
        if interrupt.static_pause { "interrupt_before" } else { "pending" },
        &checkpoint_id,
        namespace.unwrap_or(""),
        &interrupts,
    );
    Ok(Some(values))
}

