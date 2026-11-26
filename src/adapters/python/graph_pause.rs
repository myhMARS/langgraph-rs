//! Paused-run payloads shared by the sync and async bindings.
//!
//! LangGraph does not raise for a paused run: `invoke`/`resume`/`batch` return the current
//! state, and a dynamic `interrupt()` adds an `__interrupt__` list to it. These helpers build
//! that shape from the engine's interrupt error.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::adapters::python::types::PyInterrupt;
use crate::adapters::python::{json_state_to_py, value_to_py};


/// Dispatch `GraphCallbackHandler.on_interrupt` for a paused run.
///
/// Mirrors LangGraph: the event carries the loop status (`pending`), the checkpoint the paused run
/// is on and the interrupt payloads.
pub(crate) fn dispatch_interrupt(
    py: Python<'_>,
    py_config: Option<&Bound<'_, PyAny>>,
    checkpoint_id: &str,
    interrupt: &crate::Interrupt,
) -> PyResult<()> {
    dispatch_lifecycle(
        py,
        py_config,
        "interrupt",
        checkpoint_id,
        "",
        std::slice::from_ref(interrupt),
    )
}

/// Dispatch `GraphCallbackHandler.on_resume` when a run continues from a checkpoint.
pub(crate) fn dispatch_resume(
    py: Python<'_>,
    py_config: Option<&Bound<'_, PyAny>>,
    checkpoint_id: &str,
    namespace: Option<&str>,
) -> PyResult<()> {
    dispatch_lifecycle(
        py,
        py_config,
        "resume",
        checkpoint_id,
        namespace.unwrap_or(""),
        &[],
    )
}

/// Call `langgraph_rs.callbacks._dispatch_graph_lifecycle` with the event data.
fn dispatch_lifecycle(
    py: Python<'_>,
    py_config: Option<&Bound<'_, PyAny>>,
    kind: &str,
    checkpoint_id: &str,
    checkpoint_ns: &str,
    interrupts: &[crate::Interrupt],
) -> PyResult<()> {
    let Some(config) = py_config else {
        return Ok(());
    };
    let Ok(module) = py.import("langgraph_rs.callbacks") else {
        return Ok(());
    };
    let Ok(dispatcher) = module.getattr("_dispatch_graph_lifecycle") else {
        return Ok(());
    };
    let namespace: Vec<String> = if checkpoint_ns.is_empty() {
        Vec::new()
    } else {
        checkpoint_ns.split('|').map(|part| part.to_string()).collect()
    };
    let mut payloads: Vec<Py<PyAny>> = Vec::with_capacity(interrupts.len());
    for interrupt in interrupts {
        payloads.push(Py::new(
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
        )?
        .into_any());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("status", "pending")?;
    kwargs.set_item("checkpoint_id", checkpoint_id)?;
    kwargs.set_item("checkpoint_ns", pyo3::types::PyTuple::new(py, namespace)?)?;
    kwargs.set_item("interrupts", pyo3::types::PyTuple::new(py, payloads)?)?;
    let _ = dispatcher.call((config, kind), Some(&kwargs));
    Ok(())
}

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
    // With a checkpointer the paused snapshot is authoritative; without one the engine
    // attaches the paused values to the interrupt so LangGraph's shape still works.
    let checkpoint_id = snapshot
        .as_ref()
        .map(|snapshot| snapshot.checkpoint_id.clone())
        .unwrap_or_default();
    let values = match snapshot {
        Some(snapshot) => json_state_to_py(py, &snapshot.values)?,
        None => match &interrupt.values {
            Some(values) => json_state_to_py(py, &crate::JsonState(values.clone()))?,
            None => return Ok(None),
        },
    };
    if !interrupt.static_pause {
        attach_interrupts(py, values.bind(py), std::slice::from_ref(interrupt))?;
    }
    dispatch_interrupt(py, py_config, &checkpoint_id, interrupt)?;
    Ok(Some(values))
}

