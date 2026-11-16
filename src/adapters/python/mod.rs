//! Python bindings (enabled with the `python` feature).
//!
//! The Python surface mirrors the Rust API using [`JsonState`](crate::JsonState): a graph is built from
//! Python callables, every node receives the state as a `dict` and returns a `dict` of
//! channel writes (optionally with the reserved `__goto__` and `__usage__` keys), and
//! `interrupt()` inside a node pauses the run so the caller can `resume()` it later.
//!
//! ```python
//! import langgraph_rs as lg
//!
//! graph = lg.StateGraph({"count": "last_value", "log": "append"})
//! graph.add_node("inc", lambda state: {"count": state["count"] + 1, "log": [state["count"] + 1]})
//! graph.add_edge("START", "inc")
//! graph.add_conditional_edges("inc", lambda state: "inc" if state["count"] < 3 else None)
//! compiled = graph.compile()
//! compiled.invoke({"count": 0, "log": []})
//! ```

mod convert;
mod graph;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use convert::*;
use graph::{PyCompiledGraph, PyStateGraph};

create_exception!(
    langgraph_rs,
    GraphInterrupt,
    PyException,
    "Raised when a node pauses the graph through langgraph_rs.interrupt()."
);

// ---------------------------------------------------------------------------
// Module level functions
// ---------------------------------------------------------------------------

/// Pause the graph and wait for human input; returns the resume value on retry.
#[pyfunction]
fn interrupt(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = py_to_value(value)?;
    match crate::kernel::context::interrupt(value) {
        Ok(resume) => value_to_py(py, &resume),
        Err(error) => Err(to_pyerr(py, error)),
    }
}

/// Emit a custom value into the stream from inside a node.
#[pyfunction]
fn custom(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = py_to_value(value)?;
    crate::custom(value).map_err(|error| to_pyerr(py, error))
}

/// Emit a streamed message (e.g. an LLM token) from inside a node.
#[pyfunction]
fn message(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = py_to_value(value)?;
    crate::message(value).map_err(|error| to_pyerr(py, error))
}

/// Record consumption for the current node.
#[pyfunction]
fn record_usage(py: Python<'_>, usage: &Bound<'_, PyAny>) -> PyResult<()> {
    let usage = usage_from_py(usage)?;
    crate::record_usage(usage).map_err(|error| to_pyerr(py, error))
}

/// Convenience shorthand for `record_usage` with token counts.
#[pyfunction]
fn record_tokens(py: Python<'_>, input_tokens: u64, output_tokens: u64) -> PyResult<()> {
    crate::record_tokens(input_tokens, output_tokens).map_err(|error| to_pyerr(py, error))
}

/// Crate version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition.
#[pymodule]
fn langgraph_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStateGraph>()?;
    module.add_class::<PyCompiledGraph>()?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add_function(wrap_pyfunction!(interrupt, module)?)?;
    module.add_function(wrap_pyfunction!(custom, module)?)?;
    module.add_function(wrap_pyfunction!(message, module)?)?;
    module.add_function(wrap_pyfunction!(record_usage, module)?)?;
    module.add_function(wrap_pyfunction!(record_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("START", crate::START)?;
    module.add("END", crate::END)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
