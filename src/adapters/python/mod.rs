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
mod types;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use convert::*;
use graph::{PyCompiledGraph, PyStateGraph};
#[cfg(feature = "postgres")]
use types::PyPostgresSaver;
#[cfg(feature = "redis")]
use types::PyRedisSaver;
#[cfg(feature = "sqlite")]
use types::PySqliteSaver;
use types::{
    PyCachePolicy, PyCommand, PyInMemorySaver, PyInMemoryStore, PyRetryPolicy, PySend, PyStore,
    PyTask,
};

create_exception!(
    langgraph_rs,
    GraphInterrupt,
    PyException,
    "Raised when a node pauses the graph through langgraph_rs.interrupt()."
);

/// Long-term memory store attached to the current run, if any.
#[pyfunction]
fn get_store() -> Option<PyStore> {
    crate::get_store().map(PyStore::new)
}

/// User context passed to `invoke`/`run`/`stream` via `context=`.
#[pyfunction]
fn current_context(py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
    match crate::current_context() {
        Some(value) => Ok(Some(value_to_py(py, &value)?)),
        None => Ok(None),
    }
}

/// Thread id of the current run.
#[pyfunction]
fn current_thread_id() -> Option<String> {
    crate::current_thread_id()
}

/// Checkpoint namespace of the current run.
#[pyfunction]
fn current_namespace() -> String {
    crate::current_namespace()
}

/// Resume value stamped onto the current task, when replaying an interrupt.
#[pyfunction]
fn current_resume(py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
    match crate::current_resume() {
        Some(value) => Ok(Some(value_to_py(py, &value)?)),
        None => Ok(None),
    }
}

/// Wrap a Python callable so its results are memoized in the run's store.
#[pyfunction]
#[pyo3(signature = (function, name=None))]
fn task(function: Py<PyAny>, name: Option<String>) -> PyResult<PyTask> {
    PyTask::from_function(function, name)
}

/// Build a one-node graph whose `entrypoint` node runs the Python callable.
#[pyfunction]
fn entrypoint(function: Py<PyAny>) -> PyResult<PyStateGraph> {
    let inner = crate::StateGraph::<crate::JsonState>::new()
        .add_node(
            "entrypoint",
            node_from_python("entrypoint".to_string(), function),
        )
        .set_entry_point("entrypoint")
        .set_finish_point("entrypoint");
    Ok(PyStateGraph::from_inner(inner))
}

/// Remaining supersteps before the recursion limit (managed value).
#[pyfunction]
fn remaining_steps() -> Option<usize> {
    crate::remaining_steps()
}

/// Whether the current superstep is the last one (managed value).
#[pyfunction]
fn is_last_step() -> bool {
    crate::is_last_step()
}

/// Runtime information of the current run (thread, namespace, store, context).
#[pyfunction]
fn current_runtime(py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
    let Some(runtime) = crate::current_runtime() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("thread_id", runtime.thread_id)?;
    dict.set_item("namespace", runtime.namespace)?;
    dict.set_item("has_store", runtime.store.is_some())?;
    dict.set_item(
        "context",
        match runtime.context {
            Some(context) => value_to_py(py, &context)?,
            None => py.None(),
        },
    )?;
    Ok(Some(dict.into_any().unbind()))
}

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
    module.add_class::<PySend>()?;
    module.add_class::<PyCommand>()?;
    module.add_class::<PyRetryPolicy>()?;
    module.add_class::<PyCachePolicy>()?;
    module.add_class::<PyInMemorySaver>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PySqliteSaver>()?;
    module.add_class::<PyStore>()?;
    module.add_class::<PyInMemoryStore>()?;
    module.add_class::<PyTask>()?;
    #[cfg(feature = "redis")]
    module.add_class::<PyRedisSaver>()?;
    #[cfg(feature = "postgres")]
    module.add_class::<PyPostgresSaver>()?;
    module.add_function(wrap_pyfunction!(task, module)?)?;
    module.add_function(wrap_pyfunction!(entrypoint, module)?)?;
    module.add_function(wrap_pyfunction!(get_store, module)?)?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(current_thread_id, module)?)?;
    module.add_function(wrap_pyfunction!(current_namespace, module)?)?;
    module.add_function(wrap_pyfunction!(current_resume, module)?)?;
    module.add_function(wrap_pyfunction!(current_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(remaining_steps, module)?)?;
    module.add_function(wrap_pyfunction!(is_last_step, module)?)?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add_function(wrap_pyfunction!(interrupt, module)?)?;
    module.add_function(wrap_pyfunction!(custom, module)?)?;
    module.add_function(wrap_pyfunction!(message, module)?)?;
    module.add_function(wrap_pyfunction!(record_usage, module)?)?;
    module.add_function(wrap_pyfunction!(record_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("add_messages", "messages")?;
    module.add("START", crate::START)?;
    module.add("END", crate::END)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
