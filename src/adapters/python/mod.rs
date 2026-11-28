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

#[cfg(feature = "python-async")]
pub(crate) mod async_graph;
pub(crate) mod convert;
mod custom_saver;
mod custom_store;
mod graph_pause;
mod graph;
mod prebuilt;
mod types;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[cfg(feature = "python-async")]
use async_graph::{PyAsyncCompiledGraph, PyAsyncStateGraph};
use convert::*;
use graph::{PyCompiledGraph, PyStateGraph};
#[cfg(feature = "postgres")]
use types::PyPostgresSaver;
#[cfg(feature = "redis")]
use types::PyRedisSaver;
#[cfg(feature = "sqlite")]
use types::PySqliteSaver;
use types::{
    PyCachePolicy, PyCommand, PyInMemoryCache, PyInMemorySaver, PyInMemoryStore, PyInterrupt,
    PyRemoveMessage, PyRetryPolicy, PyRunControl, PySend, PyStore, PyStreamWriter, PyTask,
    PyTimeoutPolicy,
};
#[cfg(feature = "sqlite")]
use types::{PyLangGraphSqliteReader, PyLangGraphSqliteSaver, PySqliteCache, PySqliteStore};
#[cfg(feature = "postgres")]
use types::{PyPostgresCache, PyPostgresStore};
#[cfg(feature = "redis")]
use types::{PyRedisCache, PyRedisStore};

create_exception!(
    langgraph_rs,
    GraphBubbleUp,
    PyException,
    "Base class for LangGraph control-flow signals that unwind the run loop."
);
create_exception!(
    langgraph_rs,
    GraphInterrupt,
    GraphBubbleUp,
    "Raised when a node pauses the graph through langgraph_rs.interrupt()."
);
create_exception!(
    langgraph_rs,
    GraphDrained,
    GraphBubbleUp,
    "Raised when a run was asked to stop taking new work (`RunControl.request_drain`)."
);
create_exception!(
    langgraph_rs,
    GraphRecursionError,
    pyo3::exceptions::PyRecursionError,
    "Raised when the recursion limit is reached without hitting a stop condition."
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

/// Node currently executing, when called from a node or callback.
#[pyfunction]
fn current_node() -> Option<String> {
    crate::current_node()
}

/// Nodes (or `START`) that triggered the currently running task.
#[pyfunction]
fn current_triggers() -> Vec<String> {
    crate::runtime::context::current_triggers()
}

/// Superstep currently executing, when called from a node or callback.
/// Progress counter of the running task (`custom()`, `message`, usage and tool events bump it).
#[pyfunction]
fn _progress_counter() -> u64 {
    crate::adapters::python::async_graph::async_builder::current_progress()
}

#[pyfunction]
fn current_step() -> Option<usize> {
    crate::current_step()
}

/// Checkpoint namespace of the current run.
#[pyfunction]
fn current_namespace() -> String {
    crate::current_namespace()
}

/// Checkpoint namespace of the running task: `<node>:<task id>`, as LangGraph reports it.
#[pyfunction]
fn current_checkpoint_ns() -> String {
    let node = crate::runtime::context::current_node().unwrap_or_default();
    match crate::runtime::context::current_task_id() {
        Some(id) if !node.is_empty() => format!("{node}:{id}"),
        Some(id) => id,
        None => crate::current_namespace(),
    }
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

/// Create a graph whose state has a single `messages` channel (LangGraph `MessageGraph`).
#[pyfunction]
fn message_graph() -> PyStateGraph {
    PyStateGraph::from_inner(crate::StateGraph::<crate::JsonState>::with_schema(
        crate::StateSchema::new().with_reducer("messages", crate::reducers::messages()),
    ))
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

/// Writer that emits custom stream values from helpers.
#[pyfunction]
fn get_stream_writer() -> PyStreamWriter {
    PyStreamWriter
}

/// LangGraph-style config of the current run.
#[pyfunction]
fn get_config(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let config = crate::get_config();
    let value = value_to_py(py, &config)?;
    // LangGraph reports these metadata entries as tuples.
    if let Ok(dict) = value.bind(py).cast::<pyo3::types::PyDict>()
        && let Ok(metadata) = dict.get_item("metadata")
        && let Some(metadata) = metadata.and_then(|meta| meta.cast_into::<pyo3::types::PyDict>().ok())
    {
        for key in ["langgraph_triggers", "langgraph_path"] {
            if let Ok(Some(entry)) = metadata.get_item(key)
                && let Ok(items) = entry.extract::<Vec<Py<PyAny>>>()
            {
                metadata.set_item(key, pyo3::types::PyTuple::new(py, items)?)?;
            }
        }
    }
    Ok(value)
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
        "has_control",
        crate::runtime::context::current_control().is_some(),
    )?;
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

/// The run control of the current run, for `runtime.control` inside a node.
#[pyfunction]
fn current_control() -> Option<crate::adapters::python::types::PyRunControl> {
    crate::runtime::context::current_control().map(|control| {
        crate::adapters::python::types::PyRunControl::from_control((*control).clone())
    })
}

/// Pause the graph and wait for human input; returns the resume value on retry.
#[pyfunction]
#[pyo3(signature = (value, response_schema=None))]
fn interrupt(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    response_schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let value = py_to_value(value)?;
    // LangGraph accepts a JSON-schema dict or a pydantic/TypedDict class here.
    let schema_object = response_schema
        .filter(|schema| !schema.is_none())
        .map(|schema| schema.clone().unbind());
    let response_schema = match schema_object.as_ref() {
        Some(schema) => Some(convert::context_schema_value(py, schema.bind(py))?),
        None => None,
    };
    // Pydantic owns validation for Python callers, exactly as in LangGraph.
    match crate::runtime::context::interrupt_unvalidated(value, response_schema) {
        Ok(resume) => {
            let py_value = value_to_py(py, &resume)?;
            match schema_object.as_ref() {
                // A raw JSON-schema dict is used as-is and never validated (LangGraph's rule).
                Some(schema) if schema.bind(py).is_instance_of::<pyo3::types::PyDict>() => {
                    Ok(py_value)
                }
                // A class is validated (and coerced) by pydantic, which also decides the error.
                Some(schema) => coerce_response_schema(py, schema.bind(py), py_value),
                None => Ok(py_value),
            }
        }
        Err(error) => Err(to_pyerr(py, error)),
    }
}

/// Validate a resume value against `interrupt(response_schema=...)`, returning the model instance
/// (LangGraph's behaviour) and raising pydantic's error when it does not fit.
fn coerce_response_schema(
    py: Python<'_>,
    schema: &Bound<'_, PyAny>,
    value: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let adapter = py
        .import("pydantic")
        .and_then(|module| module.getattr("TypeAdapter"))
        .and_then(|class| class.call1((schema,)))?;
    match adapter.call_method1("validate_python", (value.bind(py),)) {
        Ok(value) => Ok(value.unbind()),
        Err(error) => {
            // LangGraph marks the failure so `is_invalid_resume()` and `ToolNode` can tell it
            // apart from an ordinary tool error.
            let _ = error.value(py).setattr("_langgraph_invalid_resume", true);
            let _ = error.value(py).setattr("_invalid_resume", true);
            Err(error)
        }
    }
}

/// Toggle LangChain message-object conversion for `add_messages` channels.
#[pyfunction]
fn _set_messages_objects(enabled: bool) {
    crate::runtime::context::set_messages_as_objects(enabled);
}

/// Whether message channels are handed to Python as LangChain message objects.
#[pyfunction]
fn messages_as_objects() -> bool {
    crate::runtime::context::messages_as_objects()
}

/// Write a value to a state channel from inside a node (LangGraph's `CONFIG_KEY_SEND`).
#[pyfunction]
fn _send(py: Python<'_>, channel: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = py_to_value(value)?;
    crate::runtime::context::send(channel, value).map_err(|error| to_pyerr(py, error))
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

/// Create a LangChain callback handler that forwards LLM tokens and usage into the graph stream.
///
/// The handler classes live in the Python shim (`langgraph_rs._callbacks`); it registers the factory
/// at import time so Rust never imports the package.
#[pyfunction]
#[pyo3(signature = (model=None, tags=None, streaming=false))]
pub(super) fn langchain_callback_handler(
    py: Python<'_>,
    model: Option<String>,
    tags: Option<Vec<String>>,
    streaming: bool,
) -> PyResult<Py<PyAny>> {
    // The shim registers `langchain_callback_handler` at import time; Rust never imports it.
    let factory = convert::shim_hook(py, "callback_handler")
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
            "langgraph_rs is not initialised: the callback handler factory is missing",
        ))?;
    let handler = factory
        .bind(py)
        .call1((model, tags, streaming))?;
    Ok(handler.unbind())
}

/// Emit a tool-call lifecycle event (`stream_mode="tools"`) from inside a tool.
#[pyfunction]
fn _tool_event(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = py_to_value(value)?;
    crate::runtime::context::push_tool_event(value).map_err(|error| to_pyerr(py, error))
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

#[pymodule]
fn langgraph_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStateGraph>()?;
    module.add_class::<PyCompiledGraph>()?;
    module.add_class::<super::python::types::PyStateSnapshot>()?;
    module.add(
        "CompiledStateGraph",
        module.py().get_type::<PyCompiledGraph>(),
    )?;
    module.add_class::<PySend>()?;
    module.add_class::<crate::adapters::python::types::PyStateHistory>()?;
    module.add_class::<crate::adapters::python::types::PyStreamChunks>()?;
    module.add_class::<PyCommand>()?;
    module.add_class::<PyRetryPolicy>()?;
    module.add_class::<PyCachePolicy>()?;
    module.add_class::<PyTimeoutPolicy>()?;
    module.add_class::<PyInMemorySaver>()?;
    module.add_class::<PyInMemoryCache>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PySqliteSaver>()?;
    #[cfg(feature = "redis")]
    module.add_class::<PyRedisSaver>()?;
    #[cfg(feature = "postgres")]
    module.add_class::<PyPostgresSaver>()?;
    module.add_class::<PyStore>()?;
    module.add_class::<crate::adapters::python::types::PyStoreItem>()?;
    module.add_class::<PyInMemoryStore>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PySqliteStore>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PySqliteCache>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PyLangGraphSqliteReader>()?;
    #[cfg(feature = "sqlite")]
    module.add_class::<PyLangGraphSqliteSaver>()?;
    #[cfg(feature = "redis")]
    {
        module.add_class::<PyRedisStore>()?;
        module.add_class::<PyRedisCache>()?;
    }
    #[cfg(feature = "postgres")]
    {
        module.add_class::<PyPostgresStore>()?;
        module.add_class::<PyPostgresCache>()?;
    }
    module.add_class::<PyTask>()?;
    module.add_class::<prebuilt::PyToolNode>()?;
    module.add_function(wrap_pyfunction!(prebuilt::tools_condition, module)?)?;
    module.add_function(wrap_pyfunction!(prebuilt::create_react_agent, module)?)?;
    module.add("END", crate::END)?;
    module.add_class::<PyRemoveMessage>()?;
    module.add("AsyncInMemorySaver", module.getattr("InMemorySaver")?)?;
    #[cfg(feature = "sqlite")]
    module.add("AsyncSqliteSaver", module.getattr("SqliteSaver")?)?;
    #[cfg(feature = "redis")]
    module.add("AsyncRedisSaver", module.getattr("RedisSaver")?)?;
    #[cfg(feature = "postgres")]
    module.add("AsyncPostgresSaver", module.getattr("PostgresSaver")?)?;
    #[cfg(feature = "python-async")]
    {
        module.add_class::<PyAsyncStateGraph>()?;
        module.add_class::<PyAsyncCompiledGraph>()?;
    module.add_class::<async_graph::PyAsyncStreamChunks>()?;
    }
    module.add_function(wrap_pyfunction!(message_graph, module)?)?;
    module.add_function(wrap_pyfunction!(convert::_register_error_classes, module)?)?;
    module.add_function(wrap_pyfunction!(convert::_register_shim_hook, module)?)?;
    module.add_function(wrap_pyfunction!(task, module)?)?;
    module.add_function(wrap_pyfunction!(entrypoint, module)?)?;
    module.add_function(wrap_pyfunction!(get_store, module)?)?;
    module.add_function(wrap_pyfunction!(get_config, module)?)?;
    module.add_function(wrap_pyfunction!(get_stream_writer, module)?)?;
    module.add_class::<PyInterrupt>()?;
    module.add_class::<crate::adapters::python::types::PyToolRuntime>()?;
    module.add_class::<PyStreamWriter>()?;
    module.add_class::<PyRunControl>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(current_thread_id, module)?)?;
    module.add_function(wrap_pyfunction!(current_namespace, module)?)?;
    module.add_function(wrap_pyfunction!(current_node, module)?)?;
    module.add_function(wrap_pyfunction!(current_triggers, module)?)?;
    module.add_function(wrap_pyfunction!(current_step, module)?)?;
    module.add_function(wrap_pyfunction!(_progress_counter, module)?)?;
    // Keep a reference so the async wrappers can call it without importing the shim package.
    async_graph::async_support::remember_progress_counter(
        module.getattr("_progress_counter")?.unbind(),
    );
    module.add_function(wrap_pyfunction!(current_checkpoint_ns, module)?)?;
    module.add_function(wrap_pyfunction!(current_resume, module)?)?;
    module.add_function(wrap_pyfunction!(current_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(current_control, module)?)?;
    module.add_function(wrap_pyfunction!(remaining_steps, module)?)?;
    module.add_function(wrap_pyfunction!(is_last_step, module)?)?;
    module.add("GraphBubbleUp", module.py().get_type::<GraphBubbleUp>())?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add("_set_messages_objects", wrap_pyfunction!(_set_messages_objects, module)?)?;
    module.add_function(wrap_pyfunction!(messages_as_objects, module)?)?;
    module.add_function(wrap_pyfunction!(_send, module)?)?;
    module.add("GraphDrained", module.py().get_type::<GraphDrained>())?;
    module.add("GraphRecursionError", module.py().get_type::<GraphRecursionError>())?;
    module.add_function(wrap_pyfunction!(interrupt, module)?)?;
    module.add_function(wrap_pyfunction!(custom, module)?)?;
    module.add_function(wrap_pyfunction!(message, module)?)?;
    module.add_function(wrap_pyfunction!(langchain_callback_handler, module)?)?;
    module.add_function(wrap_pyfunction!(_tool_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_usage, module)?)?;
    module.add_function(wrap_pyfunction!(record_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("add_messages", "messages")?;
    module.add("START", crate::START)?;
    module.add("END", crate::END)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
