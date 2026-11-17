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
mod async_graph;
mod convert;
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
    PyCachePolicy, PyCommand, PyInMemoryCache, PyInMemorySaver, PyInMemoryStore, PyRemoveMessage,
    PyRetryPolicy, PyRunControl, PySend, PyStore, PyStreamWriter, PyTask, PyTimeoutPolicy,
};
#[cfg(feature = "sqlite")]
use types::{PyLangGraphSqliteReader, PyLangGraphSqliteSaver, PySqliteCache, PySqliteStore};
#[cfg(feature = "postgres")]
use types::{PyPostgresCache, PyPostgresStore};
#[cfg(feature = "redis")]
use types::{PyRedisCache, PyRedisStore};

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

/// Node currently executing, when called from a node or callback.
#[pyfunction]
fn current_node() -> Option<String> {
    crate::current_node()
}

/// Superstep currently executing, when called from a node or callback.
#[pyfunction]
fn current_step() -> Option<usize> {
    crate::current_step()
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
    value_to_py(py, &config)
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
#[pyo3(signature = (value, response_schema=None))]
fn interrupt(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    response_schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let value = py_to_value(value)?;
    let response_schema = match response_schema {
        Some(schema) => Some(py_to_value(schema)?),
        None => None,
    };
    match crate::kernel::context::interrupt_with_schema(value, response_schema) {
        Ok(resume) => value_to_py(py, &resume),
        Err(crate::Error::Interrupt(interrupt)) => {
            let payload = value_to_py(py, &interrupt.value)?;
            let schema = match &interrupt.response_schema {
                Some(schema) => value_to_py(py, schema)?,
                None => py.None(),
            };
            Err(PyErr::new::<GraphInterrupt, _>((payload, schema)))
        }
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

/// Create a LangChain callback handler that forwards LLM tokens and usage into the graph stream.
static CALLBACK_HANDLER_CLASS: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();

#[pyfunction]
#[pyo3(signature = (model=None, tags=None))]
pub(super) fn langchain_callback_handler(
    py: Python<'_>,
    model: Option<String>,
    tags: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let source = r#"
class LangChainCallbackHandler:
    raise_error = False
    ignore_llm = False
    ignore_chat_model = False
    ignore_agent = False
    ignore_retriever = False
    ignore_chain = False
    ignore_retry = False
    ignore_custom_event = False

    def __init__(self, model=None, tags=None):
        self.model = model
        self.tags = list(tags or [])

    def _emit(self, token, kwargs=None):
        import langgraph_rs
        metadata = {}
        if self.model:
            metadata["model"] = self.model
        event_metadata = (kwargs or {}).get("metadata") or {}
        for key, value in event_metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata.setdefault(key, value)
        for key in ("ls_provider", "ls_model_name", "ls_model_type", "ls_integration"):
            value = (kwargs or {}).get(key)
            if value is not None:
                metadata[key] = value
        namespace = langgraph_rs.current_namespace()
        if namespace:
            metadata["langgraph_checkpoint_ns"] = namespace
        node = langgraph_rs.current_node()
        if node:
            metadata["langgraph_node"] = node
        step = langgraph_rs.current_step()
        if step is not None:
            metadata["langgraph_step"] = step
        langgraph_rs.message({"content": token, "tags": self.tags, "metadata": metadata})

    def _pick_model(self, kwargs):
        metadata = kwargs.get("metadata") or {}
        name = metadata.get("ls_model_name") or metadata.get("model")
        if name:
            self.model = name

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._pick_model(kwargs)
        if not self.model and isinstance(serialized, dict):
            self.model = serialized.get("name") or serialized.get("model")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._pick_model(kwargs)

    def on_llm_new_token(self, token, **kwargs):
        if isinstance(token, str):
            self._emit(token, kwargs)

    def on_llm_end(self, response, **kwargs):
        usage = None
        try:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
        except Exception:
            usage = None
        if not usage:
            try:
                message = response.generations[0][0].message
                usage = getattr(message, "usage_metadata", None)
            except Exception:
                usage = None
        if usage:
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            if input_tokens or output_tokens:
                import langgraph_rs
                langgraph_rs.record_tokens(int(input_tokens), int(output_tokens))

    def on_llm_error(self, error, **kwargs):
        pass

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        pass

    def on_chain_error(self, error, **kwargs):
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        pass

    def on_tool_end(self, output, **kwargs):
        pass

    def on_tool_error(self, error, **kwargs):
        pass

    def on_retriever_start(self, serialized, query, **kwargs):
        pass

    def on_retriever_end(self, documents, **kwargs):
        pass

    def on_retriever_error(self, error, **kwargs):
        pass

    def on_agent_action(self, action, **kwargs):
        pass

    def on_agent_finish(self, finish, **kwargs):
        pass

    def on_text(self, text, **kwargs):
        pass
"#;
    let source = std::ffi::CString::new(source)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    let class = CALLBACK_HANDLER_CLASS.get_or_try_init(py, || {
        let module = pyo3::types::PyModule::from_code(
            py,
            &source,
            c"langgraph_rs_callbacks.py",
            c"langgraph_rs_callbacks",
        )?;
        Ok::<_, pyo3::PyErr>(module.getattr("LangChainCallbackHandler")?.unbind())
    })?;
    let handler = class.bind(py).call1((model, tags))?;
    Ok(handler.unbind())
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
    let stream_module = pyo3::types::PyModule::from_code(
        module.py(),
        c"class AsyncStream:\n    def __init__(self, awaitable):\n        self._awaitable = awaitable\n        self._iter = None\n    def __await__(self):\n        return self._awaitable.__await__()\n    def __aiter__(self):\n        return self\n    async def __anext__(self):\n        if self._iter is None:\n            self._iter = iter(await self._awaitable)\n        try:\n            return next(self._iter)\n        except StopIteration:\n            raise StopAsyncIteration\n",
        c"langgraph_rs_async_stream.py",
        c"langgraph_rs_async_stream",
    )?;
    module.add("AsyncStream", stream_module.getattr("AsyncStream")?)?;
    module.add_class::<PyStateGraph>()?;
    module.add_class::<PyCompiledGraph>()?;
    module.add_class::<super::python::types::PyStateSnapshot>()?;
    module.add(
        "CompiledStateGraph",
        module.py().get_type::<PyCompiledGraph>(),
    )?;
    module.add_class::<PySend>()?;
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
    }
    let schema_module = pyo3::types::PyModule::from_code(
        module.py(),
        c"import typing\n\nclass MessagesState(dict):\n    __annotations__ = {'messages': typing.Annotated[list, 'add_messages']}\n\ndef _schema_from_class(cls):\n    out = {}\n    annotations = {}\n    for base in reversed(getattr(cls, '__mro__', [cls])):\n        annotations.update(getattr(base, '__annotations__', {}))\n    for name, annotation in annotations.items():\n        reducer = 'last_value'\n        if typing.get_origin(annotation) is typing.Annotated:\n            args = typing.get_args(annotation)\n            reducer = args[1]\n        out[name] = reducer\n    return out\n",
        c"langgraph_rs_schema.py",
        c"langgraph_rs_schema",
    )?;
    module.add("MessagesState", schema_module.getattr("MessagesState")?)?;
    module.add(
        "_schema_from_class",
        schema_module.getattr("_schema_from_class")?,
    )?;
    module.add_function(wrap_pyfunction!(message_graph, module)?)?;
    module.add_function(wrap_pyfunction!(task, module)?)?;
    module.add_function(wrap_pyfunction!(entrypoint, module)?)?;
    module.add_function(wrap_pyfunction!(get_store, module)?)?;
    module.add_function(wrap_pyfunction!(get_config, module)?)?;
    module.add_function(wrap_pyfunction!(get_stream_writer, module)?)?;
    module.add_class::<PyStreamWriter>()?;
    module.add_class::<PyRunControl>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(current_thread_id, module)?)?;
    module.add_function(wrap_pyfunction!(current_namespace, module)?)?;
    module.add_function(wrap_pyfunction!(current_node, module)?)?;
    module.add_function(wrap_pyfunction!(current_step, module)?)?;
    module.add_function(wrap_pyfunction!(current_resume, module)?)?;
    module.add_function(wrap_pyfunction!(current_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(remaining_steps, module)?)?;
    module.add_function(wrap_pyfunction!(is_last_step, module)?)?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add_function(wrap_pyfunction!(interrupt, module)?)?;
    module.add_function(wrap_pyfunction!(custom, module)?)?;
    module.add_function(wrap_pyfunction!(message, module)?)?;
    module.add_function(wrap_pyfunction!(langchain_callback_handler, module)?)?;
    module.add_function(wrap_pyfunction!(record_usage, module)?)?;
    module.add_function(wrap_pyfunction!(record_tokens, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("add_messages", "messages")?;
    module.add("START", crate::START)?;
    module.add("END", crate::END)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
