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
    match crate::runtime::context::interrupt_with_schema(value, response_schema) {
        Ok(resume) => value_to_py(py, &resume),
        Err(error) => Err(to_pyerr(py, error)),
    }
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
        # LLM runs that already streamed tokens; a run without tokens still has to reach
        # the `messages` stream once, the way LangGraph reports non-streaming models.
        self._streamed = set()

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
            # LangGraph counts supersteps from 1 in this metadata (its first superstep is step 1),
            # while the engine's step counter starts at 0.
            metadata["langgraph_step"] = step + 1
        triggers = langgraph_rs.current_triggers()
        if triggers:
            metadata["langgraph_triggers"] = tuple(triggers)
        if node:
            # LangGraph reports the task's pull path; subgraph namespaces extend it segment by segment.
            path = ["__pregel_pull", node]
            if namespace:
                path.extend(part for part in namespace.split("|") if part)
            metadata["langgraph_path"] = tuple(path)
        # LangGraph yields a message *chunk*; keep the fields needed to rebuild one on the
        # Python side instead of flattening the token to a string.
        if hasattr(token, "content"):
            payload = {
                "content": token.content,
                "id": getattr(token, "id", None),
                "tool_call_chunks": list(getattr(token, "tool_call_chunks", []) or []),
                "additional_kwargs": dict(getattr(token, "additional_kwargs", {}) or {}),
                "response_metadata": dict(getattr(token, "response_metadata", {}) or {}),
                "tags": self.tags,
                "metadata": metadata,
            }
        else:
            payload = {"content": token, "tags": self.tags, "metadata": metadata}
        # A live `stream_events` run reads tokens straight from this bridge, so `run.messages`
        # shows deltas while the model is still generating rather than when the node returns.
        try:
            import langgraph_rs._live as live

            active = live.active_bridge(langgraph_rs.current_thread_id())
            if active is not None:
                active.push(payload, metadata)
        except Exception:  # pragma: no cover - the shim always ships `_live`
            pass
        langgraph_rs.message(payload)

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
            run_id = kwargs.get("run_id")
            if run_id is not None:
                self._streamed.add(run_id)
            self._emit(token, kwargs)

    def _final_message_text(self, response):
        try:
            generation = response.generations[0][0]
        except Exception:
            return None
        message = getattr(generation, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "".join(parts) or None
            return None
        text = getattr(generation, "text", None)
        return text if isinstance(text, str) else None

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
        # LangGraph emits the completed message when the model did not stream tokens.
        run_id = kwargs.get("run_id")
        if run_id not in self._streamed:
            text = self._final_message_text(response)
            if text:
                self._emit(text, kwargs)

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
    }
    let schema_module = pyo3::types::PyModule::from_code(
        module.py(),
        c"import typing\n\nclass MessagesState(dict):\n    __annotations__ = {'messages': typing.Annotated[list, 'add_messages']}\n\ndef _schema_from_class(cls):\n    # `typing.get_type_hints` resolves PEP 563 string annotations (``from __future__\n    # import annotations`` maps every annotation to a string), which a plain\n    # `__annotations__` walk would silently treat as a last-value channel.\n    try:\n        annotations = typing.get_type_hints(cls, include_extras=True)\n    except Exception:\n        annotations = {}\n        for base in reversed(getattr(cls, '__mro__', [cls])):\n            annotations.update(getattr(base, '__annotations__', {}))\n    out = {}\n    for name, annotation in annotations.items():\n        reducer = 'last_value'\n        if getattr(annotation, '__name__', None) == 'IsLastStep':\n            reducer = 'is_last_step'\n        elif getattr(annotation, '__name__', None) == 'RemainingSteps':\n            reducer = 'remaining_steps'\n        elif typing.get_origin(annotation) is typing.Annotated:\n            args = typing.get_args(annotation)\n            reducer = args[1]\n        out[name] = reducer\n    return out\n",
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
    module.add_class::<PyInterrupt>()?;
    module.add_class::<PyStreamWriter>()?;
    module.add_class::<PyRunControl>()?;
    module.add_function(wrap_pyfunction!(current_context, module)?)?;
    module.add_function(wrap_pyfunction!(current_thread_id, module)?)?;
    module.add_function(wrap_pyfunction!(current_namespace, module)?)?;
    module.add_function(wrap_pyfunction!(current_node, module)?)?;
    module.add_function(wrap_pyfunction!(current_triggers, module)?)?;
    module.add_function(wrap_pyfunction!(current_step, module)?)?;
    module.add_function(wrap_pyfunction!(current_resume, module)?)?;
    module.add_function(wrap_pyfunction!(current_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(remaining_steps, module)?)?;
    module.add_function(wrap_pyfunction!(is_last_step, module)?)?;
    module.add("GraphBubbleUp", module.py().get_type::<GraphBubbleUp>())?;
    module.add("GraphInterrupt", module.py().get_type::<GraphInterrupt>())?;
    module.add_function(wrap_pyfunction!(_send, module)?)?;
    module.add("GraphRecursionError", module.py().get_type::<GraphRecursionError>())?;
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
