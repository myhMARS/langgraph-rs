//! Python bindings for the prebuilt helpers (`tools_condition`, `ToolNode`,
//! `create_react_agent`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;

use pyo3::types::{PyAny, PyDict};

use super::convert::*;
use crate::_internal::node::NodeOutput;
use crate::_internal::state::{JsonState, StateSchema};
use crate::prebuilt::{
    TOOLS, ToolDefinition, ToolNode, tools_condition as rust_tools_condition,
    tools_condition_with as rust_tools_condition_with,
};
use crate::{END, Result, Route, StateGraph, reducers};

fn tool_definitions(tools: &Bound<'_, PyAny>) -> PyResult<Vec<ToolDefinition>> {
    // (name, callable, annotation source): LangChain tools are called through `invoke` but the
    // injection markers live on the tool object itself.
    let mut entries: Vec<(String, Py<PyAny>, Py<PyAny>, bool)> = Vec::new();
    if let Ok(dict) = tools.cast::<PyDict>() {
        for (name, value) in dict.iter() {
            let name: String = name.extract()?;
            if !value.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "tool `{name}` must be callable"
                )));
            }
            entries.push((name, value.clone().unbind(), value.unbind(), false));
        }
    } else {
        let items = tools.try_iter().map_err(|_| {
            PyTypeError::new_err("tools must be a dict or an iterable of callables")
        })?;
        for (index, item) in items.enumerate() {
            let item = item?;
            if item.is_callable() {
                let name = item
                    .getattr("__name__")
                    .ok()
                    .and_then(|name| name.extract::<String>().ok())
                    .unwrap_or_else(|| format!("tool_{index}"));
                entries.push((name, item.clone().unbind(), item.unbind(), false));
                continue;
            }
            // LangChain BaseTool objects expose `name` and `invoke` instead of
            // being directly callable.
            let name = item
                .getattr("name")
                .ok()
                .and_then(|name| name.extract::<String>().ok())
                .ok_or_else(|| {
                    PyTypeError::new_err(
                        "every tool must be callable or expose `name` and `invoke`",
                    )
                })?;
            let invoke = item.getattr("invoke").map_err(|_| {
                PyTypeError::new_err(format!("tool `{name}` must expose an `invoke` method"))
            })?;
            if !invoke.is_callable() {
                return Err(PyTypeError::new_err(format!(
                    "tool `{name}` invoke attribute is not callable"
                )));
            }
            entries.push((name, invoke.unbind(), item.unbind(), true));
        }
    }
    Ok(entries
        .into_iter()
        .map(|(name, function, source, is_tool)| {
            let injected = Python::attach(|py| injection_spec(py, &source.bind(py).clone()));
            // LangChain calls a plain callable with keyword arguments named after the tool's
            // parameters; the keyword names are collected here so the call site can match that
            // without inspecting the signature on every tool call.
            // LangChain `BaseTool.invoke()` takes the argument mapping positionally, so only
            // plain callables get the keyword convention.
            let keywords = if is_tool {
                KeywordSpec {
                    names: None,
                    accepts_any: false,
                }
            } else {
                Python::attach(|py| keyword_names(py, &function.bind(py).clone()))
            };
            ToolDefinition::with_state(name, move |args, state| {
                Python::attach(|py| -> Result<serde_json::Value> {
                    let py_args =
                        value_to_py(py, &args).map_err(|error| pyerr_to_error(py, error))?;
                    // A failing tool reports the exception's `repr`, which is what LangGraph puts
                    // in the `ToolMessage` (`Error: ValueError('kaboom')`).
                    let failed = |error: pyo3::PyErr| {
                        // An `interrupt()` raised inside a tool must pause the graph, exactly as it
                        // does inside a node; only real failures become ToolMessages.
                        if error.is_instance_of::<crate::adapters::python::GraphInterrupt>(py)
                        {
                            return pyerr_to_error(py, error);
                        }
                        let repr = error
                            .value(py)
                            .repr()
                            .map(|text| text.to_string())
                            .unwrap_or_else(|_| error.to_string());
                        crate::_internal::error::Error::other(repr)
                    };
                    let result = if injected.is_empty() {
                        call_plain_tool(py, &function, py_args.bind(py), &keywords)
                            .map_err(failed)?
                    } else {
                        // LangGraph injects these parameters instead of taking them from the
                        // model: the state mapping and the long-term store object.
                        let kwargs = pyo3::types::PyDict::new(py);
                        if let Ok(mapping) = py_args.bind(py).cast::<pyo3::types::PyDict>() {
                            for (key, value) in mapping.iter() {
                                kwargs
                                    .set_item(key, value)
                                    .map_err(|error| pyerr_to_error(py, error))?;
                            }
                        }
                        for (parameter, injection) in &injected {
                            let value = match injection {
                                Injection::State => {
                                    let state = serde_json::Value::Object(state.0.clone());
                                    value_to_py(py, &state)
                                        .map_err(|error| pyerr_to_error(py, error))?
                                }
                                Injection::Store => match crate::get_store() {
                                    Some(store) => pyo3::IntoPyObjectExt::into_py_any(
                                        super::types::PyStore::new(store),
                                        py,
                                    )
                                    .map_err(|error| pyerr_to_error(py, error))?,
                                    None => py.None(),
                                },
                                // LangGraph fills a `ToolRuntime` handle with what the run knows.
                                Injection::Runtime => (|| -> PyResult<Py<PyAny>> {
                                    // Built natively: the extension owns the handle it injects.
                                    let state = serde_json::Value::Object(state.0.clone());
                                    let runtime = super::types::PyToolRuntime {
                                        state: Some(value_to_py(py, &state)?),
                                        context: match crate::current_context() {
                                            Some(context) => Some(value_to_py(py, &context)?),
                                            None => None,
                                        },
                                        config: None,
                                        stream_writer: None,
                                        tool_call_id: None,
                                        store: match crate::get_store() {
                                            Some(store) => Some(
                                                pyo3::IntoPyObjectExt::into_py_any(
                                                    super::types::PyStore::new(store),
                                                    py,
                                                )?,
                                            ),
                                            None => None,
                                        },
                                        tools: None,
                                        execution_info: None,
                                        server_info: None,
                                    };
                                    Ok(Py::new(py, runtime)?.into_any())
                                })()
                                .map_err(|error| pyerr_to_error(py, error))?
                            };
                            kwargs
                                .set_item(parameter, value)
                                .map_err(|error| pyerr_to_error(py, error))?;
                        }
                        if is_tool {
                            // LangChain tools take the argument mapping as one positional value.
                            function
                                .call1(py, (kwargs,))
                                .map_err(failed)?
                        } else {
                            function
                                .call(py, (), Some(&kwargs))
                                .map_err(failed)?
                        }
                    };
                    py_to_value(result.bind(py)).map_err(|error| pyerr_to_error(py, error))
                })
            })
        })
        .collect())
}

/// Which value LangGraph substitutes for a tool parameter the model never supplies.
#[derive(Clone, Copy)]
enum Injection {
    State,
    Store,
    Runtime,
}

/// Parameters annotated with `InjectedState` / `InjectedStore` / `ToolRuntime`.
///
/// LangChain's `BaseTool` exposes the annotation metadata through `args_schema.model_fields`;
/// plain callables are inspected with `typing.get_type_hints(..., include_extras=True)`.
fn injection_spec(py: Python<'_>, function: &Bound<'_, PyAny>) -> Vec<(String, Injection)> {
    let mut spec = Vec::new();
    let classify = |metadata: &Bound<'_, PyAny>| -> Option<Injection> {
        let name = metadata
            .getattr("__name__")
            .ok()
            .and_then(|name| name.extract::<String>().ok())
            .or_else(|| {
                metadata
                    .get_type()
                    .getattr("__name__")
                    .ok()
                    .and_then(|name| name.extract::<String>().ok())
            })?;
        match name.as_str() {
            "InjectedState" => Some(Injection::State),
            "InjectedStore" => Some(Injection::Store),
            "ToolRuntime" => Some(Injection::Runtime),
            _ => None,
        }
    };
    // `BaseTool` route: the pydantic field metadata carries the marker class.
    if let Ok(schema) = function.getattr("args_schema")
        && let Ok(fields) = schema.getattr("model_fields")
        && let Ok(fields) = fields.cast::<pyo3::types::PyDict>()
    {
        for (name, field) in fields.iter() {
            let Ok(parameter) = name.extract::<String>() else {
                continue;
            };
            let Ok(metadata) = field.getattr("metadata") else {
                continue;
            };
            if let Ok(items) = metadata.try_iter() {
                for item in items.flatten() {
                    if let Some(injection) = classify(&item) {
                        spec.push((parameter.clone(), injection));
                        break;
                    }
                }
            }
            if let Ok(annotation) = field.getattr("annotation")
                && let Some(injection) = classify(&annotation)
                && !spec.iter().any(|(known, _)| known == &parameter)
            {
                spec.push((parameter, injection));
            }
        }
        if !spec.is_empty() {
            return spec;
        }
    }
    // Plain callable route: resolve `Annotated[...]` extras.
    if let Ok(typing) = py.import("typing")
        && let Ok(hints) = typing.getattr("get_type_hints")
    {
        let kwargs = pyo3::types::PyDict::new(py);
        let _ = kwargs.set_item("include_extras", true);
        if let Ok(hints) = hints.call((function,), Some(&kwargs))
            && let Ok(hints) = hints.cast::<pyo3::types::PyDict>()
        {
            for (name, annotation) in hints.iter() {
                let Ok(parameter) = name.extract::<String>() else {
                    continue;
                };
                let Ok(args) = typing
                    .getattr("get_args")
                    .and_then(|get_args| get_args.call1((annotation,)))
                else {
                    continue;
                };
                if let Ok(items) = args.try_iter() {
                    for item in items.flatten().skip(1) {
                        if let Some(injection) = classify(&item) {
                            spec.push((parameter, injection));
                            break;
                        }
                    }
                }
            }
        }
    }
    spec
}

/// Executes tool calls from the last message (LangGraph `ToolNode`).
#[pyclass(name = "ToolNode")]
pub struct PyToolNode {
    node: ToolNode,
}

#[pymethods]
impl PyToolNode {
    #[new]
    #[pyo3(signature = (tools, messages_key=None, handle_tool_errors=None))]
    fn new(
        py: Python<'_>,
        tools: &Bound<'_, PyAny>,
        messages_key: Option<String>,
        handle_tool_errors: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut node = ToolNode::new(tool_definitions(tools)?);
        if let Some(key) = messages_key {
            node = node.messages_key(key);
        }
        if let Some(handler) = handle_tool_errors.filter(|value| !value.is_none()) {
            node = node.error_policy(tool_error_policy(py, handler)?);
        }
        Ok(Self { node })
    }

    /// Execute the pending tool calls and return message updates.
    fn __call__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // LangChain message objects carry `tool_calls` as attributes; the engine's JSON form would
        // only see the constructor payload, so the calls are flattened here first.
        let normalized = flatten_messages(py, state, self.node.messages_key_name())?;
        let state = state_from_py(Some(&normalized))?;
        let output = self
            .node
            .execute(&state)
            .map_err(|error| to_pyerr(py, error))?;
        let dict = PyDict::new(py);
        for (channel, value) in output.updates_map() {
            dict.set_item(channel, value_to_py(py, value)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "ToolNode()".to_string()
    }
}

/// Copy `state` with its message list reduced to plain dicts (`role` / `content` / `tool_calls`).
fn flatten_messages<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let Ok(dict) = state.cast::<PyDict>() else {
        return Ok(state.clone());
    };
    let copy = PyDict::new(py);
    for (name, value) in dict.iter() {
        copy.set_item(name, value)?;
    }
    let Some(messages) = dict.get_item(key)? else {
        return Ok(copy.into_any());
    };
    let Ok(items) = messages.try_iter() else {
        return Ok(copy.into_any());
    };
    let flat = pyo3::types::PyList::empty(py);
    for item in items {
        let item = item?;
        let Ok(message) = item.cast::<PyDict>() else {
            let converted = PyDict::new(py);
            for attribute in ["type", "role"] {
                if let Ok(value) = item.getattr(attribute) {
                    converted.set_item("role", value)?;
                    break;
                }
            }
            if let Ok(value) = item.getattr("content") {
                converted.set_item("content", value)?;
            }
            if let Ok(value) = item.getattr("tool_calls") {
                converted.set_item("tool_calls", value)?;
            }
            if let Ok(value) = item.getattr("id") {
                converted.set_item("id", value)?;
            }
            flat.append(converted)?;
            continue;
        };
        flat.append(message)?;
    }
    copy.set_item(key, flat)?;
    Ok(copy.into_any())
}

/// Original message behind an exception `repr` (`ValueError('kaboom')` -> `kaboom`).
fn exception_text(repr: &str) -> String {
    let Some(open) = repr.find('(') else {
        return repr.to_string();
    };
    let Some(close) = repr.rfind(')') else {
        return repr.to_string();
    };
    let inner = &repr[open + 1..close];
    let trimmed = inner.trim();
    let unquoted = trimmed
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .unwrap_or(trimmed);
    if unquoted.is_empty() {
        repr.to_string()
    } else {
        unquoted.to_string()
    }
}

/// Build LangGraph's `handle_tool_errors` policy.
///
/// `True` uses the default message, a string replaces the message, `False` propagates, and a
/// callable receives the exception and returns the message.
fn tool_error_policy(
    _py: Python<'_>,
    handler: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::prebuilt::ToolErrorPolicy>> {
    if let Ok(enabled) = handler.extract::<bool>() {
        return Ok(enabled.then(crate::prebuilt::default_tool_error_policy));
    }
    if let Ok(text) = handler.extract::<String>() {
        return Ok(Some(Arc::new(move |_message: &str| Some(text.clone()))));
    }
    if handler.is_callable() {
        let callable = handler.clone().unbind();
        return Ok(Some(Arc::new(move |message: &str| {
            Python::attach(|py| -> Option<String> {
                // LangGraph hands the handler the original exception; the engine keeps its `repr`
                // (`ValueError('kaboom')`), so the payload is unwrapped back into the message the
                // handler would have seen from `str(exc)`.
                let error = pyo3::exceptions::PyRuntimeError::new_err(exception_text(message));
                match callable.bind(py).call1((error.value(py),)) {
                    Ok(value) => value.extract::<String>().ok(),
                    Err(_) => None,
                }
            })
        })));
    }
    // A tuple/type of exceptions only filters what propagates; the extension has no exception
    // object at this point, so those all use the default message.
    Ok(Some(crate::prebuilt::default_tool_error_policy()))
}

/// Route to "tools" when the last message has tool calls, END otherwise.
#[pyfunction]
pub(super) fn tools_condition(state: &Bound<'_, PyAny>) -> PyResult<String> {
    let state = state_from_py(Some(state))?;
    let route = rust_tools_condition(&state);
    Ok(route
        .targets()
        .first()
        .cloned()
        .unwrap_or_else(|| END.to_string()))
}

/// State key LangGraph uses for create_react_agent(response_format=...) results.
const STRUCTURED_KEY: &str = "structured_response";

/// How create_react_agent(prompt=...) shapes the model input.
///
/// LangGraph accepts a string (turned into a system message), a `SystemMessage`, or a callable
/// that receives the state and returns the complete message list.
enum PromptSpec {
    None,
    Static(serde_json::Value),
    Callable(Py<PyAny>),
}

fn prompt_spec(_py: Python<'_>, prompt: Option<&Bound<'_, PyAny>>) -> PyResult<PromptSpec> {
    let Some(prompt) = prompt else {
        return Ok(PromptSpec::None);
    };
    if prompt.is_none() {
        return Ok(PromptSpec::None);
    }
    if let Ok(text) = prompt.cast::<pyo3::types::PyString>() {
        let text: String = text.extract()?;
        return Ok(PromptSpec::Static(serde_json::json!({
            "role": "system",
            "content": text,
        })));
    }
    if prompt.is_callable() {
        return Ok(PromptSpec::Callable(prompt.clone().unbind()));
    }
    let dumped = prompt
        .call_method0("model_dump")
        .or_else(|_| prompt.call_method0("dict"))?;
    let value = py_to_value(&dumped)?;
    Ok(PromptSpec::Static(value))
}

/// Build the message list handed to the model for this superstep.
fn prompt_messages(
    py: Python<'_>,
    prompt: &PromptSpec,
    state: &JsonState,
    messages_key: &str,
) -> Result<serde_json::Value> {
    let existing = state.get(messages_key).cloned();
    match prompt {
        PromptSpec::None => Ok(existing.unwrap_or_else(|| serde_json::Value::Array(Vec::new()))),
        PromptSpec::Static(prefix) => {
            let mut messages = vec![prefix.clone()];
            if let Some(Some(serde_json::Value::Array(items))) = existing.as_ref().map(Some) {
                messages.extend(items.iter().cloned());
            }
            Ok(serde_json::Value::Array(messages))
        }
        PromptSpec::Callable(callable) => {
            let py_state = value_to_py(py, &serde_json::Value::Object(state.0.clone()))
                .map_err(|error| pyerr_to_error(py, error))?;
            let produced = callable
                .bind(py)
                .call1((py_state,))
                .map_err(|error| pyerr_to_error(py, error))?;
            let items = produced
                .try_iter()
                .map_err(|error| pyerr_to_error(py, error))?;
            let mut messages = Vec::new();
            for item in items {
                let item = item.map_err(|error| pyerr_to_error(py, error))?;
                let dumped = item
                    .call_method0("model_dump")
                    .or_else(|_| item.call_method0("dict"))
                    .unwrap_or_else(|_| item.clone());
                messages.push(py_to_value(&dumped).map_err(|error| pyerr_to_error(py, error))?);
            }
            Ok(serde_json::Value::Array(messages))
        }
    }
}

/// Run the model, streaming chunks when it supports `stream`.
///
/// Returns the final message (the aggregated chunks when streaming worked, otherwise the model's
/// `invoke` result) together with whether streaming produced anything. Each chunk is forwarded on
/// the graph's `messages` stream so `stream_mode="messages"` and `run.messages` see it.
fn streamed_model_result(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    messages: &Bound<'_, PyAny>,
) -> Result<(Py<PyAny>, bool)> {
    // LangGraph calls the model once; when a streaming callback handler is installed, LangChain
    // itself switches `invoke` to the streaming path, so the tokens reach the `messages` stream and
    // the call still returns the aggregated message. Streaming here explicitly would consume an
    // extra message from models that produce a fixed sequence.
    let _ = py;
    let result = model
        .call_method1("invoke", (messages.clone(),))
        .map_err(|error| pyerr_to_error(py, error))?;
    Ok((result.unbind(), false))
}

/// How a plain callable can be called: by keyword names, or anything (`**kwargs`).
struct KeywordSpec {
    /// Parameter names callable as keywords; `None` when the signature is unavailable.
    names: Option<Vec<String>>,
    /// The callable takes `**kwargs`, so any keyword works.
    accepts_any: bool,
}

/// Inspect a plain callable's keyword parameters (excluding `self`).
fn keyword_names(py: Python<'_>, function: &Bound<'_, PyAny>) -> KeywordSpec {
    let unknown = KeywordSpec {
        names: None,
        accepts_any: false,
    };
    let Ok(inspect) = py.import("inspect") else {
        return unknown;
    };
    let Ok(signature) = inspect.call_method1("signature", (function,)) else {
        return unknown;
    };
    let Ok(parameters) = signature.getattr("parameters") else {
        return unknown;
    };
    let Ok(items) = parameters.call_method0("items") else {
        return unknown;
    };
    let Ok(iter) = items.try_iter() else {
        return unknown;
    };
    let mut names = Vec::new();
    let mut accepts_any = false;
    for item in iter {
        let Ok(item) = item else {
            return unknown;
        };
        let name = item
            .get_item(0)
            .ok()
            .and_then(|name| name.extract::<String>().ok());
        let kind = item
            .get_item(1)
            .ok()
            .and_then(|parameter| parameter.getattr("kind").ok())
            .and_then(|kind| kind.extract::<i32>().ok());
        match (name, kind) {
            // VAR_KEYWORD: every keyword is accepted.
            (_, Some(4)) => accepts_any = true,
            // POSITIONAL_ONLY cannot be called by keyword.
            (_, Some(0)) => {}
            // POSITIONAL_OR_KEYWORD and KEYWORD_ONLY are callable by name.
            (Some(name), Some(1 | 3)) if name != "self" => names.push(name),
            _ => {}
        }
    }
    KeywordSpec {
        names: Some(names),
        accepts_any,
    }
}

/// Call a plain callable the way LangChain does: keywords named after its parameters.
///
/// Falls back to passing the argument mapping as the single positional value when the tool does not
/// accept those keywords (for example `def tool(args: dict)`), which keeps the older style working.
fn call_plain_tool<'py>(
    py: Python<'py>,
    function: &Py<PyAny>,
    args: &Bound<'py, PyAny>,
    keywords: &KeywordSpec,
) -> PyResult<Py<PyAny>> {
    let mapping = args.cast::<pyo3::types::PyDict>().ok();
    let mut use_keywords = keywords.accepts_any;
    if !use_keywords
        && let (Some(names), Some(mapping)) = (keywords.names.as_ref(), mapping)
    {
        use_keywords = mapping.iter().all(|(key, _)| {
            key.extract::<String>()
                .map(|key| names.contains(&key))
                .unwrap_or(false)
        });
    }
    if use_keywords && let Some(mapping) = mapping {
        let kwargs = pyo3::types::PyDict::new(py);
        for (key, value) in mapping.iter() {
            kwargs.set_item(key, value)?;
        }
        return function.call(py, (), Some(&kwargs));
    }
    function.call1(py, (args,))
}

fn runnable_model_node(
    model: Py<PyAny>,
    messages_key: String,
    prompt: std::sync::Arc<PromptSpec>,
    structured: std::sync::Arc<Option<Py<PyAny>>>,
) -> impl Fn(&JsonState) -> Result<NodeOutput> + Send + Sync + 'static {
    let model = std::sync::Arc::new(model);
    move |state: &JsonState| {
        Python::attach(|py| -> Result<NodeOutput> {
            // This node runs inside the engine rather than through the Python node wrapper, so it
            // has to install the LangChain callback context itself; without it the model's tokens
            // never reach the `messages` stream.
            let _callbacks = super::convert::begin_callback_scope(py);
            let messages = prompt_messages(py, &prompt, state, &messages_key)?;
            let py_messages =
                value_to_py(py, &messages).map_err(|error| pyerr_to_error(py, error))?;
            // Stream the model when it supports it, forwarding each chunk on the `messages`
            // stream and aggregating the chunks into the final message (LangChain's `+`).
            let (result, streamed) = streamed_model_result(py, model.bind(py), py_messages.bind(py))?;
            let dumped = result
                .bind(py)
                .call_method0("model_dump")
                .or_else(|_| result.bind(py).call_method0("dict"))
                .map_err(|error| pyerr_to_error(py, error))?;
            let _ = streamed;
            let value = py_to_value(&dumped).map_err(|error| pyerr_to_error(py, error))?;
            let message = match value {
                serde_json::Value::Array(mut values) => {
                    values.pop().unwrap_or(serde_json::Value::Null)
                }
                other => other,
            };
            let mut output =
                NodeOutput::new().update(&messages_key, serde_json::Value::Array(vec![message.clone()]));
            // When the model finished (no tool calls) and a structured output was requested,
            // LangGraph stores the parsed object under `structured_response`.
            let has_calls = message
                .get("tool_calls")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|calls| !calls.is_empty());
            if let (Some(structured), false) = (structured.as_ref().as_ref(), has_calls) {
                let parsed = structured
                    .bind(py)
                    .call_method1("invoke", (py_messages.clone_ref(py),))
                    .map_err(|error| pyerr_to_error(py, error))?;
                let dumped = parsed
                    .call_method0("model_dump")
                    .or_else(|_| parsed.call_method0("dict"))
                    .unwrap_or_else(|_| parsed.clone());
                let value = py_to_value(&dumped).map_err(|error| pyerr_to_error(py, error))?;
                output = output.set(STRUCTURED_KEY, value);
            }
            Ok(output)
        })
    }
}

fn guard_react_output(messages_key: &str, output: NodeOutput) -> NodeOutput {
    if !crate::remaining_steps().is_some_and(|remaining| remaining <= 2) {
        return output;
    }
    let calls_tools = output
        .updates_map()
        .get(messages_key)
        .and_then(serde_json::Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("tool_calls"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|calls| !calls.is_empty());
    if !calls_tools {
        return output;
    }
    let mut guarded = NodeOutput::new();
    for (channel, value) in output.updates_map() {
        if channel != messages_key {
            guarded = guarded.update(channel.clone(), value.clone());
        }
    }
    guarded.update(
        messages_key,
        serde_json::json!([{
            "role": "assistant",
            "content": "Sorry, need more steps to process this request.",
        }]),
    )
}

/// Build a ReAct agent, matching LangGraph's `create_react_agent` signature.
#[pyfunction]
#[pyo3(signature = (model, tools, messages_key=None, checkpointer=None, prompt=None, state_schema=None, store=None, interrupt_before=None, interrupt_after=None, version=None, name=None, debug=false, response_format=None, pre_model_hook=None, post_model_hook=None, context_schema=None))]
#[allow(clippy::too_many_arguments)]
pub(super) fn create_react_agent(
    py: Python<'_>,
    model: Py<PyAny>,
    tools: &Bound<'_, PyAny>,
    messages_key: Option<String>,
    checkpointer: Option<&Bound<'_, PyAny>>,
    prompt: Option<&Bound<'_, PyAny>>,
    state_schema: Option<&Bound<'_, PyAny>>,
    store: Option<&Bound<'_, PyAny>>,
    interrupt_before: Option<Vec<String>>,
    interrupt_after: Option<Vec<String>>,
    version: Option<String>,
    name: Option<String>,
    debug: bool,
    response_format: Option<&Bound<'_, PyAny>>,
    pre_model_hook: Option<&Bound<'_, PyAny>>,
    post_model_hook: Option<&Bound<'_, PyAny>>,
    context_schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<super::graph::PyCompiledGraph> {
    let _ = debug;
    let response_format = response_format.filter(|value| !value.is_none()).cloned();
    let pre_model_hook = pre_model_hook.filter(|value| !value.is_none()).map(|value| value.clone().unbind());
    let post_model_hook = post_model_hook.filter(|value| !value.is_none()).map(|value| value.clone().unbind());
    let context_schema = context_schema.filter(|value| !value.is_none()).cloned();
    if let Some(version) = version.as_deref()
        && !matches!(version, "v1" | "v2")
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "create_react_agent version must be 'v1' or 'v2', got {version:?}"
        )));
    }
    let prompt = prompt_spec(py, prompt)?;
    let prompt = std::sync::Arc::new(prompt);
    let mut tool_node = ToolNode::new(tool_definitions(tools)?);
    if let Some(key) = messages_key.clone() {
        tool_node = tool_node.messages_key(key);
    }
    let messages_key = messages_key.unwrap_or_else(|| "messages".to_string());
    let condition_key = messages_key.clone();
    let base_schema = match state_schema {
        Some(schema) if !schema.is_none() => super::convert::schema_from_py(Some(schema))?,
        // LangGraph's default agent state declares only `messages` (+ managed values), so extra
        // input keys are dropped instead of being carried through the run.
        _ => StateSchema::new(),
    };
    let schema = base_schema
        .with_reducer(&messages_key, reducers::messages())
        .last_value(STRUCTURED_KEY)
        .with_managed_values();
    // `response_format` binds a structured-output model once; the agent node uses it to turn the
    // final answer into `structured_response`, exactly like LangGraph.
    let structured = std::sync::Arc::new(match &response_format {
        Some(format) => Some(
            model
                .bind(py)
                .call_method1("with_structured_output", (format,))?
                .unbind(),
        ),
        None => None,
    });
    let builder = StateGraph::<JsonState>::with_schema(schema);
    let inner = if model.bind(py).hasattr("bind_tools")? {
        let bound = model
            .bind(py)
            .call_method1("bind_tools", (tools,))?
            .unbind();
        let node = runnable_model_node(
            bound,
            messages_key.clone(),
            prompt.clone(),
            structured.clone(),
        );
        let key = messages_key.clone();
        builder.add_node("agent", move |state: &JsonState| {
            Ok(guard_react_output(&key, node(state)?))
        })
    } else {
        if !matches!(*prompt, PromptSpec::None) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "create_react_agent(prompt=...) needs a model with `bind_tools`/`invoke` \
                 (a LangChain Runnable); a plain callable receives the state instead",
            ));
        }
        let node = node_from_python("agent".to_string(), model);
        let key = messages_key.clone();
        builder.add_node("agent", move |state: &JsonState| {
            Ok(guard_react_output(&key, node(state)?))
        })
    }
    .add_node(TOOLS, move |state: &JsonState| tool_node.execute(state));
    // `pre_model_hook` runs before the model node, `post_model_hook` after it; the tools/END
    // decision moves to whichever node runs last.
    let mut builder = inner;
    // LangGraph routes `tools -> entrypoint`, so a `pre_model_hook` runs before every model call.
    let entry = if let Some(hook) = pre_model_hook {
        builder = builder
            .add_node(
                "pre_model_hook",
                node_from_python("pre_model_hook".to_string(), hook),
            )
            .set_entry_point("pre_model_hook")
            .add_edge("pre_model_hook", "agent");
        "pre_model_hook"
    } else {
        builder = builder.set_entry_point("agent");
        "agent"
    };
    let condition_source = if let Some(hook) = post_model_hook {
        builder = builder
            .add_node(
                "post_model_hook",
                node_from_python("post_model_hook".to_string(), hook),
            )
            .add_edge("agent", "post_model_hook");
        "post_model_hook"
    } else {
        "agent"
    };
    let inner = builder
        .add_conditional_edges(condition_source, move |state: &JsonState| {
            Ok::<Route, crate::Error>(rust_tools_condition_with(state, &condition_key))
        })
        .add_edge(TOOLS, entry);
    let mut options = crate::CompileOptions::new();
    if let Some(schema) = context_schema {
        options = options.with_context_schema(super::convert::context_schema_value(py, &schema)?);
    }
    let mut saver = None;
    if let Some(checkpointer) = checkpointer {
        if let Some(created) = super::types::checkpointer_from_py(Some(checkpointer))? {
            options = options.with_checkpointer(created.clone());
            saver = Some(created);
        }
    }
    if let Some(nodes) = interrupt_before {
        options = options.interrupt_before(nodes);
    }
    if let Some(nodes) = interrupt_after {
        options = options.interrupt_after(nodes);
    }
    if let Some(name) = name {
        options = options.with_name(name);
    }
    if let Some(store) = store
        && let Some(store) = super::types::store_from_py(Some(store))?
    {
        options = options.with_store(store);
    }
    let compiled = inner
        .compile_with(options)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(super::graph::PyCompiledGraph::from_inner_with_saver(
        compiled, saver,
    ))
}
