//! Lazy `StateHistory` sequence returned by `get_state_history`.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;

use crate::graph::CompiledGraph;
use crate::JsonState;

/// Source of history handles; implemented by the sync and async compiled graphs.
pub(crate) trait HistorySource: Send + Sync {
    fn history_handles_for(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Arc<crate::Checkpoint>>, crate::Error>;

    fn snapshot_from_handle_for(
        &self,
        handle: &Arc<crate::Checkpoint>,
    ) -> Result<crate::StateSnapshot<JsonState>, crate::Error>;
}

impl HistorySource for CompiledGraph<JsonState> {
    fn history_handles_for(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Arc<crate::Checkpoint>>, crate::Error> {
        self.history_handles(thread_id, namespace, before, limit)
    }

    fn snapshot_from_handle_for(
        &self,
        handle: &Arc<crate::Checkpoint>,
    ) -> Result<crate::StateSnapshot<JsonState>, crate::Error> {
        self.snapshot_from_handle(handle)
    }
}

impl HistorySource for crate::adapters::tokio_runtime::AsyncCompiledGraph<JsonState> {
    fn history_handles_for(
        &self,
        thread_id: &str,
        namespace: &str,
        before: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Arc<crate::Checkpoint>>, crate::Error> {
        self.history_handles(thread_id, namespace, before, limit)
    }

    fn snapshot_from_handle_for(
        &self,
        handle: &Arc<crate::Checkpoint>,
    ) -> Result<crate::StateSnapshot<JsonState>, crate::Error> {
        self.snapshot_from_handle(handle)
    }
}

struct HistoryInner {
    graph: Option<Arc<dyn HistorySource>>,
    thread_id: String,
    namespace: String,
    before: Option<String>,
    limit: Option<usize>,
    /// Checkpoint handles, loaded on first access (cheap for in-memory savers).
    handles: Option<Vec<Arc<crate::Checkpoint>>>,
    /// Snapshots converted so far; index-aligned with `handles`.
    cache: Vec<Option<Py<PyAny>>>,
    /// Filtered queries are resolved eagerly into this list instead.
    eager: Option<Vec<Py<PyAny>>>,
}

impl HistoryInner {
    fn load(&mut self) -> Result<(), crate::Error> {
        if self.handles.is_none() && self.eager.is_none() {
            let Some(graph) = &self.graph else {
                return Ok(());
            };
            let handles = graph.history_handles_for(
                &self.thread_id,
                &self.namespace,
                self.before.as_deref(),
                self.limit,
            )?;
            self.cache = (0..handles.len()).map(|_| None).collect();
            self.handles = Some(handles);
        }
        Ok(())
    }

    fn total(&mut self) -> Result<usize, crate::Error> {
        if let Some(eager) = &self.eager {
            return Ok(eager.len());
        }
        self.load()?;
        Ok(self.handles.as_ref().map(Vec::len).unwrap_or_default())
    }

    fn get(&mut self, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        if let Some(eager) = &self.eager {
            return eager
                .get(index)
                .map(|item| item.clone_ref(py))
                .ok_or_else(|| PyIndexError::new_err("history index out of range"));
        }
        self.load().map_err(|error| {
            PyRuntimeError::new_err(format!("failed to load state history: {error}"))
        })?;
        let Some(handles) = &self.handles else {
            return Err(PyIndexError::new_err("history index out of range"));
        };
        let Some(handle) = handles.get(index) else {
            return Err(PyIndexError::new_err("history index out of range"));
        };
        if let Some(cached) = self.cache.get(index).and_then(Option::as_ref) {
            return Ok(cached.clone_ref(py));
        }
        let Some(graph) = &self.graph else {
            return Err(PyIndexError::new_err("history index out of range"));
        };
        let snapshot = graph
            .snapshot_from_handle_for(handle)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        // A row's tasks ran in the step recorded by the row written *after* it (history is newest
        // first), so its per-task writes — LangGraph's `PregelTask.result` — live there.
        let converted = crate::adapters::python::convert::snapshot_to_py(py, &snapshot)?;
        if let Some(slot) = self.cache.get_mut(index) {
            *slot = Some(converted.clone_ref(py));
        }
        Ok(converted)
    }
}

/// Lazy, list-like view of a thread's checkpoint history.
///
/// Iteration, `len()` and indexing all work like they do on the list this used to return, but
/// checkpoints are only converted to Python snapshots when they are actually read. LangGraph
/// returns a lazy generator here; converting eagerly made "give me the latest checkpoint" cost
/// the whole history.
#[pyclass(name = "StateHistory")]
pub struct PyStateHistory {
    inner: Arc<Mutex<HistoryInner>>,
    /// Cursor for `next(history)`: LangGraph returns a generator, so callers may pull items.
    cursor: Mutex<usize>,
}

impl PyStateHistory {
    pub(crate) fn new(
        graph: Arc<dyn HistorySource>,
        thread_id: String,
        namespace: String,
        before: Option<String>,
        limit: Option<usize>,
    ) -> Self {
        Self {
            cursor: Mutex::new(0),
            inner: Arc::new(Mutex::new(HistoryInner {
                graph: Some(graph),
                thread_id,
                namespace,
                before,
                limit,
                handles: None,
                cache: Vec::new(),
                eager: None,
            })),
        }
    }

    pub(crate) fn from_items(items: Vec<Py<PyAny>>) -> Self {
        Self {
            cursor: Mutex::new(0),
            inner: Arc::new(Mutex::new(HistoryInner {
                graph: None,
                thread_id: String::new(),
                namespace: String::new(),
                before: None,
                limit: None,
                handles: None,
                cache: Vec::new(),
                eager: Some(items),
            })),
        }
    }
}

#[pymethods]
impl PyStateHistory {
    fn __len__(&self) -> PyResult<usize> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("state history lock poisoned"))?;
        inner
            .total()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("state history lock poisoned"))?;
        let total = inner
            .total()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))? as isize;
        let resolved = if index < 0 { total + index } else { index };
        if resolved < 0 || resolved >= total {
            return Err(PyIndexError::new_err("history index out of range"));
        }
        inner.get(py, resolved as usize)
    }

    fn __iter__(&self) -> PyStateHistoryIter {
        PyStateHistoryIter {
            inner: Arc::clone(&self.inner),
            index: 0,
        }
    }

    /// Pull the next snapshot, like `next()` on LangGraph's history generator.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut cursor = self
            .cursor
            .lock()
            .map_err(|_| PyRuntimeError::new_err("state history cursor poisoned"))?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("state history lock poisoned"))?;
        let total = inner
            .total()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        if *cursor >= total {
            return Ok(None);
        }
        let item = inner.get(py, *cursor)?;
        *cursor += 1;
        Ok(Some(item))
    }

    /// Materialise the whole history as a list (what the old return type did).
    fn to_list(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut items = Vec::new();
        let mut index = 0usize;
        loop {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("state history lock poisoned"))?;
            let total = inner
                .total()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            if index >= total {
                return Ok(items);
            }
            items.push(inner.get(py, index)?);
            drop(inner);
            index += 1;
        }
    }
}

/// Iterator over a [`PyStateHistory`].
#[pyclass(name = "StateHistoryIterator")]
pub struct PyStateHistoryIter {
    inner: Arc<Mutex<HistoryInner>>,
    index: usize,
}

#[pymethods]
impl PyStateHistoryIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("state history lock poisoned"))?;
        let total = inner
            .total()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        if self.index >= total {
            return Ok(None);
        }
        let item = inner.get(py, self.index)?;
        self.index += 1;
        Ok(Some(item))
    }
}