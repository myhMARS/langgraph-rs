//! Python-backed embedder for the vector-indexed store.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::_internal::error::{Error, Result};
use crate::store::Embedder;

/// Calls a Python `Embeddings` object or a plain callable.
pub(crate) struct PyEmbedder {
    object: Py<PyAny>,
}

impl PyEmbedder {
    pub(crate) fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }

    /// `true` when the object is a flat list of numbers (`[0.1, 0.2]`), not a list of vectors.
    fn is_flat_vector(value: &Bound<'_, PyAny>) -> bool {
        value
            .try_iter()
            .ok()
            .and_then(|mut items| items.next())
            .and_then(|first| first.ok())
            .map(|first| first.extract::<f64>().is_ok())
            .unwrap_or(false)
    }

    fn to_vectors(value: &Bound<'_, PyAny>) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::new();
        let items = value
            .try_iter()
            .map_err(|error| Error::other(format!("embedder did not return a list: {error}")))?;
        for item in items {
            let item = item.map_err(|error| Error::other(error.to_string()))?;
            let mut vector = Vec::new();
            let numbers = item
                .try_iter()
                .map_err(|error| Error::other(format!("embedding is not a list: {error}")))?;
            for number in numbers {
                let number = number.map_err(|error| Error::other(error.to_string()))?;
                let value: f64 = number
                    .extract()
                    .map_err(|error| Error::other(format!("embedding value: {error}")))?;
                vector.push(value as f32);
            }
            vectors.push(vector);
        }
        Ok(vectors)
    }

    fn call_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Python::attach(|py| {
            let object = self.object.bind(py);
            let list = PyList::new(py, texts.clone())
                .map_err(|error| Error::other(error.to_string()))?;
            if object.hasattr("embed_documents").unwrap_or(false) {
                let produced = object
                    .call_method1("embed_documents", (list,))
                    .map_err(|error| Error::other(format!("embed_documents failed: {error}")))?;
                return Self::to_vectors(&produced);
            }
            // A plain callable: first try the batch form, then fall back to one call per text.
            if let Ok(produced) = object.call1((list.clone(),))
                && let Ok(vectors) = Self::to_vectors(&produced)
                && vectors.len() == texts.len()
            {
                return Ok(vectors);
            }
            let mut vectors = Vec::with_capacity(texts.len());
            for text in texts {
                let produced = object
                    .call1((text,))
                    .map_err(|error| Error::other(format!("embedder call failed: {error}")))?;
                // A bare vector (`[0.1, 0.2, ...]`) is the common single-text shape.
                if Self::is_flat_vector(&produced) {
                    let mut vector = Vec::new();
                    for number in produced.try_iter().map_err(|error| Error::other(error.to_string()))? {
                        let number = number.map_err(|error| Error::other(error.to_string()))?;
                        vector.push(number.extract::<f64>().map_err(|error| Error::other(error.to_string()))? as f32);
                    }
                    vectors.push(vector);
                } else {
                    vectors.extend(Self::to_vectors(&produced)?);
                }
            }
            Ok(vectors)
        })
    }
}

impl Embedder for PyEmbedder {
    fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.call_batch(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let uses_query = Python::attach(|py| {
            self.object
                .bind(py)
                .hasattr("embed_query")
                .unwrap_or(false)
        });
        if uses_query {
            let vector = Python::attach(|py| {
                let produced = self
                    .object
                    .bind(py)
                    .call_method1("embed_query", (text,))
                    .map_err(|error| Error::other(format!("embed_query failed: {error}")))?;
                let mut values = Vec::new();
                if let Ok(items) = produced.try_iter() {
                    for item in items.flatten() {
                        if let Ok(number) = item.extract::<f64>() {
                            values.push(number as f32);
                        }
                    }
                }
                Ok::<Vec<f32>, Error>(values)
            })?;
            if !vector.is_empty() {
                return Ok(vector);
            }
        }
        let mut vectors = self.call_batch(vec![text.to_string()])?;
        Ok(vectors.pop().unwrap_or_default())
    }
}