//! Functional API approximation: `entrypoint` + `task`.
//!
//! LangGraph's functional API lets an entrypoint function call `@task` functions while the
//! engine checkpoints task results. This module provides the same programming model on top
//! of a single-node graph:
//!
//! * [`task`] wraps a function. When a `BaseStore` is attached to the run, every task result
//!   is keyed by `(task name, serialized input)` and stored in the store; a later call with
//!   the same input returns the stored result instead of running the function again.
//! * [`entrypoint`] returns a [`StateGraph`] with one `entrypoint` node, so checkpoints,
//!   interrupts/resume, streaming, timeouts and the rest of the engine keep working.
//!
//! Because completed task results are memoized, re-running the entrypoint after an
//! `interrupt()` does not repeat finished tasks — the practical checkpointing behaviour of
//! the functional API. Tasks are still executed inline (no parallel task scheduling), which
//! is documented as a limitation in the feature matrix.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::_internal::node::NodeOutput;
use crate::_internal::state::{State, state_to_values};
use crate::graph::StateGraph;
use crate::{Result, get_store};

/// A task function that memoizes its result in the run's long-term memory store.
#[derive(Clone)]
pub struct TaskCall<I, O> {
    name: String,
    run: Arc<dyn Fn(I) -> Result<O> + Send + Sync>,
}

impl<I, O> std::fmt::Debug for TaskCall<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskCall")
            .field("name", &self.name)
            .finish()
    }
}

impl<I, O> TaskCall<I, O>
where
    I: Serialize,
    O: DeserializeOwned + Serialize,
{
    /// Name of the task (used as the store namespace).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the task, reusing a previously stored result for the same input when possible.
    pub fn call(&self, input: I) -> Result<O> {
        let key = task_cache_key(&input)?;
        let cache_key = format!("{}:{}", self.name, key);
        if let Some(cached) = crate::runtime::context::task_result_get(&cache_key) {
            return Ok(serde_json::from_value(cached)?);
        }
        let store = get_store();
        let namespace = ["functional".to_string(), self.name.clone()];
        if let Some(store) = &store
            && let Some(cached) = store.get(&namespace, &key)?
        {
            return Ok(serde_json::from_value(cached)?);
        }
        let output = (self.run)(input)?;
        let value = serde_json::to_value(&output)?;
        crate::runtime::context::task_result_put(cache_key, value.clone());
        if let Some(store) = store {
            store.put(&namespace, &key, value)?;
        }
        Ok(output)
    }
}

/// Create a memoized task function.
pub fn task<I, O, F>(name: impl Into<String>, run: F) -> TaskCall<I, O>
where
    I: Serialize,
    O: DeserializeOwned + Serialize,
    F: Fn(I) -> Result<O> + Send + Sync + 'static,
{
    TaskCall {
        name: name.into(),
        run: Arc::new(run),
    }
}

/// Build a graph whose single `entrypoint` node runs `function`.
///
/// The function receives the current state (by value) and returns the next state; tasks it
/// calls with [`TaskCall::call`] are memoized when the graph has a `BaseStore`.
pub fn entrypoint<S, F>(function: F) -> StateGraph<S>
where
    S: State,
    F: Fn(S) -> Result<S> + Send + Sync + 'static,
{
    StateGraph::<S>::new()
        .add_node("entrypoint", move |state: &S| {
            let next = function(state.clone())?;
            let values = state_to_values(&next)?;
            let mut output = NodeOutput::new();
            for (channel, value) in values {
                output = output.replace(channel, value);
            }
            Ok(output)
        })
        .set_entry_point("entrypoint")
        .set_finish_point("entrypoint")
}

#[cfg(feature = "async")]
/// Future returned by [`AsyncTaskCall::call`].
pub type TaskFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send + 'a>>;

#[cfg(feature = "async")]
/// Async task function that memoizes its result like [`TaskCall`].
#[derive(Clone)]
pub struct AsyncTaskCall<I, O> {
    name: String,
    run: Arc<dyn Fn(I) -> TaskFuture<'static, O> + Send + Sync>,
}

#[cfg(feature = "async")]
impl<I, O> std::fmt::Debug for AsyncTaskCall<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncTaskCall")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(feature = "async")]
impl<I, O> AsyncTaskCall<I, O>
where
    I: Serialize + Send,
    O: DeserializeOwned + Serialize + Send,
{
    /// Name of the task.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run the task as a future; independent tasks can be awaited concurrently with
    /// `futures::future::join!` or `tokio::join!`.
    pub async fn call(&self, input: I) -> Result<O> {
        let key = task_cache_key(&input)?;
        let cache_key = format!("{}:{}", self.name, key);
        if let Some(cached) = crate::runtime::context::task_result_get(&cache_key) {
            return Ok(serde_json::from_value(cached)?);
        }
        let store = get_store();
        let namespace = ["functional".to_string(), self.name.clone()];
        if let Some(store) = &store
            && let Some(cached) = store.get(&namespace, &key)?
        {
            return Ok(serde_json::from_value(cached)?);
        }
        let output = (self.run)(input).await?;
        let value = serde_json::to_value(&output)?;
        crate::runtime::context::task_result_put(cache_key, value.clone());
        if let Some(store) = store {
            store.put(&namespace, &key, value)?;
        }
        Ok(output)
    }
}

#[cfg(feature = "async")]
/// Create an async memoized task.
pub fn async_task<I, O, F, Fut>(name: impl Into<String>, run: F) -> AsyncTaskCall<I, O>
where
    I: Serialize + Send + 'static,
    O: DeserializeOwned + Serialize + Send + 'static,
    F: Fn(I) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<O>> + Send + 'static,
{
    AsyncTaskCall {
        name: name.into(),
        run: Arc::new(move |input| Box::pin(run(input))),
    }
}

#[cfg(feature = "async")]
/// Build an async graph whose single `entrypoint` node awaits `function`.
///
/// The function can await several [`AsyncTaskCall::call`] futures concurrently; each task
/// still memoizes its result in the checkpoint task cache / store.
pub fn async_entrypoint<S, F, Fut>(function: F) -> crate::tokio_runtime::AsyncStateGraph<S>
where
    S: State + Send + 'static,
    F: Fn(S) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<S>> + Send + 'static,
{
    crate::tokio_runtime::AsyncStateGraph::<S>::new()
        .add_node("entrypoint", move |state: S, _context| {
            let future = function(state);
            async move {
                let next = future.await?;
                let values = state_to_values(&next)?;
                let mut output = NodeOutput::new();
                for (channel, value) in values {
                    output = output.replace(channel, value);
                }
                Ok(output)
            }
        })
        .set_entry_point("entrypoint")
        .set_finish_point("entrypoint")
}

fn task_cache_key<I: Serialize>(input: &I) -> Result<String> {
    let value = serde_json::to_value(input)?;
    let mut hasher = DefaultHasher::new();
    value.to_string().hash(&mut hasher);
    Ok(format!("{:x}", hasher.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_exposes_its_name() {
        let add = task("add", |input: i64| Ok(input + 1));
        assert_eq!(add.name(), "add");
    }
}
