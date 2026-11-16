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

use crate::kernel::graph::StateGraph;
use crate::kernel::node::NodeOutput;
use crate::kernel::state::{State, state_to_values};
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
        let Some(store) = get_store() else {
            return (self.run)(input);
        };
        let key = task_cache_key(&input)?;
        let namespace = ["functional".to_string(), self.name.clone()];
        if let Some(cached) = store.get(&namespace, &key)? {
            return Ok(serde_json::from_value(cached)?);
        }
        let output = (self.run)(input)?;
        store.put(&namespace, &key, serde_json::to_value(&output)?)?;
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
