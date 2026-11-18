//! Async functional API entrypoint (`functional.async_entrypoint` in LangGraph).
//!
//! The async graph type lives in this adapter, so the core `functional` package does not
//! depend on the Tokio runtime adapter.

use crate::_internal::node::NodeOutput;
use crate::_internal::state::{State, state_to_values};
use crate::{Result, tokio_runtime::AsyncStateGraph};

/// Build an async graph whose single `entrypoint` node awaits `function`.
///
/// The function can await several
/// [`AsyncTaskCall::call`](crate::functional::AsyncTaskCall::call) futures concurrently;
/// each task still memoizes its result in the checkpoint task cache / store.
pub fn async_entrypoint<S, F, Fut>(function: F) -> AsyncStateGraph<S>
where
    S: State + Send + 'static,
    F: Fn(S) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<S>> + Send + 'static,
{
    AsyncStateGraph::<S>::new()
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
