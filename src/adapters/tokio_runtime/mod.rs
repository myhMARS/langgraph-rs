//! Tokio async runtime adapter (feature `async`).
//!
//! Async graphs mirror the sync kernel: nodes are async callables, a superstep awaits
//! all currently scheduled nodes concurrently, writes merge through the same channels,
//! and scheduling reuses the kernel's edge evaluation. The adapter is deliberately
//! separate from the kernel so the sync engine has no runtime dependency:
//!
//! - **No blocking.** Every node returns a `Send` future and supersteps run through a
//!   bounded `FuturesUnordered` window, so pending I/O overlaps up to `max_concurrency`.
//! - **Explicit context.** Because futures interleave on one task, the thread-local
//!   context of the sync API is not available; nodes receive an [`AsyncContext`](crate::tokio_runtime::AsyncContext) with
//!   `interrupt`, `record_usage`, `record_tokens` and `custom`.
//! - **Feature parity with the sync engine** for checkpoints, human-in-the-loop,
//!   retries, error routing, streaming, goto and `Send`.
//!
//! ```
//! use langgraph_rs::tokio_runtime::AsyncStateGraph;
//! use langgraph_rs::{JsonState, NodeOutput};
//!
//! # async fn run() -> langgraph_rs::Result<()> {
//! let graph = AsyncStateGraph::<JsonState>::new()
//!     .add_node("increment", |state: JsonState, _ctx| async move {
//!         let value = state.get("value").and_then(|value| value.as_i64()).unwrap_or(0);
//!         Ok(NodeOutput::new().set("value", value + 1))
//!     })
//!     .set_entry_point("increment")
//!     .set_finish_point("increment")
//!     .compile()?;
//!
//! let input = JsonState::default().with("value", serde_json::json!(0));
//! let output = graph.invoke(&input).await?;
//! assert_eq!(output.get("value"), Some(&serde_json::json!(1)));
//! # Ok(())
//! # }
//! ```

mod backend_impl;
mod builder;
mod compiled;
mod context;
mod functional;
mod save;

pub use builder::AsyncStateGraph;
pub use compiled::AsyncCompiledGraph;
pub use context::AsyncContext;
pub use functional::async_entrypoint;

/// Boxed future returned by an async node.
pub type BoxNodeFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<crate::NodeOutput>> + Send>>;

/// Async execution function of a graph node.
pub type AsyncNodeFn<S> = std::sync::Arc<
    dyn Fn(S, AsyncContext) -> BoxNodeFuture + std::marker::Send + std::marker::Sync,
>;
