//! # langgraph-rs
//!
//! A Rust port of the core runtime of [LangGraph](https://github.com/langchain-ai/langgraph):
//! a stateful, graph-based execution engine for orchestrating LLM agents and workflows.
//!
//! The engine runs a [`StateGraph`] in **supersteps**. Each superstep executes the
//! currently scheduled nodes (in parallel when there is more than one), merges their
//! writes into the graph state through [`Channel`] merge policies, evaluates outgoing
//! edges and schedules the next superstep until no work is left or [`END`] is reached.
//!
//! ## Features
//!
//! - typed or JSON state with `LastValue`, `Topic` and custom reducer channels
//! - static edges, conditional routing, symbolic path maps and multi-target fan-out
//! - parallel supersteps executed on scoped threads
//! - [`Send`] dynamic fan-out for map/reduce workflows
//! - checkpointing via the [`Checkpointer`] trait with an in-memory saver, persistent
//!   SQLite/Redis/Postgres adapters (`sqlite` / `redis` / `postgres` features), thread
//!   history and time travel
//! - human-in-the-loop interrupts ([`interrupt`] / [`CompiledGraph::resume`]) plus
//!   `interrupt_before` / `interrupt_after` hooks
//! - streaming of values, per-node updates, custom events, interrupts, checkpoints and
//!   per-node usage records
//! - per-node consumption tracking ([`record_usage`], [`Usage`], [`RunReport`]): tokens,
//!   cost, wall-clock duration and aggregated totals per node and per run
//! - retry policies ([`RetryPolicy`]) with exponential backoff, and error routing to
//!   handler nodes ([`StateGraph::add_error_edge`] / [`last_error`])
//! - subgraphs: embed a compiled graph as a node with [`StateGraph::add_subgraph`]
//! - node-level caching ([`CachePolicy`], [`NodeCache`], [`InMemoryCache`]): opt-in per node,
//!   TTLs and custom keys, cached executions flagged in run reports
//! - an optional Tokio async adapter (`async` feature, module [`tokio_runtime`]) with the
//!   same checkpointing, HITL, retry and error-routing capabilities
//! - optional Python bindings behind the `python` feature (PyO3), exposing the same engine
//!   to Python callables
//!
//! ## Architecture
//!
//! The crate follows a **microkernel + hexagonal** design. The domain kernel
//! (`kernel`) owns the graph model, superstep engine, channels, state, checkpoints and
//! usage records. It depends only on **ports** — [`Checkpointer`] and [`TraceSink`] — plus
//! a microkernel registry: [`Kernel`], [`KernelBuilder`], [`Plugin`] and
//! [`ChannelFactory`]. **Adapters** (`adapters`) implement those ports: the in-memory
//! checkpoint saver and the optional PyO3 bindings. Channel merge policies are plugins
//! resolved by name, so downstream crates extend the engine without touching it.
//!
//! ## Quick start
//!
//! ```
//! use langgraph_rs::{NodeOutput, Route, RunConfig, State, StateGraph, StateSchema};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Clone, Default, Serialize, Deserialize)]
//! struct Counter {
//!     value: i64,
//! }
//!
//! impl State for Counter {
//!     fn schema() -> StateSchema {
//!         StateSchema::new().last_value("value")
//!     }
//! }
//!
//! let graph = StateGraph::<Counter>::new()
//!     .add_node("increment", |state: &Counter| {
//!         Ok(NodeOutput::new().set("value", state.value + 1))
//!     })
//!     .set_entry_point("increment")
//!     .add_conditional_edges("increment", |state: &Counter| {
//!         Ok(if state.value < 3 {
//!             Route::node("increment")
//!         } else {
//!             Route::end()
//!         })
//!     })
//!     .compile()
//!     .unwrap();
//!
//! let output = graph.invoke(&Counter::default(), &RunConfig::new()).unwrap();
//! assert_eq!(output.value, 3);
//! ```
//!
//! ## Execution model
//!
//! ```text
//!  input ──► [ superstep 1: run nodes ] ──► merge channels ──► route edges ──► [ superstep 2 ] ──► ... ──► END
//!                     │                          │
//!                     └── checkpoint ────────────┘
//! ```
//!
//! A node is any `Fn(&State) -> Result<NodeOutput>`; it returns channel writes instead
//! of mutating state, which is what makes checkpointing, resuming and time travel
//! possible.

// Interrupts carry their payload inline (public API); keeping the error small enough for clippy would require boxing it.
#![allow(clippy::result_large_err)]

mod adapters;
pub mod application;
pub mod config;
pub mod constants;
pub mod domain;
pub use adapters::interop;
pub mod managed;
pub mod runtime;
pub mod types;
pub use application::{functional, graph, prebuilt, pregel};
pub use domain as kernel;
pub use domain::{channels, checkpoint, errors, store};

#[cfg(feature = "python")]
pub use adapters::python;

#[cfg(feature = "async")]
pub use adapters::tokio_runtime;

pub use adapters::memory::{InMemoryCache, InMemorySaver, InMemoryStore, MemorySaver};

/// Async checkpointer adapter that offloads a synchronous saver to a blocking pool.
#[cfg(feature = "async")]
pub use adapters::async_checkpointer::BlockingSaver;
#[cfg(feature = "sqlite")]
pub use adapters::sqlite::SqliteSaver;

#[cfg(feature = "sqlite")]
/// SQLite-backed long-term memory store.
pub use adapters::sqlite::SqliteStore;

#[cfg(feature = "sqlite")]
/// SQLite-backed node cache.
pub use adapters::sqlite::SqliteCache;

#[cfg(feature = "sqlite")]
/// Read-only reader for LangGraph SQLite checkpoint databases.
pub use adapters::sqlite::LangGraphSqliteReader;

#[cfg(feature = "sqlite")]
/// Checkpointer writing LangGraph's SQLite table layout.
pub use adapters::sqlite::LangGraphSqliteSaver;

/// Alias for [`InMemorySaver`] used from async graphs (same synchronous implementation).
#[cfg(feature = "async")]
pub type AsyncInMemorySaver = InMemorySaver;
/// Alias for [`SqliteSaver`] used from async graphs (same synchronous implementation).
#[cfg(all(feature = "async", feature = "sqlite"))]
pub type AsyncSqliteSaver = SqliteSaver;
/// Alias for [`PostgresSaver`] used from async graphs (same synchronous implementation).
#[cfg(all(feature = "async", feature = "postgres"))]
/// Native async Postgres checkpointer.
pub use adapters::postgres::AsyncPostgresSaver;
/// Alias for [`RedisSaver`] used from async graphs (same synchronous implementation).
#[cfg(all(feature = "async", feature = "redis"))]
/// Native async Redis checkpointer.
pub use adapters::redis::AsyncRedisSaver;

#[cfg(feature = "redis")]
pub use adapters::redis::RedisSaver;

#[cfg(feature = "redis")]
/// Redis-backed long-term memory store.
pub use adapters::redis::RedisStore;

#[cfg(feature = "redis")]
/// Redis-backed node cache.
pub use adapters::redis::RedisCache;

#[cfg(feature = "postgres")]
pub use adapters::postgres::PostgresSaver;

#[cfg(feature = "postgres")]
/// Postgres-backed long-term memory store.
pub use adapters::postgres::PostgresStore;

#[cfg(feature = "postgres")]
/// Postgres-backed node cache.
pub use adapters::postgres::PostgresCache;
pub use application::context::{
    current_context, current_namespace, current_node, current_resume, current_runtime,
    current_step, current_task_id, current_thread_id, custom, get_config, get_store,
    get_stream_writer, interrupt, interrupt_with_schema, is_last_step, last_error, message,
    push_custom, push_message, record_tokens, record_usage, remaining_steps,
};
pub use application::control::RunControl;
pub use kernel::cache::{CachePolicy, CachedOutput, NodeCache};
pub use kernel::channel::{Channel, Reducer, RemoveMessage, reducers};
pub use kernel::checkpoint::Checkpoint;
pub use kernel::command::Command;
pub use kernel::error::{Error, Interrupt, NodeFailure, Result};
/// Convenience graph whose state is [`MessagesState`] (LangGraph `MessageGraph`).
pub type MessageGraph = StateGraph<MessagesState>;

pub use application::graph::{
    CompileOptions, CompiledGraph, DEFAULT_RECURSION_LIMIT, Durability, RunConfig, StateGraph,
    StateSnapshot, Targets,
};
pub use kernel::node::{ConditionalEdge, NodeFn, NodeOutput, PathMap, Route, Send, Task};
pub use kernel::plugin::{BuiltinChannels, ChannelFactory, Kernel, KernelBuilder, Plugin};
/// Async checkpoint storage port (feature `async`).
#[cfg(feature = "async")]
pub use kernel::ports::AsyncCheckpointer;
pub use kernel::ports::{Checkpointer, TraceSink};
pub use kernel::retry::RetryPolicy;
pub use kernel::state::{JsonState, MessagesState, State, StateSchema};
pub use kernel::store::{BaseStore, Namespace, Runtime, StoreItem};
pub use kernel::stream::{StreamChunk, StreamMode};
pub use kernel::timeout::TimeoutPolicy;
pub use kernel::usage::{NodeExecution, RunOutcome, RunReport, Usage};

pub use indexmap::IndexMap;
pub use serde_json::{Value, json};

/// Reserved node name used as the source of entry-point edges.
pub const START: &str = "__start__";
/// Reserved node name that terminates a run.
pub const END: &str = "__end__";
