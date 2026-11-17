//! Adapters (hexagonal architecture): concrete implementations of kernel ports.

pub mod memory;

#[cfg(feature = "async")]
pub mod async_checkpointer;

#[cfg(feature = "python")]
pub mod python;
/// Tokio async runtime adapter (feature `async`).
#[cfg(feature = "async")]
pub mod tokio_runtime;

/// SQLite checkpoint, store, cache and LangGraph-compatible adapters.
#[cfg(feature = "sqlite")]
pub mod sqlite;

/// Redis checkpoint, store, cache and async checkpointer adapters.
#[cfg(feature = "redis")]
pub mod redis;

/// Postgres checkpoint, store, cache and async checkpointer adapters.
#[cfg(feature = "postgres")]
pub mod postgres;
