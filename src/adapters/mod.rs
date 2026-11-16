//! Adapters (hexagonal architecture): concrete implementations of kernel ports.

pub mod memory;

#[cfg(feature = "async")]
pub mod async_checkpointer;

#[cfg(feature = "python")]
pub mod python;
/// Tokio async runtime adapter (feature `async`).
#[cfg(feature = "async")]
pub mod tokio_runtime;

/// SQLite checkpoint persistence (feature `sqlite`).
#[cfg(feature = "sqlite")]
pub mod sqlite;

/// SQLite long-term memory store (feature `sqlite`).
#[cfg(feature = "sqlite")]
pub mod sqlite_store;

/// Redis checkpoint persistence (feature `redis`).
#[cfg(feature = "redis")]
pub mod redis;

/// Postgres checkpoint persistence (feature `postgres`).
#[cfg(feature = "postgres")]
pub mod postgres;
