//! Adapters (hexagonal architecture): concrete implementations of kernel ports.

pub mod memory;

#[cfg(feature = "python")]
pub mod python;
/// Tokio async runtime adapter (feature `async`).
#[cfg(feature = "async")]
pub mod tokio_runtime;
