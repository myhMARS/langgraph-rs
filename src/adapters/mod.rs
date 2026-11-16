//! Adapters (hexagonal architecture): concrete implementations of kernel ports.

pub mod memory;

#[cfg(feature = "python")]
pub mod python;
