//! Kernel: the domain core (hexagonal architecture).
//!
//! Everything in this module is independent of concrete adapters: persistence,
//! tracing sinks, language bindings and other infrastructure live in
//! [`crate::adapters`] and implement the ports declared in [`ports`].

pub mod cache;
pub mod channel;
pub mod checkpoint;
pub mod command;
pub mod context;
pub mod error;
pub mod executor;
pub mod graph;
pub mod node;
pub mod plugin;
pub mod ports;
pub mod retry;
pub mod state;
pub mod store;
pub mod stream;
pub mod timeout;
pub mod usage;
