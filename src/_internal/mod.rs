//! Internal primitives shared by the graph, Pregel and adapter layers.
//!
//! Mirrors LangGraph's `_internal` package: these modules are implementation details
//! (state containers, commands, errors, ports, streaming values, retry/timeout/usage
//! helpers) rather than part of the public package surface.

pub mod cache;
pub mod command;
pub mod error;
pub mod interop;
pub mod lifecycle;
pub mod node;
pub mod plugin;
pub mod ports;
pub mod retry;
pub mod state;
pub mod stream;
pub mod timeout;
pub mod usage;
