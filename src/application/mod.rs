//! Application layer: graph orchestration and the Pregel execution engine.
//!
//! This layer depends on the domain core in [`crate::kernel`] (state, channels,
//! checkpoints, ports) and is independent of concrete adapters.

pub mod context;
pub mod control;
pub mod executor;
pub mod graph;
