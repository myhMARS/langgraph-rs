//! Managed values, mirroring `langgraph.managed`.
//!
//! [`remaining_steps`] and [`is_last_step`] are injected into node state when the
//! schema enables managed values.

pub use crate::runtime::context::{is_last_step, remaining_steps};
