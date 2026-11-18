#![allow(unused_imports)]
//! Compatibility path for the former application layer.
//!
//! Canonical packages are now [`crate::graph`], [`crate::pregel`],
//! [`crate::functional`], [`crate::prebuilt`] and [`crate::runtime`].

pub use crate::runtime::{context, control};
pub use crate::{functional, graph, prebuilt, pregel};
