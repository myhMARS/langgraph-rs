//! Microkernel: the plugin registry that composes the engine's capabilities.
//!
//! The kernel itself knows how to execute supersteps but not how a channel merges
//! writes or where trace records go. Those capabilities are registered as named
//! factories / ports by [`Plugin`]s, so downstream crates can add new channel
//! semantics without touching the engine:
//!
//! ```
//! use langgraph_rs::{Channel, Kernel, Plugin, StateSchema};
//!
//! struct DoubleSum;
//!
//! impl Plugin for DoubleSum {
//!     fn name(&self) -> &'static str {
//!         "double-sum"
//!     }
//!
//!     fn install(&self, kernel: &mut Kernel) -> langgraph_rs::Result<()> {
//!         kernel.register_channel("double_sum", || {
//!             Channel::reducer(|current, incoming| {
//!                 let left = current.as_i64().unwrap_or_default();
//!                 let right = incoming.as_i64().unwrap_or_default();
//!                 Ok(serde_json::json!(left + right * 2))
//!             })
//!         });
//!         Ok(())
//!     }
//! }
//!
//! fn main() -> langgraph_rs::Result<()> {
//!     let kernel = Kernel::builder().with_plugin(DoubleSum)?.build();
//!     let schema: StateSchema = kernel.schema([("total".to_string(), "double_sum".to_string())])?;
//!     assert!(schema.contains("total"));
//!     Ok(())
//! }
//! ```

use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;

use super::channel::{Channel, reducers};
use super::error::{Error, Result};
use super::ports::TraceSink;
use super::state::StateSchema;

/// Factory that creates a channel merge policy on demand.
#[derive(Clone)]
pub struct ChannelFactory(Arc<dyn Fn() -> Channel + Send + Sync>);

impl ChannelFactory {
    /// Wrap a factory function.
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> Channel + Send + Sync + 'static,
    {
        Self(Arc::new(factory))
    }

    /// Create a fresh channel value.
    pub fn create(&self) -> Channel {
        (self.0)()
    }
}

impl fmt::Debug for ChannelFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ChannelFactory(..)")
    }
}

/// A microkernel plugin: registers capabilities into a [`Kernel`].
pub trait Plugin: Send + Sync {
    /// Stable plugin name, used in diagnostics.
    fn name(&self) -> &'static str;

    /// Register channels, trace sinks or other capabilities.
    fn install(&self, kernel: &mut Kernel) -> Result<()>;
}

/// Registry of named channel factories and trace sinks.
///
/// The kernel (graph compilation + executor) only resolves capabilities through this
/// registry, which keeps it independent of the concrete implementations.
#[derive(Clone, Default)]
pub struct Kernel {
    factories: IndexMap<String, ChannelFactory>,
    trace_sinks: Vec<Arc<dyn TraceSink>>,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("channels", &self.factories.keys().collect::<Vec<_>>())
            .field("trace_sinks", &self.trace_sinks.len())
            .finish()
    }
}

impl Kernel {
    /// An empty kernel without any channel policy.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The kernel used by default: the crate's built-in channel policies.
    pub fn builtin() -> Self {
        let mut kernel = Self::empty();
        BuiltinChannels
            .install(&mut kernel)
            .expect("the built-in channel plugin is infallible");
        kernel
    }

    /// Start building a kernel with the built-in capabilities.
    pub fn builder() -> KernelBuilder {
        KernelBuilder {
            kernel: Self::builtin(),
        }
    }

    /// Register a named channel factory and return the kernel for chaining.
    pub fn register_channel<F>(&mut self, name: impl Into<String>, factory: F) -> &mut Self
    where
        F: Fn() -> Channel + Send + Sync + 'static,
    {
        self.factories
            .insert(name.into(), ChannelFactory::new(factory));
        self
    }

    /// Install a plugin.
    pub fn install<P: Plugin>(&mut self, plugin: P) -> Result<&mut Self> {
        plugin.install(self)?;
        Ok(self)
    }

    /// Register a trace sink adapter.
    pub fn add_trace_sink(&mut self, sink: Arc<dyn TraceSink>) -> &mut Self {
        self.trace_sinks.push(sink);
        self
    }

    /// Resolve a channel spec name to a fresh channel value.
    pub fn resolve_channel(&self, spec: &str) -> Option<Channel> {
        self.factories.get(spec).map(ChannelFactory::create)
    }

    /// Registered channel spec names.
    pub fn channel_specs(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }

    /// Registered trace sinks.
    pub fn trace_sinks(&self) -> &[Arc<dyn TraceSink>] {
        &self.trace_sinks
    }

    /// Build a schema from `(channel name, plugin spec)` pairs.
    pub fn schema<I>(&self, entries: I) -> Result<StateSchema>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let mut schema = StateSchema::dynamic();
        for (name, spec) in entries {
            let channel = self.resolve_channel(&spec).ok_or_else(|| {
                Error::validation(format!(
                    "unknown channel plugin `{spec}` for `{name}`; registered plugins: {}",
                    self.channel_specs().collect::<Vec<_>>().join(", ")
                ))
            })?;
            schema = schema.with_channel(name, channel);
        }
        Ok(schema)
    }
}

/// Builder for a [`Kernel`].
pub struct KernelBuilder {
    kernel: Kernel,
}

impl KernelBuilder {
    /// Install a plugin.
    pub fn with_plugin<P: Plugin>(mut self, plugin: P) -> Result<Self> {
        plugin.install(&mut self.kernel)?;
        Ok(self)
    }

    /// Register a channel factory.
    pub fn with_channel<F>(mut self, name: impl Into<String>, factory: F) -> Self
    where
        F: Fn() -> Channel + Send + Sync + 'static,
    {
        self.kernel.register_channel(name, factory);
        self
    }

    /// Register a trace sink.
    pub fn with_trace_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.kernel.add_trace_sink(sink);
        self
    }

    /// Finish building.
    pub fn build(self) -> Kernel {
        self.kernel
    }
}

/// Registers the channel policies that ship with the crate.
pub struct BuiltinChannels;

impl Plugin for BuiltinChannels {
    fn name(&self) -> &'static str {
        "builtin-channels"
    }

    fn install(&self, kernel: &mut Kernel) -> Result<()> {
        kernel.register_channel("last_value", || Channel::LastValue);
        kernel.register_channel("topic", || Channel::Topic);
        kernel.register_channel("append", || Channel::with_reducer(reducers::append()));
        kernel.register_channel("add", || Channel::with_reducer(reducers::add()));
        kernel.register_channel("messages", || Channel::with_reducer(reducers::messages()));
        kernel.register_channel("any_value", || Channel::AnyValue);
        kernel.register_channel("ephemeral_value", || Channel::EphemeralValue);
        kernel.register_channel("untracked_value", || Channel::UntrackedValue);
        kernel.register_channel("delta", || Channel::delta(reducers::append()));
        kernel.register_channel("last_value_after_finish", || Channel::LastValueAfterFinish);
        Ok(())
    }
}
