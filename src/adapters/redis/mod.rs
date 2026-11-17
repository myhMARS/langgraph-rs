//! Redis adapters grouped by responsibility.

#[cfg(feature = "async")]
mod async_checkpointer;
mod cache;
mod checkpointer;
mod store;

#[cfg(feature = "async")]
pub use async_checkpointer::AsyncRedisSaver;
pub use cache::RedisCache;
pub use checkpointer::RedisSaver;
pub use store::RedisStore;
