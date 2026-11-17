//! Redis adapters grouped by responsibility.

mod async_checkpointer;
mod cache;
mod checkpointer;
mod store;

pub use async_checkpointer::AsyncRedisSaver;
pub use cache::RedisCache;
pub use checkpointer::RedisSaver;
pub use store::RedisStore;
