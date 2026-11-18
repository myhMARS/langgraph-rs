//! Native async Redis checkpointer (features `async` + `redis`).

use redis::AsyncCommands;

use crate::_internal::error::{Error, Result};
use crate::_internal::ports::{AsyncCheckpointer, BoxFuture};
use crate::checkpoint::Checkpoint;

const THREADS_KEY: &str = "langgraph_rs:threads";

fn checkpoint_key(thread_id: &str, checkpoint_id: &str) -> String {
    format!("langgraph_rs:checkpoint:{thread_id}:{checkpoint_id}")
}

fn index_key(thread_id: &str) -> String {
    format!("langgraph_rs:thread:{thread_id}:index")
}

/// Async checkpointer backed by a Tokio Redis connection manager.
#[derive(Clone)]
pub struct AsyncRedisSaver {
    connection: redis::aio::ConnectionManager,
}

impl std::fmt::Debug for AsyncRedisSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncRedisSaver").finish_non_exhaustive()
    }
}

impl AsyncRedisSaver {
    /// Connect to Redis (`redis://...` URL).
    pub async fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|error| Error::Checkpoint(format!("async redis: open client: {error}")))?;
        let connection = redis::aio::ConnectionManager::new(client)
            .await
            .map_err(|error| Error::Checkpoint(format!("async redis: connect: {error}")))?;
        Ok(Self { connection })
    }

    fn decode(payload: String) -> Result<Checkpoint> {
        serde_json::from_str(&payload)
            .map_err(|error| Error::Checkpoint(format!("async redis: decode: {error}")))
    }
}

async fn load_many(
    connection: &mut redis::aio::ConnectionManager,
    thread_id: &str,
    ids: Vec<String>,
) -> Result<Vec<Checkpoint>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut command = redis::cmd("MGET");
    for id in &ids {
        command.arg(checkpoint_key(thread_id, id));
    }
    let payloads: Vec<Option<String>> = command
        .query_async(connection)
        .await
        .map_err(|error| Error::Checkpoint(format!("async redis: history: {error}")))?;
    let mut checkpoints = Vec::with_capacity(payloads.len());
    for payload in payloads.into_iter().flatten() {
        checkpoints.push(AsyncRedisSaver::decode(payload)?);
    }
    Ok(checkpoints)
}

impl AsyncCheckpointer for AsyncRedisSaver {
    fn put(&self, checkpoint: Checkpoint) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let payload = serde_json::to_string(&checkpoint)
                .map_err(|error| Error::Checkpoint(format!("async redis: encode: {error}")))?;
            let key = checkpoint_key(&checkpoint.thread_id, &checkpoint.id);
            let index = index_key(&checkpoint.thread_id);
            let score = if checkpoint.created_at_millis > 0 {
                checkpoint.created_at_millis as f64
            } else {
                checkpoint.step as f64
            };
            let mut connection = self.connection.clone();
            redis::pipe()
                .atomic()
                .set(&key, payload)
                .zadd(&index, &checkpoint.id, score)
                .sadd(THREADS_KEY, &checkpoint.thread_id)
                .query_async::<()>(&mut connection)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: put: {error}")))?;
            Ok(())
        })
    }

    fn get(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        let checkpoint_id = checkpoint_id.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let payload: Option<String> = connection
                .get(checkpoint_key(&thread_id, &checkpoint_id))
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: get: {error}")))?;
            payload.map(Self::decode).transpose()
        })
    }

    fn latest(&self, thread_id: &str) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrevrange(index_key(&thread_id), 0, 0)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: latest id: {error}")))?;
            let Some(id) = ids.first() else {
                return Ok(None);
            };
            let payload: Option<String> = connection
                .get(checkpoint_key(&thread_id, id))
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: latest get: {error}")))?;
            payload.map(Self::decode).transpose()
        })
    }

    fn history(&self, thread_id: &str) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrevrange(index_key(&thread_id), 0, -1)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: history ids: {error}")))?;
            load_many(&mut connection, &thread_id, ids).await
        })
    }

    fn latest_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        let namespace = namespace.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrevrange(index_key(&thread_id), 0, -1)
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async redis: latest_in ids: {error}"))
                })?;
            for id in ids {
                let payload: Option<String> = connection
                    .get(checkpoint_key(&thread_id, &id))
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async redis: latest_in get: {error}"))
                    })?;
                if let Some(payload) = payload {
                    let checkpoint = Self::decode(payload)?;
                    if checkpoint.namespace == namespace {
                        return Ok(Some(checkpoint));
                    }
                }
            }
            Ok(None)
        })
    }

    fn history_in(
        &self,
        thread_id: &str,
        namespace: &str,
    ) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        let namespace = namespace.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrevrange(index_key(&thread_id), 0, -1)
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async redis: history_in ids: {error}"))
                })?;
            let mut checkpoints = load_many(&mut connection, &thread_id, ids).await?;
            checkpoints.retain(|checkpoint| checkpoint.namespace == namespace);
            Ok(checkpoints)
        })
    }

    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> BoxFuture<'_, Result<()>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrevrange(index_key(&thread_id), 0, -1)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: prune ids: {error}")))?;
            for id in ids.iter().skip(keep_last) {
                let _: () = connection
                    .del(checkpoint_key(&thread_id, id))
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async redis: prune del: {error}"))
                    })?;
                let _: () = connection
                    .zrem(index_key(&thread_id), id)
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async redis: prune zrem: {error}"))
                    })?;
            }
            if keep_last == 0 {
                let _: () = connection
                    .srem(THREADS_KEY, &thread_id)
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async redis: prune srem: {error}"))
                    })?;
            }
            Ok(())
        })
    }

    fn delete_thread(&self, thread_id: &str) -> BoxFuture<'_, Result<()>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let mut connection = self.connection.clone();
            let ids: Vec<String> = connection
                .zrange(index_key(&thread_id), 0, -1)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: delete ids: {error}")))?;
            for id in ids {
                let _: () = connection
                    .del(checkpoint_key(&thread_id, &id))
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async redis: delete del: {error}"))
                    })?;
            }
            let _: () = connection
                .del(index_key(&thread_id))
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async redis: delete index: {error}"))
                })?;
            let _: () = connection
                .srem(THREADS_KEY, &thread_id)
                .await
                .map_err(|error| Error::Checkpoint(format!("async redis: delete srem: {error}")))?;
            Ok(())
        })
    }

    fn copy_thread(&self, source: &str, target: &str) -> BoxFuture<'_, Result<()>> {
        let source = source.to_string();
        let target = target.to_string();
        Box::pin(async move {
            let checkpoints = self.history(&source).await?;
            for mut checkpoint in checkpoints {
                checkpoint.thread_id = target.clone();
                self.put(checkpoint).await?;
            }
            Ok(())
        })
    }
}
