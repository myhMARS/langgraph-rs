//! Native async Postgres checkpointer (features `async` + `postgres`).

use std::sync::Arc;

use tokio_postgres::NoTls;

use crate::kernel::checkpoint::Checkpoint;
use crate::kernel::error::{Error, Result};
use crate::kernel::ports::{AsyncCheckpointer, BoxFuture};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS langgraph_rs_checkpoints (
    thread_id TEXT NOT NULL,
    id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    step BIGINT NOT NULL,
    created_at BIGINT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, id)
);";

/// Async checkpointer backed by `tokio-postgres` (no blocking pool).
pub struct AsyncPostgresSaver {
    client: Arc<tokio_postgres::Client>,
}

impl std::fmt::Debug for AsyncPostgresSaver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncPostgresSaver").finish_non_exhaustive()
    }
}

impl AsyncPostgresSaver {
    /// Connect using a libpq URL/DSN and create the checkpoint table.
    pub async fn connect(url: &str) -> Result<Self> {
        let (client, connection) = tokio_postgres::connect(url, NoTls)
            .await
            .map_err(|error| Error::Checkpoint(format!("async postgres: connect: {error}")))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let saver = Self {
            client: Arc::new(client),
        };
        saver
            .client
            .batch_execute(SCHEMA)
            .await
            .map_err(|error| Error::Checkpoint(format!("async postgres: schema: {error}")))?;
        Ok(saver)
    }

    fn decode(row: &tokio_postgres::Row) -> Result<Checkpoint> {
        let payload: String = row
            .try_get("payload")
            .map_err(|error| Error::Checkpoint(format!("async postgres: row: {error}")))?;
        serde_json::from_str(&payload)
            .map_err(|error| Error::Checkpoint(format!("async postgres: decode: {error}")))
    }
}

impl AsyncCheckpointer for AsyncPostgresSaver {
    fn put(&self, checkpoint: Checkpoint) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let payload = serde_json::to_string(&checkpoint).map_err(|error| {
                Error::Checkpoint(format!("async postgres: serialize: {error}"))
            })?;
            self.client
                .execute(
                    "INSERT INTO langgraph_rs_checkpoints
                     (thread_id, id, namespace, step, created_at, payload)
                     VALUES ($1, $2, $3, $4, $5, $6)
                     ON CONFLICT (thread_id, id) DO UPDATE SET payload = excluded.payload",
                    &[
                        &checkpoint.thread_id,
                        &checkpoint.id,
                        &checkpoint.namespace,
                        &(checkpoint.step as i64),
                        &(checkpoint.created_at_millis as i64),
                        &payload,
                    ],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: put: {error}")))?;
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
            let row = self
                .client
                .query_opt(
                    "SELECT payload FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1 AND id = $2",
                    &[&thread_id, &checkpoint_id],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: get: {error}")))?;
            row.as_ref().map(Self::decode).transpose()
        })
    }

    fn latest(&self, thread_id: &str) -> BoxFuture<'_, Result<Option<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let row = self
                .client
                .query_opt(
                    "SELECT payload FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1
                     ORDER BY step DESC, created_at DESC, id DESC
                     LIMIT 1",
                    &[&thread_id],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: latest: {error}")))?;
            row.as_ref().map(Self::decode).transpose()
        })
    }

    fn history(&self, thread_id: &str) -> BoxFuture<'_, Result<Vec<Checkpoint>>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            let rows = self
                .client
                .query(
                    "SELECT payload FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1
                     ORDER BY step DESC, created_at DESC, id DESC",
                    &[&thread_id],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: history: {error}")))?;
            rows.iter().map(Self::decode).collect()
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
            let row = self
                .client
                .query_opt(
                    "SELECT payload FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1 AND namespace = $2
                     ORDER BY step DESC, created_at DESC, id DESC
                     LIMIT 1",
                    &[&thread_id, &namespace],
                )
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async postgres: latest_in: {error}"))
                })?;
            row.as_ref().map(Self::decode).transpose()
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
            let rows = self
                .client
                .query(
                    "SELECT payload FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1 AND namespace = $2
                     ORDER BY step DESC, created_at DESC, id DESC",
                    &[&thread_id, &namespace],
                )
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async postgres: history_in: {error}"))
                })?;
            rows.iter().map(Self::decode).collect()
        })
    }

    fn threads(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            let rows = self
                .client
                .query(
                    "SELECT DISTINCT thread_id FROM langgraph_rs_checkpoints",
                    &[],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: threads: {error}")))?;
            Ok(rows.iter().map(|row| row.get(0)).collect())
        })
    }

    fn prune_thread(&self, thread_id: &str, keep_last: usize) -> BoxFuture<'_, Result<()>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            self.client
                .execute(
                    "DELETE FROM langgraph_rs_checkpoints
                     WHERE thread_id = $1
                       AND id NOT IN (
                           SELECT id FROM langgraph_rs_checkpoints
                           WHERE thread_id = $1
                           ORDER BY step DESC, created_at DESC, id DESC
                           LIMIT $2
                       )",
                    &[&thread_id, &(keep_last as i64)],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: prune: {error}")))?;
            Ok(())
        })
    }

    fn delete_thread(&self, thread_id: &str) -> BoxFuture<'_, Result<()>> {
        let thread_id = thread_id.to_string();
        Box::pin(async move {
            self.client
                .execute(
                    "DELETE FROM langgraph_rs_checkpoints WHERE thread_id = $1",
                    &[&thread_id],
                )
                .await
                .map_err(|error| Error::Checkpoint(format!("async postgres: delete: {error}")))?;
            Ok(())
        })
    }

    fn copy_thread(&self, source: &str, target: &str) -> BoxFuture<'_, Result<()>> {
        let source = source.to_string();
        let target = target.to_string();
        Box::pin(async move {
            let rows = self
                .client
                .query(
                    "SELECT payload FROM langgraph_rs_checkpoints WHERE thread_id = $1",
                    &[&source],
                )
                .await
                .map_err(|error| {
                    Error::Checkpoint(format!("async postgres: copy read: {error}"))
                })?;
            for row in rows {
                let mut checkpoint = Self::decode(&row)?;
                checkpoint.thread_id = target.clone();
                let payload = serde_json::to_string(&checkpoint).map_err(|error| {
                    Error::Checkpoint(format!("async postgres: copy encode: {error}"))
                })?;
                self.client
                    .execute(
                        "INSERT INTO langgraph_rs_checkpoints
                         (thread_id, id, namespace, step, created_at, payload)
                         VALUES ($1, $2, $3, $4, $5, $6)
                         ON CONFLICT (thread_id, id) DO UPDATE SET payload = excluded.payload",
                        &[
                            &checkpoint.thread_id,
                            &checkpoint.id,
                            &checkpoint.namespace,
                            &(checkpoint.step as i64),
                            &(checkpoint.created_at_millis as i64),
                            &payload,
                        ],
                    )
                    .await
                    .map_err(|error| {
                        Error::Checkpoint(format!("async postgres: copy write: {error}"))
                    })?;
            }
            Ok(())
        })
    }
}
