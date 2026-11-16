//! Live native async Redis/Postgres checkpointer tests (env-gated).

#![cfg(feature = "async")]

#[cfg(all(feature = "async", feature = "redis"))]
#[tokio::test]
async fn async_redis_saver_roundtrip() {
    use std::sync::Arc;

    use langgraph_rs::{
        AsyncCheckpointer, AsyncRedisSaver, JsonState, NodeOutput, RunConfig, StateSchema, json,
        tokio_runtime::AsyncStateGraph,
    };

    let Ok(url) = std::env::var("REDIS_URL") else {
        return;
    };
    let saver = Arc::new(AsyncRedisSaver::connect(&url).await.unwrap());
    let thread = format!("async-redis-{}", std::process::id());
    saver.delete_thread(&thread).await.unwrap();

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .with_async_checkpointer(saver.clone())
        .compile()
        .unwrap();

    graph
        .run(&JsonState::new(), &RunConfig::new().with_thread(&thread))
        .await
        .unwrap();
    let latest = saver.latest(&thread).await.unwrap().unwrap();
    assert_eq!(latest.values.get("value"), Some(&json!(1)));
    assert!(saver.history(&thread).await.unwrap().len() >= 2);
    assert!(saver.threads().await.unwrap().contains(&thread));
    saver.prune_thread(&thread, 1).await.unwrap();
    assert_eq!(saver.history(&thread).await.unwrap().len(), 1);
    saver.delete_thread(&thread).await.unwrap();
    assert!(saver.latest(&thread).await.unwrap().is_none());
}

#[cfg(all(feature = "async", feature = "postgres"))]
#[tokio::test]
async fn async_postgres_saver_roundtrip() {
    use std::sync::Arc;

    use langgraph_rs::{
        AsyncCheckpointer, AsyncPostgresSaver, JsonState, NodeOutput, RunConfig, StateSchema, json,
        tokio_runtime::AsyncStateGraph,
    };

    let Ok(url) = std::env::var("POSTGRES_URL") else {
        return;
    };
    let saver = Arc::new(AsyncPostgresSaver::connect(&url).await.unwrap());
    let thread = format!("async-postgres-{}", std::process::id());
    saver.delete_thread(&thread).await.unwrap();

    let graph = AsyncStateGraph::<JsonState>::with_schema(StateSchema::dynamic())
        .add_node("inc", |state: JsonState, _ctx| async move {
            let value = state
                .get("value")
                .and_then(|value| value.as_i64())
                .unwrap_or_default();
            Ok(NodeOutput::new().update("value", json!(value + 1)))
        })
        .set_entry_point("inc")
        .set_finish_point("inc")
        .with_async_checkpointer(saver.clone())
        .compile()
        .unwrap();

    graph
        .run(&JsonState::new(), &RunConfig::new().with_thread(&thread))
        .await
        .unwrap();
    let latest = saver.latest(&thread).await.unwrap().unwrap();
    assert_eq!(latest.values.get("value"), Some(&json!(1)));
    assert!(saver.history(&thread).await.unwrap().len() >= 2);
    assert!(saver.threads().await.unwrap().contains(&thread));
    saver
        .copy_thread(&thread, &format!("{thread}-copy"))
        .await
        .unwrap();
    assert!(
        saver
            .latest(&format!("{thread}-copy"))
            .await
            .unwrap()
            .is_some()
    );
    saver.prune_thread(&thread, 1).await.unwrap();
    assert_eq!(saver.history(&thread).await.unwrap().len(), 1);
    saver.delete_thread(&thread).await.unwrap();
    saver
        .delete_thread(&format!("{thread}-copy"))
        .await
        .unwrap();
}
