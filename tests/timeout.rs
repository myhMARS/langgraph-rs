#![cfg(feature = "async")]

use std::time::Duration;
use langgraph_rs::tokio_runtime::AsyncStateGraph;
use langgraph_rs::{NodeOutput, RunConfig, State, StateSchema, TimeoutPolicy};

#[derive(Clone, Default, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct TState {
    value: i64,
}
impl State for TState {
    fn schema() -> StateSchema {
        StateSchema::new().last_value("value")
    }
}

#[tokio::test]
async fn async_timeout_fires() {
    let graph = AsyncStateGraph::<TState>::new()
        .add_node_with_timeout(
            "slow",
            TimeoutPolicy::new().with_run_timeout(Duration::from_millis(10)),
            |_state: TState, _ctx| async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(NodeOutput::new().set("value", 1))
            },
        )
        .set_entry_point("slow")
        .set_finish_point("slow")
        .compile()
        .unwrap();
    let error = graph
        .run(&TState::default(), &RunConfig::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error}");
}
