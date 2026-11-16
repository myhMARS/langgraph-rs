//! `RemoveMessage`, `MessagesState` and node input-schema metadata.

use langgraph_rs::{
    JsonState, MessagesState, NodeOutput, RemoveMessage, RunConfig, StateGraph, StateSchema, json,
    reducers,
};

fn state_with_messages() -> JsonState {
    JsonState::new()
}

#[test]
fn remove_message_deletes_by_id_from_json_state() {
    let schema = StateSchema::new().with_reducer("messages", reducers::messages());
    let graph = StateGraph::<JsonState>::with_schema(schema)
        .add_node("append", |_state: &JsonState| {
            Ok(NodeOutput::new().set(
                "messages",
                vec![
                    json!({ "id": "1", "content": "a" }),
                    json!({ "id": "2", "content": "b" }),
                ],
            ))
        })
        .add_node("remove", |_state: &JsonState| {
            Ok(NodeOutput::new().set("messages", vec![RemoveMessage::new("1").to_value()]))
        })
        .set_entry_point("append")
        .add_edge("append", "remove")
        .set_finish_point("remove")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&state_with_messages(), &RunConfig::new())
        .unwrap();
    let messages = output.get("messages").unwrap().as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["id"], json!("2"));
}

#[test]
fn messages_state_merges_and_removes() {
    let graph = StateGraph::<MessagesState>::new()
        .add_node("first", |_state: &MessagesState| {
            Ok(NodeOutput::new().set("messages", vec![json!({ "id": "a", "content": "hello" })]))
        })
        .add_node("second", |_state: &MessagesState| {
            Ok(NodeOutput::new().set(
                "messages",
                vec![
                    json!({ "id": "b", "content": "world" }),
                    json!({ "id": "a", "content": "hello!" }),
                ],
            ))
        })
        .add_node("remove-b", |_state: &MessagesState| {
            Ok(NodeOutput::new().set("messages", vec![RemoveMessage::new("b").to_value()]))
        })
        .set_entry_point("first")
        .add_edge("first", "second")
        .add_edge("second", "remove-b")
        .set_finish_point("remove-b")
        .compile()
        .unwrap();

    let output = graph
        .invoke(&MessagesState::default(), &RunConfig::new())
        .unwrap();
    assert_eq!(output.messages.len(), 1);
    assert_eq!(output.messages[0]["id"], json!("a"));
    assert_eq!(output.messages[0]["content"], json!("hello!"));
}

#[test]
fn node_input_schema_is_stored_as_metadata() {
    let schema = json!({ "type": "object", "properties": { "value": { "type": "integer" } } });
    let graph = StateGraph::<MessagesState>::new()
        .add_node_with_input_schema("work", schema.clone(), |_state: &MessagesState| {
            Ok(NodeOutput::new())
        })
        .set_entry_point("work")
        .set_finish_point("work")
        .compile()
        .unwrap();

    assert_eq!(graph.node_input_schema("work"), Some(&schema));
    assert!(graph.node_input_schema("missing").is_none());
}
