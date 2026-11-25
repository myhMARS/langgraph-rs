//! Long-term memory store — Rust native API.
//!
//! Same operations as the Python demos, on `InMemoryStore` through the `BaseStore` trait:
//! `put` / `get` / `search` / `namespaces`, all keyed by a namespace slice.
//!
//! ```text
//! cargo run --release --example store_native
//! cargo run --release --example store_native -- --bench 2000
//! ```

use std::time::Instant;

use langgraph_rs::{BaseStore, InMemoryStore};
use serde_json::{Value, json};

fn namespace(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_string()).collect()
}

fn run_once(store: &InMemoryStore) -> Value {
    let users_1 = namespace(&["users", "1"]);
    let users_2 = namespace(&["users", "2"]);
    let teams = namespace(&["teams", "core"]);

    store
        .put(&users_1, "name", json!({"first": "Ada", "last": "Lovelace"}))
        .expect("put");
    store
        .put(&users_1, "email", json!({"address": "ada@example.com"}))
        .expect("put");
    store
        .put(&users_2, "name", json!({"first": "Grace", "last": "Hopper"}))
        .expect("put");
    store
        .put(&teams, "lead", json!({"name": "ada"}))
        .expect("put");

    let read = store.get(&users_1, "name").expect("get");
    let users = store.search(&namespace(&["users"]), None, None).expect("search");

    let mut items: Vec<String> = users
        .iter()
        .map(|item| format!("{}:{}", item.namespace.join("/"), item.key))
        .collect();
    items.sort();

    let mut names: Vec<Value> = users
        .iter()
        .filter(|item| item.key == "name")
        .map(|item| {
            item.value
                .get("first")
                .cloned()
                .unwrap_or(Value::Null)
        })
        .collect();
    names.sort_by_key(Value::to_string);

    let mut namespaces: Vec<String> = store
        .namespaces()
        .expect("namespaces")
        .into_iter()
        .map(|namespace| namespace.join("/"))
        .collect();
    namespaces.sort();

    json!({
        "names": names,
        "namespaces": namespaces,
        "read": read,
        "search": items,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--bench") {
        let iterations: u32 = args
            .get(index + 1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(2000);
        for _ in 0..50 {
            let _ = run_once(&InMemoryStore::new());
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = run_once(&InMemoryStore::new());
            }
            samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        println!(
            "{{\"engine\":\"rust native\",\"ns_per_op\":{}}}",
            samples[samples.len() / 2]
        );
        return;
    }
    println!("{}", run_once(&InMemoryStore::new()));
}