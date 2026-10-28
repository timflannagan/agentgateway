use bytes::Bytes;
use serde_json::{Value, json};

use super::to_responses;

fn translate(input: &[u8]) -> Value {
	let response =
		to_responses::translate_response(&Bytes::copy_from_slice(input), "input-model", None)
			.expect("Chat Completions response should translate");
	serde_json::from_slice(&response.serialize().expect("response should serialize"))
		.expect("translated response should be JSON")
}

#[test]
fn buffered_failed_response_preserves_error() {
	let mut input: Value =
		serde_json::from_slice(include_bytes!("../tests/response/completions/basic.json")).unwrap();
	input["choices"][0]["finish_reason"] = json!("content_filter");
	let input = serde_json::to_vec(&input).unwrap();

	let response = translate(&input);

	assert_eq!(response["status"], json!("failed"));
	assert_eq!(
		response["error"],
		json!({"code": "content_filter", "message": "Content filtered"})
	);
}

#[test]
fn gemini_thought_signature_round_trips_through_responses_reasoning_item() {
	let signature = "CqUBAbc123def456GHI789jklMNOpqrSTUvwxYZ0123456789+/aBcDeFgHiJkLmNoPqRsTuVwXyZ==";
	let mut upstream: Value = serde_json::from_slice(include_bytes!(
		"../tests/response/completions/tool_call.json"
	))
	.unwrap();
	upstream["choices"][0]["message"]["tool_calls"][0]["extra_content"] = json!({
		"google": {"thought_signature": signature}
	});

	let first_response = translate(&serde_json::to_vec(&upstream).unwrap());
	let output = first_response["output"].as_array().unwrap();
	assert_eq!(output[0]["type"], "reasoning");
	assert_eq!(output[1]["type"], "function_call");
	assert_eq!(output[1]["call_id"], "call_abc123");
	assert_eq!(output[2]["type"], "function_call");
	assert_eq!(output[2]["call_id"], "call_xyz789");

	let mut input = output.clone();
	input.push(json!({
		"type": "function_call_output",
		"call_id": "call_abc123",
		"output": "sunny"
	}));
	input.push(json!({
		"type": "function_call_output",
		"call_id": "call_xyz789",
		"output": "done"
	}));
	let request: crate::types::responses::Request = serde_json::from_value(json!({
		"model": "gemini-3.8-flash",
		"input": input,
		"tools": [],
		"store": false
	}))
	.unwrap();
	let mut translated = super::from_responses::translate_request(&request).unwrap();
	let next_request: Value = serde_json::to_value(&translated.request).unwrap();
	let calls = next_request["messages"][0]["tool_calls"]
		.as_array()
		.unwrap();

	assert_eq!(calls[0]["id"], "call_abc123");
	assert_eq!(
		calls[0]["extra_content"]["google"]["thought_signature"],
		signature
	);
	assert_eq!(calls[1]["id"], "call_xyz789");
	assert!(calls[1].get("extra_content").is_none());

	translated.request.strip_tool_call_extra_content();
	let other_provider: Value = serde_json::to_value(translated.request).unwrap();
	let calls = other_provider["messages"][0]["tool_calls"]
		.as_array()
		.unwrap();
	assert!(calls[0].get("extra_content").is_none());
}

#[test]
fn gemini_thought_signature_pairs_with_the_next_call_when_call_ids_repeat() {
	let carrier = |sig: &str| crate::conversion::GeminiThoughtSignature::encode("call_dup", sig);
	let turn = |sig: &str| {
		vec![
			json!({"type":"reasoning","id":"rs","summary":[],"encrypted_content": carrier(sig)}),
			json!({"type":"function_call","call_id":"call_dup","name":"f","arguments":"{}"}),
			json!({"type":"function_call_output","call_id":"call_dup","output":"ok"}),
		]
	};
	let mut input = turn("sig1");
	input.extend(turn("sig2"));
	let request: crate::types::responses::Request = serde_json::from_value(
		json!({"model":"gemini-3.8-flash","input":input,"tools":[],"store":false}),
	)
	.unwrap();
	let translated = super::from_responses::translate_request(&request).unwrap();
	let v: Value = serde_json::to_value(&translated.request).unwrap();
	let sigs: Vec<_> = v["messages"]
		.as_array()
		.unwrap()
		.iter()
		.filter_map(|m| {
			m["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
				.as_str()
				.map(str::to_owned)
		})
		.collect();
	assert_eq!(sigs, ["sig1", "sig2"]);
}

fn translate_responses_request(body: Value) -> Value {
	let request: crate::types::responses::Request = serde_json::from_value(body).unwrap();
	let translated = super::from_responses::translate_request(&request).unwrap();
	serde_json::to_value(&translated.request).unwrap()
}

fn roles(request: &Value) -> Vec<&str> {
	request["messages"]
		.as_array()
		.unwrap()
		.iter()
		.map(|m| m["role"].as_str().unwrap())
		.collect()
}

// Codex records a turn's text after the function calls it made. Each tool result must still
// directly follow the assistant message that made the call.
#[test]
fn assistant_text_after_function_call_stays_with_the_call() {
	let request = translate_responses_request(json!({
		"model": "claude-sonnet-5",
		"input": [
			{"role": "user", "content": "run ls"},
			{"type": "function_call", "call_id": "call_1", "name": "exec", "arguments": "{}"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Still running."}]},
			{"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
			{"type": "function_call", "call_id": "call_2", "name": "exec", "arguments": "{}"},
			{"type": "function_call", "call_id": "call_3", "name": "exec", "arguments": "{}"},
			{"type": "message", "role": "assistant", "content": []},
			{"type": "function_call_output", "call_id": "call_2", "output": "b"},
			{"type": "function_call_output", "call_id": "call_3", "output": "c"}
		]
	}));

	assert_eq!(
		roles(&request),
		["user", "assistant", "tool", "assistant", "tool", "tool"]
	);
	let messages = &request["messages"];
	assert_eq!(messages[1]["content"], "Still running.");
	assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
	assert_eq!(messages[3]["tool_calls"].as_array().unwrap().len(), 2);
	assert!(messages[3].get("content").is_none());
}

#[test]
fn assistant_text_before_function_call_is_one_message() {
	let request = translate_responses_request(json!({
		"model": "claude-sonnet-5",
		"input": [
			{"role": "user", "content": "run ls"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Running."}]},
			{"type": "function_call", "call_id": "call_1", "name": "exec", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "a.txt"}
		]
	}));

	assert_eq!(roles(&request), ["user", "assistant", "tool"]);
	assert_eq!(request["messages"][1]["content"], "Running.");
	assert_eq!(request["messages"][1]["tool_calls"][0]["id"], "call_1");
}

// Some providers reject a tool choice without tools. Non-function tools are dropped during
// translation, so a request can arrive with tools and a tool choice but leave with no tools.
#[test]
fn tool_choice_is_dropped_when_no_tools_survive() {
	for tools in [json!([]), json!([{"type": "web_search"}])] {
		let request = translate_responses_request(json!({
			"model": "claude-sonnet-5",
			"input": "hi",
			"tools": tools,
			"tool_choice": "auto",
			"parallel_tool_calls": true
		}));
		assert!(request.get("tools").is_none(), "{request}");
		assert!(request.get("tool_choice").is_none(), "{request}");
		assert!(request.get("parallel_tool_calls").is_none(), "{request}");
	}

	let request = translate_responses_request(json!({
		"model": "claude-sonnet-5",
		"input": "hi",
		"tool_choice": "auto"
	}));
	assert!(request.get("tool_choice").is_none(), "{request}");
}

#[test]
fn tool_choice_is_kept_with_function_tools() {
	let request = translate_responses_request(json!({
		"model": "claude-sonnet-5",
		"input": "hi",
		"tools": [
			{"type": "web_search"},
			{"type": "function", "name": "exec", "parameters": {"type": "object", "properties": {}}}
		],
		"tool_choice": "auto",
		"parallel_tool_calls": true
	}));
	assert_eq!(request["tools"].as_array().unwrap().len(), 1);
	assert_eq!(request["tool_choice"], "auto");
	assert_eq!(request["parallel_tool_calls"], true);
}
