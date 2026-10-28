use agent_core::strng;
use agent_core::strng::Strng;
use http::{HeaderMap, HeaderValue, header};

use crate::{ChatFormat, RouteType, apply};

#[apply(schema!)]
#[cfg_attr(feature = "schema", schemars(rename = "CopilotProvider"))]
pub struct Provider {
	/// Model ID to send to GitHub Copilot, overriding the model in the client request.
	#[serde(default, rename = "model", skip_serializing_if = "Option::is_none")]
	pub model_override: Option<Strng>,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("copilot");
}

impl Provider {
	pub fn is_anthropic_model(request_model: &str) -> bool {
		request_model.to_ascii_lowercase().starts_with("claude-")
	}
	pub fn set_required_headers(headers: &mut HeaderMap) {
		headers.insert(
			header::CONTENT_TYPE,
			HeaderValue::from_static("application/json"),
		);
		headers.insert(
			"editor-version",
			HeaderValue::from_static(concat!("agentgateway/", env!("CARGO_PKG_VERSION"))),
		);
		headers.insert(
			"openai-intent",
			HeaderValue::from_static("conversation-agent"),
		);
		headers.insert(
			"x-github-api-version",
			HeaderValue::from_static("2025-10-01"),
		);
		headers.insert("x-initiator", HeaderValue::from_static("agent"));
		headers.insert(
			"x-interaction-type",
			HeaderValue::from_static("conversation-agent"),
		);
	}
	pub fn supported_formats_for_model(
		request_model: &str,
		catalog: crate::model_catalog::Catalog<'_>,
	) -> Vec<ChatFormat> {
		let m = request_model;
		let normalized_model = m.to_ascii_lowercase();
		// TODO: also support endpoint parsing from copilot models and add a tool to grab specific setups in agctl
		if let Some(tags) = catalog.and_then(|c| c.get_model_tags(&normalized_model)) {
			let formats: Vec<ChatFormat> = [
				ChatFormat::OpenAICompletions,
				ChatFormat::OpenAIResponses,
				ChatFormat::AnthropicMessages,
				ChatFormat::BedrockConverse,
				ChatFormat::VertexGemini,
			]
			.into_iter()
			.filter(|f| tags.contains(f.tag()))
			.collect();
			if !formats.is_empty() {
				tracing::debug!(model = %m, ?formats, "copilot formats from modelcatalog tags");
				return formats;
			}
		}
		// Truth table from `curl https://api.githubcopilot.com/models -H "Authorization: Bearer ghu_..." | '.data[] | {id,supported_endpoints}'`
		match normalized_model.as_str() {
			m if m.starts_with("claude-") => {
				// Copilot supports Completions even for Anthropic
				// This is enabled so we can do Responses --> Completions [--> Anthropic, within copilot, presumably].
				// If we add native Responses --> Anthropic we should drop this
				vec![ChatFormat::AnthropicMessages, ChatFormat::OpenAICompletions]
			},
			m if m.starts_with("grok-") || m.starts_with("mai-") => {
				vec![ChatFormat::OpenAIResponses]
			},
			m if m.starts_with("gemini-") => {
				vec![ChatFormat::OpenAICompletions]
			},
			m if m.starts_with("gpt-3") || m.starts_with("gpt-4") => {
				vec![ChatFormat::OpenAICompletions]
			},
			"gpt-5.4" | "gpt-5-mini" => {
				vec![ChatFormat::OpenAICompletions, ChatFormat::OpenAIResponses]
			},
			m if m.starts_with("gpt-") => {
				vec![ChatFormat::OpenAIResponses]
			},
			_ => vec![ChatFormat::OpenAICompletions],
		}
	}
}

pub const DEFAULT_HOST_STR: &str = "api.githubcopilot.com";
pub const DEFAULT_HOST: Strng = strng::literal!(DEFAULT_HOST_STR);

pub fn path_suffix(route: RouteType) -> &'static str {
	match route {
		RouteType::Messages => "/v1/messages",
		RouteType::Responses => "/responses",
		RouteType::Embeddings => "/embeddings",
		RouteType::Rerank => "/rerank",
		RouteType::Models => "/models",
		_ => "/chat/completions",
	}
}

/// Copilot uses different opaque IDs for events belonging to one Responses output item.
/// Anchor them to the first ID instead of inventing IDs that cannot be replayed upstream.
pub fn normalize_responses_stream(body: agent_http::Body, buffer_limit: usize) -> agent_http::Body {
	use bytes::Bytes;
	use tokio_sse_codec::{Frame, SseDecoder, SseEncoder};

	use crate::parse::{sse::DoneTolerantSseDecoder, transform};

	let mut ids = ResponseItemIds::default();
	transform::parser(
		body,
		DoneTolerantSseDecoder(SseDecoder::<Bytes>::with_max_size(buffer_limit)),
		SseEncoder::new(),
		move |event| {
			let transform::TransformEvent::Item(mut frame) = event else {
				return None;
			};
			if let Frame::Event(event) = &mut frame
				&& let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&event.data)
				&& ids.normalize(&mut value)
			{
				event.data = Bytes::from(serde_json::to_vec(&value).expect("JSON value serializes"));
			}
			Some(frame)
		},
	)
}

#[derive(Default)]
struct ResponseItemIds(std::collections::BTreeMap<u64, String>);

impl ResponseItemIds {
	fn normalize(&mut self, event: &mut serde_json::Value) -> bool {
		use serde_json::Value;

		let terminal = matches!(
			event.get("type").and_then(Value::as_str),
			Some("response.completed" | "response.incomplete" | "response.failed")
		);
		if let Some(index) = event.get("output_index").and_then(Value::as_u64) {
			if event.get("type").and_then(Value::as_str) == Some("response.output_item.added")
				&& let Some(id) = event.pointer("/item/id").and_then(Value::as_str)
			{
				self.0.entry(index).or_insert_with(|| id.to_owned());
			}
		}

		let mut changed = false;
		if let Some(id) = event
			.get("output_index")
			.and_then(Value::as_u64)
			.and_then(|index| self.0.get(&index))
		{
			changed |= replace_id(event.get_mut("item_id"), id);
			changed |= replace_id(event.pointer_mut("/item/id"), id);
		}
		// Keep the final output array consistent with the incrementally emitted items,
		// including incomplete responses. Encrypted reasoning and call IDs stay intact.
		if terminal
			&& let Some(output) = event
				.pointer_mut("/response/output")
				.and_then(Value::as_array_mut)
		{
			for (index, item) in output.iter_mut().enumerate() {
				if let Some(id) = self.0.get(&(index as u64)) {
					changed |= replace_id(item.get_mut("id"), id);
				}
			}
		}
		changed
	}
}

fn replace_id(value: Option<&mut serde_json::Value>, id: &str) -> bool {
	if let Some(serde_json::Value::String(current)) = value
		&& current != id
	{
		*current = id.to_owned();
		return true;
	}
	false
}

#[cfg(test)]
mod tests {
	#[test]
	fn response_item_ids_normalize_interleaved_items_and_terminal_outputs() {
		use serde_json::json;

		let mut ids = ResponseItemIds::default();
		for (index, kind) in [(0, "reasoning"), (1, "function_call"), (2, "message")] {
			let mut added = json!({
				"type": "response.output_item.added", "output_index": index,
				"item": {"type": kind, "id": format!("anchor-{index}")}
			});
			assert!(!ids.normalize(&mut added));
		}
		for (index, kind) in [
			(1, "response.function_call_arguments.delta"),
			(0, "response.reasoning_summary_part.added"),
			(2, "response.content_part.added"),
			(0, "response.reasoning_summary_text.delta"),
			(2, "response.output_text.delta"),
			(1, "response.function_call_arguments.done"),
		] {
			let mut delta = json!({
				"type": kind, "output_index": index, "item_id": "changing-id",
				"sequence_number": 12, "delta": "opaque delta", "provider_extension": true
			});
			let mut expected = delta.clone();
			expected["item_id"] = json!(format!("anchor-{index}"));
			assert!(ids.normalize(&mut delta));
			assert_eq!(delta, expected);
			assert!(!ids.normalize(&mut delta));
		}
		let mut done = json!({
			"type": "response.output_item.done", "output_index": 0,
			"item": {"id": "done-id", "encrypted_content": "opaque-reasoning"}
		});
		assert!(ids.normalize(&mut done));
		assert_eq!(done["item"]["id"], "anchor-0");
		assert_eq!(done["item"]["encrypted_content"], "opaque-reasoning");
		for kind in [
			"response.completed",
			"response.incomplete",
			"response.failed",
		] {
			let mut terminal = json!({"type": kind, "response": {"output": [
				{"id": "final-0", "encrypted_content": "opaque-reasoning"},
				{"id": "final-1", "call_id": "call-id", "arguments": "{}"},
				{"id": "final-2", "content": [{"type": "output_text", "text": "OK"}]},
				{"id": "unregistered"}
			]}});
			let mut expected = terminal.clone();
			for index in 0..3 {
				expected["response"]["output"][index]["id"] = json!(format!("anchor-{index}"));
			}
			assert!(ids.normalize(&mut terminal));
			assert_eq!(terminal, expected);
		}
	}

	#[test]
	fn response_item_ids_preserve_unknown_or_missing_ids() {
		use serde_json::json;

		let mut ids = ResponseItemIds::default();
		for mut event in [
			json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "unknown"}),
			json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}),
			json!({"type": "response.completed", "response": {"output": [{"id": "unknown"}]}}),
			json!({"type": "response.output_text.delta", "item_id": "unindexed"}),
			json!({"provider_extension": true}),
		] {
			let expected = event.clone();
			assert!(!ids.normalize(&mut event));
			assert_eq!(event, expected);
		}
	}

	#[tokio::test]
	async fn response_item_ids_preserve_sse_metadata_and_fragmented_tool_replay() {
		use bytes::{Bytes, BytesMut};
		use http_body_util::BodyExt;
		use serde_json::json;
		use tokio_sse_codec::{Event, Frame, SseDecoder};
		use tokio_util::codec::Decoder;

		let input = concat!(
			": heartbeat\nretry: 1000\nid: cursor\nevent: response.output_item.added\n",
			"data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
			"\"item\":{\"type\":\"function_call\",\"id\":\"anchor\",\"call_id\":\"call\"}}\n\n",
			"event: response.function_call_arguments.delta\n",
			"data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,",
			"\"item_id\":\"rotating\",\"delta\":\"{}\"}\n\n",
			"event: future-event\ndata: {\"future\":true}\n\n",
			"data: non-json\n\n",
			"data: [DONE]"
		);
		let chunks = input
			.as_bytes()
			.chunks(7)
			.map(|part| Ok::<_, std::io::Error>(Bytes::copy_from_slice(part)))
			.collect::<Vec<_>>();
		let body = agent_http::Body::from_stream(futures_util::stream::iter(chunks));
		let output = normalize_responses_stream(body, 4096)
			.collect()
			.await
			.unwrap()
			.to_bytes();
		let mut decoder = SseDecoder::<Bytes>::new();
		let mut buffer = BytesMut::from(output.as_ref());
		let mut frames = vec![];
		while let Some(frame) = decoder.decode_eof(&mut buffer).unwrap() {
			frames.push(frame);
		}
		assert!(matches!(&frames[0], Frame::Comment(c) if c.as_ref() == b"heartbeat"));
		assert!(matches!(&frames[1], Frame::Retry(d) if *d == std::time::Duration::from_secs(1)));
		let events = frames
			.iter()
			.filter_map(|frame| match frame {
				Frame::Event(event) => Some(event),
				_ => None,
			})
			.collect::<Vec<&Event<Bytes>>>();
		assert_eq!(events.len(), 5);
		assert_eq!(events[0].name, "response.output_item.added");
		assert_eq!(events[0].id.as_deref(), Some("cursor"));
		assert_eq!(events[1].name, "response.function_call_arguments.delta");
		let delta: serde_json::Value = serde_json::from_slice(&events[1].data).unwrap();
		assert_eq!(delta["item_id"], "anchor");
		assert_eq!(delta["delta"], "{}");
		assert_eq!(
			serde_json::from_slice::<serde_json::Value>(&events[2].data).unwrap(),
			json!({"future": true})
		);
		assert_eq!(events[3].data.as_ref(), b"non-json");
		assert_eq!(events[4].data.as_ref(), b"[DONE]");
	}

	use super::*;
	use crate::model_catalog::{Catalog, TestCatalog, tags};

	#[test]
	fn catalog_format_tags_override_builtins() {
		// grok-* defaults to Responses; a catalog tag forces Completions instead.
		let cat = TestCatalog::new([("grok-2", &[tags::OPENAI_COMPLETIONS][..])]);
		let catalog: Catalog = Some(&cat);
		assert_eq!(
			Provider::supported_formats_for_model("grok-2", catalog),
			vec![ChatFormat::OpenAICompletions]
		);
	}

	#[test]
	fn catalog_format_tags_are_case_insensitive() {
		let cat = TestCatalog::new([("grok-2", &[tags::OPENAI_COMPLETIONS][..])]);
		let catalog: Catalog = Some(&cat);
		assert_eq!(
			Provider::supported_formats_for_model("Grok-2", catalog),
			vec![ChatFormat::OpenAICompletions]
		);
	}

	#[test]
	fn untagged_model_falls_back_to_builtins() {
		let cat = TestCatalog::new([("grok-2", &[][..])]);
		let catalog: Catalog = Some(&cat);
		assert_eq!(
			Provider::supported_formats_for_model("grok-2", catalog),
			vec![ChatFormat::OpenAIResponses]
		);
	}
}
