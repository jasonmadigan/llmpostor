# Provider Response Reference

Response shapes for OpenAI, Anthropic, and Gemini as implemented by llmpostor. Primary audience: anyone implementing token extraction in a gateway (e.g. Kuadrant's TokenRateLimitPolicy).

## OpenAI

### POST /v1/chat/completions

Non-streaming response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gpt-4",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}
  }
}
```

**Token extraction:** `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`. All three always present. `total_tokens` = prompt + completion.

**Streaming:** SSE with `data: {chunk}` lines. Chunks have `object: "chat.completion.chunk"` and `choices[].delta` instead of `choices[].message`. Usage only appears in the final chunk when the request includes `stream_options.include_usage: true` -- otherwise no usage in the stream at all. Stream terminates with `data: [DONE]`.

The usage chunk has an empty `choices` array and carries the full `usage` object identical to the non-streaming shape.

### POST /v1/responses

```json
{
  "id": "resp-abc123",
  "object": "response",
  "model": "gpt-4",
  "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}],
  "usage": {
    "input_tokens": 10,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 15
  }
}
```

**Token extraction:** `usage.input_tokens`, `usage.output_tokens`, `usage.total_tokens`. Note the field names differ from chat/completions (`input_tokens` vs `prompt_tokens`, `output_tokens` vs `completion_tokens`).

**Streaming:** Not implemented.

### POST /v1/completions (legacy)

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "model": "gpt-3.5-turbo-instruct",
  "choices": [{"text": "...", "index": 0, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

**Token extraction:** Same field names as chat/completions. No detail sub-objects.

### POST /v1/embeddings

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0023, -0.0091]}],
  "model": "text-embedding-ada-002",
  "usage": {"prompt_tokens": 10, "total_tokens": 10}
}
```

**Token extraction:** `usage.prompt_tokens` and `usage.total_tokens` only. No `completion_tokens` (embeddings produce no output tokens).

### Error shape

```json
{"error": {"message": "simulated error", "type": "rate_limit_error", "code": "simulated"}}
```

Error `type` values: `authentication_error` (401), `rate_limit_error` (429), `server_error` (500), `invalid_request_error` (default).

## Anthropic

### POST /v1/messages

Non-streaming response:

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "model": "claude-sonnet-4-6",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 25,
    "output_tokens": 15,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**Token extraction:** `usage.input_tokens`, `usage.output_tokens`. There is **no `total_tokens` field** -- the gateway must compute `input_tokens + output_tokens` itself. Cache token fields are always present (zeroed when no caching).

**Streaming:** Uses named SSE events (`event:` + `data:` pairs). The sequence:

| # | Event | Carries |
|-|-|-|
| 1 | `message_start` | Full `message` object with `usage.input_tokens` (output_tokens is 0) |
| 2 | `content_block_start` | Empty content block `{"type":"text","text":""}` |
| 3 | `ping` | Heartbeat, no data |
| 4 | `content_block_delta` | `delta.text` with content |
| 5 | `content_block_stop` | Block index |
| 6 | `message_delta` | `usage.output_tokens` (cumulative final count) |
| 7 | `message_stop` | Nothing |

For token extraction from streaming: `input_tokens` comes from event 1 (`message_start`), `output_tokens` comes from event 6 (`message_delta`). These are in different events, so the extractor must track state across the stream.

### Error shape

```json
{"type": "error", "error": {"type": "rate_limit_error", "message": "simulated error"}}
```

Error `type` values: `invalid_request_error` (400), `authentication_error` (401), `rate_limit_error` (429), `overloaded_error` (529), `api_error` (default).

## Gemini

### POST /v1beta/models/{model}:generateContent

Non-streaming response:

```json
{
  "candidates": [{
    "content": {"parts": [{"text": "..."}], "role": "model"},
    "finishReason": "STOP",
    "index": 0,
    "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}]
  }],
  "usageMetadata": {
    "promptTokenCount": 4,
    "candidatesTokenCount": 12,
    "totalTokenCount": 16,
    "cachedContentTokenCount": 0
  },
  "modelVersion": "gemini-2.5-flash"
}
```

**Token extraction:** `usageMetadata.promptTokenCount`, `usageMetadata.candidatesTokenCount`, `usageMetadata.totalTokenCount`. Note camelCase field names. The usage object is named `usageMetadata`, not `usage`.

**Streaming:** `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`. SSE with `data: {json}` lines (no named events). Intermediate chunks carry text in `candidates[].content.parts` but no `usageMetadata`. The **final chunk only** includes `usageMetadata` with the full counts and `finishReason: "STOP"`.

### Error shape

```json
{"error": {"code": 429, "message": "simulated error", "status": "RESOURCE_EXHAUSTED"}}
```

Status values map HTTP codes to gRPC-style strings: `INVALID_ARGUMENT` (400), `UNAUTHENTICATED` (401), `PERMISSION_DENIED` (403), `NOT_FOUND` (404), `RESOURCE_EXHAUSTED` (429), `INTERNAL` (500), `UNAVAILABLE` (503).

## Cross-provider comparison

### Token field mapping

| Concept | OpenAI (chat) | OpenAI (responses) | Anthropic | Gemini |
|-|-|-|-|-|
| Usage object path | `usage` | `usage` | `usage` | `usageMetadata` |
| Input tokens | `usage.prompt_tokens` | `usage.input_tokens` | `usage.input_tokens` | `usageMetadata.promptTokenCount` |
| Output tokens | `usage.completion_tokens` | `usage.output_tokens` | `usage.output_tokens` | `usageMetadata.candidatesTokenCount` |
| Total tokens | `usage.total_tokens` | `usage.total_tokens` | not provided | `usageMetadata.totalTokenCount` |
| Cached tokens | `usage.prompt_tokens_details.cached_tokens` | `usage.input_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` | `usageMetadata.cachedContentTokenCount` |

### Streaming token delivery

| Provider | Format | Where usage appears | Events to watch |
|-|-|-|-|
| OpenAI (chat) | `data: {json}` + `data: [DONE]` | Final chunk (only if `stream_options.include_usage: true`) | Last chunk before `[DONE]` |
| Anthropic | Named SSE events (`event:` + `data:`) | Split: `input_tokens` in `message_start`, `output_tokens` in `message_delta` | Events 1 and 6 |
| Gemini | `data: {json}` | Final chunk only | Last SSE line |

### Gateway implementation notes

- **OpenAI chat streaming** requires the client to set `stream_options.include_usage: true` in the request. If absent, no usage data appears in the stream. A gateway may need to inject this option into the request body.
- **Anthropic** never provides `total_tokens`. The gateway must sum `input_tokens + output_tokens`.
- **Anthropic streaming** splits usage across two events. The gateway must buffer `input_tokens` from `message_start` and `output_tokens` from `message_delta` to get the full picture.
- **Gemini** uses camelCase field names and nests usage under `usageMetadata` rather than `usage`. Its streaming endpoint is a different action (`:streamGenerateContent`) rather than a `stream: true` body flag.
- **OpenAI responses API** uses different field names (`input_tokens`/`output_tokens`) than the chat API (`prompt_tokens`/`completion_tokens`). A gateway must handle both.
