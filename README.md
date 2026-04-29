# llmpostor

An LLM API that does no thinking whatsoever.

llmpostor returns correctly-shaped responses for OpenAI, Anthropic, and Gemini APIs with accurate token metadata, but performs no actual inference. It exists to test token extraction in [Kuadrant's TokenRateLimitPolicy](https://github.com/Kuadrant/kuadrant-operator/issues/1864), which needs to parse usage data from multiple provider response formats. Think of it as a stunt double: it looks right on camera, but there's nobody home.

## Usage

```sh
docker run --rm -p 8080:8080 ghcr.io/jasonmadigan/llmpostor:latest
```

Or build from source:

```sh
make build
./bin/llmpostor --port 8080
```

## Endpoints

| Provider | Endpoint | Streaming |
|-|-|-|
| OpenAI | `POST /v1/chat/completions` | Yes |
| OpenAI | `POST /v1/responses` | Yes |
| OpenAI | `POST /v1/completions` | No |
| OpenAI | `POST /v1/embeddings` | No |
| Anthropic | `POST /v1/messages` | Yes |
| Gemini | `POST /v1beta/models/{model}:generateContent` | Yes |
| Gemini | `POST /v1beta/models/{model}:streamGenerateContent` | Yes |

Health checks at `/healthz` and `/readyz`.

## Configuration

| Flag | Default | Description |
|-|-|-|
| `-port` | `8080` | Listen port |
| `-model` | `gpt-4` | Default model name in responses |
| `-input-tokens` | `10` | Default input token count |
| `-output-tokens` | `5` | Default output token count |
| `-response-content` | `Hello` | Static text returned in all responses |
| `-config` | | Path to JSON config file |
| `-ttft` | `0s` | Mean time to first token (e.g. `200ms`) |
| `-ttft-stddev` | `0s` | TTFT jitter standard deviation (e.g. `30ms`) |
| `-itl` | `0s` | Mean inter-token latency (e.g. `20ms`) |
| `-itl-stddev` | `0s` | ITL jitter standard deviation (e.g. `5ms`) |
| `-load-factor` | `1.0` | Max latency multiplier under concurrent load |
| `-max-concurrent` | `32` | Concurrent request count at which load-factor peaks |

Flags override config file values.

## Controlling responses

Per-request token counts are set via headers, overriding the defaults above:

- `X-Sim-Input-Tokens` -- override input/prompt token count
- `X-Sim-Output-Tokens` -- override output/completion token count

```sh
# openai
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Sim-Input-Tokens: 25" \
  -H "X-Sim-Output-Tokens: 40" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}'

# openai (streaming)
curl -s -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}'

# openai responses api (streaming)
curl -s -N http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","input":"hi","stream":true}'

# anthropic
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-real" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}],"max_tokens":100}'

# anthropic (streaming)
curl -s -N http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-real" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"stream":true}'

# gemini
curl -s http://localhost:8080/v1beta/models/gemini-2.5-flash:generateContent?key=not-real \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"hi"}]}]}'

# gemini (streaming)
curl -s -N 'http://localhost:8080/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=not-real' \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"hi"}]}]}'
```

Auth headers and API keys are accepted but not validated. Obviously.

## Token usage in responses

Every response includes token usage metadata in the provider's native format. This is what a gateway needs to parse for token rate limiting.

| Provider | Location | Input field | Output field | Total field |
|-|-|-|-|-|
| OpenAI (chat/completions) | `usage` | `prompt_tokens` | `completion_tokens` | `total_tokens` |
| OpenAI (responses) | `usage` | `input_tokens` | `output_tokens` | `total_tokens` |
| OpenAI (embeddings) | `usage` | `prompt_tokens` | -- | `total_tokens` |
| Anthropic | `usage` | `input_tokens` | `output_tokens` | not provided |
| Gemini | `usageMetadata` | `promptTokenCount` | `candidatesTokenCount` | `totalTokenCount` |

Anthropic does not return a total -- the gateway must sum `input_tokens + output_tokens`.

In streaming responses, token usage appears at different points per provider:

| Provider | Where usage appears |
|-|-|
| OpenAI (chat) | Final SSE chunk (when `stream_options.include_usage: true`) |
| OpenAI (responses) | `response.completed` event |
| Anthropic | `message_start` has `input_tokens`, `message_delta` has `output_tokens` |
| Gemini | Final SSE chunk only |

## Latency simulation

By default, responses are instant. To simulate realistic inference timing:

```sh
./bin/llmpostor --port 8080 \
  --ttft 200ms \
  --ttft-stddev 30ms \
  --itl 20ms \
  --itl-stddev 5ms \
  --load-factor 2.0 \
  --max-concurrent 32
```

- **TTFT** (time to first token) -- delay before the first token or response. Applied to both streaming and non-streaming requests.
- **ITL** (inter-token latency) -- delay between each streamed chunk. Only applies to streaming responses.
- **Jitter** -- both TTFT and ITL support a standard deviation parameter. Values are sampled from a normal distribution, clamped to zero (no negative delays).
- **Load factor** -- scales latency based on concurrent requests. At 1 concurrent request, factor is 1.0 (no scaling). At `max-concurrent`, factor reaches the configured maximum. Formula: `1 + (factor - 1) * (running - 1) / (max - 1)`.

## Error simulation

Set `X-Sim-Error` to an HTTP status code to get a provider-native error response:

```sh
curl -s http://localhost:8080/v1/chat/completions \
  -H "X-Sim-Error: 429" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}'
```

Each provider returns its own error shape (OpenAI's `error` object, Anthropic's `{"type":"error",...}`, Gemini's `{"error":{...}}`). Errors are deterministic and per-request.

## Building

```sh
make build          # bin/llmpostor
make test           # go test ./...
make docker-build   # IMAGE=llmpostor:latest
make docker-run     # runs on :8080
```

## Deploying

Kubernetes manifests in `deploy/k8s/`. Apply with kustomize:

```sh
kubectl apply -k deploy/k8s/
```


