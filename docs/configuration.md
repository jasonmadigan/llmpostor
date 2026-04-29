# Configuration

llmpostor is configured through three layers, applied in order of increasing precedence:

**Config file < CLI flags < Request headers**

Values set later override earlier ones. For example, a config file setting `default_input_tokens` to 50 is overridden by the `-input-tokens 100` flag, which is in turn overridden by a request carrying `X-Sim-Input-Tokens: 200`.

## Config file

Pass a JSON file with `-config path/to/config.json`. All fields are optional; unset fields use defaults.

```json
{
  "port": 8080,
  "enabled_providers": ["openai", "anthropic", "gemini"],
  "default_model": "gpt-4",
  "default_input_tokens": 10,
  "default_output_tokens": 5,
  "response_content": "Hello",
  "ttft": "200ms",
  "ttft_stddev": "30ms",
  "itl": "20ms",
  "itl_stddev": "5ms",
  "load_factor": 2.0,
  "max_concurrent": 32
}
```

## CLI flags

| Flag | Default | Description |
|-|-|
| `-config` | (none) | Path to JSON config file |
| `-port` | `8080` | Listen port |
| `-model` | `gpt-4` | Default model name returned in responses |
| `-input-tokens` | `10` | Default input/prompt token count |
| `-output-tokens` | `5` | Default output/completion token count |
| `-response-content` | `Hello` | Static content returned in responses |
| `-ttft` | `0s` | Mean time to first token (e.g. `200ms`) |
| `-ttft-stddev` | `0s` | TTFT jitter standard deviation |
| `-itl` | `0s` | Mean inter-token latency (e.g. `20ms`) |
| `-itl-stddev` | `0s` | ITL jitter standard deviation |
| `-load-factor` | `1.0` | Max latency multiplier under concurrent load |
| `-max-concurrent` | `32` | Concurrent request count at which load-factor peaks |

Note: `enabled_providers` is only configurable via the config file, not via flags.

## Request headers

Per-request overrides, applied by middleware. These take highest precedence.

| Header | Type | Effect |
|-|-|-|
| `X-Sim-Input-Tokens` | integer | Override input/prompt token count |
| `X-Sim-Output-Tokens` | integer | Override output/completion token count |
| `X-Sim-Error` | integer | Force a provider-specific error response (HTTP status code) |

### Examples

Override token counts:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Sim-Input-Tokens: 500" \
  -H "X-Sim-Output-Tokens: 100" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}'
```

Force an error response:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Sim-Error: 429" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}'
```

## Enabling providers

The `enabled_providers` array controls which provider endpoints are registered. Valid values: `openai`, `anthropic`, `gemini`.

To simulate only one provider:

```json
{
  "enabled_providers": ["anthropic"]
}
```

Only the routes for enabled providers are mounted. Requests to disabled provider endpoints will 404.

## Latency simulation

Latency is disabled by default (all values `0s`). When configured, it makes responses behave like real inference.

- **TTFT** -- delay before the first token. Applied to streaming (before first chunk) and non-streaming (before response) requests.
- **ITL** -- delay between each streamed chunk. Streaming only.
- **Jitter** -- TTFT and ITL each accept a stddev. Values are sampled from a normal distribution and clamped to zero.
- **Load factor** -- multiplies both TTFT and ITL based on concurrent request count. At 1 request, no scaling. At `max-concurrent`, latency is multiplied by `load-factor`.

```bash
# realistic openai-like timing
./bin/llmpostor -ttft 300ms -ttft-stddev 50ms -itl 25ms -itl-stddev 5ms

# heavier load simulation
./bin/llmpostor -ttft 200ms -itl 20ms -load-factor 3.0 -max-concurrent 16
```

## Deployment

### Local

```bash
go run ./cmd/llmpostor -port 9090 -model claude-3-opus -input-tokens 100
```

### Docker

```bash
docker build -t llmpostor .
docker run -p 8080:8080 llmpostor -model gpt-4o -output-tokens 50
```

With a config file:

```bash
docker run -p 8080:8080 -v $(pwd)/config.json:/config.json llmpostor -config /config.json
```

### Kubernetes

Manifests are in `deploy/k8s/`. The deployment runs as non-root (UID 65532) with a read-only root filesystem.

To inject configuration, create a ConfigMap and mount it:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmpostor-config
data:
  config.json: |
    {
      "default_model": "claude-3-opus",
      "default_input_tokens": 100,
      "enabled_providers": ["anthropic"]
    }
```

Then add the volume and args to the deployment:

```yaml
containers:
  - name: llmpostor
    args: ["-config", "/etc/llmpostor/config.json"]
    volumeMounts:
      - name: config
        mountPath: /etc/llmpostor
volumes:
  - name: config
    configMap:
      name: llmpostor-config
```
