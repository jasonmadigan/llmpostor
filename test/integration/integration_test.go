//go:build integration

package integration

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
	"github.com/jasonmadigan/llmpostor/internal/provider"
	"github.com/jasonmadigan/llmpostor/internal/provider/anthropic"
	"github.com/jasonmadigan/llmpostor/internal/provider/gemini"
	"github.com/jasonmadigan/llmpostor/internal/provider/openai"
	"github.com/jasonmadigan/llmpostor/internal/server"
)

func startServer(t *testing.T) string {
	t.Helper()
	cfg := config.DefaultConfig()

	lat := latency.NewCalculator(latency.Config{})

	registry := provider.NewRegistry()
	registry.Register(openai.New(cfg, lat))
	registry.Register(anthropic.New(cfg, lat))
	registry.Register(gemini.New(cfg, lat))

	handler, err := server.New(cfg, registry)
	if err != nil {
		t.Fatalf("creating server: %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	srv := &http.Server{Handler: handler}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })

	// wait for server to be ready
	base := fmt.Sprintf("http://%s", ln.Addr().String())
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(base + "/healthz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return base
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("server did not become ready")
	return ""
}

// post sends a POST request and returns the raw body and status.
func post(t *testing.T, url string, body any, headers map[string]string) ([]byte, int) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		t.Fatalf("creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading body: %v", err)
	}
	return data, resp.StatusCode
}

// postStream sends a POST and returns the raw response for SSE parsing.
func postStream(t *testing.T, url string, body any, headers map[string]string) *http.Response {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		t.Fatalf("creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	return resp
}

// unmarshalMap decodes JSON bytes into a generic map.
func unmarshalMap(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal: %v\nbody: %s", err, string(data))
	}
	return m
}

// requireField asserts a field exists in a map and returns it.
func requireField(t *testing.T, m map[string]any, key string) any {
	t.Helper()
	v, ok := m[key]
	if !ok {
		t.Fatalf("missing field %q in %v", key, m)
	}
	return v
}

// requireNoField asserts a field does NOT exist in a map.
func requireNoField(t *testing.T, m map[string]any, key string) {
	t.Helper()
	if _, ok := m[key]; ok {
		t.Fatalf("unexpected field %q present in %v", key, m)
	}
}

// requireMapField asserts a field is a map and returns it.
func requireMapField(t *testing.T, m map[string]any, key string) map[string]any {
	t.Helper()
	v := requireField(t, m, key)
	sub, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("field %q is not a map: %T", key, v)
	}
	return sub
}

// --- OpenAI chat/completions ---

func TestOpenAIChatCompletionsNonStreaming(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model":    "gpt-4",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}
	data, status := post(t, base+"/v1/chat/completions", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "id")
	requireField(t, m, "object")
	requireField(t, m, "model")
	requireField(t, m, "choices")

	usage := requireMapField(t, m, "usage")
	requireField(t, usage, "prompt_tokens")
	requireField(t, usage, "completion_tokens")
	requireField(t, usage, "total_tokens")
	requireMapField(t, usage, "prompt_tokens_details")
	requireMapField(t, usage, "completion_tokens_details")

	details := requireMapField(t, usage, "prompt_tokens_details")
	requireField(t, details, "cached_tokens")
	requireField(t, details, "audio_tokens")

	compDetails := requireMapField(t, usage, "completion_tokens_details")
	requireField(t, compDetails, "reasoning_tokens")
	requireField(t, compDetails, "audio_tokens")
	requireField(t, compDetails, "accepted_prediction_tokens")
	requireField(t, compDetails, "rejected_prediction_tokens")
}

func TestOpenAIChatCompletionsStreaming(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model":          "gpt-4",
		"messages":       []map[string]string{{"role": "user", "content": "hi"}},
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	resp := postStream(t, base+"/v1/chat/completions", body, nil)
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("expected text/event-stream, got %s", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var chunks []string
	var lastDataLine string
	var sawDone bool

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			payload := strings.TrimPrefix(line, "data: ")
			chunks = append(chunks, payload)
			lastDataLine = payload
		}
	}

	if !sawDone {
		t.Fatal("stream did not end with data: [DONE]")
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// last json chunk should have usage
	lastChunk := unmarshalMap(t, []byte(lastDataLine))
	usage := requireMapField(t, lastChunk, "usage")
	requireField(t, usage, "prompt_tokens")
	requireField(t, usage, "completion_tokens")
	requireField(t, usage, "total_tokens")

	// every chunk should be valid json with the SSE data: prefix format
	for _, c := range chunks {
		var m map[string]any
		if err := json.Unmarshal([]byte(c), &m); err != nil {
			t.Errorf("invalid json chunk: %s", c)
		}
	}
}

// --- OpenAI /v1/responses ---

func TestOpenAIResponses(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model": "gpt-4",
		"input": "hi",
	}
	data, status := post(t, base+"/v1/responses", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "id")
	requireField(t, m, "object")
	requireField(t, m, "output")

	usage := requireMapField(t, m, "usage")

	// responses API uses input_tokens/output_tokens, NOT prompt_tokens/completion_tokens
	requireField(t, usage, "input_tokens")
	requireField(t, usage, "output_tokens")
	requireField(t, usage, "total_tokens")
	requireNoField(t, usage, "prompt_tokens")
	requireNoField(t, usage, "completion_tokens")

	requireMapField(t, usage, "input_tokens_details")
	requireMapField(t, usage, "output_tokens_details")
}

// --- OpenAI /v1/completions ---

func TestOpenAICompletions(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model":  "gpt-3.5-turbo-instruct",
		"prompt": "hello",
	}
	data, status := post(t, base+"/v1/completions", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "id")
	requireField(t, m, "object")
	requireField(t, m, "choices")

	usage := requireMapField(t, m, "usage")
	requireField(t, usage, "prompt_tokens")
	requireField(t, usage, "completion_tokens")
	requireField(t, usage, "total_tokens")

	// no detail sub-objects
	requireNoField(t, usage, "prompt_tokens_details")
	requireNoField(t, usage, "completion_tokens_details")
}

// --- OpenAI /v1/embeddings ---

func TestOpenAIEmbeddings(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model": "text-embedding-ada-002",
		"input": "hello",
	}
	data, status := post(t, base+"/v1/embeddings", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "object")
	requireField(t, m, "data")
	requireField(t, m, "model")

	usage := requireMapField(t, m, "usage")
	requireField(t, usage, "prompt_tokens")
	requireField(t, usage, "total_tokens")
	requireNoField(t, usage, "completion_tokens")
}

// --- Anthropic /v1/messages ---

func TestAnthropicMessagesNonStreaming(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
	}
	data, status := post(t, base+"/v1/messages", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "id")
	requireField(t, m, "type")
	requireField(t, m, "role")
	requireField(t, m, "content")
	requireField(t, m, "model")
	requireField(t, m, "stop_reason")

	usage := requireMapField(t, m, "usage")
	requireField(t, usage, "input_tokens")
	requireField(t, usage, "output_tokens")
	requireField(t, usage, "cache_creation_input_tokens")
	requireField(t, usage, "cache_read_input_tokens")

	// anthropic does not include total_tokens
	requireNoField(t, usage, "total_tokens")
}

func TestAnthropicMessagesStreaming(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"stream":     true,
	}
	resp := postStream(t, base+"/v1/messages", body, nil)
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("expected text/event-stream, got %s", ct)
	}

	// parse anthropic SSE format: event: <type>\ndata: <json>\n\n
	scanner := bufio.NewScanner(resp.Body)
	var events []string
	var currentEvent string

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			events = append(events, currentEvent)
			// validate each data line is valid json
			payload := strings.TrimPrefix(line, "data: ")
			var m map[string]any
			if err := json.Unmarshal([]byte(payload), &m); err != nil {
				t.Errorf("invalid json for event %q: %s", currentEvent, payload)
			}
		}
	}

	// validate required event types are present
	required := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	eventSet := make(map[string]bool)
	for _, e := range events {
		eventSet[e] = true
	}
	for _, req := range required {
		if !eventSet[req] {
			t.Errorf("missing required event type: %s (got: %v)", req, events)
		}
	}
}

// --- Gemini generateContent ---

func TestGeminiGenerateContentNonStreaming(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]string{{"text": "hi"}}},
		},
	}
	data, status := post(t, base+"/v1beta/models/gemini-2.5-flash:generateContent", body, nil)
	if status != 200 {
		t.Fatalf("expected 200, got %d: %s", status, data)
	}

	m := unmarshalMap(t, data)
	requireField(t, m, "candidates")
	requireField(t, m, "modelVersion")

	// gemini uses usageMetadata, NOT usage
	requireNoField(t, m, "usage")
	meta := requireMapField(t, m, "usageMetadata")

	// camelCase field names
	requireField(t, meta, "promptTokenCount")
	requireField(t, meta, "candidatesTokenCount")
	requireField(t, meta, "totalTokenCount")
	requireField(t, meta, "cachedContentTokenCount")

	// must NOT have snake_case variants
	requireNoField(t, meta, "prompt_tokens")
	requireNoField(t, meta, "completion_tokens")
	requireNoField(t, meta, "total_tokens")
}

func TestGeminiStreamGenerateContent(t *testing.T) {
	base := startServer(t)
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]string{{"text": "hi"}}},
		},
	}
	resp := postStream(t, base+"/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", body, nil)
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	var chunks []map[string]any

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		m := unmarshalMap(t, []byte(payload))
		chunks = append(chunks, m)
	}

	if len(chunks) == 0 {
		t.Fatal("no chunks received")
	}

	// only the final chunk should have usageMetadata
	lastChunk := chunks[len(chunks)-1]
	meta := requireMapField(t, lastChunk, "usageMetadata")
	requireField(t, meta, "promptTokenCount")
	requireField(t, meta, "candidatesTokenCount")
	requireField(t, meta, "totalTokenCount")

	// intermediate chunks should not have usageMetadata
	for i := 0; i < len(chunks)-1; i++ {
		requireNoField(t, chunks[i], "usageMetadata")
	}
}

// --- cross-cutting: X-Sim-Input-Tokens / X-Sim-Output-Tokens ---

func TestCustomTokenHeaders(t *testing.T) {
	base := startServer(t)
	headers := map[string]string{
		"X-Sim-Input-Tokens":  "42",
		"X-Sim-Output-Tokens": "17",
	}

	t.Run("openai_chat", func(t *testing.T) {
		body := map[string]any{
			"model":    "gpt-4",
			"messages": []map[string]string{{"role": "user", "content": "hi"}},
		}
		data, status := post(t, base+"/v1/chat/completions", body, headers)
		if status != 200 {
			t.Fatalf("expected 200, got %d", status)
		}
		m := unmarshalMap(t, data)
		usage := requireMapField(t, m, "usage")
		if v := usage["prompt_tokens"].(float64); v != 42 {
			t.Errorf("expected prompt_tokens=42, got %v", v)
		}
		if v := usage["completion_tokens"].(float64); v != 17 {
			t.Errorf("expected completion_tokens=17, got %v", v)
		}
		if v := usage["total_tokens"].(float64); v != 59 {
			t.Errorf("expected total_tokens=59, got %v", v)
		}
	})

	t.Run("openai_responses", func(t *testing.T) {
		body := map[string]any{"model": "gpt-4", "input": "hi"}
		data, status := post(t, base+"/v1/responses", body, headers)
		if status != 200 {
			t.Fatalf("expected 200, got %d", status)
		}
		m := unmarshalMap(t, data)
		usage := requireMapField(t, m, "usage")
		if v := usage["input_tokens"].(float64); v != 42 {
			t.Errorf("expected input_tokens=42, got %v", v)
		}
		if v := usage["output_tokens"].(float64); v != 17 {
			t.Errorf("expected output_tokens=17, got %v", v)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		body := map[string]any{
			"model":      "claude-sonnet-4-6",
			"max_tokens": 100,
			"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		}
		data, status := post(t, base+"/v1/messages", body, headers)
		if status != 200 {
			t.Fatalf("expected 200, got %d", status)
		}
		m := unmarshalMap(t, data)
		usage := requireMapField(t, m, "usage")
		if v := usage["input_tokens"].(float64); v != 42 {
			t.Errorf("expected input_tokens=42, got %v", v)
		}
		if v := usage["output_tokens"].(float64); v != 17 {
			t.Errorf("expected output_tokens=17, got %v", v)
		}
	})

	t.Run("gemini", func(t *testing.T) {
		body := map[string]any{
			"contents": []map[string]any{
				{"parts": []map[string]string{{"text": "hi"}}},
			},
		}
		data, status := post(t, base+"/v1beta/models/gemini-2.5-flash:generateContent", body, headers)
		if status != 200 {
			t.Fatalf("expected 200, got %d", status)
		}
		m := unmarshalMap(t, data)
		meta := requireMapField(t, m, "usageMetadata")
		if v := meta["promptTokenCount"].(float64); v != 42 {
			t.Errorf("expected promptTokenCount=42, got %v", v)
		}
		if v := meta["candidatesTokenCount"].(float64); v != 17 {
			t.Errorf("expected candidatesTokenCount=17, got %v", v)
		}
		if v := meta["totalTokenCount"].(float64); v != 59 {
			t.Errorf("expected totalTokenCount=59, got %v", v)
		}
	})
}

// --- cross-cutting: X-Sim-Error ---

func TestSimErrorOpenAI(t *testing.T) {
	base := startServer(t)
	headers := map[string]string{"X-Sim-Error": "429"}
	body := map[string]any{
		"model":    "gpt-4",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}
	data, status := post(t, base+"/v1/chat/completions", body, headers)
	if status != 429 {
		t.Fatalf("expected 429, got %d", status)
	}

	m := unmarshalMap(t, data)
	errObj := requireMapField(t, m, "error")
	requireField(t, errObj, "message")
	requireField(t, errObj, "type")
}

func TestSimErrorAnthropic(t *testing.T) {
	base := startServer(t)
	headers := map[string]string{"X-Sim-Error": "429"}
	body := map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
	}
	data, status := post(t, base+"/v1/messages", body, headers)
	if status != 429 {
		t.Fatalf("expected 429, got %d", status)
	}

	m := unmarshalMap(t, data)

	// anthropic top-level has type: "error"
	if v := m["type"]; v != "error" {
		t.Errorf("expected type=error, got %v", v)
	}

	errObj := requireMapField(t, m, "error")
	requireField(t, errObj, "type")
	requireField(t, errObj, "message")
}

func TestSimErrorGemini(t *testing.T) {
	base := startServer(t)
	headers := map[string]string{"X-Sim-Error": "429"}
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]string{{"text": "hi"}}},
		},
	}
	data, status := post(t, base+"/v1beta/models/gemini-2.5-flash:generateContent", body, headers)
	if status != 429 {
		t.Fatalf("expected 429, got %d", status)
	}

	m := unmarshalMap(t, data)
	errObj := requireMapField(t, m, "error")

	requireField(t, errObj, "code")
	requireField(t, errObj, "message")
	requireField(t, errObj, "status")

	if v := errObj["code"].(float64); v != 429 {
		t.Errorf("expected error.code=429, got %v", v)
	}
	if v := errObj["status"].(string); v != "RESOURCE_EXHAUSTED" {
		t.Errorf("expected error.status=RESOURCE_EXHAUSTED, got %v", v)
	}
}
