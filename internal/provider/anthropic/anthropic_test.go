package anthropic

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
)

func newTestProvider() *Provider {
	return New(&config.Config{
		DefaultModel:        "claude-sonnet-4-6",
		DefaultInputTokens:  25,
		DefaultOutputTokens: 15,
		ResponseContent:     "Hello",
	}, latency.NewCalculator(latency.Config{}))
}

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	p := newTestProvider()
	mux := http.NewServeMux()
	p.RegisterRoutes(mux)
	return httptest.NewServer(config.SimHeaderMiddleware(mux))
}

func postMessages(t *testing.T, ts *httptest.Server, body string, headers map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatalf("creating request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("sending request: %v", err)
	}
	return resp
}

func TestNonStreamingResponseShape(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decoding response: %v", err)
	}

	// verify required fields
	for _, field := range []string{"id", "type", "role", "content", "model", "stop_reason", "usage"} {
		if _, ok := result[field]; !ok {
			t.Errorf("missing field %q", field)
		}
	}

	if result["type"] != "message" {
		t.Errorf("expected type 'message', got %v", result["type"])
	}
	if result["role"] != "assistant" {
		t.Errorf("expected role 'assistant', got %v", result["role"])
	}
	if result["model"] != "claude-sonnet-4-6" {
		t.Errorf("expected model 'claude-sonnet-4-6', got %v", result["model"])
	}

	// stop_sequence must be null
	if result["stop_sequence"] != nil {
		t.Errorf("expected stop_sequence null, got %v", result["stop_sequence"])
	}

	// content must be array with text block
	contentArr, ok := result["content"].([]any)
	if !ok || len(contentArr) != 1 {
		t.Fatalf("expected content array with 1 element, got %v", result["content"])
	}
	block, ok := contentArr[0].(map[string]any)
	if !ok {
		t.Fatalf("expected content block to be object, got %T", contentArr[0])
	}
	if block["type"] != "text" || block["text"] != "Hello" {
		t.Errorf("unexpected content block: %v", block)
	}
}

func TestNoTotalTokensField(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var raw json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// parse usage as raw map to check exact fields
	var result struct {
		Usage map[string]any `json:"usage"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if _, exists := result.Usage["total_tokens"]; exists {
		t.Error("usage must NOT contain total_tokens for Anthropic responses")
	}
}

func TestAllUsageFieldsPresent(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var result struct {
		Usage map[string]any `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}

	required := []string{
		"input_tokens",
		"output_tokens",
		"cache_creation_input_tokens",
		"cache_read_input_tokens",
	}
	for _, field := range required {
		if _, ok := result.Usage[field]; !ok {
			t.Errorf("missing usage field %q", field)
		}
	}
}

func TestTokenCountsFromHeaders(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	headers := map[string]string{
		"X-Sim-Input-Tokens":  "100",
		"X-Sim-Output-Tokens": "50",
	}
	resp := postMessages(t, ts, `{"messages":[{"role":"user","content":"hi"}]}`, headers)
	defer func() { _ = resp.Body.Close() }()

	var result struct {
		Usage usage `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if result.Usage.InputTokens != 100 {
		t.Errorf("expected input_tokens 100, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 50 {
		t.Errorf("expected output_tokens 50, got %d", result.Usage.OutputTokens)
	}
}

func TestErrorResponseShape(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	tests := []struct {
		code     string
		status   int
		errType  string
	}{
		{"400", 400, "invalid_request_error"},
		{"401", 401, "authentication_error"},
		{"429", 429, "rate_limit_error"},
		{"500", 500, "api_error"},
		{"529", 529, "overloaded_error"},
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			headers := map[string]string{"X-Sim-Error": tt.code}
			resp := postMessages(t, ts, `{"messages":[{"role":"user","content":"hi"}]}`, headers)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != tt.status {
				t.Errorf("expected status %d, got %d", tt.status, resp.StatusCode)
			}

			var result apiError
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				t.Fatalf("decode: %v", err)
			}

			if result.Type != "error" {
				t.Errorf("expected type 'error', got %q", result.Type)
			}
			if result.Error.Type != tt.errType {
				t.Errorf("expected error type %q, got %q", tt.errType, result.Error.Type)
			}
			if result.Error.Message == "" {
				t.Error("expected non-empty error message")
			}
		})
	}
}

// sseEvent holds a parsed SSE event with both event type and data payload.
type sseEvent struct {
	Event string
	Data  string
}

func parseSSEEvents(t *testing.T, body []byte) []sseEvent {
	t.Helper()
	var events []sseEvent
	scanner := bufio.NewScanner(bytes.NewReader(body))

	var current sseEvent
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			current.Event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			current.Data = strings.TrimPrefix(line, "data: ")
		case line == "":
			if current.Event != "" || current.Data != "" {
				events = append(events, current)
				current = sseEvent{}
			}
		}
	}
	return events
}

func TestStreamingContentType(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	ct := resp.Header.Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got %q", ct)
	}
}

func TestStreamingEventSequence(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read body: %v", err)
	}
	events := parseSSEEvents(t, buf.Bytes())

	expectedEvents := []string{
		"message_start",
		"content_block_start",
		"ping",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedEvents) {
		t.Fatalf("expected %d events, got %d: %v", len(expectedEvents), len(events), events)
	}

	for i, expected := range expectedEvents {
		if events[i].Event != expected {
			t.Errorf("event %d: expected %q, got %q", i, expected, events[i].Event)
		}
	}
}

func TestStreamingEventsHaveEventField(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read body: %v", err)
	}
	events := parseSSEEvents(t, buf.Bytes())

	for i, ev := range events {
		if ev.Event == "" {
			t.Errorf("event %d missing 'event:' field", i)
		}
		if ev.Data == "" {
			t.Errorf("event %d missing 'data:' field", i)
		}
	}
}

func TestStreamingMessageStartUsage(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	headers := map[string]string{
		"X-Sim-Input-Tokens":  "42",
		"X-Sim-Output-Tokens": "7",
	}
	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, headers)
	defer func() { _ = resp.Body.Close() }()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read body: %v", err)
	}
	events := parseSSEEvents(t, buf.Bytes())

	// message_start should carry input_tokens
	var msgStart streamMessageStart
	if err := json.Unmarshal([]byte(events[0].Data), &msgStart); err != nil {
		t.Fatalf("unmarshal message_start: %v", err)
	}

	if msgStart.Message.Usage.InputTokens != 42 {
		t.Errorf("expected input_tokens 42 in message_start, got %d", msgStart.Message.Usage.InputTokens)
	}
	if msgStart.Message.Usage.OutputTokens != 0 {
		t.Errorf("expected output_tokens 0 in message_start, got %d", msgStart.Message.Usage.OutputTokens)
	}

	// message_delta should carry output_tokens
	var msgDelta streamMessageDelta
	if err := json.Unmarshal([]byte(events[5].Data), &msgDelta); err != nil {
		t.Fatalf("unmarshal message_delta: %v", err)
	}

	if msgDelta.Usage.OutputTokens != 7 {
		t.Errorf("expected output_tokens 7 in message_delta, got %d", msgDelta.Usage.OutputTokens)
	}
}

func TestStreamingContentBlockDelta(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read body: %v", err)
	}
	events := parseSSEEvents(t, buf.Bytes())

	// content_block_delta is event index 3
	var delta streamContentBlockDelta
	if err := json.Unmarshal([]byte(events[3].Data), &delta); err != nil {
		t.Fatalf("unmarshal delta: %v", err)
	}

	if delta.Delta.Type != "text_delta" {
		t.Errorf("expected delta type 'text_delta', got %q", delta.Delta.Type)
	}
	if delta.Delta.Text != "Hello" {
		t.Errorf("expected delta text 'Hello', got %q", delta.Delta.Text)
	}
}

func TestStreamingNoTotalTokens(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer func() { _ = resp.Body.Close() }()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatalf("read body: %v", err)
	}

	if bytes.Contains(buf.Bytes(), []byte("total_tokens")) {
		t.Error("streaming response must NOT contain total_tokens")
	}
}

func TestProviderName(t *testing.T) {
	p := newTestProvider()
	if p.Name() != "anthropic" {
		t.Errorf("expected name 'anthropic', got %q", p.Name())
	}
}

func TestInvalidRequestBody(t *testing.T) {
	ts := newTestServer(t)
	defer ts.Close()

	resp := postMessages(t, ts, `not json`, nil)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}

	var result apiError
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if result.Type != "error" {
		t.Errorf("expected type 'error', got %q", result.Type)
	}
}
