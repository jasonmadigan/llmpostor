package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
)

func setup(t *testing.T) (*Provider, *http.ServeMux) {
	t.Helper()
	cfg := config.DefaultConfig()
	lat := latency.NewCalculator(latency.Config{})
	p := New(cfg, lat)
	mux := http.NewServeMux()
	p.RegisterRoutes(mux)
	return p, mux
}

func withSimHeaders(mux http.Handler) http.Handler {
	return config.SimHeaderMiddleware(mux)
}

func doPost(t *testing.T, handler http.Handler, path string, body any, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest("POST", path, &buf)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	return rr
}

func TestChatCompletionsResponseShape(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"model":    "gpt-4",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected content-type application/json, got %q", ct)
	}

	var resp chatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.ID != "chatcmpl-abc123" {
		t.Errorf("unexpected id: %q", resp.ID)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("unexpected object: %q", resp.Object)
	}
	if resp.Model != "gpt-4" {
		t.Errorf("unexpected model: %q", resp.Model)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message == nil {
		t.Fatal("expected message in choice")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("unexpected role: %q", resp.Choices[0].Message.Role)
	}
	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("unexpected content: %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Error("expected finish_reason 'stop'")
	}
	if resp.Usage == nil {
		t.Fatal("expected usage in response")
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("expected prompt_tokens 10, got %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("expected completion_tokens 5, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("expected total_tokens 15, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatCompletionsTokensFromHeaders(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, map[string]string{
		"X-Sim-Input-Tokens":  "42",
		"X-Sim-Output-Tokens": "17",
	})

	var resp chatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.Usage.PromptTokens != 42 {
		t.Errorf("expected prompt_tokens 42, got %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 17 {
		t.Errorf("expected completion_tokens 17, got %d", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 59 {
		t.Errorf("expected total_tokens 59, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatCompletionsUsesDefaultModel(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	// no model in request body
	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	var resp chatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.Model != "gpt-4" {
		t.Errorf("expected default model gpt-4, got %q", resp.Model)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
		"messages":       []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected text/event-stream, got %q", ct)
	}

	// parse SSE lines
	scanner := bufio.NewScanner(rr.Body)
	var chunks []chatResponse
	var gotDone bool

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if line == "data: [DONE]" {
			gotDone = true
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			t.Errorf("unexpected line format: %q", line)
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var chunk chatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("failed to decode chunk: %v (data: %q)", err, data)
		}
		chunks = append(chunks, chunk)
	}

	if !gotDone {
		t.Error("missing data: [DONE] terminator")
	}

	// expect: role chunk, content chunk, finish chunk, usage chunk = 4
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d", len(chunks))
	}

	// role chunk
	if chunks[0].Choices[0].Delta == nil || chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Error("first chunk should have role 'assistant'")
	}

	// content chunk
	if chunks[1].Choices[0].Delta == nil || chunks[1].Choices[0].Delta.Content != "Hello" {
		t.Error("second chunk should have content 'Hello'")
	}

	// finish chunk
	if chunks[2].Choices[0].FinishReason == nil || *chunks[2].Choices[0].FinishReason != "stop" {
		t.Error("third chunk should have finish_reason 'stop'")
	}

	// usage chunk
	if chunks[3].Usage == nil {
		t.Fatal("fourth chunk should have usage")
	}
	if chunks[3].Usage.PromptTokens != 10 {
		t.Errorf("expected prompt_tokens 10 in usage chunk, got %d", chunks[3].Usage.PromptTokens)
	}
	if chunks[3].Usage.TotalTokens != 15 {
		t.Errorf("expected total_tokens 15 in usage chunk, got %d", chunks[3].Usage.TotalTokens)
	}
}

func TestChatCompletionsStreamingNoUsage(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	scanner := bufio.NewScanner(rr.Body)
	var chunks int
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") && line != "data: [DONE]" {
			chunks++
		}
	}

	// without include_usage: role, content, finish = 3 chunks (no usage chunk)
	if chunks != 3 {
		t.Errorf("expected 3 chunks without include_usage, got %d", chunks)
	}
}

func TestResponsesResponseShape(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model": "gpt-4",
		"input": "hi",
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	var resp responsesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if resp.ID != "resp-abc123" {
		t.Errorf("unexpected id: %q", resp.ID)
	}
	if resp.Object != "response" {
		t.Errorf("unexpected object: %q", resp.Object)
	}
	if len(resp.Output) != 1 {
		t.Fatalf("expected 1 output, got %d", len(resp.Output))
	}
	if resp.Output[0].Type != "message" {
		t.Errorf("unexpected output type: %q", resp.Output[0].Type)
	}
	if resp.Output[0].Role != "assistant" {
		t.Errorf("unexpected role: %q", resp.Output[0].Role)
	}
	if len(resp.Output[0].Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(resp.Output[0].Content))
	}
	if resp.Output[0].Content[0].Type != "output_text" {
		t.Errorf("unexpected content type: %q", resp.Output[0].Content[0].Type)
	}
	if resp.Output[0].Content[0].Text != "Hello" {
		t.Errorf("unexpected text: %q", resp.Output[0].Content[0].Text)
	}
}

func TestResponsesUsageFieldNames(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"input": "hi",
	}, map[string]string{
		"X-Sim-Input-Tokens":  "20",
		"X-Sim-Output-Tokens": "8",
	})

	// verify the JSON uses input_tokens/output_tokens, not prompt_tokens/completion_tokens
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw["usage"], &usage); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	if _, ok := usage["input_tokens"]; !ok {
		t.Error("responses usage should have input_tokens")
	}
	if _, ok := usage["output_tokens"]; !ok {
		t.Error("responses usage should have output_tokens")
	}
	if _, ok := usage["prompt_tokens"]; ok {
		t.Error("responses usage should NOT have prompt_tokens")
	}
	if _, ok := usage["completion_tokens"]; ok {
		t.Error("responses usage should NOT have completion_tokens")
	}

	var resp responsesResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal resp: %v", err)
	}
	if resp.Usage.InputTokens != 20 {
		t.Errorf("expected input_tokens 20, got %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 8 {
		t.Errorf("expected output_tokens 8, got %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 28 {
		t.Errorf("expected total_tokens 28, got %d", resp.Usage.TotalTokens)
	}
}

func TestCompletionsResponseShape(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/completions", map[string]any{
		"model":  "gpt-3.5-turbo-instruct",
		"prompt": "say hello",
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	var resp completionsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if resp.ID != "cmpl-abc123" {
		t.Errorf("unexpected id: %q", resp.ID)
	}
	if resp.Object != "text_completion" {
		t.Errorf("unexpected object: %q", resp.Object)
	}
	if resp.Model != "gpt-3.5-turbo-instruct" {
		t.Errorf("unexpected model: %q", resp.Model)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Text != "Hello" {
		t.Errorf("unexpected text: %q", resp.Choices[0].Text)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("unexpected finish_reason: %q", resp.Choices[0].FinishReason)
	}
}

func TestEmbeddingsResponseShape(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/embeddings", map[string]any{
		"model": "text-embedding-ada-002",
		"input": "hello",
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	var resp embeddingsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("unexpected object: %q", resp.Object)
	}
	if resp.Model != "text-embedding-ada-002" {
		t.Errorf("unexpected model: %q", resp.Model)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Data))
	}
	if resp.Data[0].Object != "embedding" {
		t.Errorf("unexpected embedding object: %q", resp.Data[0].Object)
	}
	if len(resp.Data[0].Embedding) != 2 {
		t.Errorf("expected 2 embedding values, got %d", len(resp.Data[0].Embedding))
	}
}

func TestEmbeddingsHasNoCompletionTokens(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/embeddings", map[string]any{
		"input": "hello",
	}, map[string]string{
		"X-Sim-Input-Tokens": "25",
	})

	// check raw JSON to ensure completion_tokens is absent
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw["usage"], &usage); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	if _, ok := usage["completion_tokens"]; ok {
		t.Error("embeddings usage should not have completion_tokens")
	}

	var resp embeddingsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal resp: %v", err)
	}

	if resp.Usage.PromptTokens != 25 {
		t.Errorf("expected prompt_tokens 25, got %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.TotalTokens != 25 {
		t.Errorf("expected total_tokens 25, got %d", resp.Usage.TotalTokens)
	}
}

func TestEmbeddingsIgnoresOutputTokenHeader(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/embeddings", map[string]any{
		"input": "hello",
	}, map[string]string{
		"X-Sim-Input-Tokens":  "30",
		"X-Sim-Output-Tokens": "99",
	})

	var resp embeddingsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// output tokens header should be ignored for embeddings
	if resp.Usage.TotalTokens != 30 {
		t.Errorf("expected total_tokens 30 (input only), got %d", resp.Usage.TotalTokens)
	}
}

func TestErrorResponse(t *testing.T) {
	endpoints := []string{
		"/v1/chat/completions",
		"/v1/responses",
		"/v1/completions",
		"/v1/embeddings",
	}

	for _, ep := range endpoints {
		t.Run(ep, func(t *testing.T) {
			_, mux := setup(t)
			handler := withSimHeaders(mux)

			rr := doPost(t, handler, ep, map[string]any{}, map[string]string{
				"X-Sim-Error": "429",
			})

			if rr.Code != 429 {
				t.Fatalf("expected 429, got %d", rr.Code)
			}
			if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
				t.Errorf("expected content-type application/json, got %q", ct)
			}

			var errResp apiError
			if err := json.Unmarshal(rr.Body.Bytes(), &errResp); err != nil {
				t.Fatalf("failed to decode error: %v", err)
			}
			if errResp.Error.Message != "simulated error" {
				t.Errorf("unexpected message: %q", errResp.Error.Message)
			}
			if errResp.Error.Type != "rate_limit_error" {
				t.Errorf("unexpected type: %q", errResp.Error.Type)
			}
			if errResp.Error.Code != "simulated" {
				t.Errorf("unexpected code: %q", errResp.Error.Code)
			}
		})
	}
}

func TestErrorTypeMapping(t *testing.T) {
	tests := []struct {
		code     string
		wantType string
	}{
		{"400", "invalid_request_error"},
		{"401", "authentication_error"},
		{"429", "rate_limit_error"},
		{"500", "server_error"},
		{"403", "invalid_request_error"}, // unmapped codes default to invalid_request_error
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			_, mux := setup(t)
			handler := withSimHeaders(mux)

			rr := doPost(t, handler, "/v1/chat/completions", map[string]any{}, map[string]string{
				"X-Sim-Error": tt.code,
			})

			var errResp apiError
			if err := json.Unmarshal(rr.Body.Bytes(), &errResp); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if errResp.Error.Type != tt.wantType {
				t.Errorf("code %s: expected type %q, got %q", tt.code, tt.wantType, errResp.Error.Type)
			}
		})
	}
}

func TestProviderName(t *testing.T) {
	p := New(config.DefaultConfig(), latency.NewCalculator(latency.Config{}))
	if p.Name() != "openai" {
		t.Errorf("expected name 'openai', got %q", p.Name())
	}
}

func TestChatCompletionsInvalidBody(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}

	var errResp apiError
	if err := json.Unmarshal(rr.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("unexpected error type: %q", errResp.Error.Type)
	}
}

func TestChatCompletionsPromptTokensDetails(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	// verify prompt_tokens_details and completion_tokens_details exist in raw JSON
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw["usage"], &usage); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	if _, ok := usage["prompt_tokens_details"]; !ok {
		t.Error("expected prompt_tokens_details in usage")
	}
	if _, ok := usage["completion_tokens_details"]; !ok {
		t.Error("expected completion_tokens_details in usage")
	}
}

func TestResponsesInputTokensDetails(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"input": "hi",
	}, nil)

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	var usage map[string]json.RawMessage
	if err := json.Unmarshal(raw["usage"], &usage); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	if _, ok := usage["input_tokens_details"]; !ok {
		t.Error("expected input_tokens_details in usage")
	}
	if _, ok := usage["output_tokens_details"]; !ok {
		t.Error("expected output_tokens_details in usage")
	}
}

func TestStreamingChunkObjectType(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	scanner := bufio.NewScanner(rr.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var chunk chatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("unmarshal chunk: %v", err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("expected object 'chat.completion.chunk', got %q", chunk.Object)
		}
	}
}

// verify all chunks use delta, not message
func TestStreamingChunksUseDelta(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/chat/completions", map[string]any{
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}, nil)

	body := rr.Body.Bytes()
	reader := bufio.NewReader(bytes.NewReader(body))
	for {
		line, err := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if err == io.EOF {
			break
		}
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &raw); err != nil {
			t.Fatalf("unmarshal raw: %v", err)
		}

		var choices []map[string]json.RawMessage
		if err := json.Unmarshal(raw["choices"], &choices); err != nil {
			t.Fatalf("unmarshal choices: %v", err)
		}
		for _, c := range choices {
			if _, ok := c["message"]; ok {
				t.Error("streaming chunks should use 'delta', not 'message'")
			}
		}
	}
}

// parseResponsesSSE parses "event: <type>\ndata: <json>" pairs from SSE output
func parseResponsesSSE(t *testing.T, body []byte) []responsesStreamEvent {
	t.Helper()
	var events []responsesStreamEvent
	scanner := bufio.NewScanner(bytes.NewReader(body))
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var ev responsesStreamEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				t.Fatalf("failed to decode event %q: %v (data: %q)", currentEvent, err, data)
			}
			events = append(events, ev)
			currentEvent = ""
		}
	}
	return events
}

func TestResponsesStreamingEventSequence(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model":  "gpt-4",
		"input":  "hi",
		"stream": true,
	}, nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected text/event-stream, got %q", ct)
	}

	events := parseResponsesSSE(t, rr.Body.Bytes())

	wantTypes := []string{
		"response.created",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}

	if len(events) != len(wantTypes) {
		t.Fatalf("expected %d events, got %d", len(wantTypes), len(events))
	}

	for i, want := range wantTypes {
		if events[i].Type != want {
			t.Errorf("event %d: expected type %q, got %q", i, want, events[i].Type)
		}
	}
}

func TestResponsesStreamingEventTypes(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model":  "gpt-4",
		"input":  "hi",
		"stream": true,
	}, nil)

	// verify event: lines match data type fields
	scanner := bufio.NewScanner(rr.Body)
	var currentEvent string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") && currentEvent != "" {
			data := strings.TrimPrefix(line, "data: ")
			var ev responsesStreamEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				t.Fatalf("unmarshal event: %v", err)
			}
			if ev.Type != currentEvent {
				t.Errorf("event line %q does not match data type %q", currentEvent, ev.Type)
			}
			currentEvent = ""
		}
	}
}

func TestResponsesStreamingUsage(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model":  "gpt-4",
		"input":  "hi",
		"stream": true,
	}, map[string]string{
		"X-Sim-Input-Tokens":  "20",
		"X-Sim-Output-Tokens": "8",
	})

	events := parseResponsesSSE(t, rr.Body.Bytes())

	// usage should only appear in the final response.completed event
	last := events[len(events)-1]
	if last.Type != "response.completed" {
		t.Fatalf("last event should be response.completed, got %q", last.Type)
	}
	if last.Response == nil || last.Response.Usage == nil {
		t.Fatal("response.completed should have usage")
	}

	usage := last.Response.Usage
	if usage.InputTokens != 20 {
		t.Errorf("expected input_tokens 20, got %d", usage.InputTokens)
	}
	if usage.OutputTokens != 8 {
		t.Errorf("expected output_tokens 8, got %d", usage.OutputTokens)
	}
	if usage.TotalTokens != 28 {
		t.Errorf("expected total_tokens 28, got %d", usage.TotalTokens)
	}

	// non-final events should not have usage
	for i, ev := range events[:len(events)-1] {
		if ev.Response != nil && ev.Response.Usage != nil {
			t.Errorf("event %d (%s) should not have usage", i, ev.Type)
		}
	}
}

func TestResponsesStreamingUsesInputOutputNaming(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"input":  "hi",
		"stream": true,
	}, nil)

	events := parseResponsesSSE(t, rr.Body.Bytes())
	last := events[len(events)-1]

	// check raw JSON of the completed event for field naming
	scanner := bufio.NewScanner(bytes.NewReader(rr.Body.Bytes()))
	var completedData string
	var foundCompleted bool
	for scanner.Scan() {
		line := scanner.Text()
		if line == "event: response.completed" {
			foundCompleted = true
			continue
		}
		if foundCompleted && strings.HasPrefix(line, "data: ") {
			completedData = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	if completedData == "" {
		t.Fatal("could not find response.completed data")
	}

	// parse the response object inside the event
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(completedData), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(raw["response"], &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	var usageRaw map[string]json.RawMessage
	if err := json.Unmarshal(resp["usage"], &usageRaw); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}

	if _, ok := usageRaw["input_tokens"]; !ok {
		t.Error("streaming usage should have input_tokens")
	}
	if _, ok := usageRaw["output_tokens"]; !ok {
		t.Error("streaming usage should have output_tokens")
	}
	if _, ok := usageRaw["prompt_tokens"]; ok {
		t.Error("streaming usage should NOT have prompt_tokens")
	}
	if _, ok := usageRaw["completion_tokens"]; ok {
		t.Error("streaming usage should NOT have completion_tokens")
	}

	_ = last // used above via events
}

func TestResponsesStreamingNoDoneTerminator(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"input":  "hi",
		"stream": true,
	}, nil)

	if strings.Contains(rr.Body.String(), "data: [DONE]") {
		t.Error("responses streaming should not have data: [DONE] terminator")
	}
}

func TestResponsesStreamingDelta(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model":  "gpt-4",
		"input":  "hi",
		"stream": true,
	}, nil)

	events := parseResponsesSSE(t, rr.Body.Bytes())

	// find the delta event
	var found bool
	for _, ev := range events {
		if ev.Type == "response.output_text.delta" {
			found = true
			if ev.Delta != "Hello" {
				t.Errorf("expected delta 'Hello', got %q", ev.Delta)
			}
		}
	}
	if !found {
		t.Error("expected response.output_text.delta event")
	}
}

func TestResponsesStreamingCreatedStatus(t *testing.T) {
	_, mux := setup(t)
	handler := withSimHeaders(mux)

	rr := doPost(t, handler, "/v1/responses", map[string]any{
		"model":  "gpt-4",
		"input":  "hi",
		"stream": true,
	}, nil)

	events := parseResponsesSSE(t, rr.Body.Bytes())

	// response.created should have in_progress status
	if events[0].Response == nil {
		t.Fatal("response.created should have response object")
	}
	if events[0].Response.Status != "in_progress" {
		t.Errorf("response.created status should be in_progress, got %q", events[0].Response.Status)
	}

	// response.completed should have completed status
	last := events[len(events)-1]
	if last.Response == nil {
		t.Fatal("response.completed should have response object")
	}
	if last.Response.Status != "completed" {
		t.Errorf("response.completed status should be completed, got %q", last.Response.Status)
	}
}
