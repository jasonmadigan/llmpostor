package gemini

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
	"github.com/jasonmadigan/llmpostor/internal/provider"
)

func testHandler(t *testing.T, cfg *config.Config) http.Handler {
	t.Helper()
	if cfg == nil {
		cfg = config.DefaultConfig()
	}
	p := New(cfg, latency.NewCalculator(latency.Config{}))
	reg := provider.NewRegistry()
	reg.Register(p)
	mux := http.NewServeMux()
	if err := reg.RegisterRoutes(mux, []string{"gemini"}); err != nil {
		t.Fatal(err)
	}
	return config.SimHeaderMiddleware(mux)
}

func TestResponseShape(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.UsageMetadata == nil {
		t.Fatal("usageMetadata must be present")
	}

	// verify field is usageMetadata, not usage
	raw := make(map[string]json.RawMessage)
	if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}
	if _, ok := raw["usageMetadata"]; !ok {
		t.Error("response must use 'usageMetadata' key, not 'usage'")
	}
	if _, ok := raw["usage"]; ok {
		t.Error("response must not contain 'usage' key")
	}
}

func TestCamelCaseFieldNames(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	body := w.Body.String()
	for _, field := range []string{"promptTokenCount", "candidatesTokenCount", "totalTokenCount"} {
		if !strings.Contains(body, field) {
			t.Errorf("expected camelCase field %q in response", field)
		}
	}
	// must not contain snake_case variants
	for _, field := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if strings.Contains(body, field) {
			t.Errorf("response must not contain snake_case field %q", field)
		}
	}
}

func TestModelFromPath(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.0-flash-lite:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.ModelVersion != "gemini-2.0-flash-lite" {
		t.Errorf("expected model gemini-2.0-flash-lite, got %q", resp.ModelVersion)
	}
}

func TestTokenCountsFromHeaders(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	req.Header.Set("X-Sim-Input-Tokens", "42")
	req.Header.Set("X-Sim-Output-Tokens", "17")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	u := resp.UsageMetadata
	if u.PromptTokenCount != 42 {
		t.Errorf("promptTokenCount: got %d, want 42", u.PromptTokenCount)
	}
	if u.CandidatesTokenCount != 17 {
		t.Errorf("candidatesTokenCount: got %d, want 17", u.CandidatesTokenCount)
	}
	if u.TotalTokenCount != 59 {
		t.Errorf("totalTokenCount: got %d, want 59", u.TotalTokenCount)
	}
}

func TestTokenCountsFromConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.DefaultInputTokens = 100
	cfg.DefaultOutputTokens = 50
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	u := resp.UsageMetadata
	if u.PromptTokenCount != 100 || u.CandidatesTokenCount != 50 || u.TotalTokenCount != 150 {
		t.Errorf("token counts from config: got %d/%d/%d, want 100/50/150",
			u.PromptTokenCount, u.CandidatesTokenCount, u.TotalTokenCount)
	}
}

func TestResponseContent(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ResponseContent = "test response"
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		t.Fatal("no candidates or parts")
	}
	if resp.Candidates[0].Content.Parts[0].Text != "test response" {
		t.Errorf("got %q, want 'test response'", resp.Candidates[0].Content.Parts[0].Text)
	}
}

func TestCandidateFields(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp response
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	c := resp.Candidates[0]
	if c.Content.Role != "model" {
		t.Errorf("role: got %q, want 'model'", c.Content.Role)
	}
	if c.FinishReason == nil || *c.FinishReason != "STOP" {
		t.Error("finishReason should be STOP")
	}
	if len(c.SafetyRatings) == 0 {
		t.Error("safetyRatings should be present")
	}
}

func TestErrorResponseShape(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	req.Header.Set("X-Sim-Error", "429")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 429 {
		t.Fatalf("expected 429, got %d", w.Code)
	}

	var errResp errorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if errResp.Error.Code != 429 {
		t.Errorf("error code: got %d, want 429", errResp.Error.Code)
	}
	if errResp.Error.Status != "RESOURCE_EXHAUSTED" {
		t.Errorf("status: got %q, want RESOURCE_EXHAUSTED", errResp.Error.Status)
	}

	// must not match openai or anthropic error shapes
	raw := make(map[string]json.RawMessage)
	if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}
	if _, ok := raw["type"]; ok {
		t.Error("gemini errors must not have 'type' (that is anthropic)")
	}
}

func TestErrorStatusMapping(t *testing.T) {
	tests := []struct {
		code   string
		status string
	}{
		{"400", "INVALID_ARGUMENT"},
		{"401", "UNAUTHENTICATED"},
		{"429", "RESOURCE_EXHAUSTED"},
		{"500", "INTERNAL"},
		{"503", "UNAVAILABLE"},
	}

	h := testHandler(t, nil)
	for _, tt := range tests {
		req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
		req.Header.Set("X-Sim-Error", tt.code)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)

		var errResp errorResponse
		if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if errResp.Error.Status != tt.status {
			t.Errorf("code %s: got status %q, want %q", tt.code, errResp.Error.Status, tt.status)
		}
	}
}

func TestStreamGenerateContentRoute(t *testing.T) {
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type: got %q, want text/event-stream", ct)
	}
}

func TestStreamUsageOnlyInFinalChunk(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ResponseContent = "Hello world"
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	chunks := parseSSEChunks(t, w.Body.String())
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}

	// intermediate chunks must not have usageMetadata
	for i := 0; i < len(chunks)-1; i++ {
		raw := make(map[string]json.RawMessage)
		if err := json.Unmarshal(chunks[i], &raw); err != nil {
			t.Fatalf("unmarshal chunk %d: %v", i, err)
		}
		if _, ok := raw["usageMetadata"]; ok {
			t.Errorf("chunk %d: intermediate chunk must not have usageMetadata", i)
		}
	}

	// final chunk must have usageMetadata
	lastRaw := make(map[string]json.RawMessage)
	if err := json.Unmarshal(chunks[len(chunks)-1], &lastRaw); err != nil {
		t.Fatalf("unmarshal last raw: %v", err)
	}
	if _, ok := lastRaw["usageMetadata"]; !ok {
		t.Error("final chunk must have usageMetadata")
	}

	// final chunk must have finishReason
	var last streamChunk
	if err := json.Unmarshal(chunks[len(chunks)-1], &last); err != nil {
		t.Fatalf("unmarshal last: %v", err)
	}
	if len(last.Candidates) == 0 || last.Candidates[0].FinishReason == nil {
		t.Error("final chunk must have finishReason")
	}
}

func TestStreamTokenCountsFromHeaders(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ResponseContent = "Hello world"
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", strings.NewReader("{}"))
	req.Header.Set("X-Sim-Input-Tokens", "20")
	req.Header.Set("X-Sim-Output-Tokens", "30")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	chunks := parseSSEChunks(t, w.Body.String())
	var last streamChunk
	if err := json.Unmarshal(chunks[len(chunks)-1], &last); err != nil {
		t.Fatalf("unmarshal last: %v", err)
	}
	u := last.UsageMetadata
	if u == nil {
		t.Fatal("final chunk missing usageMetadata")
	}
	if u.PromptTokenCount != 20 || u.CandidatesTokenCount != 30 || u.TotalTokenCount != 50 {
		t.Errorf("stream tokens: got %d/%d/%d, want 20/30/50",
			u.PromptTokenCount, u.CandidatesTokenCount, u.TotalTokenCount)
	}
}

func TestStreamModelVersion(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ResponseContent = "Hello world"
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	chunks := parseSSEChunks(t, w.Body.String())
	for i, data := range chunks {
		var chunk streamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			t.Fatalf("unmarshal chunk %d: %v", i, err)
		}
		if chunk.ModelVersion != "gemini-2.5-flash" {
			t.Errorf("chunk %d: modelVersion got %q, want gemini-2.5-flash", i, chunk.ModelVersion)
		}
	}
}

func TestStreamReconstructsFullContent(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ResponseContent = "Hello world"
	h := testHandler(t, cfg)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	chunks := parseSSEChunks(t, w.Body.String())
	var combined string
	for i, data := range chunks {
		var chunk streamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			t.Fatalf("unmarshal chunk %d: %v", i, err)
		}
		if len(chunk.Candidates) > 0 && len(chunk.Candidates[0].Content.Parts) > 0 {
			combined += chunk.Candidates[0].Content.Parts[0].Text
		}
	}
	if combined != "Hello world" {
		t.Errorf("reconstructed content: got %q, want 'Hello world'", combined)
	}
}

func TestGenerateContentRouteWithoutSSE(t *testing.T) {
	// generateContent without alt=sse should be non-streaming
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("content-type: got %q, want application/json", ct)
	}
}

func TestStreamGenerateContentRouteIsStreaming(t *testing.T) {
	// streamGenerateContent is always streaming, even without alt=sse
	h := testHandler(t, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("streamGenerateContent should stream: content-type got %q", ct)
	}
}

func TestProviderName(t *testing.T) {
	p := New(config.DefaultConfig(), latency.NewCalculator(latency.Config{}))
	if p.Name() != "gemini" {
		t.Errorf("name: got %q, want 'gemini'", p.Name())
	}
}

func parseSSEChunks(t *testing.T, body string) [][]byte {
	t.Helper()
	var chunks [][]byte
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			chunks = append(chunks, []byte(strings.TrimPrefix(line, "data: ")))
		}
	}
	return chunks
}
