package gemini

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
)

type Provider struct {
	cfg     *config.Config
	latency *latency.Calculator
}

func New(cfg *config.Config, lat *latency.Calculator) *Provider {
	return &Provider{cfg: cfg, latency: lat}
}

func (p *Provider) Name() string { return "gemini" }

const routePrefix = "/v1beta/models/"

func (p *Provider) RegisterRoutes(mux *http.ServeMux) {
	// prefix match -- Go's mux {path...} wildcard doesn't handle colons in
	// segments (e.g. "gemini-2.5-flash:generateContent"), so we match the
	// prefix and parse model:action from the raw path.
	mux.HandleFunc("POST "+routePrefix, p.handle)
}

func (p *Provider) handle(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	suffix := strings.TrimPrefix(r.URL.Path, routePrefix)
	model, action := parseModelAction(suffix)
	if model == "" {
		model = p.cfg.DefaultModel
	}

	if code := config.ErrorCodeFromContext(r.Context()); code != 0 {
		writeError(w, code)
		return
	}

	switch action {
	case "generateContent":
		p.handleGenerate(w, r, model, false)
	case "streamGenerateContent":
		p.handleGenerate(w, r, model, true)
	default:
		writeError(w, http.StatusBadRequest)
	}
}

func (p *Provider) handleGenerate(w http.ResponseWriter, r *http.Request, model string, stream bool) {
	// alt=sse also triggers streaming
	if r.URL.Query().Get("alt") == "sse" {
		stream = true
	}

	inputTokens := config.InputTokensFromContext(r.Context(), p.cfg.DefaultInputTokens)
	outputTokens := config.OutputTokensFromContext(r.Context(), p.cfg.DefaultOutputTokens)
	content := p.cfg.ResponseContent

	if stream {
		p.writeStream(w, model, content, inputTokens, outputTokens)
		return
	}

	// non-streaming: apply ttft as thinking time
	time.Sleep(p.latency.TTFT())

	resp := generateResponse(model, content, inputTokens, outputTokens)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}

func (p *Provider) writeStream(w http.ResponseWriter, model, content string, inputTokens, outputTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError)
		return
	}

	// split content roughly in half for two chunks
	mid := len(content) / 2
	if mid == 0 {
		mid = len(content)
	}
	parts := []string{content[:mid]}
	if mid < len(content) {
		parts = append(parts, content[mid:])
	}

	// ttft delay before first chunk
	time.Sleep(p.latency.TTFT())

	// intermediate chunks: partial text, no usage, no finish reason
	for i := 0; i < len(parts)-1; i++ {
		chunk := streamChunk{
			Candidates: []candidate{{
				Content: contentBlock{
					Parts: []part{{Text: parts[i]}},
					Role:  "model",
				},
				Index: 0,
			}},
			ModelVersion: model,
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		// itl delay between chunks
		time.Sleep(p.latency.ITL())
	}

	// final chunk: last text part, finish reason, safety ratings, usage
	lastPart := parts[len(parts)-1]
	final := streamChunk{
		Candidates: []candidate{{
			Content: contentBlock{
				Parts: []part{{Text: lastPart}},
				Role:  "model",
			},
			Index:         0,
			FinishReason:  strPtr("STOP"),
			SafetyRatings: defaultSafetyRatings(),
		}},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:        inputTokens,
			CandidatesTokenCount:    outputTokens,
			TotalTokenCount:         inputTokens + outputTokens,
			CachedContentTokenCount: 0,
		},
		ModelVersion: model,
	}
	data, _ := json.Marshal(final)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// parseModelAction splits "gemini-2.5-flash:generateContent" into model and action.
func parseModelAction(path string) (string, string) {
	// path is everything after /v1beta/models/, e.g. "gemini-2.5-flash:generateContent"
	idx := strings.LastIndex(path, ":")
	if idx < 0 {
		return path, ""
	}
	return path[:idx], path[idx+1:]
}

func generateResponse(model, content string, inputTokens, outputTokens int) response {
	return response{
		Candidates: []candidate{{
			Content: contentBlock{
				Parts: []part{{Text: content}},
				Role:  "model",
			},
			FinishReason:  strPtr("STOP"),
			Index:         0,
			SafetyRatings: defaultSafetyRatings(),
		}},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:        inputTokens,
			CandidatesTokenCount:    outputTokens,
			TotalTokenCount:         inputTokens + outputTokens,
			CachedContentTokenCount: 0,
		},
		ModelVersion: model,
	}
}

func defaultSafetyRatings() []safetyRating {
	return []safetyRating{
		{Category: "HARM_CATEGORY_HARASSMENT", Probability: "NEGLIGIBLE"},
	}
}

func strPtr(s string) *string { return &s }

// error handling

var statusToGRPC = map[int]string{
	400: "INVALID_ARGUMENT",
	401: "UNAUTHENTICATED",
	403: "PERMISSION_DENIED",
	404: "NOT_FOUND",
	429: "RESOURCE_EXHAUSTED",
	500: "INTERNAL",
	503: "UNAVAILABLE",
}

func writeError(w http.ResponseWriter, code int) {
	status, ok := statusToGRPC[code]
	if !ok {
		status = "INTERNAL"
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	//nolint:errcheck // best-effort error response
	json.NewEncoder(w).Encode(errorResponse{
		Error: errorDetail{
			Code:    code,
			Message: "simulated error",
			Status:  status,
		},
	})
}

// types

type response struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string         `json:"modelVersion"`
}

type streamChunk struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string         `json:"modelVersion"`
}

type candidate struct {
	Content       contentBlock   `json:"content"`
	FinishReason  *string        `json:"finishReason,omitempty"`
	Index         int            `json:"index"`
	SafetyRatings []safetyRating `json:"safetyRatings,omitempty"`
}

type contentBlock struct {
	Parts []part `json:"parts"`
	Role  string `json:"role"`
}

type part struct {
	Text string `json:"text"`
}

type safetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}
