package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
)

// Provider implements the Anthropic Messages API shape.
type Provider struct {
	cfg     *config.Config
	latency *latency.Calculator
}

func New(cfg *config.Config, lat *latency.Calculator) *Provider {
	return &Provider{cfg: cfg, latency: lat}
}

func (p *Provider) Name() string { return "anthropic" }

func (p *Provider) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/messages", p.handleMessages)
}

// response types

type usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type messageResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []contentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        usage          `json:"usage"`
}

// error types

type apiErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type apiError struct {
	Type  string         `json:"type"`
	Error apiErrorDetail `json:"error"`
}

// streaming types

type streamMessageStart struct {
	Type    string          `json:"type"`
	Message messageResponse `json:"message"`
}

type streamContentBlockStart struct {
	Type         string       `json:"type"`
	Index        int          `json:"index"`
	ContentBlock contentBlock `json:"content_block"`
}

type streamPing struct {
	Type string `json:"type"`
}

type textDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type streamContentBlockDelta struct {
	Type  string    `json:"type"`
	Index int       `json:"index"`
	Delta textDelta `json:"delta"`
}

type streamContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type messageDeltaBody struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type messageDeltaUsage struct {
	OutputTokens int `json:"output_tokens"`
}

type streamMessageDelta struct {
	Type  string            `json:"type"`
	Delta messageDeltaBody  `json:"delta"`
	Usage messageDeltaUsage `json:"usage"`
}

type streamMessageStop struct {
	Type string `json:"type"`
}

func (p *Provider) handleMessages(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	ctx := r.Context()

	// check for simulated error
	if code := config.ErrorCodeFromContext(ctx); code != 0 {
		writeError(w, code)
		return
	}

	// decode request body for stream and model fields
	var req struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest)
		return
	}

	inputTokens := config.InputTokensFromContext(ctx, p.cfg.DefaultInputTokens)
	outputTokens := config.OutputTokensFromContext(ctx, p.cfg.DefaultOutputTokens)
	model := p.cfg.DefaultModel
	if req.Model != "" {
		model = req.Model
	}
	content := p.cfg.ResponseContent

	if req.Stream {
		p.handleStream(w, model, content, inputTokens, outputTokens)
		return
	}

	// non-streaming: apply ttft as thinking time
	time.Sleep(p.latency.TTFT())

	resp := messageResponse{
		ID:           "msg_01XFDUDYJgAACzvnptvVoYEL",
		Type:         "message",
		Role:         "assistant",
		Content:      []contentBlock{{Type: "text", Text: content}},
		Model:        model,
		StopReason:   "end_turn",
		StopSequence: nil,
		Usage: usage{
			InputTokens:              inputTokens,
			OutputTokens:             outputTokens,
			CacheCreationInputTokens: 0,
			CacheReadInputTokens:     0,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}

func (p *Provider) handleStream(w http.ResponseWriter, model, content string, inputTokens, outputTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError)
		return
	}

	// ttft delay before first chunk
	time.Sleep(p.latency.TTFT())

	// message_start
	writeSSE(w, "message_start", streamMessageStart{
		Type: "message_start",
		Message: messageResponse{
			ID:           "msg_01XFDUDYJgAACzvnptvVoYEL",
			Type:         "message",
			Role:         "assistant",
			Content:      []contentBlock{},
			Model:        model,
			StopReason:   "",
			StopSequence: nil,
			Usage: usage{
				InputTokens:              inputTokens,
				OutputTokens:             0,
				CacheCreationInputTokens: 0,
				CacheReadInputTokens:     0,
			},
		},
	})
	flusher.Flush()

	// itl delay
	time.Sleep(p.latency.ITL())

	// content_block_start
	writeSSE(w, "content_block_start", streamContentBlockStart{
		Type:         "content_block_start",
		Index:        0,
		ContentBlock: contentBlock{Type: "text", Text: ""},
	})
	flusher.Flush()

	// ping
	writeSSE(w, "ping", streamPing{Type: "ping"})
	flusher.Flush()

	// itl delay
	time.Sleep(p.latency.ITL())

	// content_block_delta
	writeSSE(w, "content_block_delta", streamContentBlockDelta{
		Type:  "content_block_delta",
		Index: 0,
		Delta: textDelta{Type: "text_delta", Text: content},
	})
	flusher.Flush()

	// itl delay
	time.Sleep(p.latency.ITL())

	// content_block_stop
	writeSSE(w, "content_block_stop", streamContentBlockStop{
		Type:  "content_block_stop",
		Index: 0,
	})
	flusher.Flush()

	// itl delay
	time.Sleep(p.latency.ITL())

	// message_delta
	writeSSE(w, "message_delta", streamMessageDelta{
		Type: "message_delta",
		Delta: messageDeltaBody{
			StopReason:   "end_turn",
			StopSequence: nil,
		},
		Usage: messageDeltaUsage{
			OutputTokens: outputTokens,
		},
	})
	flusher.Flush()

	// message_stop
	writeSSE(w, "message_stop", streamMessageStop{Type: "message_stop"})
	flusher.Flush()
}

func writeSSE(w http.ResponseWriter, event string, data any) {
	b, _ := json.Marshal(data)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
}

func writeError(w http.ResponseWriter, code int) {
	errType := errorTypeForStatus(code)
	resp := apiError{
		Type: "error",
		Error: apiErrorDetail{
			Type:    errType,
			Message: "simulated error",
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	//nolint:errcheck // best-effort error response
	json.NewEncoder(w).Encode(resp)
}

func errorTypeForStatus(code int) string {
	switch code {
	case 400:
		return "invalid_request_error"
	case 401:
		return "authentication_error"
	case 429:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}
