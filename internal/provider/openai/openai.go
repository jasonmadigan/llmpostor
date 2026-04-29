package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
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

func (p *Provider) Name() string { return "openai" }

func (p *Provider) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", p.handleChatCompletions)
	mux.HandleFunc("POST /v1/responses", p.handleResponses)
	mux.HandleFunc("POST /v1/completions", p.handleCompletions)
	mux.HandleFunc("POST /v1/embeddings", p.handleEmbeddings)
}

// error shapes

type apiError struct {
	Error apiErrorBody `json:"error"`
}

type apiErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func errorTypeForStatus(code int) string {
	switch code {
	case 401:
		return "authentication_error"
	case 429:
		return "rate_limit_error"
	case 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

// writeSimError writes an openai-shaped error and returns true if X-Sim-Error was set.
func writeSimError(w http.ResponseWriter, r *http.Request) bool {
	code := config.ErrorCodeFromContext(r.Context())
	if code == 0 {
		return false
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	//nolint:errcheck // best-effort error response
	json.NewEncoder(w).Encode(apiError{
		Error: apiErrorBody{
			Message: "simulated error",
			Type:    errorTypeForStatus(code),
			Code:    "simulated",
		},
	})
	return true
}

// chat/completions types

type chatRequest struct {
	Stream        bool          `json:"stream"`
	StreamOptions *streamOpts   `json:"stream_options,omitempty"`
	Model         string        `json:"model,omitempty"`
}

type streamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage,omitempty"`
}

type chatChoice struct {
	Index        int          `json:"index"`
	Message      *chatMsg     `json:"message,omitempty"`
	Delta        *chatMsg     `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatUsage struct {
	PromptTokens            int                    `json:"prompt_tokens"`
	CompletionTokens        int                    `json:"completion_tokens"`
	TotalTokens             int                    `json:"total_tokens"`
	PromptTokensDetails     promptTokensDetails    `json:"prompt_tokens_details"`
	CompletionTokensDetails completionTokenDetails `json:"completion_tokens_details"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens"`
}

type completionTokenDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens"`
	AudioTokens              int `json:"audio_tokens"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
}

func (p *Provider) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	if writeSimError(w, r) {
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		//nolint:errcheck // best-effort error response
		json.NewEncoder(w).Encode(apiError{
			Error: apiErrorBody{
				Message: "invalid request body",
				Type:    "invalid_request_error",
				Code:    "invalid_json",
			},
		})
		return
	}

	ctx := r.Context()
	inputTokens := config.InputTokensFromContext(ctx, p.cfg.DefaultInputTokens)
	outputTokens := config.OutputTokensFromContext(ctx, p.cfg.DefaultOutputTokens)
	model := p.cfg.DefaultModel
	if req.Model != "" {
		model = req.Model
	}
	content := p.cfg.ResponseContent

	if req.Stream {
		p.handleChatStream(w, model, content, inputTokens, outputTokens, req.StreamOptions)
		return
	}

	// non-streaming: apply ttft as thinking time
	time.Sleep(p.latency.TTFT())

	stop := "stop"
	resp := chatResponse{
		ID:     "chatcmpl-abc123",
		Object: "chat.completion",
		Model:  model,
		Choices: []chatChoice{
			{
				Index:        0,
				Message:      &chatMsg{Role: "assistant", Content: content},
				FinishReason: &stop,
			},
		},
		Usage: &chatUsage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}

func (p *Provider) handleChatStream(w http.ResponseWriter, model, content string, inputTokens, outputTokens int, opts *streamOpts) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// ttft delay before first chunk
	time.Sleep(p.latency.TTFT())

	// role chunk
	roleChunk := chatResponse{
		ID:     "chatcmpl-abc123",
		Object: "chat.completion.chunk",
		Model:  model,
		Choices: []chatChoice{
			{
				Index:        0,
				Delta:        &chatMsg{Role: "assistant", Content: ""},
				FinishReason: nil,
			},
		},
	}
	writeSSEChunk(w, roleChunk)
	flusher.Flush()

	// itl delay between chunks
	time.Sleep(p.latency.ITL())

	// content chunk
	contentChunk := chatResponse{
		ID:     "chatcmpl-abc123",
		Object: "chat.completion.chunk",
		Model:  model,
		Choices: []chatChoice{
			{
				Index:        0,
				Delta:        &chatMsg{Role: "", Content: content},
				FinishReason: nil,
			},
		},
	}
	writeSSEChunk(w, contentChunk)
	flusher.Flush()

	// itl delay before finish
	time.Sleep(p.latency.ITL())

	// finish chunk
	stop := "stop"
	finishChunk := chatResponse{
		ID:     "chatcmpl-abc123",
		Object: "chat.completion.chunk",
		Model:  model,
		Choices: []chatChoice{
			{
				Index:        0,
				Delta:        &chatMsg{},
				FinishReason: &stop,
			},
		},
	}
	writeSSEChunk(w, finishChunk)
	flusher.Flush()

	// usage chunk if requested
	if opts != nil && opts.IncludeUsage {
		usageChunk := chatResponse{
			ID:      "chatcmpl-abc123",
			Object:  "chat.completion.chunk",
			Model:   model,
			Choices: []chatChoice{},
			Usage: &chatUsage{
				PromptTokens:     inputTokens,
				CompletionTokens: outputTokens,
				TotalTokens:      inputTokens + outputTokens,
			},
		}
		writeSSEChunk(w, usageChunk)
		flusher.Flush()
	}

	_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeSSEChunk(w http.ResponseWriter, v any) {
	data, _ := json.Marshal(v)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}

// responses API types

type responsesRequest struct {
	Stream bool   `json:"stream"`
	Model  string `json:"model,omitempty"`
}

type responsesResponse struct {
	ID     string            `json:"id"`
	Object string            `json:"object"`
	Status string            `json:"status,omitempty"`
	Model  string            `json:"model"`
	Output []responsesOutput `json:"output"`
	Usage  *responsesUsage   `json:"usage"`
}

type responsesOutput struct {
	Type    string             `json:"type"`
	Role    string             `json:"role"`
	Status  string             `json:"status,omitempty"`
	Content []responsesContent `json:"content"`
}

type responsesContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responsesUsage struct {
	InputTokens         int                    `json:"input_tokens"`
	InputTokensDetails  responsesInputDetails  `json:"input_tokens_details"`
	OutputTokens        int                    `json:"output_tokens"`
	OutputTokensDetails responsesOutputDetails `json:"output_tokens_details"`
	TotalTokens         int                    `json:"total_tokens"`
}

type responsesInputDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type responsesOutputDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// streaming event types for the responses API

type responsesStreamEvent struct {
	Type         string           `json:"type"`
	Response     *responsesResponse `json:"response,omitempty"`
	OutputIndex  *int             `json:"output_index,omitempty"`
	ContentIndex *int             `json:"content_index,omitempty"`
	Item         *responsesOutput `json:"item,omitempty"`
	Part         *responsesContent `json:"part,omitempty"`
	Delta        string           `json:"delta,omitempty"`
}

func (p *Provider) handleResponses(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	if writeSimError(w, r) {
		return
	}

	time.Sleep(p.latency.TTFT())

	var req responsesRequest
	_ = json.NewDecoder(r.Body).Decode(&req)

	ctx := r.Context()
	inputTokens := config.InputTokensFromContext(ctx, p.cfg.DefaultInputTokens)
	outputTokens := config.OutputTokensFromContext(ctx, p.cfg.DefaultOutputTokens)

	model := p.cfg.DefaultModel
	if req.Model != "" {
		model = req.Model
	}

	if req.Stream {
		p.handleResponsesStream(w, model, p.cfg.ResponseContent, inputTokens, outputTokens)
		return
	}

	resp := responsesResponse{
		ID:     "resp-abc123",
		Object: "response",
		Model:  model,
		Output: []responsesOutput{
			{
				Type: "message",
				Role: "assistant",
				Content: []responsesContent{
					{Type: "output_text", Text: p.cfg.ResponseContent},
				},
			},
		},
		Usage: &responsesUsage{
			InputTokens:         inputTokens,
			InputTokensDetails:  responsesInputDetails{},
			OutputTokens:        outputTokens,
			OutputTokensDetails: responsesOutputDetails{},
			TotalTokens:         inputTokens + outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}

func (p *Provider) handleResponsesStream(w http.ResponseWriter, model, content string, inputTokens, outputTokens int) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	idx0 := 0

	// response.created
	writeResponsesSSE(w, "response.created", responsesStreamEvent{
		Type: "response.created",
		Response: &responsesResponse{
			ID:     "resp-abc123",
			Object: "response",
			Status: "in_progress",
			Model:  model,
			Output: []responsesOutput{},
			Usage:  nil,
		},
	})
	flusher.Flush()

	// response.output_item.added
	writeResponsesSSE(w, "response.output_item.added", responsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: &idx0,
		Item: &responsesOutput{
			Type:    "message",
			Role:    "assistant",
			Status:  "in_progress",
			Content: []responsesContent{},
		},
	})
	flusher.Flush()

	// response.content_part.added
	writeResponsesSSE(w, "response.content_part.added", responsesStreamEvent{
		Type:         "response.content_part.added",
		OutputIndex:  &idx0,
		ContentIndex: &idx0,
		Part:         &responsesContent{Type: "output_text", Text: ""},
	})
	flusher.Flush()

	// response.output_text.delta
	writeResponsesSSE(w, "response.output_text.delta", responsesStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  &idx0,
		ContentIndex: &idx0,
		Delta:        content,
	})
	flusher.Flush()

	// response.content_part.done
	writeResponsesSSE(w, "response.content_part.done", responsesStreamEvent{
		Type:         "response.content_part.done",
		OutputIndex:  &idx0,
		ContentIndex: &idx0,
		Part:         &responsesContent{Type: "output_text", Text: content},
	})
	flusher.Flush()

	// response.output_item.done
	writeResponsesSSE(w, "response.output_item.done", responsesStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: &idx0,
		Item: &responsesOutput{
			Type:   "message",
			Role:   "assistant",
			Status: "completed",
			Content: []responsesContent{
				{Type: "output_text", Text: content},
			},
		},
	})
	flusher.Flush()

	// response.completed
	writeResponsesSSE(w, "response.completed", responsesStreamEvent{
		Type: "response.completed",
		Response: &responsesResponse{
			ID:     "resp-abc123",
			Object: "response",
			Status: "completed",
			Model:  model,
			Output: []responsesOutput{
				{
					Type: "message",
					Role: "assistant",
					Content: []responsesContent{
						{Type: "output_text", Text: content},
					},
				},
			},
			Usage: &responsesUsage{
				InputTokens:         inputTokens,
				InputTokensDetails:  responsesInputDetails{},
				OutputTokens:        outputTokens,
				OutputTokensDetails: responsesOutputDetails{},
				TotalTokens:         inputTokens + outputTokens,
			},
		},
	})
	flusher.Flush()
}

func writeResponsesSSE(w http.ResponseWriter, event string, v any) {
	data, _ := json.Marshal(v)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
}

// completions (legacy) types

type completionsResponse struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Model   string              `json:"model"`
	Choices []completionsChoice `json:"choices"`
	Usage   completionsUsage    `json:"usage"`
}

type completionsChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type completionsUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (p *Provider) handleCompletions(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	if writeSimError(w, r) {
		return
	}

	time.Sleep(p.latency.TTFT())

	ctx := r.Context()
	inputTokens := config.InputTokensFromContext(ctx, p.cfg.DefaultInputTokens)
	outputTokens := config.OutputTokensFromContext(ctx, p.cfg.DefaultOutputTokens)

	var req struct {
		Model string `json:"model,omitempty"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	model := p.cfg.DefaultModel
	if req.Model != "" {
		model = req.Model
	}

	resp := completionsResponse{
		ID:     "cmpl-abc123",
		Object: "text_completion",
		Model:  model,
		Choices: []completionsChoice{
			{Text: p.cfg.ResponseContent, Index: 0, FinishReason: "stop"},
		},
		Usage: completionsUsage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}

// embeddings types

type embeddingsResponse struct {
	Object string           `json:"object"`
	Data   []embeddingData  `json:"data"`
	Model  string           `json:"model"`
	Usage  embeddingsUsage  `json:"usage"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (p *Provider) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	p.latency.Acquire()
	defer p.latency.Release()

	if writeSimError(w, r) {
		return
	}

	time.Sleep(p.latency.TTFT())

	ctx := r.Context()
	inputTokens := config.InputTokensFromContext(ctx, p.cfg.DefaultInputTokens)

	var req struct {
		Model string `json:"model,omitempty"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	model := p.cfg.DefaultModel
	if req.Model != "" {
		model = req.Model
	}

	resp := embeddingsResponse{
		Object: "list",
		Data: []embeddingData{
			{
				Object:    "embedding",
				Index:     0,
				Embedding: []float64{0.0023, -0.0091},
			},
		},
		Model: model,
		Usage: embeddingsUsage{
			PromptTokens: inputTokens,
			TotalTokens:  inputTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
		return
	}
}
