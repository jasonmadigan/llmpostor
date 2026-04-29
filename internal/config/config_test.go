package config

import (
	"context"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Port != 8080 {
		t.Errorf("expected port 8080, got %d", cfg.Port)
	}
	if cfg.DefaultInputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", cfg.DefaultInputTokens)
	}
	if cfg.DefaultOutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", cfg.DefaultOutputTokens)
	}
	if len(cfg.EnabledProviders) != 3 {
		t.Errorf("expected 3 enabled providers, got %d", len(cfg.EnabledProviders))
	}
}

func TestLoadFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")

	data := `{"port": 9090, "default_model": "claude-sonnet-4-20250514", "default_input_tokens": 20}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := DefaultConfig()
	if err := LoadFromFile(path, cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Port != 9090 {
		t.Errorf("expected port 9090, got %d", cfg.Port)
	}
	if cfg.DefaultModel != "claude-sonnet-4-20250514" {
		t.Errorf("expected model claude-sonnet-4-20250514, got %s", cfg.DefaultModel)
	}
	if cfg.DefaultInputTokens != 20 {
		t.Errorf("expected 20 input tokens, got %d", cfg.DefaultInputTokens)
	}
	// unset fields keep defaults
	if cfg.DefaultOutputTokens != 5 {
		t.Errorf("expected 5 output tokens (default), got %d", cfg.DefaultOutputTokens)
	}
}

func TestLoadFromFileMissing(t *testing.T) {
	cfg := DefaultConfig()
	err := LoadFromFile("/nonexistent/path.json", cfg)
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestFlagsPrecedenceOverFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := `{"port": 9090, "default_model": "file-model"}`
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := DefaultConfig()
	if err := LoadFromFile(path, cfg); err != nil {
		t.Fatal(err)
	}

	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	BindFlags(fs, cfg)
	if err := fs.Parse([]string{"-port", "3000", "-model", "flag-model"}); err != nil {
		t.Fatal(err)
	}

	if cfg.Port != 3000 {
		t.Errorf("expected flag port 3000, got %d", cfg.Port)
	}
	if cfg.DefaultModel != "flag-model" {
		t.Errorf("expected flag model, got %s", cfg.DefaultModel)
	}
}

func TestSimHeaderMiddleware(t *testing.T) {
	var captured context.Context
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Context()
		w.WriteHeader(http.StatusOK)
	})

	handler := SimHeaderMiddleware(inner)

	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	req.Header.Set("X-Sim-Input-Tokens", "42")
	req.Header.Set("X-Sim-Output-Tokens", "17")
	req.Header.Set("X-Sim-Error", "429")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if v := InputTokensFromContext(captured, 0); v != 42 {
		t.Errorf("expected input tokens 42, got %d", v)
	}
	if v := OutputTokensFromContext(captured, 0); v != 17 {
		t.Errorf("expected output tokens 17, got %d", v)
	}
	if v := ErrorCodeFromContext(captured); v != 429 {
		t.Errorf("expected error code 429, got %d", v)
	}
}

func TestSimHeaderMiddlewareNoHeaders(t *testing.T) {
	var captured context.Context
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Context()
		w.WriteHeader(http.StatusOK)
	})

	handler := SimHeaderMiddleware(inner)
	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if v := InputTokensFromContext(captured, 99); v != 99 {
		t.Errorf("expected fallback 99, got %d", v)
	}
	if v := OutputTokensFromContext(captured, 88); v != 88 {
		t.Errorf("expected fallback 88, got %d", v)
	}
	if v := ErrorCodeFromContext(captured); v != 0 {
		t.Errorf("expected 0 error code, got %d", v)
	}
}

func TestSimHeaderMiddlewareInvalidValues(t *testing.T) {
	var captured context.Context
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.Context()
		w.WriteHeader(http.StatusOK)
	})

	handler := SimHeaderMiddleware(inner)
	req := httptest.NewRequest(http.MethodPost, "/test", nil)
	req.Header.Set("X-Sim-Input-Tokens", "not-a-number")
	req.Header.Set("X-Sim-Error", "abc")

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if v := InputTokensFromContext(captured, 10); v != 10 {
		t.Errorf("expected fallback for invalid header, got %d", v)
	}
	if v := ErrorCodeFromContext(captured); v != 0 {
		t.Errorf("expected 0 for invalid error header, got %d", v)
	}
}
