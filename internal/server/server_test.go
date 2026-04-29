package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/provider"
)

func newTestHandler(t *testing.T) http.Handler {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.EnabledProviders = nil // no providers needed for health tests
	registry := provider.NewRegistry()
	h, err := New(cfg, registry)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return h
}

func TestHealthz(t *testing.T) {
	h := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if body := strings.TrimSpace(w.Body.String()); body != "ok" {
		t.Errorf("expected 'ok', got %q", body)
	}
}

func TestReadyz(t *testing.T) {
	h := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if body := strings.TrimSpace(w.Body.String()); body != "ok" {
		t.Errorf("expected 'ok', got %q", body)
	}
}

func TestHealthEndpointsRejectPost(t *testing.T) {
	h := newTestHandler(t)
	for _, path := range []string{"/healthz", "/readyz"} {
		req := httptest.NewRequest(http.MethodPost, path, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code == http.StatusOK {
			t.Errorf("POST %s should not return 200", path)
		}
	}
}

func TestLoggingMiddlewareDoesNotBreakResponse(t *testing.T) {
	h := newTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("logging middleware broke response: got %d", w.Code)
	}
}
