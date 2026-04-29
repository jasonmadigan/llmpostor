package provider

import (
	"net/http"
	"testing"
)

type stubProvider struct {
	name    string
	called  bool
}

func (s *stubProvider) Name() string { return s.name }

func (s *stubProvider) RegisterRoutes(mux *http.ServeMux) {
	s.called = true
	mux.HandleFunc("GET /stub", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func TestRegistryRegisterAndEnable(t *testing.T) {
	r := NewRegistry()
	p := &stubProvider{name: "test"}
	r.Register(p)

	enabled := r.EnabledProviders([]string{"test"})
	if len(enabled) != 1 {
		t.Fatalf("expected 1 enabled provider, got %d", len(enabled))
	}
	if enabled[0].Name() != "test" {
		t.Errorf("expected provider name 'test', got %q", enabled[0].Name())
	}
}

func TestRegistryEnabledFiltersUnregistered(t *testing.T) {
	r := NewRegistry()
	r.Register(&stubProvider{name: "a"})

	enabled := r.EnabledProviders([]string{"a", "b"})
	if len(enabled) != 1 {
		t.Fatalf("expected 1 enabled provider, got %d", len(enabled))
	}
}

func TestRegistryRegisterRoutes(t *testing.T) {
	r := NewRegistry()
	p := &stubProvider{name: "test"}
	r.Register(p)

	mux := http.NewServeMux()
	if err := r.RegisterRoutes(mux, []string{"test"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !p.called {
		t.Error("RegisterRoutes was not called on the provider")
	}
}

func TestRegistryRegisterRoutesUnknownProvider(t *testing.T) {
	r := NewRegistry()
	mux := http.NewServeMux()
	if err := r.RegisterRoutes(mux, []string{"nonexistent"}); err == nil {
		t.Error("expected error for unknown provider, got nil")
	}
}

func TestRegistryEmptyEnabled(t *testing.T) {
	r := NewRegistry()
	r.Register(&stubProvider{name: "a"})

	enabled := r.EnabledProviders(nil)
	if len(enabled) != 0 {
		t.Errorf("expected 0 enabled providers, got %d", len(enabled))
	}
}
