package provider

import (
	"fmt"
	"net/http"
)

// Provider defines the interface that each LLM API mock must implement.
type Provider interface {
	Name() string
	RegisterRoutes(mux *http.ServeMux)
}

// Registry holds registered providers and wires them to a mux.
type Registry struct {
	providers map[string]Provider
}

func NewRegistry() *Registry {
	return &Registry{providers: make(map[string]Provider)}
}

// Register adds a provider to the registry.
func (r *Registry) Register(p Provider) {
	r.providers[p.Name()] = p
}

// EnabledProviders returns the subset of registered providers whose names
// appear in the enabled list.
func (r *Registry) EnabledProviders(enabled []string) []Provider {
	var out []Provider
	for _, name := range enabled {
		if p, ok := r.providers[name]; ok {
			out = append(out, p)
		}
	}
	return out
}

// RegisterRoutes wires enabled providers to the mux. Returns an error if an
// enabled provider name has no registration.
func (r *Registry) RegisterRoutes(mux *http.ServeMux, enabled []string) error {
	for _, name := range enabled {
		p, ok := r.providers[name]
		if !ok {
			return fmt.Errorf("provider %q enabled but not registered", name)
		}
		p.RegisterRoutes(mux)
	}
	return nil
}
