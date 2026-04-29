package server

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/provider"
)

// New creates an http.Handler with health endpoints, sim-header middleware,
// request logging, and provider routes.
func New(cfg *config.Config, registry *provider.Registry) (http.Handler, error) {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintln(w, "ok")
	})

	if err := registry.RegisterRoutes(mux, cfg.EnabledProviders); err != nil {
		return nil, err
	}

	var handler http.Handler = mux
	handler = config.SimHeaderMiddleware(handler)
	handler = loggingMiddleware(handler)

	return handler, nil
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, rw.status, time.Since(start))
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

// delegate flusher to the underlying writer for SSE streaming
func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
