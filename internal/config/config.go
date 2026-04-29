package config

import (
	"context"
	"encoding/json"
	"flag"
	"net/http"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port                int      `json:"port"`
	EnabledProviders    []string `json:"enabled_providers"`
	DefaultModel        string   `json:"default_model"`
	DefaultInputTokens  int      `json:"default_input_tokens"`
	DefaultOutputTokens int      `json:"default_output_tokens"`
	ResponseContent     string   `json:"response_content"`
	TTFT                string   `json:"ttft"`
	TTFTStdDev          string   `json:"ttft_stddev"`
	ITL                 string   `json:"itl"`
	ITLStdDev           string   `json:"itl_stddev"`
	LoadFactor          float64  `json:"load_factor"`
	MaxConcurrent       int      `json:"max_concurrent"`
}

func DefaultConfig() *Config {
	return &Config{
		Port:                8080,
		EnabledProviders:    []string{"openai", "anthropic", "gemini"},
		DefaultModel:        "gpt-4",
		DefaultInputTokens:  10,
		DefaultOutputTokens: 5,
		ResponseContent:     "Hello",
		TTFT:                "0s",
		TTFTStdDev:          "0s",
		ITL:                 "0s",
		ITLStdDev:           "0s",
		LoadFactor:          1.0,
		MaxConcurrent:       32,
	}
}

// LoadFromFile reads a JSON config file, applying values on top of cfg.
func LoadFromFile(path string, cfg *Config) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, cfg)
}

// BindFlags registers CLI flags that write into cfg. Call flag.Parse() after.
func BindFlags(fs *flag.FlagSet, cfg *Config) {
	fs.IntVar(&cfg.Port, "port", cfg.Port, "listen port")
	fs.StringVar(&cfg.DefaultModel, "model", cfg.DefaultModel, "default model name")
	fs.IntVar(&cfg.DefaultInputTokens, "input-tokens", cfg.DefaultInputTokens, "default input token count")
	fs.IntVar(&cfg.DefaultOutputTokens, "output-tokens", cfg.DefaultOutputTokens, "default output token count")
	fs.StringVar(&cfg.ResponseContent, "response-content", cfg.ResponseContent, "static response content")
	fs.StringVar(&cfg.TTFT, "ttft", cfg.TTFT, "mean time to first token (e.g. 200ms)")
	fs.StringVar(&cfg.TTFTStdDev, "ttft-stddev", cfg.TTFTStdDev, "ttft jitter stddev (e.g. 30ms)")
	fs.StringVar(&cfg.ITL, "itl", cfg.ITL, "mean inter-token latency (e.g. 20ms)")
	fs.StringVar(&cfg.ITLStdDev, "itl-stddev", cfg.ITLStdDev, "itl jitter stddev (e.g. 5ms)")
	fs.Float64Var(&cfg.LoadFactor, "load-factor", cfg.LoadFactor, "max latency multiplier under load")
	fs.IntVar(&cfg.MaxConcurrent, "max-concurrent", cfg.MaxConcurrent, "max concurrent requests for load scaling")
}

// ParseLatencyDuration parses a duration string, returning 0 on error.
func ParseLatencyDuration(s string) time.Duration {
	d, _ := time.ParseDuration(s)
	return d
}

type contextKey int

const (
	simInputTokensKey contextKey = iota
	simOutputTokensKey
	simErrorKey
)

// SimValues holds per-request overrides extracted from X-Sim-* headers.
type SimValues struct {
	InputTokens  *int
	OutputTokens *int
	ErrorCode    *int
}

// SimHeaderMiddleware extracts X-Sim-Input-Tokens, X-Sim-Output-Tokens, and
// X-Sim-Error headers and stores them in the request context.
func SimHeaderMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		if v := r.Header.Get("X-Sim-Input-Tokens"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				ctx = context.WithValue(ctx, simInputTokensKey, n)
			}
		}
		if v := r.Header.Get("X-Sim-Output-Tokens"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				ctx = context.WithValue(ctx, simOutputTokensKey, n)
			}
		}
		if v := r.Header.Get("X-Sim-Error"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				ctx = context.WithValue(ctx, simErrorKey, n)
			}
		}

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// InputTokensFromContext returns the per-request input token override, or the
// config default if no header was set.
func InputTokensFromContext(ctx context.Context, fallback int) int {
	if v, ok := ctx.Value(simInputTokensKey).(int); ok {
		return v
	}
	return fallback
}

// OutputTokensFromContext returns the per-request output token override, or the
// config default if no header was set.
func OutputTokensFromContext(ctx context.Context, fallback int) int {
	if v, ok := ctx.Value(simOutputTokensKey).(int); ok {
		return v
	}
	return fallback
}

// ErrorCodeFromContext returns the simulated error code, or 0 if none set.
func ErrorCodeFromContext(ctx context.Context) int {
	if v, ok := ctx.Value(simErrorKey).(int); ok {
		return v
	}
	return 0
}
