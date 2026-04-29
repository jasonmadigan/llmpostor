package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/jasonmadigan/llmpostor/internal/config"
	"github.com/jasonmadigan/llmpostor/internal/latency"
	"github.com/jasonmadigan/llmpostor/internal/provider"
	"github.com/jasonmadigan/llmpostor/internal/provider/anthropic"
	"github.com/jasonmadigan/llmpostor/internal/provider/gemini"
	"github.com/jasonmadigan/llmpostor/internal/provider/openai"
	"github.com/jasonmadigan/llmpostor/internal/server"
)

func main() {
	cfg := config.DefaultConfig()

	configFile := ""
	flag.StringVar(&configFile, "config", "", "path to JSON config file")
	config.BindFlags(flag.CommandLine, cfg)
	flag.Parse()

	if configFile != "" {
		if err := config.LoadFromFile(configFile, cfg); err != nil {
			log.Fatalf("loading config: %v", err)
		}
		// re-parse so flags override file values
		flag.Parse()
	}

	lat := latency.NewCalculator(latency.Config{
		TTFT:          config.ParseLatencyDuration(cfg.TTFT),
		TTFTStdDev:    config.ParseLatencyDuration(cfg.TTFTStdDev),
		ITL:           config.ParseLatencyDuration(cfg.ITL),
		ITLStdDev:     config.ParseLatencyDuration(cfg.ITLStdDev),
		LoadFactor:    cfg.LoadFactor,
		MaxConcurrent: cfg.MaxConcurrent,
	})

	registry := provider.NewRegistry()
	registry.Register(openai.New(cfg, lat))
	registry.Register(anthropic.New(cfg, lat))
	registry.Register(gemini.New(cfg, lat))

	handler, err := server.New(cfg, registry)
	if err != nil {
		log.Fatalf("creating server: %v", err)
	}

	addr := fmt.Sprintf(":%d", cfg.Port)
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
