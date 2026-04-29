package latency

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// Config holds latency simulation parameters.
type Config struct {
	TTFT          time.Duration // mean time to first token
	TTFTStdDev    time.Duration // stddev for ttft jitter
	ITL           time.Duration // mean inter-token latency
	ITLStdDev     time.Duration // stddev for itl jitter
	LoadFactor    float64       // max multiplier under load (e.g. 2.0)
	MaxConcurrent int           // max concurrent requests for load scaling
}

// Calculator produces jittered latency values scaled by current load.
type Calculator struct {
	config  Config
	running atomic.Int64
	mu      sync.Mutex
	rng     *rand.Rand
}

// NewCalculator creates a calculator with the given config.
func NewCalculator(cfg Config) *Calculator {
	if cfg.MaxConcurrent < 1 {
		cfg.MaxConcurrent = 1
	}
	if cfg.LoadFactor < 1.0 {
		cfg.LoadFactor = 1.0
	}
	return &Calculator{
		config: cfg,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Acquire increments the running request count.
func (c *Calculator) Acquire() {
	c.running.Add(1)
}

// Release decrements the running request count.
func (c *Calculator) Release() {
	c.running.Add(-1)
}

// TTFT returns a jittered time-to-first-token delay scaled by load.
func (c *Calculator) TTFT() time.Duration {
	return c.jittered(c.config.TTFT, c.config.TTFTStdDev)
}

// ITL returns a jittered inter-token latency delay scaled by load.
func (c *Calculator) ITL() time.Duration {
	return c.jittered(c.config.ITL, c.config.ITLStdDev)
}

// Enabled returns true if any latency simulation is configured.
func (c *Calculator) Enabled() bool {
	return c.config.TTFT > 0 || c.config.ITL > 0
}

func (c *Calculator) jittered(mean, stddev time.Duration) time.Duration {
	if mean == 0 && stddev == 0 {
		return 0
	}
	factor := c.loadFactor()
	base := float64(mean) * factor

	var jitter float64
	if stddev > 0 {
		c.mu.Lock()
		jitter = c.rng.NormFloat64() * float64(stddev) * factor
		c.mu.Unlock()
	}

	d := base + jitter
	// clamp to zero
	if d < 0 {
		return 0
	}
	return time.Duration(math.Round(d))
}

// loadFactor returns the current scaling multiplier based on concurrent requests.
// formula: 1 + (factor - 1) * (running - 1) / (maxConcurrent - 1)
func (c *Calculator) loadFactor() float64 {
	if c.config.LoadFactor <= 1.0 || c.config.MaxConcurrent <= 1 {
		return 1.0
	}
	running := c.running.Load()
	if running <= 1 {
		return 1.0
	}
	if running >= int64(c.config.MaxConcurrent) {
		return c.config.LoadFactor
	}
	return 1.0 + (c.config.LoadFactor-1.0)*float64(running-1)/float64(c.config.MaxConcurrent-1)
}
