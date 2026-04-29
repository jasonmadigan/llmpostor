package latency

import (
	"testing"
	"time"
)

func TestZeroConfigProducesZeroDelays(t *testing.T) {
	c := NewCalculator(Config{})
	for range 100 {
		if d := c.TTFT(); d != 0 {
			t.Fatalf("expected 0 TTFT, got %v", d)
		}
		if d := c.ITL(); d != 0 {
			t.Fatalf("expected 0 ITL, got %v", d)
		}
	}
}

func TestTTFTWithinExpectedRange(t *testing.T) {
	c := NewCalculator(Config{
		TTFT:       200 * time.Millisecond,
		TTFTStdDev: 20 * time.Millisecond,
	})

	for range 100 {
		d := c.TTFT()
		// with stddev=20ms, values outside mean +/- 5*stddev are extremely unlikely
		if d < 100*time.Millisecond || d > 300*time.Millisecond {
			t.Fatalf("TTFT %v out of expected range [100ms, 300ms]", d)
		}
	}
}

func TestITLWithinExpectedRange(t *testing.T) {
	c := NewCalculator(Config{
		ITL:       30 * time.Millisecond,
		ITLStdDev: 5 * time.Millisecond,
	})

	for range 100 {
		d := c.ITL()
		if d < 5*time.Millisecond || d > 55*time.Millisecond {
			t.Fatalf("ITL %v out of expected range [5ms, 55ms]", d)
		}
	}
}

func TestLoadFactorScaling(t *testing.T) {
	c := NewCalculator(Config{
		TTFT:          100 * time.Millisecond,
		LoadFactor:    3.0,
		MaxConcurrent: 10,
	})

	// single request: factor = 1.0
	c.running.Store(1)
	single := avgTTFT(c, 200)

	// max requests: factor = 3.0
	c.running.Store(10)
	max := avgTTFT(c, 200)

	// max should be roughly 3x single
	ratio := float64(max) / float64(single)
	if ratio < 2.5 || ratio > 3.5 {
		t.Fatalf("expected ratio ~3.0, got %.2f (single=%v max=%v)", ratio, single, max)
	}
}

func TestLoadFactorNoScalingAtOneRequest(t *testing.T) {
	c := NewCalculator(Config{
		TTFT:          100 * time.Millisecond,
		LoadFactor:    5.0,
		MaxConcurrent: 32,
	})
	c.running.Store(1)
	f := c.loadFactor()
	if f != 1.0 {
		t.Fatalf("expected factor 1.0 at 1 running, got %f", f)
	}
}

func TestLoadFactorAtMaxConcurrent(t *testing.T) {
	c := NewCalculator(Config{
		TTFT:          100 * time.Millisecond,
		LoadFactor:    2.0,
		MaxConcurrent: 32,
	})
	c.running.Store(32)
	f := c.loadFactor()
	if f != 2.0 {
		t.Fatalf("expected factor 2.0 at max concurrent, got %f", f)
	}
}

func TestJitterProducesVariation(t *testing.T) {
	c := NewCalculator(Config{
		TTFT:       100 * time.Millisecond,
		TTFTStdDev: 30 * time.Millisecond,
	})

	seen := make(map[time.Duration]bool)
	for range 50 {
		seen[c.TTFT()] = true
	}
	if len(seen) < 5 {
		t.Fatalf("expected variation in TTFT values, got only %d distinct values", len(seen))
	}
}

func TestNegativeClamp(t *testing.T) {
	// large stddev relative to mean -- should never produce negative
	c := NewCalculator(Config{
		TTFT:       10 * time.Millisecond,
		TTFTStdDev: 100 * time.Millisecond,
	})

	for range 1000 {
		if d := c.TTFT(); d < 0 {
			t.Fatalf("got negative TTFT: %v", d)
		}
	}
}

func TestEnabledFlag(t *testing.T) {
	zero := NewCalculator(Config{})
	if zero.Enabled() {
		t.Fatal("expected disabled for zero config")
	}

	ttftOnly := NewCalculator(Config{TTFT: time.Millisecond})
	if !ttftOnly.Enabled() {
		t.Fatal("expected enabled with TTFT set")
	}

	itlOnly := NewCalculator(Config{ITL: time.Millisecond})
	if !itlOnly.Enabled() {
		t.Fatal("expected enabled with ITL set")
	}
}

func TestAcquireRelease(t *testing.T) {
	c := NewCalculator(Config{})
	c.Acquire()
	c.Acquire()
	if n := c.running.Load(); n != 2 {
		t.Fatalf("expected 2 running, got %d", n)
	}
	c.Release()
	if n := c.running.Load(); n != 1 {
		t.Fatalf("expected 1 running, got %d", n)
	}
}

func avgTTFT(c *Calculator, n int) time.Duration {
	var total time.Duration
	for range n {
		total += c.TTFT()
	}
	return total / time.Duration(n)
}
