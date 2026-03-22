package gradient

import (
	"math/rand"
)

// RNG is an interface for random number generation.
type RNG interface {
	Intn(n int) int
	Float64() float64
	NormFloat64() float64
}

// stdRNG wraps the standard library rand.Rand.
type stdRNG struct {
	*rand.Rand
}

func newRNG(seed int64) RNG {
	return &stdRNG{rand.New(rand.NewSource(seed))} //nolint:gosec // seeded for reproducibility
}
