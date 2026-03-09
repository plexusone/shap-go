// Package rand provides random number generation utilities for SHAP computation.
package rand

import (
	"math/rand"
)

// Permutation generates a random permutation of [0, n).
func Permutation(n int, rng *rand.Rand) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	return perm
}

// Coalition generates a random coalition (subset) of features.
// Each feature is included with the given probability.
func Coalition(n int, prob float64, rng *rand.Rand) []bool {
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = rng.Float64() < prob
	}
	return mask
}

// CoalitionSize generates a random coalition of exactly k features.
func CoalitionSize(n, k int, rng *rand.Rand) []bool {
	if k > n {
		k = n
	}
	if k < 0 {
		k = 0
	}

	// Generate permutation and take first k
	perm := Permutation(n, rng)
	mask := make([]bool, n)
	for i := 0; i < k; i++ {
		mask[perm[i]] = true
	}
	return mask
}

// SampleIndices returns k random indices from [0, n) without replacement.
func SampleIndices(n, k int, rng *rand.Rand) []int {
	if k > n {
		k = n
	}
	if k < 0 {
		k = 0
	}

	perm := Permutation(n, rng)
	return perm[:k]
}
