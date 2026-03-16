package rand

import (
	"math/rand"
	"sort"
	"testing"
)

func TestPermutation(t *testing.T) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests //nolint:gosec // deterministic seed for reproducible tests

	t.Run("length", func(t *testing.T) {
		for _, n := range []int{0, 1, 5, 10, 100} {
			perm := Permutation(n, rng)
			if len(perm) != n {
				t.Errorf("Permutation(%d): got length %d, want %d", n, len(perm), n)
			}
		}
	})

	t.Run("contains_all_elements", func(t *testing.T) {
		n := 10
		perm := Permutation(n, rng)

		// Sort and verify it's [0, 1, 2, ..., n-1]
		sorted := make([]int, len(perm))
		copy(sorted, perm)
		sort.Ints(sorted)

		for i := 0; i < n; i++ {
			if sorted[i] != i {
				t.Errorf("Permutation missing element %d", i)
			}
		}
	})

	t.Run("is_shuffled", func(t *testing.T) {
		// With high probability, a permutation of 10 elements should not be sorted
		n := 10
		identityCount := 0
		trials := 100

		for i := 0; i < trials; i++ {
			perm := Permutation(n, rng)
			isIdentity := true
			for j := 0; j < n; j++ {
				if perm[j] != j {
					isIdentity = false
					break
				}
			}
			if isIdentity {
				identityCount++
			}
		}

		// Probability of identity is 1/n! = 1/3628800, so this should almost never happen
		if identityCount > 1 {
			t.Errorf("Permutation returned identity %d times out of %d (expected ~0)", identityCount, trials)
		}
	})

	t.Run("deterministic_with_seed", func(t *testing.T) {
		rng1 := rand.New(rand.NewSource(123)) //nolint:gosec // deterministic seed for reproducible tests
		rng2 := rand.New(rand.NewSource(123)) //nolint:gosec // deterministic seed for reproducible tests

		perm1 := Permutation(10, rng1)
		perm2 := Permutation(10, rng2)

		for i := range perm1 {
			if perm1[i] != perm2[i] {
				t.Errorf("Same seed produced different permutations")
				break
			}
		}
	})
}

func TestCoalition(t *testing.T) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests

	t.Run("length", func(t *testing.T) {
		for _, n := range []int{0, 1, 5, 10} {
			mask := Coalition(n, 0.5, rng)
			if len(mask) != n {
				t.Errorf("Coalition(%d, 0.5): got length %d, want %d", n, len(mask), n)
			}
		}
	})

	t.Run("prob_zero", func(t *testing.T) {
		mask := Coalition(10, 0.0, rng)
		for i, v := range mask {
			if v {
				t.Errorf("Coalition with prob=0 has true at index %d", i)
			}
		}
	})

	t.Run("prob_one", func(t *testing.T) {
		mask := Coalition(10, 1.0, rng)
		for i, v := range mask {
			if !v {
				t.Errorf("Coalition with prob=1 has false at index %d", i)
			}
		}
	})

	t.Run("approximate_probability", func(t *testing.T) {
		n := 1000
		prob := 0.3
		trials := 100
		totalTrue := 0

		for i := 0; i < trials; i++ {
			mask := Coalition(n, prob, rng)
			for _, v := range mask {
				if v {
					totalTrue++
				}
			}
		}

		// Expected: trials * n * prob = 100 * 1000 * 0.3 = 30000
		expected := float64(trials*n) * prob
		actual := float64(totalTrue)
		// Allow 5% deviation
		if actual < expected*0.95 || actual > expected*1.05 {
			t.Errorf("Coalition probability off: got %.0f true, expected ~%.0f", actual, expected)
		}
	})
}

func TestCoalitionSize(t *testing.T) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests

	t.Run("exact_size", func(t *testing.T) {
		for _, tc := range []struct{ n, k int }{
			{10, 0},
			{10, 1},
			{10, 5},
			{10, 10},
		} {
			mask := CoalitionSize(tc.n, tc.k, rng)
			if len(mask) != tc.n {
				t.Errorf("CoalitionSize(%d, %d): got length %d, want %d", tc.n, tc.k, len(mask), tc.n)
			}

			count := 0
			for _, v := range mask {
				if v {
					count++
				}
			}
			if count != tc.k {
				t.Errorf("CoalitionSize(%d, %d): got %d true elements, want %d", tc.n, tc.k, count, tc.k)
			}
		}
	})

	t.Run("k_greater_than_n", func(t *testing.T) {
		mask := CoalitionSize(5, 10, rng)
		count := 0
		for _, v := range mask {
			if v {
				count++
			}
		}
		if count != 5 {
			t.Errorf("CoalitionSize(5, 10): got %d true elements, want 5", count)
		}
	})

	t.Run("negative_k", func(t *testing.T) {
		mask := CoalitionSize(5, -1, rng)
		count := 0
		for _, v := range mask {
			if v {
				count++
			}
		}
		if count != 0 {
			t.Errorf("CoalitionSize(5, -1): got %d true elements, want 0", count)
		}
	})

	t.Run("randomness", func(t *testing.T) {
		// Multiple calls should produce different coalitions
		n, k := 10, 5
		coalitions := make([][]bool, 10)
		for i := range coalitions {
			coalitions[i] = CoalitionSize(n, k, rng)
		}

		// Check that not all coalitions are identical
		allSame := true
		for i := 1; i < len(coalitions); i++ {
			for j := 0; j < n; j++ {
				if coalitions[i][j] != coalitions[0][j] {
					allSame = false
					break
				}
			}
			if !allSame {
				break
			}
		}
		if allSame {
			t.Error("CoalitionSize produced identical coalitions")
		}
	})
}

func TestSampleIndices(t *testing.T) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests

	t.Run("correct_count", func(t *testing.T) {
		for _, tc := range []struct{ n, k int }{
			{10, 0},
			{10, 1},
			{10, 5},
			{10, 10},
		} {
			indices := SampleIndices(tc.n, tc.k, rng)
			if len(indices) != tc.k {
				t.Errorf("SampleIndices(%d, %d): got %d indices, want %d", tc.n, tc.k, len(indices), tc.k)
			}
		}
	})

	t.Run("valid_range", func(t *testing.T) {
		n, k := 10, 5
		indices := SampleIndices(n, k, rng)
		for _, idx := range indices {
			if idx < 0 || idx >= n {
				t.Errorf("SampleIndices(%d, %d): index %d out of range [0, %d)", n, k, idx, n)
			}
		}
	})

	t.Run("no_duplicates", func(t *testing.T) {
		n, k := 10, 5
		indices := SampleIndices(n, k, rng)
		seen := make(map[int]bool)
		for _, idx := range indices {
			if seen[idx] {
				t.Errorf("SampleIndices(%d, %d): duplicate index %d", n, k, idx)
			}
			seen[idx] = true
		}
	})

	t.Run("k_greater_than_n", func(t *testing.T) {
		indices := SampleIndices(5, 10, rng)
		if len(indices) != 5 {
			t.Errorf("SampleIndices(5, 10): got %d indices, want 5", len(indices))
		}
	})

	t.Run("negative_k", func(t *testing.T) {
		indices := SampleIndices(5, -1, rng)
		if len(indices) != 0 {
			t.Errorf("SampleIndices(5, -1): got %d indices, want 0", len(indices))
		}
	})
}

func BenchmarkPermutation(b *testing.B) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests
	for i := 0; i < b.N; i++ {
		Permutation(100, rng)
	}
}

func BenchmarkCoalition(b *testing.B) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests
	for i := 0; i < b.N; i++ {
		Coalition(100, 0.5, rng)
	}
}

func BenchmarkCoalitionSize(b *testing.B) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests
	for i := 0; i < b.N; i++ {
		CoalitionSize(100, 50, rng)
	}
}

func BenchmarkSampleIndices(b *testing.B) {
	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic seed for reproducible tests
	for i := 0; i < b.N; i++ {
		SampleIndices(100, 50, rng)
	}
}
