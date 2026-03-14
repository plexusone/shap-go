package masker

import (
	"fmt"
	"math/rand"
)

// IndependentMasker implements marginal/independent masking using background samples.
// When a feature is masked, its value is replaced with a randomly sampled value
// from the background dataset (assumes feature independence).
type IndependentMasker struct {
	// background contains the background dataset (rows x features).
	background [][]float64

	// numFeatures is the number of features.
	numFeatures int

	// rng is the random number generator for sampling.
	rng *rand.Rand
}

// NewIndependentMasker creates a new IndependentMasker with the given background data.
// The background data should be a slice of feature vectors, each with the same length.
func NewIndependentMasker(background [][]float64) (*IndependentMasker, error) {
	if len(background) == 0 {
		return nil, fmt.Errorf("background dataset cannot be empty")
	}

	numFeatures := len(background[0])
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("background row %d has %d features, expected %d", i, len(row), numFeatures)
		}
	}

	return &IndependentMasker{
		background:  background,
		numFeatures: numFeatures,
		rng:         rand.New(rand.NewSource(rand.Int63())), //nolint:gosec // crypto not needed for sampling
	}, nil
}

// NewIndependentMaskerWithSeed creates a new IndependentMasker with a specific random seed.
func NewIndependentMaskerWithSeed(background [][]float64, seed int64) (*IndependentMasker, error) {
	masker, err := NewIndependentMasker(background)
	if err != nil {
		return nil, err
	}
	masker.rng = rand.New(rand.NewSource(seed)) //nolint:gosec // seeded for reproducibility
	return masker, nil
}

// Mask creates a masked version of the instance.
// For each feature where mask[i] is true, the feature value is replaced
// with a randomly sampled value from the background dataset.
// Returns an error if instance or mask dimensions don't match the expected feature count.
func (m *IndependentMasker) Mask(instance []float64, mask []bool) ([]float64, error) {
	if len(instance) != m.numFeatures {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d", ErrInstanceFeatureMismatch, len(instance), m.numFeatures)
	}
	if len(mask) != m.numFeatures {
		return nil, fmt.Errorf("%w: mask has %d elements, expected %d", ErrMaskFeatureMismatch, len(mask), m.numFeatures)
	}

	result := make([]float64, m.numFeatures)
	for i := 0; i < m.numFeatures; i++ {
		if mask[i] {
			// Sample from background
			bgIdx := m.rng.Intn(len(m.background))
			result[i] = m.background[bgIdx][i]
		} else {
			// Keep original value
			result[i] = instance[i]
		}
	}

	return result, nil
}

// MaskWithBackground creates a masked version using a specific background sample.
// This is useful when you want deterministic behavior or when computing
// SHAP values with a fixed background reference.
// Returns an error if instance, mask, or background sample dimensions don't match.
func (m *IndependentMasker) MaskWithBackground(instance []float64, mask []bool, bgSample []float64) ([]float64, error) {
	if len(instance) != m.numFeatures {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d", ErrInstanceFeatureMismatch, len(instance), m.numFeatures)
	}
	if len(mask) != m.numFeatures {
		return nil, fmt.Errorf("%w: mask has %d elements, expected %d", ErrMaskFeatureMismatch, len(mask), m.numFeatures)
	}
	if len(bgSample) != m.numFeatures {
		return nil, fmt.Errorf("%w: background sample has %d features, expected %d", ErrBackgroundFeatureMismatch, len(bgSample), m.numFeatures)
	}

	result := make([]float64, m.numFeatures)
	for i := 0; i < m.numFeatures; i++ {
		if mask[i] {
			result[i] = bgSample[i]
		} else {
			result[i] = instance[i]
		}
	}

	return result, nil
}

// NumFeatures returns the number of features.
func (m *IndependentMasker) NumFeatures() int {
	return m.numFeatures
}

// BackgroundSize returns the number of background samples.
func (m *IndependentMasker) BackgroundSize() int {
	return len(m.background)
}

// Background returns the background dataset.
func (m *IndependentMasker) Background() [][]float64 {
	return m.background
}

// SampleBackground returns a randomly sampled background instance.
func (m *IndependentMasker) SampleBackground() []float64 {
	idx := m.rng.Intn(len(m.background))
	result := make([]float64, m.numFeatures)
	copy(result, m.background[idx])
	return result
}

// MeanBackground returns the mean of each feature across all background samples.
func (m *IndependentMasker) MeanBackground() []float64 {
	means := make([]float64, m.numFeatures)
	for _, row := range m.background {
		for i, v := range row {
			means[i] += v
		}
	}
	n := float64(len(m.background))
	for i := range means {
		means[i] /= n
	}
	return means
}

// Ensure IndependentMasker implements Masker.
var _ Masker = (*IndependentMasker)(nil)
