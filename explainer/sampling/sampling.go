// Package sampling provides a Monte Carlo sampling-based SHAP explainer.
package sampling

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Explainer implements SHAP value computation using Monte Carlo sampling.
// This is a simple but effective approach that samples random coalitions
// and estimates each feature's marginal contribution.
type Explainer struct {
	model        model.Model
	background   [][]float64
	featureNames []string
	baseValue    float64
	config       explainer.Config
	rng          *rand.Rand
}

// New creates a new sampling-based SHAP explainer.
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error) {
	if m == nil {
		return nil, fmt.Errorf("model cannot be nil")
	}
	if len(background) == 0 {
		return nil, fmt.Errorf("background data cannot be empty")
	}

	numFeatures := m.NumFeatures()
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("background row %d has %d features, expected %d",
				i, len(row), numFeatures)
		}
	}

	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

	// Compute base value (expected prediction on background)
	ctx := context.Background()
	predictions, err := m.PredictBatch(ctx, background)
	if err != nil {
		return nil, fmt.Errorf("failed to compute base value: %w", err)
	}

	var baseValue float64
	for _, p := range predictions {
		baseValue += p
	}
	baseValue /= float64(len(predictions))

	return &Explainer{
		model:        m,
		background:   background,
		featureNames: config.FeatureNames,
		baseValue:    baseValue,
		config:       config,
		rng:          config.GetRNG(),
	}, nil
}

// Explain computes SHAP values for a single instance.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	if len(instance) != e.model.NumFeatures() {
		return nil, fmt.Errorf("instance has %d features, expected %d",
			len(instance), e.model.NumFeatures())
	}

	// Get the prediction for the instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute SHAP values using Monte Carlo sampling
	shapValues, err := e.computeSHAPValues(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to compute SHAP values: %w", err)
	}

	// Build the explanation
	values := make(map[string]float64)
	featureValues := make(map[string]float64)
	for i, name := range e.featureNames {
		values[name] = shapValues[i]
		featureValues[name] = instance[i]
	}

	exp := &explanation.Explanation{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Values:        values,
		FeatureNames:  e.featureNames,
		FeatureValues: featureValues,
		Timestamp:     time.Now(),
		ModelID:       e.config.ModelID,
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "sampling",
			NumSamples:     e.config.NumSamples,
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}

	return exp, nil
}

// ExplainBatch computes SHAP explanations for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, inst := range instances {
		exp, err := e.Explain(ctx, inst)
		if err != nil {
			return nil, fmt.Errorf("failed to explain instance %d: %w", i, err)
		}
		results[i] = exp
	}
	return results, nil
}

// BaseValue returns the expected model output on the background dataset.
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// FeatureNames returns the names of the features.
func (e *Explainer) FeatureNames() []string {
	return e.featureNames
}

// computeSHAPValues computes SHAP values using Monte Carlo sampling.
// For each feature, we estimate its contribution by averaging over random coalitions.
func (e *Explainer) computeSHAPValues(ctx context.Context, instance []float64) ([]float64, error) {
	numFeatures := len(instance)
	shapValues := make([]float64, numFeatures)
	counts := make([]int, numFeatures)

	// Sample random permutations and compute marginal contributions
	for sample := 0; sample < e.config.NumSamples; sample++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Generate a random permutation
		perm := e.randomPermutation(numFeatures)

		// Pick a random background sample
		bgIdx := e.rng.Intn(len(e.background))
		bgSample := e.background[bgIdx]

		// Start with background (all features masked)
		current := make([]float64, numFeatures)
		copy(current, bgSample)

		prevPred, err := e.model.Predict(ctx, current)
		if err != nil {
			return nil, fmt.Errorf("prediction failed: %w", err)
		}

		// Add features one by one in permutation order
		for _, featureIdx := range perm {
			// Replace the masked value with the instance value
			current[featureIdx] = instance[featureIdx]

			newPred, err := e.model.Predict(ctx, current)
			if err != nil {
				return nil, fmt.Errorf("prediction failed: %w", err)
			}

			// Contribution = prediction after - prediction before
			contribution := newPred - prevPred
			shapValues[featureIdx] += contribution
			counts[featureIdx]++

			prevPred = newPred
		}
	}

	// Average the contributions
	for i := range shapValues {
		if counts[i] > 0 {
			shapValues[i] /= float64(counts[i])
		}
	}

	return shapValues, nil
}

// randomPermutation generates a random permutation of [0, n).
func (e *Explainer) randomPermutation(n int) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	e.rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	return perm
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
