// Package permutation provides a permutation-based SHAP explainer with antithetic sampling.
package permutation

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Explainer implements SHAP value computation using permutation-based algorithm
// with antithetic sampling for variance reduction.
//
// The algorithm works as follows:
// 1. For each sample permutation:
//   - Forward pass: Start with all features masked (background), add features one by one
//   - Reverse pass (antithetic): Start with all features present, remove features one by one
//   - Average contributions from both passes
//
// 2. Average over all samples
//
// This approach guarantees local accuracy (SHAP values sum to prediction - baseline).
type Explainer struct {
	model        model.Model
	background   [][]float64
	featureNames []string
	baseValue    float64
	config       explainer.Config
	rng          *rand.Rand
	mu           sync.Mutex // protects rng
}

// New creates a new permutation-based SHAP explainer.
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

// Explain computes SHAP values for a single instance using antithetic permutation sampling.
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

	// Compute SHAP values
	var shapValues []float64
	if e.config.NumWorkers > 1 {
		shapValues, err = e.computeSHAPValuesParallel(ctx, instance)
	} else {
		shapValues, err = e.computeSHAPValues(ctx, instance)
	}
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
			Algorithm:      "permutation",
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

// computeSHAPValues computes SHAP values using antithetic permutation sampling.
func (e *Explainer) computeSHAPValues(ctx context.Context, instance []float64) ([]float64, error) {
	numFeatures := len(instance)
	shapValues := make([]float64, numFeatures)

	for sample := 0; sample < e.config.NumSamples; sample++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		contributions, err := e.sampleContributions(ctx, instance)
		if err != nil {
			return nil, err
		}

		for i := range shapValues {
			shapValues[i] += contributions[i]
		}
	}

	// Average over samples
	for i := range shapValues {
		shapValues[i] /= float64(e.config.NumSamples)
	}

	return shapValues, nil
}

// computeSHAPValuesParallel computes SHAP values using parallel workers.
func (e *Explainer) computeSHAPValuesParallel(ctx context.Context, instance []float64) ([]float64, error) {
	numFeatures := len(instance)
	numWorkers := e.config.NumWorkers
	numSamples := e.config.NumSamples

	// Distribute samples across workers
	samplesPerWorker := numSamples / numWorkers
	extraSamples := numSamples % numWorkers

	type result struct {
		values []float64
		err    error
	}

	results := make(chan result, numWorkers)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		workerSamples := samplesPerWorker
		if w < extraSamples {
			workerSamples++
		}

		// Create a worker-specific RNG
		e.mu.Lock()
		workerSeed := e.rng.Int63()
		e.mu.Unlock()

		go func(numSamples int, seed int64) {
			defer wg.Done()

			workerRNG := rand.New(rand.NewSource(seed))
			values := make([]float64, numFeatures)

			for sample := 0; sample < numSamples; sample++ {
				select {
				case <-ctx.Done():
					results <- result{nil, ctx.Err()}
					return
				default:
				}

				contributions, err := e.sampleContributionsWithRNG(ctx, instance, workerRNG)
				if err != nil {
					results <- result{nil, err}
					return
				}

				for i := range values {
					values[i] += contributions[i]
				}
			}

			results <- result{values, nil}
		}(workerSamples, workerSeed)
	}

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Aggregate results
	shapValues := make([]float64, numFeatures)
	for res := range results {
		if res.err != nil {
			return nil, res.err
		}
		for i := range shapValues {
			shapValues[i] += res.values[i]
		}
	}

	// Average over total samples
	for i := range shapValues {
		shapValues[i] /= float64(numSamples)
	}

	return shapValues, nil
}

// sampleContributions computes feature contributions for a single permutation sample.
func (e *Explainer) sampleContributions(ctx context.Context, instance []float64) ([]float64, error) {
	e.mu.Lock()
	rng := e.rng
	e.mu.Unlock()
	return e.sampleContributionsWithRNG(ctx, instance, rng)
}

// sampleContributionsWithRNG computes feature contributions using a specific RNG.
func (e *Explainer) sampleContributionsWithRNG(ctx context.Context, instance []float64, rng *rand.Rand) ([]float64, error) {
	numFeatures := len(instance)
	contributions := make([]float64, numFeatures)

	// Generate a random permutation
	perm := randomPermutation(numFeatures, rng)

	// Pick a random background sample
	bgIdx := rng.Intn(len(e.background))
	bgSample := e.background[bgIdx]

	// Forward pass: start with background, add features
	forward := make([]float64, numFeatures)
	copy(forward, bgSample)

	forwardPreds := make([]float64, numFeatures+1)
	var err error
	forwardPreds[0], err = e.model.Predict(ctx, forward)
	if err != nil {
		return nil, fmt.Errorf("forward pass prediction failed: %w", err)
	}

	for i, featureIdx := range perm {
		forward[featureIdx] = instance[featureIdx]
		forwardPreds[i+1], err = e.model.Predict(ctx, forward)
		if err != nil {
			return nil, fmt.Errorf("forward pass prediction failed: %w", err)
		}
	}

	// Reverse pass: start with instance, remove features (antithetic)
	reverse := make([]float64, numFeatures)
	copy(reverse, instance)

	reversePreds := make([]float64, numFeatures+1)
	reversePreds[0], err = e.model.Predict(ctx, reverse)
	if err != nil {
		return nil, fmt.Errorf("reverse pass prediction failed: %w", err)
	}

	for i := numFeatures - 1; i >= 0; i-- {
		featureIdx := perm[i]
		reverse[featureIdx] = bgSample[featureIdx]
		reversePreds[numFeatures-i], err = e.model.Predict(ctx, reverse)
		if err != nil {
			return nil, fmt.Errorf("reverse pass prediction failed: %w", err)
		}
	}

	// Combine forward and reverse contributions
	for i, featureIdx := range perm {
		// Forward contribution: pred[i+1] - pred[i] (after adding feature - before)
		forwardContrib := forwardPreds[i+1] - forwardPreds[i]

		// Reverse contribution: pred[before removing] - pred[after removing]
		// reversePreds[0] is full instance, reversePreds[numFeatures] is background
		// Feature at perm[i] is removed at step (numFeatures - i - 1)
		reverseStep := numFeatures - i - 1
		reverseContrib := reversePreds[reverseStep] - reversePreds[reverseStep+1]

		// Average the two
		contributions[featureIdx] = (forwardContrib + reverseContrib) / 2
	}

	return contributions, nil
}

// randomPermutation generates a random permutation of [0, n).
func randomPermutation(n int, rng *rand.Rand) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	return perm
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
