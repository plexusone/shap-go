// Package exact provides ExactSHAP for computing exact Shapley values.
//
// ExactSHAP computes exact SHAP values by enumerating all 2^n possible feature
// coalitions. This is the brute-force approach that guarantees mathematically
// correct Shapley values, but is only practical for small feature sets (≤15 features)
// due to O(n * 2^n) complexity.
//
// Use cases:
//   - Validating other SHAP implementations
//   - Small feature sets where exact values are critical
//   - Reference/educational purposes
//
// For each feature i, the Shapley value is computed as:
//
//	φᵢ = Σ_{S ⊆ N\{i}} [|S|! * (n-|S|-1)! / n!] * [f(S ∪ {i}) - f(S)]
//
// where N is the set of all features and S ranges over all subsets not containing i.
package exact

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// MaxFeatures is the maximum number of features supported.
// Beyond this, the exponential complexity becomes impractical.
const MaxFeatures = 20

// Common errors returned by ExactSHAP.
var (
	ErrNilModel        = errors.New("model cannot be nil")
	ErrNoBackground    = errors.New("background data cannot be empty")
	ErrFeatureMismatch = errors.New("feature count mismatch")
	ErrTooManyFeatures = errors.New("too many features for exact computation")
)

// Explainer implements ExactSHAP by enumerating all feature coalitions.
type Explainer struct {
	model        model.Model
	background   [][]float64
	baseValue    float64
	featureNames []string
	config       explainer.Config

	// Precomputed Shapley weights: weights[s] = s! * (n-s-1)! / n!
	// where s is coalition size and n is total features
	shapleyWeights []float64
}

// New creates a new ExactSHAP explainer.
//
// Parameters:
//   - m: The model to explain (implements model.Model interface)
//   - background: Representative samples for baseline/masking
//   - opts: Configuration options (WithFeatureNames, WithModelID, etc.)
//
// Returns an error if the number of features exceeds MaxFeatures (20).
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error) {
	if m == nil {
		return nil, ErrNilModel
	}
	if len(background) == 0 {
		return nil, ErrNoBackground
	}

	numFeatures := m.NumFeatures()
	if numFeatures > MaxFeatures {
		return nil, fmt.Errorf("%w: %d features (max %d)", ErrTooManyFeatures, numFeatures, MaxFeatures)
	}

	// Validate background dimensions
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("%w: background row %d has %d features, expected %d",
				ErrFeatureMismatch, i, len(row), numFeatures)
		}
	}

	// Apply configuration
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

	// Precompute Shapley weights
	weights := computeShapleyWeights(numFeatures)

	return &Explainer{
		model:          m,
		background:     background,
		baseValue:      baseValue,
		featureNames:   config.FeatureNames,
		config:         config,
		shapleyWeights: weights,
	}, nil
}

// Explain computes exact SHAP values for a single instance.
//
// This enumerates all 2^(n-1) coalitions for each feature to compute
// mathematically exact Shapley values.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.model.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d",
			ErrFeatureMismatch, len(instance), numFeatures)
	}

	// Get the prediction for the instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute exact SHAP values
	shapValues, err := e.computeExactSHAP(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to compute SHAP values: %w", err)
	}

	// Build the explanation
	values := make(map[string]float64)
	featureValues := make(map[string]float64)
	for i, name := range e.featureNames {
		values[name] = shapValues[i]      //nolint:gosec // bounds checked: len(shapValues) == len(featureNames)
		featureValues[name] = instance[i] //nolint:gosec // bounds checked: len(instance) == numFeatures validated above
	}

	return &explanation.Explanation{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Values:        values,
		FeatureNames:  e.featureNames,
		FeatureValues: featureValues,
		Timestamp:     time.Now(),
		ModelID:       e.config.ModelID,
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "exact",
			NumSamples:     1 << numFeatures, // 2^n coalitions evaluated
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}, nil
}

// ExplainBatch computes SHAP explanations for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, inst := range instances {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

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

// computeExactSHAP computes exact Shapley values by enumerating all coalitions.
//
// For each feature i:
//
//	φᵢ = Σ_{S ⊆ N\{i}} w(|S|) * [f(S ∪ {i}) - f(S)]
//
// where w(s) = s! * (n-s-1)! / n!
func (e *Explainer) computeExactSHAP(ctx context.Context, instance []float64) ([]float64, error) {
	numFeatures := len(instance)
	shapValues := make([]float64, numFeatures)

	// Cache coalition predictions to avoid redundant computations
	// Key: bitmask of features in coalition
	predCache := make(map[uint64]float64)

	// Iterate over each feature
	for i := 0; i < numFeatures; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Compute SHAP value for feature i by iterating over all coalitions S ⊆ N\{i}
		// There are 2^(n-1) such coalitions
		numCoalitions := uint64(1) << (numFeatures - 1)

		for coalitionIdx := uint64(0); coalitionIdx < numCoalitions; coalitionIdx++ {
			// Build the coalition mask (excluding feature i)
			// coalitionIdx encodes which of the other features are in S
			coalition := buildCoalition(coalitionIdx, i, numFeatures)
			coalitionSize := popCount(coalition)

			// Get f(S) - prediction without feature i
			predWithout, err := e.getCoalitionPrediction(ctx, instance, coalition, predCache)
			if err != nil {
				return nil, err
			}

			// Get f(S ∪ {i}) - prediction with feature i
			coalitionWith := coalition | (1 << i)
			predWith, err := e.getCoalitionPrediction(ctx, instance, coalitionWith, predCache)
			if err != nil {
				return nil, err
			}

			// Add weighted marginal contribution
			weight := e.shapleyWeights[coalitionSize]
			shapValues[i] += weight * (predWith - predWithout)
		}
	}

	return shapValues, nil
}

// getCoalitionPrediction returns the model prediction for a coalition.
// Uses caching to avoid redundant predictions.
func (e *Explainer) getCoalitionPrediction(ctx context.Context, instance []float64, coalition uint64, cache map[uint64]float64) (float64, error) {
	if pred, ok := cache[coalition]; ok {
		return pred, nil
	}

	pred, err := e.evaluateCoalition(ctx, instance, coalition)
	if err != nil {
		return 0, err
	}

	cache[coalition] = pred
	return pred, nil
}

// evaluateCoalition computes the expected prediction for a coalition.
// Features in the coalition use instance values; others use background values.
func (e *Explainer) evaluateCoalition(ctx context.Context, instance []float64, coalition uint64) (float64, error) {
	// Use batched predictions if enabled
	if e.config.UseBatchedPredictions {
		return e.evaluateCoalitionBatched(ctx, instance, coalition)
	}

	numFeatures := len(instance)

	// Average prediction over all background samples
	totalPred := 0.0
	for _, bgSample := range e.background {
		input := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if coalition&(1<<j) != 0 {
				input[j] = instance[j]
			} else {
				input[j] = bgSample[j]
			}
		}

		pred, err := e.model.Predict(ctx, input)
		if err != nil {
			return 0, err
		}
		totalPred += pred
	}

	return totalPred / float64(len(e.background)), nil
}

// evaluateCoalitionBatched computes coalition prediction using batched model inference.
// This is more efficient when the model has optimized batch prediction.
func (e *Explainer) evaluateCoalitionBatched(ctx context.Context, instance []float64, coalition uint64) (float64, error) {
	numFeatures := len(instance)
	numBackground := len(e.background)

	// Build all masked inputs at once
	inputs := make([][]float64, numBackground)
	for b, bgSample := range e.background {
		input := make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if coalition&(1<<j) != 0 {
				input[j] = instance[j]
			} else {
				input[j] = bgSample[j]
			}
		}
		inputs[b] = input
	}

	// Batch prediction
	predictions, err := e.model.PredictBatch(ctx, inputs)
	if err != nil {
		return 0, err
	}

	// Average predictions
	totalPred := 0.0
	for _, pred := range predictions {
		totalPred += pred
	}

	return totalPred / float64(numBackground), nil
}

// buildCoalition converts a coalition index to a bitmask, skipping feature i.
//
// coalitionIdx encodes which of the (n-1) features (excluding i) are in the coalition.
// We map this to a full n-bit mask.
func buildCoalition(coalitionIdx uint64, skipFeature, numFeatures int) uint64 {
	var coalition uint64
	bitPos := 0

	for j := 0; j < numFeatures; j++ {
		if j == skipFeature {
			continue
		}
		if coalitionIdx&(1<<bitPos) != 0 {
			coalition |= 1 << j
		}
		bitPos++
	}

	return coalition
}

// popCount returns the number of 1 bits in x (population count / Hamming weight).
func popCount(x uint64) int {
	count := 0
	for x != 0 {
		count++
		x &= x - 1
	}
	return count
}

// computeShapleyWeights precomputes Shapley weights for all coalition sizes.
//
// weight[s] = s! * (n-s-1)! / n!
//
// These are the coefficients in the Shapley value formula.
func computeShapleyWeights(n int) []float64 {
	weights := make([]float64, n)

	// Precompute factorials
	factorial := make([]float64, n+1)
	factorial[0] = 1
	for i := 1; i <= n; i++ {
		factorial[i] = factorial[i-1] * float64(i)
	}

	nFactorial := factorial[n]

	for s := 0; s < n; s++ {
		// weight = s! * (n-s-1)! / n!
		weights[s] = factorial[s] * factorial[n-s-1] / nFactorial
	}

	return weights
}

// Ensure Explainer implements explainer.Explainer interface.
var _ explainer.Explainer = (*Explainer)(nil)
