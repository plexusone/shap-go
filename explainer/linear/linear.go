// Package linear provides LinearSHAP for exact SHAP values on linear models.
//
// LinearSHAP computes exact SHAP values for linear models using a closed-form
// solution. For a linear model f(x) = bias + Σᵢ wᵢxᵢ, the SHAP value for
// feature i is simply:
//
//	SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
//
// where E[xᵢ] is the mean of feature i in the background data.
//
// This is exact (not an approximation) and very fast - O(n) where n is the
// number of features.
package linear

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
)

// Common errors returned by LinearSHAP.
var (
	ErrNoWeights        = errors.New("weights cannot be empty")
	ErrNoBackground     = errors.New("background data cannot be empty")
	ErrFeatureMismatch  = errors.New("feature count mismatch")
	ErrInconsistentData = errors.New("inconsistent background data dimensions")
)

// Explainer implements LinearSHAP for linear models.
// It provides exact SHAP values using the closed-form solution.
type Explainer struct {
	weights      []float64 // Model weights (coefficients)
	bias         float64   // Model bias (intercept)
	featureMeans []float64 // Mean of each feature in background
	baseValue    float64   // E[f(x)] = bias + Σ wᵢ × E[xᵢ]
	config       explainer.Config
}

// New creates a new LinearSHAP explainer.
//
// Parameters:
//   - weights: The model coefficients (one per feature)
//   - bias: The model intercept/bias term
//   - background: Representative samples from the data distribution
//   - opts: Optional configuration options
//
// For logistic regression, pass the log-odds coefficients. The SHAP values
// will explain contributions to the log-odds (before sigmoid transformation).
func New(weights []float64, bias float64, background [][]float64, opts ...explainer.Option) (*Explainer, error) {
	if len(weights) == 0 {
		return nil, ErrNoWeights
	}
	if len(background) == 0 {
		return nil, ErrNoBackground
	}

	numFeatures := len(weights)

	// Validate background data dimensions
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("%w: row %d has %d features, expected %d",
				ErrInconsistentData, i, len(row), numFeatures)
		}
	}

	// Compute feature means from background
	featureMeans := computeMeans(background, numFeatures)

	// Compute base value: E[f(x)] = bias + Σ wᵢ × E[xᵢ]
	baseValue := bias
	for i, w := range weights {
		baseValue += w * featureMeans[i]
	}

	// Apply configuration
	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

	return &Explainer{
		weights:      weights,
		bias:         bias,
		featureMeans: featureMeans,
		baseValue:    baseValue,
		config:       config,
	}, nil
}

// NewFromModel creates a LinearSHAP explainer from weight and bias slices.
// This is a convenience constructor for models that provide coefficients directly.
func NewFromModel(weights []float64, bias float64, background [][]float64, opts ...explainer.Option) (*Explainer, error) {
	return New(weights, bias, background, opts...)
}

// Explain computes exact SHAP values for a single instance.
//
// For each feature i: SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
//
// This satisfies local accuracy: Σ SHAP[i] = f(x) - E[f(x)]
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	start := time.Now()

	if len(instance) != len(e.weights) {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d",
			ErrFeatureMismatch, len(instance), len(e.weights))
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Compute prediction: f(x) = bias + Σ wᵢ × xᵢ
	// Note: len(instance) == len(e.weights) is validated above
	prediction := e.bias
	for i, w := range e.weights {
		prediction += w * instance[i] //nolint:gosec // bounds checked above
	}

	// Compute SHAP values: SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
	shapValues := make(map[string]float64, len(e.weights))
	featureValues := make(map[string]float64, len(e.weights))

	for i, w := range e.weights {
		name := e.config.FeatureNames[i]
		shapValues[name] = w * (instance[i] - e.featureMeans[i]) //nolint:gosec // bounds checked above
		featureValues[name] = instance[i]                        //nolint:gosec // bounds checked above
	}

	elapsed := time.Since(start)

	return &explanation.Explanation{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Values:        shapValues,
		FeatureNames:  e.config.FeatureNames,
		FeatureValues: featureValues,
		ModelID:       e.config.ModelID,
		Timestamp:     time.Now(),
		Metadata: explanation.ExplanationMetadata{
			Algorithm:     "linear",
			ComputeTimeMS: elapsed.Milliseconds(),
		},
	}, nil
}

// ExplainBatch computes SHAP values for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))

	for i, instance := range instances {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		exp, err := e.Explain(ctx, instance)
		if err != nil {
			return nil, fmt.Errorf("instance %d: %w", i, err)
		}
		results[i] = exp
	}

	return results, nil
}

// BaseValue returns the expected model output E[f(x)].
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// Weights returns the model weights.
func (e *Explainer) Weights() []float64 {
	result := make([]float64, len(e.weights))
	copy(result, e.weights)
	return result
}

// Bias returns the model bias/intercept.
func (e *Explainer) Bias() float64 {
	return e.bias
}

// FeatureMeans returns the mean of each feature in the background data.
func (e *Explainer) FeatureMeans() []float64 {
	result := make([]float64, len(e.featureMeans))
	copy(result, e.featureMeans)
	return result
}

// NumFeatures returns the number of features.
func (e *Explainer) NumFeatures() int {
	return len(e.weights)
}

// computeMeans computes the mean of each feature across background samples.
func computeMeans(background [][]float64, numFeatures int) []float64 {
	means := make([]float64, numFeatures)
	n := float64(len(background))

	for _, row := range background {
		for i, v := range row {
			means[i] += v
		}
	}

	for i := range means {
		means[i] /= n
	}

	return means
}
