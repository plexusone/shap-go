package additive

import (
	"context"
	"fmt"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Explainer implements SHAP value computation for additive models (GAMs).
//
// For additive models where f(x) = Σ fᵢ(xᵢ), SHAP values are computed exactly as:
//
//	φᵢ = fᵢ(xᵢ) - E[fᵢ(Xᵢ)]
//
// This is much faster than model-agnostic methods since no sampling is required.
type Explainer struct {
	model        model.Model
	background   [][]float64
	reference    []float64 // mean of background data
	baseValue    float64
	featureNames []string
	config       explainer.Config

	// Precomputed expected effects for each feature: E[fᵢ(Xᵢ)]
	// expectedEffects[i] = mean effect of feature i over background
	expectedEffects []float64
}

// New creates a new AdditiveExplainer.
//
// Parameters:
//   - m: The additive model to explain (must have no feature interactions)
//   - background: Representative samples for computing expected feature effects
//   - opts: Configuration options (WithFeatureNames, WithModelID, etc.)
//
// The explainer precomputes expected feature effects from the background data,
// making subsequent Explain calls very fast.
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

	// Apply configuration
	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

	// Compute reference point (mean of background)
	reference := computeMean(background, numFeatures)

	// Compute base value: E[f(X)] = mean prediction over background
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

	// Precompute expected effects for each feature
	expectedEffects, err := computeExpectedEffects(ctx, m, background, reference)
	if err != nil {
		return nil, fmt.Errorf("failed to compute expected effects: %w", err)
	}

	return &Explainer{
		model:           m,
		background:      background,
		reference:       reference,
		baseValue:       baseValue,
		featureNames:    config.FeatureNames,
		config:          config,
		expectedEffects: expectedEffects,
	}, nil
}

// computeMean computes the element-wise mean of the background data.
func computeMean(background [][]float64, numFeatures int) []float64 {
	mean := make([]float64, numFeatures)
	n := float64(len(background))

	for _, row := range background {
		for j, val := range row {
			mean[j] += val
		}
	}

	for j := range mean {
		mean[j] /= n
	}

	return mean
}

// computeExpectedEffects precomputes E[fᵢ(Xᵢ)] for each feature.
//
// For each feature i, we compute the mean effect by:
// 1. For each background sample, create input with only feature i from that sample
// 2. Compute the model prediction
// 3. Subtract the base value (prediction at reference)
// 4. Average across all background samples
func computeExpectedEffects(ctx context.Context, m model.Model, background [][]float64, reference []float64) ([]float64, error) {
	numFeatures := len(reference)
	numBackground := len(background)
	expectedEffects := make([]float64, numFeatures)

	// Get base prediction at reference point
	basePred, err := m.Predict(ctx, reference)
	if err != nil {
		return nil, err
	}

	// For each feature, compute its expected effect
	for i := 0; i < numFeatures; i++ {
		sumEffect := 0.0

		for _, bgSample := range background {
			// Create input with only feature i from background sample
			input := make([]float64, numFeatures)
			copy(input, reference)
			input[i] = bgSample[i]

			pred, err := m.Predict(ctx, input)
			if err != nil {
				return nil, err
			}

			// Effect = prediction - base prediction
			sumEffect += pred - basePred
		}

		expectedEffects[i] = sumEffect / float64(numBackground)
	}

	return expectedEffects, nil
}

// Explain computes SHAP values for a single instance.
//
// For additive models, this is O(n) where n is the number of features.
// The SHAP value for feature i is: φᵢ = effect_i(instance) - E[effect_i]
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.model.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("instance has %d features, expected %d",
			len(instance), numFeatures)
	}

	// Get prediction for this instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute SHAP values
	shapValues, err := e.computeSHAPValues(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to compute SHAP values: %w", err)
	}

	// Build result
	values := make(map[string]float64, numFeatures)
	featureValues := make(map[string]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		name := e.featureNames[i]
		values[name] = shapValues[i]
		featureValues[name] = instance[i]
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
			Algorithm:      "additive",
			NumSamples:     numFeatures, // One evaluation per feature
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}, nil
}

// computeSHAPValues computes SHAP values for the given instance.
//
// φᵢ = effect_i(instance) - E[effect_i]
//
// where effect_i(x) = f(ref₀, ..., xᵢ, ..., refₙ) - f(ref)
func (e *Explainer) computeSHAPValues(ctx context.Context, instance []float64) ([]float64, error) {
	numFeatures := len(instance)
	shapValues := make([]float64, numFeatures)

	// Get base prediction at reference
	basePred, err := e.model.Predict(ctx, e.reference)
	if err != nil {
		return nil, err
	}

	// For each feature, compute its effect at the instance value
	for i := 0; i < numFeatures; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Create input with only feature i from instance
		input := make([]float64, numFeatures)
		copy(input, e.reference)
		input[i] = instance[i]

		pred, err := e.model.Predict(ctx, input)
		if err != nil {
			return nil, err
		}

		// Effect at instance value
		instanceEffect := pred - basePred

		// SHAP value = instance effect - expected effect
		shapValues[i] = instanceEffect - e.expectedEffects[i]
	}

	return shapValues, nil
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

// BaseValue returns the expected model output (prediction at the reference point).
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// FeatureNames returns the names of the features.
func (e *Explainer) FeatureNames() []string {
	return e.featureNames
}

// Reference returns the reference point (mean of background data).
func (e *Explainer) Reference() []float64 {
	return e.reference
}

// ExpectedEffects returns the precomputed expected effects for each feature.
// These are E[fᵢ(Xᵢ)] computed from the background data.
func (e *Explainer) ExpectedEffects() []float64 {
	return e.expectedEffects
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
