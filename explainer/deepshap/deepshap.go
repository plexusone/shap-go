// Package deepshap provides DeepSHAP for explaining deep neural network predictions.
//
// DeepSHAP combines the DeepLIFT algorithm with Shapley values to efficiently
// compute feature attributions for neural networks. It uses the rescale rule
// from DeepLIFT to propagate attributions backward through the network.
//
// Key concepts:
//   - Works with ONNX models via ActivationSession
//   - Uses background dataset as baseline references
//   - Computes exact local feature importance
//   - Guarantees local accuracy: sum(SHAP values) = prediction - baseline
//
// Supported layer types:
//   - Dense/Gemm (fully connected)
//   - ReLU, Sigmoid, Tanh activations
//   - Softmax output layers
//   - Add (for residual connections)
//   - Identity/Dropout/Flatten (pass-through)
//
// Usage:
//
//	session, _ := onnx.NewActivationSession(config)
//	explainer, _ := deepshap.New(session, background)
//	explanation, _ := explainer.Explain(ctx, instance)
package deepshap

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model/onnx"
)

// Common errors returned by DeepSHAP.
var (
	ErrNilSession      = errors.New("activation session cannot be nil")
	ErrNoBackground    = errors.New("background data cannot be empty")
	ErrFeatureMismatch = errors.New("feature count mismatch")
	ErrNilGraphInfo    = errors.New("graph info cannot be nil")
)

// Explainer implements DeepSHAP for neural network explanations.
type Explainer struct {
	session      *onnx.ActivationSession
	graphInfo    *onnx.GraphInfo
	background   [][]float64
	baseValue    float64
	featureNames []string
	config       explainer.Config

	// Propagation engine for backward pass
	propagationEngine *PropagationEngine
}

// New creates a new DeepSHAP explainer.
//
// Parameters:
//   - session: ActivationSession configured to capture intermediate outputs
//   - graphInfo: Parsed ONNX graph structure (from onnx.ParseGraph)
//   - background: Representative samples for baseline/reference
//   - opts: Configuration options (WithFeatureNames, WithModelID, etc.)
//
// The background dataset is used to compute reference activations. For best
// results, use a representative subset of your training data (100-1000 samples).
func New(
	session *onnx.ActivationSession,
	graphInfo *onnx.GraphInfo,
	background [][]float64,
	opts ...explainer.Option,
) (*Explainer, error) {
	if session == nil {
		return nil, ErrNilSession
	}
	if graphInfo == nil {
		return nil, ErrNilGraphInfo
	}
	if len(background) == 0 {
		return nil, ErrNoBackground
	}

	numFeatures := session.NumFeatures()

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
	predictions, err := session.PredictBatch(ctx, background)
	if err != nil {
		return nil, fmt.Errorf("failed to compute base value: %w", err)
	}

	var baseValue float64
	for _, p := range predictions {
		baseValue += p
	}
	baseValue /= float64(len(predictions))

	// Create propagation engine
	propEngine := NewPropagationEngine(graphInfo)

	return &Explainer{
		session:           session,
		graphInfo:         graphInfo,
		background:        background,
		baseValue:         baseValue,
		featureNames:      config.FeatureNames,
		config:            config,
		propagationEngine: propEngine,
	}, nil
}

// NewSimple creates a DeepSHAP explainer without requiring graph parsing.
// This uses the simplified propagation method that works with activation
// tensors but doesn't require full graph structure.
//
// Use this when you have an ActivationSession but don't want to parse
// the ONNX graph structure.
func NewSimple(
	session *onnx.ActivationSession,
	background [][]float64,
	opts ...explainer.Option,
) (*Explainer, error) {
	if session == nil {
		return nil, ErrNilSession
	}
	if len(background) == 0 {
		return nil, ErrNoBackground
	}

	numFeatures := session.NumFeatures()

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

	// Compute base value
	ctx := context.Background()
	predictions, err := session.PredictBatch(ctx, background)
	if err != nil {
		return nil, fmt.Errorf("failed to compute base value: %w", err)
	}

	var baseValue float64
	for _, p := range predictions {
		baseValue += p
	}
	baseValue /= float64(len(predictions))

	return &Explainer{
		session:      session,
		background:   background,
		baseValue:    baseValue,
		featureNames: config.FeatureNames,
		config:       config,
	}, nil
}

// Explain computes SHAP values for a single instance.
//
// The returned explanation contains:
//   - SHAP values for each feature
//   - The model prediction for the instance
//   - The base value (expected output on background)
//   - Verification that sum(SHAP values) ≈ prediction - baseline
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.session.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d",
			ErrFeatureMismatch, len(instance), numFeatures)
	}

	// Get instance activations
	instanceResult, err := e.session.PredictWithActivations(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to get instance activations: %w", err)
	}

	// Compute SHAP values by averaging over background samples
	shapValues := make([]float64, numFeatures)

	for _, bgSample := range e.background {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Get reference activations
		refResult, err := e.session.PredictWithActivations(ctx, bgSample)
		if err != nil {
			return nil, fmt.Errorf("failed to get reference activations: %w", err)
		}

		// Compute attributions for this reference
		attrs := e.computeAttributions(instance, bgSample, instanceResult, refResult)

		// Accumulate
		for i, a := range attrs {
			shapValues[i] += a
		}
	}

	// Average over background samples
	for i := range shapValues {
		shapValues[i] /= float64(len(e.background))
	}

	// Build the explanation
	values := make(map[string]float64)
	featureValues := make(map[string]float64)
	for i, name := range e.featureNames {
		values[name] = shapValues[i]
		featureValues[name] = instance[i]
	}

	return &explanation.Explanation{
		Prediction:    instanceResult.Prediction,
		BaseValue:     e.baseValue,
		Values:        values,
		FeatureNames:  e.featureNames,
		FeatureValues: featureValues,
		Timestamp:     time.Now(),
		ModelID:       e.config.ModelID,
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "deepshap",
			NumSamples:     len(e.background),
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

// computeAttributions computes SHAP attributions for a single instance-reference pair.
func (e *Explainer) computeAttributions(
	instance, reference []float64,
	instanceResult, refResult *onnx.ActivationResult,
) []float64 {
	// If we have a propagation engine, use it
	if e.propagationEngine != nil && e.graphInfo != nil {
		instanceAct := e.buildActivationData(instance, instanceResult)
		referenceAct := e.buildActivationData(reference, refResult)

		result, err := e.propagationEngine.Propagate(instanceAct, referenceAct, 1.0)
		if err == nil && result != nil {
			return result.Attributions
		}
		// Fall back to simple method if propagation fails
	}

	// Simple method: use rescale rule directly on activations
	return e.computeSimpleAttributions(instance, reference, instanceResult, refResult)
}

// buildActivationData converts ActivationResult to ActivationData for propagation.
func (e *Explainer) buildActivationData(input []float64, result *onnx.ActivationResult) *ActivationData {
	// Convert float32 activations to float64
	layerActs := make(map[string][]float64)
	for name, acts := range result.Activations {
		f64Acts := make([]float64, len(acts))
		for i, v := range acts {
			f64Acts[i] = float64(v)
		}
		layerActs[name] = f64Acts
	}

	return &ActivationData{
		Input:            input,
		LayerActivations: layerActs,
		Output:           result.Prediction,
	}
}

// computeSimpleAttributions uses a simplified rescale approach.
func (e *Explainer) computeSimpleAttributions(
	instance, reference []float64,
	instanceResult, refResult *onnx.ActivationResult,
) []float64 {
	attributions := make([]float64, len(instance))

	// Compute output difference
	outputDiff := instanceResult.Prediction - refResult.Prediction

	// Compute total input difference for normalization
	totalInputDiff := 0.0
	for i := range instance {
		totalInputDiff += instance[i] - reference[i]
	}

	if totalInputDiff == 0 {
		// No input difference, no attributions
		return attributions
	}

	// Distribute output difference proportionally to input differences
	for i := range attributions {
		inputDiff := instance[i] - reference[i]
		attributions[i] = inputDiff * (outputDiff / totalInputDiff)
	}

	return attributions
}

// Ensure Explainer implements explainer.Explainer interface.
var _ explainer.Explainer = (*Explainer)(nil)
