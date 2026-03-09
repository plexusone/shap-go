package sampling

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model"
)

// linearModel implements a simple linear model: y = sum(x)
func linearModel(ctx context.Context, input []float64) (float64, error) {
	var sum float64
	for _, v := range input {
		sum += v
	}
	return sum, nil
}

// weightedModel implements: y = w[0]*x[0] + w[1]*x[1] + ...
type weightedModel struct {
	weights []float64
}

func (m *weightedModel) predict(ctx context.Context, input []float64) (float64, error) {
	var sum float64
	for i, v := range input {
		sum += m.weights[i] * v
	}
	return sum, nil
}

func TestSamplingExplainer_LinearModel(t *testing.T) {
	// Create a linear model: y = x0 + x1 + x2
	fm := model.NewFuncModel(linearModel, 3)

	// Background data: all zeros
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	// Create explainer
	exp, err := New(fm, background,
		explainer.WithNumSamples(500),
		explainer.WithSeed(42),
		explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Explain instance [1, 2, 3]
	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-6.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 6.0", explanation.Prediction)
	}

	// Check base value (should be 0 since background is all zeros)
	if math.Abs(explanation.BaseValue-0.0) > 1e-10 {
		t.Errorf("BaseValue = %f, want 0.0", explanation.BaseValue)
	}

	// Verify local accuracy
	result := explanation.Verify(0.1) // Allow some tolerance for Monte Carlo
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}

	// For a linear model, SHAP values should equal feature values
	// (with some Monte Carlo noise)
	tolerance := 0.2 // Monte Carlo has variance
	if math.Abs(explanation.Values["x0"]-1.0) > tolerance {
		t.Errorf("SHAP(x0) = %f, want ~1.0", explanation.Values["x0"])
	}
	if math.Abs(explanation.Values["x1"]-2.0) > tolerance {
		t.Errorf("SHAP(x1) = %f, want ~2.0", explanation.Values["x1"])
	}
	if math.Abs(explanation.Values["x2"]-3.0) > tolerance {
		t.Errorf("SHAP(x2) = %f, want ~3.0", explanation.Values["x2"])
	}
}

func TestSamplingExplainer_WeightedModel(t *testing.T) {
	// Create a weighted model: y = 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background: mean is [0, 0]
	background := [][]float64{
		{0.0, 0.0},
	}

	exp, err := New(fm, background,
		explainer.WithNumSamples(500),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 1.0} // Prediction = 2*1 + 3*1 = 5

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// For linear model with all-zero background:
	// SHAP(x0) should be 2.0 (weight * value)
	// SHAP(x1) should be 3.0 (weight * value)
	tolerance := 0.3
	if math.Abs(explanation.Values["feature_0"]-2.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want ~2.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want ~3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(0.1)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestSamplingExplainer_BaseValue(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)

	// Background with non-zero mean: mean = [1, 2] => base value = 3
	background := [][]float64{
		{0.0, 0.0},
		{2.0, 4.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Base value should be mean prediction on background
	// mean([0, 0]) = 0, mean([2, 4]) = 6 => base = 3
	if math.Abs(exp.BaseValue()-3.0) > 1e-10 {
		t.Errorf("BaseValue() = %f, want 3.0", exp.BaseValue())
	}
}

func TestSamplingExplainer_FeatureNames(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	// With custom names
	exp, err := New(fm, background,
		explainer.WithFeatureNames([]string{"a", "b", "c"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	names := exp.FeatureNames()
	if len(names) != 3 || names[0] != "a" || names[1] != "b" || names[2] != "c" {
		t.Errorf("FeatureNames() = %v, want [a, b, c]", names)
	}
}

func TestSamplingExplainer_ExplainBatch(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{1.0, 1.0},
		{2.0, 2.0},
	}

	explanations, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch() error = %v", err)
	}

	if len(explanations) != 2 {
		t.Fatalf("ExplainBatch() returned %d explanations, want 2", len(explanations))
	}

	if math.Abs(explanations[0].Prediction-2.0) > 1e-10 {
		t.Errorf("Explanation[0].Prediction = %f, want 2.0", explanations[0].Prediction)
	}
	if math.Abs(explanations[1].Prediction-4.0) > 1e-10 {
		t.Errorf("Explanation[1].Prediction = %f, want 4.0", explanations[1].Prediction)
	}
}

func TestSamplingExplainer_Metadata(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithModelID("test-model"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 1.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if explanation.ModelID != "test-model" {
		t.Errorf("ModelID = %s, want test-model", explanation.ModelID)
	}
	if explanation.Metadata.Algorithm != "sampling" {
		t.Errorf("Metadata.Algorithm = %s, want sampling", explanation.Metadata.Algorithm)
	}
	if explanation.Metadata.NumSamples != 50 {
		t.Errorf("Metadata.NumSamples = %d, want 50", explanation.Metadata.NumSamples)
	}
	if explanation.Metadata.BackgroundSize != 1 {
		t.Errorf("Metadata.BackgroundSize = %d, want 1", explanation.Metadata.BackgroundSize)
	}
}

func TestSamplingExplainer_ContextCancellation(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(10000), // Many samples
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.Explain(ctx, []float64{1.0, 1.0})
	if err == nil {
		t.Error("Explain() should error with cancelled context")
	}
}

func TestNew_NilModel(t *testing.T) {
	_, err := New(nil, [][]float64{{1.0}})
	if err == nil {
		t.Error("New() should error with nil model")
	}
}

func TestNew_EmptyBackground(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 1)
	_, err := New(fm, [][]float64{})
	if err == nil {
		t.Error("New() should error with empty background")
	}
}

func TestNew_MismatchedFeatures(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{1.0, 2.0}} // Only 2 features

	_, err := New(fm, background)
	if err == nil {
		t.Error("New() should error with mismatched features")
	}
}
