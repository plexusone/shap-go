package permutation

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

func TestPermutationExplainer_LinearModel(t *testing.T) {
	// Create a linear model: y = x0 + x1 + x2
	fm := model.NewFuncModel(linearModel, 3)

	// Background data: all zeros
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	// Create explainer
	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
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

	// Verify local accuracy - permutation with antithetic sampling should be exact
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}

	// For a linear model with single background, SHAP values should equal feature values exactly
	if math.Abs(explanation.Values["x0"]-1.0) > 1e-10 {
		t.Errorf("SHAP(x0) = %f, want 1.0", explanation.Values["x0"])
	}
	if math.Abs(explanation.Values["x1"]-2.0) > 1e-10 {
		t.Errorf("SHAP(x1) = %f, want 2.0", explanation.Values["x1"])
	}
	if math.Abs(explanation.Values["x2"]-3.0) > 1e-10 {
		t.Errorf("SHAP(x2) = %f, want 3.0", explanation.Values["x2"])
	}
}

func TestPermutationExplainer_WeightedModel(t *testing.T) {
	// Create a weighted model: y = 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background: mean is [0, 0]
	background := [][]float64{
		{0.0, 0.0},
	}

	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
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
	// SHAP(x0) should be exactly 2.0 (weight * value)
	// SHAP(x1) should be exactly 3.0 (weight * value)
	if math.Abs(explanation.Values["feature_0"]-2.0) > 1e-10 {
		t.Errorf("SHAP(feature_0) = %f, want 2.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > 1e-10 {
		t.Errorf("SHAP(feature_1) = %f, want 3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestPermutationExplainer_MultipleBackground(t *testing.T) {
	// Test with multiple background samples
	fm := model.NewFuncModel(linearModel, 2)

	background := [][]float64{
		{0.0, 0.0},
		{2.0, 2.0},
	}

	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Base value should be mean of background predictions: (0+4)/2 = 2
	if math.Abs(exp.BaseValue()-2.0) > 1e-10 {
		t.Errorf("BaseValue() = %f, want 2.0", exp.BaseValue())
	}

	ctx := context.Background()
	instance := []float64{3.0, 3.0} // Prediction = 6

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Verify local accuracy
	result := explanation.Verify(0.1) // Allow some tolerance due to averaging over backgrounds
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestPermutationExplainer_AntitheticSampling(t *testing.T) {
	// Test that antithetic sampling reduces variance
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	// Run multiple times and check variance
	instance := []float64{1.0, 2.0, 3.0}
	ctx := context.Background()

	var results []float64
	for seed := int64(0); seed < 10; seed++ {
		exp, err := New(fm, background,
			explainer.WithNumSamples(50),
			explainer.WithSeed(seed),
		)
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}

		explanation, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Explain() error = %v", err)
		}

		// Check local accuracy for each run
		result := explanation.Verify(1e-10)
		if !result.Valid {
			t.Errorf("Local accuracy failed for seed %d", seed)
		}

		results = append(results, explanation.Values["feature_0"])
	}

	// All results should be exactly 1.0 for linear model
	for i, r := range results {
		if math.Abs(r-1.0) > 1e-10 {
			t.Errorf("Run %d: SHAP(feature_0) = %f, want 1.0", i, r)
		}
	}
}

func TestPermutationExplainer_Parallel(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithNumWorkers(4),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestPermutationExplainer_ExplainBatch(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
	}

	explanations, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch() error = %v", err)
	}

	if len(explanations) != 3 {
		t.Fatalf("ExplainBatch() returned %d explanations, want 3", len(explanations))
	}

	for i, e := range explanations {
		result := e.Verify(1e-10)
		if !result.Valid {
			t.Errorf("Explanation[%d] local accuracy failed", i)
		}
	}
}

func TestPermutationExplainer_Metadata(t *testing.T) {
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
	if explanation.Metadata.Algorithm != "permutation" {
		t.Errorf("Metadata.Algorithm = %s, want permutation", explanation.Metadata.Algorithm)
	}
	if explanation.Metadata.NumSamples != 50 {
		t.Errorf("Metadata.NumSamples = %d, want 50", explanation.Metadata.NumSamples)
	}
}

func TestPermutationExplainer_ContextCancellation(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(10000),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = exp.Explain(ctx, []float64{1.0, 1.0})
	if err == nil {
		t.Error("Explain() should error with cancelled context")
	}
}

func TestNew_Errors(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)

	tests := []struct {
		name       string
		model      model.Model
		background [][]float64
		wantErr    bool
	}{
		{
			name:       "nil model",
			model:      nil,
			background: [][]float64{{1.0}},
			wantErr:    true,
		},
		{
			name:       "empty background",
			model:      fm,
			background: [][]float64{},
			wantErr:    true,
		},
		{
			name:       "mismatched features",
			model:      fm,
			background: [][]float64{{1.0, 2.0, 3.0}},
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.model, tt.background)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPermutationExplainer_ConfidenceIntervals(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	// Create explainer with confidence intervals enabled
	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
		explainer.WithSeed(42),
		explainer.WithConfidenceLevel(0.95),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check that confidence intervals are present
	if !explanation.HasConfidenceIntervals() {
		t.Fatal("Expected confidence intervals to be present")
	}

	ci := explanation.Metadata.ConfidenceIntervals
	if ci.Level != 0.95 {
		t.Errorf("Confidence level = %f, want 0.95", ci.Level)
	}

	// Check that intervals are computed for all features
	for _, name := range exp.FeatureNames() {
		lower, ok := ci.Lower[name]
		if !ok {
			t.Errorf("Missing lower bound for %s", name)
		}
		upper, ok := ci.Upper[name]
		if !ok {
			t.Errorf("Missing upper bound for %s", name)
		}
		se, ok := ci.StandardErrors[name]
		if !ok {
			t.Errorf("Missing standard error for %s", name)
		}

		// Lower should be less than or equal to upper
		if lower > upper {
			t.Errorf("Lower bound %f > upper bound %f for %s", lower, upper, name)
		}

		// SHAP value should be within the interval
		shapVal := explanation.Values[name]
		if shapVal < lower || shapVal > upper {
			t.Errorf("SHAP value %f not in interval [%f, %f] for %s",
				shapVal, lower, upper, name)
		}

		// Standard error should be non-negative
		if se < 0 {
			t.Errorf("Standard error %f is negative for %s", se, name)
		}
	}

	// Test GetConfidenceInterval helper
	lower, upper, ok := explanation.GetConfidenceInterval("feature_0")
	if !ok {
		t.Error("GetConfidenceInterval returned false for feature_0")
	}
	if lower != ci.Lower["feature_0"] || upper != ci.Upper["feature_0"] {
		t.Error("GetConfidenceInterval values don't match")
	}
}

func TestPermutationExplainer_ConfidenceIntervals_Parallel(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	// Test confidence intervals with parallel workers
	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
		explainer.WithSeed(42),
		explainer.WithConfidenceLevel(0.95),
		explainer.WithNumWorkers(4),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check that confidence intervals are present
	if !explanation.HasConfidenceIntervals() {
		t.Fatal("Expected confidence intervals to be present with parallel workers")
	}

	ci := explanation.Metadata.ConfidenceIntervals
	if ci.Level != 0.95 {
		t.Errorf("Confidence level = %f, want 0.95", ci.Level)
	}

	// Verify structure
	if len(ci.Lower) != 2 || len(ci.Upper) != 2 || len(ci.StandardErrors) != 2 {
		t.Errorf("Unexpected CI map sizes: lower=%d, upper=%d, se=%d",
			len(ci.Lower), len(ci.Upper), len(ci.StandardErrors))
	}
}

func TestPermutationExplainer_NoConfidenceIntervals(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	// Create explainer WITHOUT confidence intervals (default)
	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check that confidence intervals are NOT present
	if explanation.HasConfidenceIntervals() {
		t.Error("Expected no confidence intervals when not configured")
	}
	if explanation.Metadata.ConfidenceIntervals != nil {
		t.Error("ConfidenceIntervals should be nil when not configured")
	}
}
