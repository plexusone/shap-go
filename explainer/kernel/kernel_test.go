package kernel

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
	bias    float64
}

func (m *weightedModel) predict(ctx context.Context, input []float64) (float64, error) {
	sum := m.bias
	for i, v := range input {
		if i < len(m.weights) {
			sum += m.weights[i] * v
		}
	}
	return sum, nil
}

func TestKernelExplainer_LinearModel(t *testing.T) {
	// Create a linear model: y = x0 + x1 + x2
	fm := model.NewFuncModel(linearModel, 3)

	// Background data: all zeros
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	// Create explainer
	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
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

	// Verify local accuracy - KernelSHAP with constraint should satisfy this exactly
	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}

	// For a linear model with zero background, SHAP values should equal feature values
	tolerance := 0.15 // KernelSHAP should be fairly accurate
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

func TestKernelExplainer_WeightedModel(t *testing.T) {
	// Create a weighted model: y = 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background: mean is [0, 0]
	background := [][]float64{
		{0.0, 0.0},
	}

	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
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
	tolerance := 0.2
	if math.Abs(explanation.Values["feature_0"]-2.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want ~2.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want ~3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestKernelExplainer_WithBias(t *testing.T) {
	// Model: y = 10 + 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}, bias: 10.0}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background with feature means [1, 1]
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

	// Base value = 10 + 2*1 + 3*1 = 15
	expectedBaseValue := 15.0
	if math.Abs(exp.BaseValue()-expectedBaseValue) > 1e-6 {
		t.Errorf("BaseValue() = %f, want %f", exp.BaseValue(), expectedBaseValue)
	}

	ctx := context.Background()
	instance := []float64{3.0, 2.0} // Prediction = 10 + 2*3 + 3*2 = 22

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// SHAP values: deviation from mean * weight
	// SHAP(x0) = 2 * (3 - 1) = 4
	// SHAP(x1) = 3 * (2 - 1) = 3
	tolerance := 0.3
	if math.Abs(explanation.Values["feature_0"]-4.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want ~4.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want ~3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestKernelExplainer_LocalAccuracyAlwaysSatisfied(t *testing.T) {
	// Test that local accuracy is always satisfied due to constraint-based solving
	fm := model.NewFuncModel(linearModel, 4)

	background := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{4.0, 3.0, 2.0, 1.0},
	}

	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()

	// Test multiple instances
	instances := [][]float64{
		{5.0, 5.0, 5.0, 5.0},
		{0.0, 0.0, 0.0, 0.0},
		{10.0, 0.0, 10.0, 0.0},
	}

	for i, instance := range instances {
		explanation, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Instance %d: Explain() error = %v", i, err)
		}

		// Local accuracy should be satisfied exactly
		result := explanation.Verify(1e-6)
		if !result.Valid {
			t.Errorf("Instance %d: Local accuracy failed: sum=%f, expected=%f, diff=%f",
				i, result.SumSHAP, result.Expected, result.Difference)
		}
	}
}

func TestKernelExplainer_SingleFeature(t *testing.T) {
	// Edge case: single feature model
	fm := model.NewFuncModel(linearModel, 1)
	background := [][]float64{{0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{5.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// For single feature, SHAP value equals prediction - base value
	if math.Abs(explanation.Values["feature_0"]-5.0) > 1e-6 {
		t.Errorf("SHAP(feature_0) = %f, want 5.0", explanation.Values["feature_0"])
	}

	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed: diff=%f", result.Difference)
	}
}

func TestKernelExplainer_TwoFeatures(t *testing.T) {
	// Edge case: two features
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
	explanation, err := exp.Explain(ctx, []float64{3.0, 7.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-10.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 10.0", explanation.Prediction)
	}

	// SHAP values should be close to feature values for linear model
	tolerance := 0.3
	if math.Abs(explanation.Values["feature_0"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want ~3.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-7.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want ~7.0", explanation.Values["feature_1"])
	}

	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed: diff=%f", result.Difference)
	}
}

func TestKernelExplainer_BaseValue(t *testing.T) {
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

func TestKernelExplainer_FeatureNames(t *testing.T) {
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

func TestKernelExplainer_ExplainBatch(t *testing.T) {
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

	// Both should satisfy local accuracy
	for i, exp := range explanations {
		result := exp.Verify(1e-6)
		if !result.Valid {
			t.Errorf("Explanation[%d]: Local accuracy failed: diff=%f", i, result.Difference)
		}
	}
}

func TestKernelExplainer_Metadata(t *testing.T) {
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
	if explanation.Metadata.Algorithm != "kernel" {
		t.Errorf("Metadata.Algorithm = %s, want kernel", explanation.Metadata.Algorithm)
	}
	if explanation.Metadata.NumSamples != 50 {
		t.Errorf("Metadata.NumSamples = %d, want 50", explanation.Metadata.NumSamples)
	}
	if explanation.Metadata.BackgroundSize != 1 {
		t.Errorf("Metadata.BackgroundSize = %d, want 1", explanation.Metadata.BackgroundSize)
	}
}

func TestKernelExplainer_ParallelWorkers(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithNumSamples(200),
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

	// Should still satisfy local accuracy
	result := explanation.Verify(1e-6)
	if !result.Valid {
		t.Errorf("Local accuracy failed with parallel workers: diff=%f", result.Difference)
	}

	// SHAP values should be reasonable
	tolerance := 0.3
	if math.Abs(explanation.Values["feature_0"]-1.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want ~1.0", explanation.Values["feature_0"])
	}
}

func TestKernelExplainer_ContextCancellation(t *testing.T) {
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

func TestKernelExplainer_Explain_MismatchedInstance(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	_, err = exp.Explain(ctx, []float64{1.0, 2.0}) // Only 2 features, expected 3
	if err == nil {
		t.Error("Explain() should error with mismatched instance size")
	}
}

// Test kernel weight computation
func TestShapleyKernelWeight(t *testing.T) {
	testCases := []struct {
		numFeatures   int
		coalitionSize int
		wantZero      bool
	}{
		{5, 0, true},  // Empty coalition
		{5, 5, true},  // Full coalition
		{5, 1, false}, // Size 1
		{5, 4, false}, // Size M-1
		{5, 2, false}, // Size 2
	}

	for _, tc := range testCases {
		weight := shapleyKernelWeight(tc.numFeatures, tc.coalitionSize)
		if tc.wantZero && weight != 0 {
			t.Errorf("shapleyKernelWeight(%d, %d) = %f, want 0", tc.numFeatures, tc.coalitionSize, weight)
		}
		if !tc.wantZero && weight <= 0 {
			t.Errorf("shapleyKernelWeight(%d, %d) = %f, want > 0", tc.numFeatures, tc.coalitionSize, weight)
		}
	}

	// Verify symmetry: w(s) == w(M-s)
	for numFeatures := 3; numFeatures <= 6; numFeatures++ {
		for s := 1; s < numFeatures; s++ {
			w1 := shapleyKernelWeight(numFeatures, s)
			w2 := shapleyKernelWeight(numFeatures, numFeatures-s)
			if math.Abs(w1-w2) > 1e-10 {
				t.Errorf("Kernel weight not symmetric: w(%d, %d)=%f != w(%d, %d)=%f",
					numFeatures, s, w1, numFeatures, numFeatures-s, w2)
			}
		}
	}
}

// Test binomial coefficient
func TestBinomialCoefficient(t *testing.T) {
	testCases := []struct {
		n, k int
		want float64
	}{
		{5, 0, 1},
		{5, 5, 1},
		{5, 1, 5},
		{5, 2, 10},
		{5, 3, 10},
		{10, 3, 120},
		{10, 5, 252},
	}

	for _, tc := range testCases {
		got := binomialCoefficient(tc.n, tc.k)
		if math.Abs(got-tc.want) > 1e-6 {
			t.Errorf("binomialCoefficient(%d, %d) = %f, want %f", tc.n, tc.k, got, tc.want)
		}
	}
}

// Test next combination generator
func TestNextCombination(t *testing.T) {
	// Test generating all C(4,2) = 6 combinations
	indices := []int{0, 1}
	combinations := [][]int{{0, 1}}

	for nextCombination(indices, 4) {
		combo := make([]int, len(indices))
		copy(combo, indices)
		combinations = append(combinations, combo)
	}

	expected := [][]int{
		{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
	}

	if len(combinations) != len(expected) {
		t.Fatalf("Got %d combinations, want %d", len(combinations), len(expected))
	}

	for i, combo := range combinations {
		if combo[0] != expected[i][0] || combo[1] != expected[i][1] {
			t.Errorf("Combination %d: got %v, want %v", i, combo, expected[i])
		}
	}
}

// Benchmark
func BenchmarkKernelExplainer_Explain(b *testing.B) {
	fm := model.NewFuncModel(linearModel, 5)
	background := [][]float64{{0.0, 0.0, 0.0, 0.0, 0.0}}

	exp, _ := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkKernelExplainer_Explain_Parallel(b *testing.B) {
	fm := model.NewFuncModel(linearModel, 5)
	background := [][]float64{{0.0, 0.0, 0.0, 0.0, 0.0}}

	exp, _ := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithNumWorkers(4),
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

// Tests for batched predictions

func TestKernelExplainer_BatchedPredictions_SameResults(t *testing.T) {
	// Verify that batched predictions produce the same results as non-batched
	fm := model.NewFuncModel(linearModel, 3)

	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	// Create non-batched explainer
	expNonBatched, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() non-batched error = %v", err)
	}

	// Create batched explainer with same seed
	expBatched, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)
	if err != nil {
		t.Fatalf("New() batched error = %v", err)
	}

	// Get explanations
	resultNonBatched, err := expNonBatched.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() non-batched error = %v", err)
	}

	resultBatched, err := expBatched.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() batched error = %v", err)
	}

	// Results should be identical
	tolerance := 1e-10
	for name, val := range resultNonBatched.Values {
		batchedVal := resultBatched.Values[name]
		if math.Abs(val-batchedVal) > tolerance {
			t.Errorf("SHAP(%s): non-batched=%f, batched=%f, diff=%f",
				name, val, batchedVal, math.Abs(val-batchedVal))
		}
	}

	// Both should satisfy local accuracy
	if !resultNonBatched.Verify(1e-6).Valid {
		t.Error("Non-batched explanation failed local accuracy")
	}
	if !resultBatched.Verify(1e-6).Valid {
		t.Error("Batched explanation failed local accuracy")
	}
}

func TestKernelExplainer_BatchedPredictions_WeightedModel(t *testing.T) {
	// Test batched predictions with a more complex weighted model
	wm := &weightedModel{weights: []float64{2.0, 3.0, 1.0}, bias: 0.5}
	fm := model.NewFuncModel(wm.predict, 3)

	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 0.5, 0.25},
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	// Test with batched predictions
	exp, err := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Verify local accuracy
	verifyResult := result.Verify(1e-6)
	if !verifyResult.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
	}

	// Check prediction is correct: 2*1 + 3*2 + 1*3 + 0.5 = 11.5
	expectedPred := 11.5
	if math.Abs(result.Prediction-expectedPred) > 1e-10 {
		t.Errorf("Prediction = %f, want %f", result.Prediction, expectedPred)
	}
}

func TestKernelExplainer_BatchedPredictions_SingleFeature(t *testing.T) {
	// Test batched predictions with single feature (edge case)
	fm := model.NewFuncModel(func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * 2, nil
	}, 1)

	background := [][]float64{
		{0.0},
		{1.0},
	}

	ctx := context.Background()
	instance := []float64{3.0}

	exp, err := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Single feature: SHAP = prediction - baseline
	expectedSHAP := result.Prediction - result.BaseValue
	tolerance := 1e-10
	if math.Abs(result.Values["feature_0"]-expectedSHAP) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want %f", result.Values["feature_0"], expectedSHAP)
	}
}

func TestKernelExplainer_BatchedPredictions_LargeBackground(t *testing.T) {
	// Test batched predictions with larger background (benefits more from batching)
	fm := model.NewFuncModel(linearModel, 4)

	// 10 background samples
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, 4)
		for j := range background[i] {
			background[i][j] = float64(i) * 0.1
		}
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0}

	// Test both batched and non-batched
	expNonBatched, _ := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
	)

	expBatched, _ := New(fm, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)

	resultNonBatched, err := expNonBatched.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() non-batched error = %v", err)
	}

	resultBatched, err := expBatched.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() batched error = %v", err)
	}

	// Results should be identical
	tolerance := 1e-10
	for name, val := range resultNonBatched.Values {
		batchedVal := resultBatched.Values[name]
		if math.Abs(val-batchedVal) > tolerance {
			t.Errorf("SHAP(%s): non-batched=%f, batched=%f", name, val, batchedVal)
		}
	}
}

func BenchmarkKernelExplainer_Batched(b *testing.B) {
	fm := model.NewFuncModel(linearModel, 5)
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, 5)
	}

	exp, _ := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkKernelExplainer_NonBatched(b *testing.B) {
	fm := model.NewFuncModel(linearModel, 5)
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, 5)
	}

	exp, _ := New(fm, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(false),
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}
