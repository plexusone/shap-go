package exact

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

// weightedModel implements: y = w[0]*x[0] + w[1]*x[1] + ... + bias
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

func TestExactExplainer_LinearModel(t *testing.T) {
	// Create a linear model: y = x0 + x1 + x2
	fm := model.NewFuncModel(linearModel, 3)

	// Background data: all zeros
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	// Create explainer
	exp, err := New(fm, background,
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

	// Verify local accuracy - ExactSHAP should satisfy this exactly
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}

	// For a linear model with zero background, SHAP values should equal feature values exactly
	tolerance := 1e-10 // ExactSHAP gives exact values
	if math.Abs(explanation.Values["x0"]-1.0) > tolerance {
		t.Errorf("SHAP(x0) = %f, want 1.0", explanation.Values["x0"])
	}
	if math.Abs(explanation.Values["x1"]-2.0) > tolerance {
		t.Errorf("SHAP(x1) = %f, want 2.0", explanation.Values["x1"])
	}
	if math.Abs(explanation.Values["x2"]-3.0) > tolerance {
		t.Errorf("SHAP(x2) = %f, want 3.0", explanation.Values["x2"])
	}
}

func TestExactExplainer_WeightedModel(t *testing.T) {
	// Create a weighted model: y = 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background: all zeros
	background := [][]float64{
		{0.0, 0.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 1.0} // Prediction = 2*1 + 3*1 = 5

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-5.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 5.0", explanation.Prediction)
	}

	// For linear model with zero background:
	// SHAP(x0) should be 2.0 (weight * value)
	// SHAP(x1) should be 3.0 (weight * value)
	tolerance := 1e-10
	if math.Abs(explanation.Values["feature_0"]-2.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want 2.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want 3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestExactExplainer_WithBias(t *testing.T) {
	// Model: y = 10 + 2*x0 + 3*x1
	wm := &weightedModel{weights: []float64{2.0, 3.0}, bias: 10.0}
	fm := model.NewFuncModel(wm.predict, 2)

	// Background with feature means [1, 1]
	background := [][]float64{
		{0.0, 0.0},
		{2.0, 2.0},
	}
	// Mean = [1, 1], base value = 10 + 2*1 + 3*1 = 15

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0, 3.0} // Prediction = 10 + 2*2 + 3*3 = 23

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-23.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 23.0", explanation.Prediction)
	}

	// Check base value = 15
	if math.Abs(explanation.BaseValue-15.0) > 1e-10 {
		t.Errorf("BaseValue = %f, want 15.0", explanation.BaseValue)
	}

	// SHAP values:
	// SHAP(x0) = 2 * (2 - 1) = 2
	// SHAP(x1) = 3 * (3 - 1) = 6
	tolerance := 1e-10
	if math.Abs(explanation.Values["feature_0"]-2.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want 2.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-6.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want 6.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy: sum = 2 + 6 = 8 = 23 - 15
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestExactExplainer_SingleFeature(t *testing.T) {
	// Single feature model
	singleModel := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * 5.0, nil
	}
	fm := model.NewFuncModel(singleModel, 1)

	background := [][]float64{
		{0.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0} // Prediction = 10

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// For single feature, SHAP value equals the full contribution
	if math.Abs(explanation.Values["feature_0"]-10.0) > 1e-10 {
		t.Errorf("SHAP(feature_0) = %f, want 10.0", explanation.Values["feature_0"])
	}

	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed")
	}
}

func TestExactExplainer_NonlinearModel(t *testing.T) {
	// Nonlinear model: y = x0 * x1
	multiplyModel := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * input[1], nil
	}
	fm := model.NewFuncModel(multiplyModel, 2)

	background := [][]float64{
		{0.0, 0.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{3.0, 4.0} // Prediction = 12

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-12.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 12.0", explanation.Prediction)
	}

	// For multiplication with zero background:
	// f({}) = 0, f({x0}) = 0, f({x1}) = 0, f({x0,x1}) = 12
	// By symmetry in this case, SHAP values should be equal
	// SHAP(x0) = SHAP(x1) = 6
	tolerance := 1e-10
	if math.Abs(explanation.Values["feature_0"]-6.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want 6.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-6.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want 6.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestExactExplainer_NonlinearModelWithBackground(t *testing.T) {
	// Nonlinear model: y = x0 * x1
	multiplyModel := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * input[1], nil
	}
	fm := model.NewFuncModel(multiplyModel, 2)

	// Non-zero background
	background := [][]float64{
		{1.0, 2.0}, // prediction = 2
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{3.0, 4.0} // Prediction = 12

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-12.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 12.0", explanation.Prediction)
	}

	// Base value = 1 * 2 = 2
	if math.Abs(explanation.BaseValue-2.0) > 1e-10 {
		t.Errorf("BaseValue = %f, want 2.0", explanation.BaseValue)
	}

	// Verify local accuracy: SHAP values should sum to 12 - 2 = 10
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestExactExplainer_ExplainBatch(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 2)
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	}

	explanations, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch() error = %v", err)
	}

	if len(explanations) != 3 {
		t.Fatalf("got %d explanations, want 3", len(explanations))
	}

	// Verify each explanation
	for i, exp := range explanations {
		result := exp.Verify(1e-10)
		if !result.Valid {
			t.Errorf("Instance %d: local accuracy failed", i)
		}
	}
}

func TestExactExplainer_Errors(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	t.Run("nil model", func(t *testing.T) {
		_, err := New(nil, background)
		if err != ErrNilModel {
			t.Errorf("expected ErrNilModel, got %v", err)
		}
	})

	t.Run("empty background", func(t *testing.T) {
		_, err := New(fm, [][]float64{})
		if err != ErrNoBackground {
			t.Errorf("expected ErrNoBackground, got %v", err)
		}
	})

	t.Run("wrong background dimensions", func(t *testing.T) {
		badBackground := [][]float64{{0.0, 0.0}} // 2 features instead of 3
		_, err := New(fm, badBackground)
		if err == nil {
			t.Error("expected error for mismatched dimensions")
		}
	})

	t.Run("wrong instance dimensions", func(t *testing.T) {
		exp, _ := New(fm, background)
		_, err := exp.Explain(context.Background(), []float64{1.0, 2.0}) // 2 features instead of 3
		if err == nil {
			t.Error("expected error for mismatched instance dimensions")
		}
	})
}

func TestExactExplainer_TooManyFeatures(t *testing.T) {
	// Create model with too many features
	manyFeatures := func(ctx context.Context, input []float64) (float64, error) {
		return 0, nil
	}
	fm := model.NewFuncModel(manyFeatures, MaxFeatures+1)

	background := make([][]float64, 1)
	background[0] = make([]float64, MaxFeatures+1)

	_, err := New(fm, background)
	if err == nil {
		t.Error("expected error for too many features")
	}
}

func TestExactExplainer_FourFeatures(t *testing.T) {
	// Test with 4 features to exercise more coalitions
	weightedModel := func(ctx context.Context, input []float64) (float64, error) {
		return 1*input[0] + 2*input[1] + 3*input[2] + 4*input[3], nil
	}
	fm := model.NewFuncModel(weightedModel, 4)

	background := [][]float64{
		{0.0, 0.0, 0.0, 0.0},
	}

	exp, err := New(fm, background,
		explainer.WithFeatureNames([]string{"a", "b", "c", "d"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 1.0, 1.0, 1.0} // Prediction = 10

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check SHAP values
	tolerance := 1e-10
	if math.Abs(explanation.Values["a"]-1.0) > tolerance {
		t.Errorf("SHAP(a) = %f, want 1.0", explanation.Values["a"])
	}
	if math.Abs(explanation.Values["b"]-2.0) > tolerance {
		t.Errorf("SHAP(b) = %f, want 2.0", explanation.Values["b"])
	}
	if math.Abs(explanation.Values["c"]-3.0) > tolerance {
		t.Errorf("SHAP(c) = %f, want 3.0", explanation.Values["c"])
	}
	if math.Abs(explanation.Values["d"]-4.0) > tolerance {
		t.Errorf("SHAP(d) = %f, want 4.0", explanation.Values["d"])
	}

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f",
			result.SumSHAP, result.Expected)
	}
}

func TestExactExplainer_Metadata(t *testing.T) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background,
		explainer.WithModelID("test-model"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if explanation.Metadata.Algorithm != "exact" {
		t.Errorf("Algorithm = %s, want exact", explanation.Metadata.Algorithm)
	}

	if explanation.ModelID != "test-model" {
		t.Errorf("ModelID = %s, want test-model", explanation.ModelID)
	}

	// NumSamples should be 2^n = 8 for 3 features
	if explanation.Metadata.NumSamples != 8 {
		t.Errorf("NumSamples = %d, want 8", explanation.Metadata.NumSamples)
	}
}

func TestComputeShapleyWeights(t *testing.T) {
	// Test Shapley weights computation
	// For n features, weight[s] = s! * (n-s-1)! / n!
	// Sum of weights should equal 1 (since it's a probability distribution)

	for n := 2; n <= 5; n++ {
		weights := computeShapleyWeights(n)

		if len(weights) != n {
			t.Errorf("n=%d: got %d weights, want %d", n, len(weights), n)
			continue
		}

		// Sum of weights * C(n-1, s) for each s should equal 1
		// This is because each feature has 2^(n-1) coalitions
		totalWeight := 0.0
		for s := 0; s < n; s++ {
			// Number of coalitions of size s is C(n-1, s)
			numCoalitions := binomial(n-1, s)
			totalWeight += weights[s] * float64(numCoalitions)
		}

		if math.Abs(totalWeight-1.0) > 1e-10 {
			t.Errorf("n=%d: total weight = %f, want 1.0", n, totalWeight)
		}
	}
}

func TestBuildCoalition(t *testing.T) {
	tests := []struct {
		coalitionIdx uint64
		skipFeature  int
		numFeatures  int
		expected     uint64
	}{
		// 2 features, skip feature 0
		{0b0, 0, 2, 0b00}, // empty coalition
		{0b1, 0, 2, 0b10}, // feature 1 in coalition

		// 2 features, skip feature 1
		{0b0, 1, 2, 0b00}, // empty coalition
		{0b1, 1, 2, 0b01}, // feature 0 in coalition

		// 3 features, skip feature 1
		{0b00, 1, 3, 0b000}, // empty
		{0b01, 1, 3, 0b001}, // feature 0
		{0b10, 1, 3, 0b100}, // feature 2
		{0b11, 1, 3, 0b101}, // features 0 and 2
	}

	for _, tc := range tests {
		result := buildCoalition(tc.coalitionIdx, tc.skipFeature, tc.numFeatures)
		if result != tc.expected {
			t.Errorf("buildCoalition(%b, %d, %d) = %b, want %b",
				tc.coalitionIdx, tc.skipFeature, tc.numFeatures, result, tc.expected)
		}
	}
}

func TestPopCount(t *testing.T) {
	tests := []struct {
		x        uint64
		expected int
	}{
		{0b0, 0},
		{0b1, 1},
		{0b10, 1},
		{0b11, 2},
		{0b111, 3},
		{0b1010, 2},
		{0b11111111, 8},
	}

	for _, tc := range tests {
		result := popCount(tc.x)
		if result != tc.expected {
			t.Errorf("popCount(%b) = %d, want %d", tc.x, result, tc.expected)
		}
	}
}

// binomial computes C(n, k) = n! / (k! * (n-k)!)
func binomial(n, k int) int {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	result := 1
	for i := 0; i < k; i++ {
		result = result * (n - i) / (i + 1)
	}
	return result
}

func BenchmarkExactExplainer_3Features(b *testing.B) {
	fm := model.NewFuncModel(linearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}
	exp, _ := New(fm, background)
	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkExactExplainer_5Features(b *testing.B) {
	model5 := func(ctx context.Context, input []float64) (float64, error) {
		sum := 0.0
		for _, v := range input {
			sum += v
		}
		return sum, nil
	}
	fm := model.NewFuncModel(model5, 5)
	background := [][]float64{{0.0, 0.0, 0.0, 0.0, 0.0}}
	exp, _ := New(fm, background)
	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkExactExplainer_10Features(b *testing.B) {
	model10 := func(ctx context.Context, input []float64) (float64, error) {
		sum := 0.0
		for _, v := range input {
			sum += v
		}
		return sum, nil
	}
	fm := model.NewFuncModel(model10, 10)
	background := [][]float64{make([]float64, 10)}
	exp, _ := New(fm, background)
	ctx := context.Background()
	instance := make([]float64, 10)
	for i := range instance {
		instance[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}
