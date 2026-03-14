package linear

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
)

func TestNew_Errors(t *testing.T) {
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	tests := []struct {
		name       string
		weights    []float64
		bias       float64
		background [][]float64
		wantErr    error
	}{
		{
			name:       "empty weights",
			weights:    []float64{},
			bias:       0.0,
			background: background,
			wantErr:    ErrNoWeights,
		},
		{
			name:       "nil weights",
			weights:    nil,
			bias:       0.0,
			background: background,
			wantErr:    ErrNoWeights,
		},
		{
			name:       "empty background",
			weights:    []float64{1.0, 2.0},
			bias:       0.0,
			background: [][]float64{},
			wantErr:    ErrNoBackground,
		},
		{
			name:       "nil background",
			weights:    []float64{1.0, 2.0},
			bias:       0.0,
			background: nil,
			wantErr:    ErrNoBackground,
		},
		{
			name:    "mismatched dimensions",
			weights: []float64{1.0, 2.0},
			bias:    0.0,
			background: [][]float64{
				{1.0, 2.0, 3.0}, // Wrong size
			},
			wantErr: ErrInconsistentData,
		},
		{
			name:    "inconsistent background rows",
			weights: []float64{1.0, 2.0},
			bias:    0.0,
			background: [][]float64{
				{1.0, 2.0},
				{3.0}, // Wrong size
			},
			wantErr: ErrInconsistentData,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.weights, tt.bias, tt.background)
			if err == nil {
				t.Error("New() should return error")
				return
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("New() error = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestNew_Success(t *testing.T) {
	weights := []float64{2.0, 3.0}
	bias := 1.0
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if exp.NumFeatures() != 2 {
		t.Errorf("NumFeatures() = %d, want 2", exp.NumFeatures())
	}

	if exp.Bias() != 1.0 {
		t.Errorf("Bias() = %f, want 1.0", exp.Bias())
	}

	// Check weights are copied
	w := exp.Weights()
	if len(w) != 2 || w[0] != 2.0 || w[1] != 3.0 {
		t.Errorf("Weights() = %v, want [2.0, 3.0]", w)
	}

	// Check feature means: E[x0] = (1+3)/2 = 2, E[x1] = (2+4)/2 = 3
	means := exp.FeatureMeans()
	if len(means) != 2 || means[0] != 2.0 || means[1] != 3.0 {
		t.Errorf("FeatureMeans() = %v, want [2.0, 3.0]", means)
	}

	// Check base value: bias + w0*E[x0] + w1*E[x1] = 1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
	if exp.BaseValue() != 14.0 {
		t.Errorf("BaseValue() = %f, want 14.0", exp.BaseValue())
	}
}

func TestExplain_SimpleLinear(t *testing.T) {
	// Model: f(x) = 1 + 2*x0 + 3*x1
	weights := []float64{2.0, 3.0}
	bias := 1.0
	background := [][]float64{
		{0.0, 0.0},
		{2.0, 2.0},
	}
	// E[x0] = 1.0, E[x1] = 1.0
	// Base value = 1 + 2*1 + 3*1 = 6

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{3.0, 2.0}
	// f(x) = 1 + 2*3 + 3*2 = 1 + 6 + 6 = 13
	// SHAP[0] = 2 * (3 - 1) = 4
	// SHAP[1] = 3 * (2 - 1) = 3

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check prediction
	if math.Abs(explanation.Prediction-13.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 13.0", explanation.Prediction)
	}

	// Check base value
	if math.Abs(explanation.BaseValue-6.0) > 1e-10 {
		t.Errorf("BaseValue = %f, want 6.0", explanation.BaseValue)
	}

	// Check SHAP values
	if math.Abs(explanation.Values["feature_0"]-4.0) > 1e-10 {
		t.Errorf("SHAP[feature_0] = %f, want 4.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-3.0) > 1e-10 {
		t.Errorf("SHAP[feature_1] = %f, want 3.0", explanation.Values["feature_1"])
	}

	// Verify local accuracy: sum(SHAP) = prediction - base_value
	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			result.SumSHAP, result.Expected, result.Difference)
	}
}

func TestExplain_NegativeWeights(t *testing.T) {
	// Model: f(x) = 5 - 2*x0 + 4*x1
	weights := []float64{-2.0, 4.0}
	bias := 5.0
	background := [][]float64{
		{1.0, 1.0},
		{3.0, 3.0},
	}
	// E[x0] = 2.0, E[x1] = 2.0
	// Base value = 5 + (-2)*2 + 4*2 = 5 - 4 + 8 = 9

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{4.0, 1.0}
	// f(x) = 5 + (-2)*4 + 4*1 = 5 - 8 + 4 = 1
	// SHAP[0] = -2 * (4 - 2) = -4
	// SHAP[1] = 4 * (1 - 2) = -4

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if math.Abs(explanation.Prediction-1.0) > 1e-10 {
		t.Errorf("Prediction = %f, want 1.0", explanation.Prediction)
	}

	if math.Abs(explanation.BaseValue-9.0) > 1e-10 {
		t.Errorf("BaseValue = %f, want 9.0", explanation.BaseValue)
	}

	if math.Abs(explanation.Values["feature_0"]-(-4.0)) > 1e-10 {
		t.Errorf("SHAP[feature_0] = %f, want -4.0", explanation.Values["feature_0"])
	}
	if math.Abs(explanation.Values["feature_1"]-(-4.0)) > 1e-10 {
		t.Errorf("SHAP[feature_1] = %f, want -4.0", explanation.Values["feature_1"])
	}

	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: diff=%f", result.Difference)
	}
}

func TestExplain_ZeroWeight(t *testing.T) {
	// Model: f(x) = 10 + 0*x0 + 5*x1 (x0 has no effect)
	weights := []float64{0.0, 5.0}
	bias := 10.0
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{100.0, 5.0} // x0 varies wildly but shouldn't affect SHAP

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// SHAP[0] should be 0 regardless of instance value
	if explanation.Values["feature_0"] != 0.0 {
		t.Errorf("SHAP[feature_0] = %f, want 0.0", explanation.Values["feature_0"])
	}

	result := explanation.Verify(1e-10)
	if !result.Valid {
		t.Errorf("Local accuracy failed: diff=%f", result.Difference)
	}
}

func TestExplain_InstanceAtMean(t *testing.T) {
	// When instance equals the mean, all SHAP values should be 0
	weights := []float64{2.0, 3.0}
	bias := 1.0
	background := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	// E[x0] = 2.0, E[x1] = 3.0

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0, 3.0} // At the mean

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// All SHAP values should be 0
	for name, val := range explanation.Values {
		if val != 0.0 {
			t.Errorf("SHAP[%s] = %f, want 0.0", name, val)
		}
	}

	// Prediction should equal base value
	if explanation.Prediction != explanation.BaseValue {
		t.Errorf("Prediction = %f, BaseValue = %f, should be equal at mean",
			explanation.Prediction, explanation.BaseValue)
	}
}

func TestExplain_FeatureMismatch(t *testing.T) {
	weights := []float64{1.0, 2.0}
	bias := 0.0
	background := [][]float64{{1.0, 2.0}}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()

	// Wrong number of features
	_, err = exp.Explain(ctx, []float64{1.0})
	if err == nil {
		t.Error("Explain() should return error for wrong feature count")
	}
	if !errors.Is(err, ErrFeatureMismatch) {
		t.Errorf("Explain() error = %v, want ErrFeatureMismatch", err)
	}
}

func TestExplain_ContextCancellation(t *testing.T) {
	weights := []float64{1.0}
	bias := 0.0
	background := [][]float64{{1.0}}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.Explain(ctx, []float64{2.0})
	if err == nil {
		t.Error("Explain() should return error when context cancelled")
	}
}

func TestExplainBatch(t *testing.T) {
	weights := []float64{2.0, 3.0}
	bias := 1.0
	background := [][]float64{
		{1.0, 1.0},
		{3.0, 3.0},
	}

	exp, err := New(weights, bias, background)
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
		t.Errorf("ExplainBatch() returned %d explanations, want 3", len(explanations))
	}

	// Verify each explanation
	for i, explanation := range explanations {
		result := explanation.Verify(1e-10)
		if !result.Valid {
			t.Errorf("Explanation %d failed local accuracy: diff=%f", i, result.Difference)
		}
	}
}

func TestExplainBatch_ContextCancellation(t *testing.T) {
	weights := []float64{1.0}
	bias := 0.0
	background := [][]float64{{1.0}}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = exp.ExplainBatch(ctx, [][]float64{{1.0}, {2.0}})
	if err == nil {
		t.Error("ExplainBatch() should return error when context cancelled")
	}
}

func TestExplain_WithFeatureNames(t *testing.T) {
	weights := []float64{2.0, 3.0}
	bias := 1.0
	background := [][]float64{{0.0, 0.0}}

	exp, err := New(weights, bias, background,
		explainer.WithFeatureNames([]string{"age", "income"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Check feature names are used
	if _, ok := explanation.Values["age"]; !ok {
		t.Error("Expected 'age' in SHAP values")
	}
	if _, ok := explanation.Values["income"]; !ok {
		t.Error("Expected 'income' in SHAP values")
	}

	// Check feature values are recorded
	if explanation.FeatureValues["age"] != 1.0 {
		t.Errorf("FeatureValues[age] = %f, want 1.0", explanation.FeatureValues["age"])
	}
}

func TestExplain_WithModelID(t *testing.T) {
	weights := []float64{1.0}
	bias := 0.0
	background := [][]float64{{0.0}}

	exp, err := New(weights, bias, background,
		explainer.WithModelID("linear-model-v1"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if explanation.ModelID != "linear-model-v1" {
		t.Errorf("ModelID = %q, want %q", explanation.ModelID, "linear-model-v1")
	}
}

func TestExplain_Metadata(t *testing.T) {
	weights := []float64{1.0}
	bias := 0.0
	background := [][]float64{{0.0}}

	exp, err := New(weights, bias, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if explanation.Metadata.Algorithm != "linear" {
		t.Errorf("Metadata.Algorithm = %q, want %q", explanation.Metadata.Algorithm, "linear")
	}
}

func TestComputeMeans(t *testing.T) {
	background := [][]float64{
		{1.0, 4.0, 7.0},
		{2.0, 5.0, 8.0},
		{3.0, 6.0, 9.0},
	}

	means := computeMeans(background, 3)

	expected := []float64{2.0, 5.0, 8.0}
	for i, m := range means {
		if m != expected[i] {
			t.Errorf("means[%d] = %f, want %f", i, m, expected[i])
		}
	}
}

// Benchmark tests
func BenchmarkExplain(b *testing.B) {
	// 20 features
	numFeatures := 20
	weights := make([]float64, numFeatures)
	for i := range weights {
		weights[i] = float64(i + 1)
	}
	bias := 1.0

	background := make([][]float64, 100)
	for i := range background {
		background[i] = make([]float64, numFeatures)
		for j := range background[i] {
			background[i][j] = float64(i*j) / 100.0
		}
	}

	exp, _ := New(weights, bias, background)
	instance := make([]float64, numFeatures)
	for i := range instance {
		instance[i] = float64(i) * 0.1
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkExplainBatch(b *testing.B) {
	numFeatures := 20
	batchSize := 100

	weights := make([]float64, numFeatures)
	for i := range weights {
		weights[i] = float64(i + 1)
	}
	bias := 1.0

	background := make([][]float64, 50)
	for i := range background {
		background[i] = make([]float64, numFeatures)
	}

	exp, _ := New(weights, bias, background)

	instances := make([][]float64, batchSize)
	for i := range instances {
		instances[i] = make([]float64, numFeatures)
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.ExplainBatch(ctx, instances)
	}
}
