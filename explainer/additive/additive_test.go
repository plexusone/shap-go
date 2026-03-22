package additive

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model"
)

// additiveModel implements a true additive model: f(x) = Σ fᵢ(xᵢ)
// Each feature has an independent effect with no interactions.
func additiveLinearModel(ctx context.Context, input []float64) (float64, error) {
	// f(x) = 2*x₀ + 3*x₁ + 1*x₂
	return 2*input[0] + 3*input[1] + 1*input[2], nil
}

// additiveNonlinearModel has independent nonlinear effects for each feature
func additiveNonlinearModel(ctx context.Context, input []float64) (float64, error) {
	// f(x) = x₀² + sin(x₁) + exp(x₂/10)
	return input[0]*input[0] + math.Sin(input[1]) + math.Exp(input[2]/10), nil
}

func TestAdditiveExplainer_LinearModel(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)

	// Background with mean at origin
	background := [][]float64{
		{-1.0, -1.0, -1.0},
		{1.0, 1.0, 1.0},
	}

	exp, err := New(fm, background,
		explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// For linear additive model with zero-mean background:
	// SHAP(x0) = 2*1 - 2*0 = 2
	// SHAP(x1) = 3*2 - 3*0 = 6
	// SHAP(x2) = 1*3 - 1*0 = 3
	tolerance := 1e-10
	if math.Abs(result.Values["x0"]-2.0) > tolerance {
		t.Errorf("SHAP(x0) = %f, want 2.0", result.Values["x0"])
	}
	if math.Abs(result.Values["x1"]-6.0) > tolerance {
		t.Errorf("SHAP(x1) = %f, want 6.0", result.Values["x1"])
	}
	if math.Abs(result.Values["x2"]-3.0) > tolerance {
		t.Errorf("SHAP(x2) = %f, want 3.0", result.Values["x2"])
	}

	// Verify local accuracy
	verifyResult := result.Verify(1e-10)
	if !verifyResult.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
	}
}

func TestAdditiveExplainer_NonlinearModel(t *testing.T) {
	fm := model.NewFuncModel(additiveNonlinearModel, 3)

	// Background samples
	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
	}

	exp, err := New(fm, background,
		explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0, 1.5, 5.0}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Verify local accuracy - sum of SHAP values should equal prediction - baseline
	verifyResult := result.Verify(1e-10)
	if !verifyResult.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
			verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
	}

	// Verify metadata
	if result.Metadata.Algorithm != "additive" {
		t.Errorf("Algorithm = %s, want additive", result.Metadata.Algorithm)
	}
}

func TestAdditiveExplainer_SingleFeature(t *testing.T) {
	// Single feature model: f(x) = x²
	singleFeatureModel := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * input[0], nil
	}
	fm := model.NewFuncModel(singleFeatureModel, 1)

	background := [][]float64{
		{0.0},
		{1.0},
		{2.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{3.0}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Single feature should get all the credit
	// SHAP = prediction - baseline
	expected := result.Prediction - result.BaseValue
	tolerance := 1e-10
	if math.Abs(result.Values["feature_0"]-expected) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want %f", result.Values["feature_0"], expected)
	}
}

func TestAdditiveExplainer_ZeroBackground(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)

	// Single zero background sample
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// With zero background, SHAP values equal the linear coefficients times feature values
	// SHAP(x0) = 2*1 = 2
	// SHAP(x1) = 3*2 = 6
	// SHAP(x2) = 1*3 = 3
	tolerance := 1e-10
	if math.Abs(result.Values["feature_0"]-2.0) > tolerance {
		t.Errorf("SHAP(feature_0) = %f, want 2.0", result.Values["feature_0"])
	}
	if math.Abs(result.Values["feature_1"]-6.0) > tolerance {
		t.Errorf("SHAP(feature_1) = %f, want 6.0", result.Values["feature_1"])
	}
	if math.Abs(result.Values["feature_2"]-3.0) > tolerance {
		t.Errorf("SHAP(feature_2) = %f, want 3.0", result.Values["feature_2"])
	}

	// Verify local accuracy
	verifyResult := result.Verify(1e-10)
	if !verifyResult.Valid {
		t.Errorf("Local accuracy failed: sum=%f, expected=%f",
			verifyResult.SumSHAP, verifyResult.Expected)
	}
}

func TestAdditiveExplainer_ExplainBatch(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}

	results, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch() error = %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("ExplainBatch() returned %d results, want 3", len(results))
	}

	// Verify each result satisfies local accuracy
	for i, result := range results {
		verifyResult := result.Verify(1e-10)
		if !verifyResult.Valid {
			t.Errorf("Instance %d: local accuracy failed: sum=%f, expected=%f",
				i, verifyResult.SumSHAP, verifyResult.Expected)
		}
	}
}

func TestAdditiveExplainer_Errors(t *testing.T) {
	tests := []struct {
		name       string
		model      model.Model
		background [][]float64
		wantErr    bool
	}{
		{
			name:       "nil model",
			model:      nil,
			background: [][]float64{{0.0}},
			wantErr:    true,
		},
		{
			name:       "empty background",
			model:      model.NewFuncModel(additiveLinearModel, 3),
			background: [][]float64{},
			wantErr:    true,
		},
		{
			name:       "wrong background dimensions",
			model:      model.NewFuncModel(additiveLinearModel, 3),
			background: [][]float64{{0.0, 0.0}}, // 2 features, model expects 3
			wantErr:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.model, tc.background)
			if (err != nil) != tc.wantErr {
				t.Errorf("New() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestAdditiveExplainer_WrongInstanceDimensions(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0} // 2 features, model expects 3

	_, err = exp.Explain(ctx, instance)
	if err == nil {
		t.Error("Explain() should return error for wrong instance dimensions")
	}
}

func TestAdditiveExplainer_ContextCancellation(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.Explain(ctx, []float64{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("Explain() should return error for cancelled context")
	}
}

func TestAdditiveExplainer_HelperMethods(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{
		{1.0, 2.0, 3.0},
		{3.0, 4.0, 5.0},
	}

	exp, err := New(fm, background,
		explainer.WithModelID("test-model"),
		explainer.WithFeatureNames([]string{"a", "b", "c"}),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	// Test Reference()
	ref := exp.Reference()
	if len(ref) != 3 {
		t.Errorf("Reference() length = %d, want 3", len(ref))
	}
	// Reference should be mean: [2, 3, 4]
	expectedRef := []float64{2.0, 3.0, 4.0}
	for i, v := range ref {
		if math.Abs(v-expectedRef[i]) > 1e-10 {
			t.Errorf("Reference()[%d] = %f, want %f", i, v, expectedRef[i])
		}
	}

	// Test FeatureNames()
	names := exp.FeatureNames()
	expectedNames := []string{"a", "b", "c"}
	for i, name := range names {
		if name != expectedNames[i] {
			t.Errorf("FeatureNames()[%d] = %s, want %s", i, name, expectedNames[i])
		}
	}

	// Test ExpectedEffects()
	effects := exp.ExpectedEffects()
	if len(effects) != 3 {
		t.Errorf("ExpectedEffects() length = %d, want 3", len(effects))
	}
}

func TestAdditiveExplainer_Metadata(t *testing.T) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}

	exp, err := New(fm, background,
		explainer.WithModelID("test-gam"),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	result, err := exp.Explain(ctx, []float64{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	if result.Metadata.Algorithm != "additive" {
		t.Errorf("Algorithm = %s, want additive", result.Metadata.Algorithm)
	}

	if result.ModelID != "test-gam" {
		t.Errorf("ModelID = %s, want test-gam", result.ModelID)
	}

	if result.Metadata.BackgroundSize != 2 {
		t.Errorf("BackgroundSize = %d, want 2", result.Metadata.BackgroundSize)
	}

	// NumSamples should equal number of features (one eval per feature)
	if result.Metadata.NumSamples != 3 {
		t.Errorf("NumSamples = %d, want 3", result.Metadata.NumSamples)
	}
}

// Test that additive explainer gives same results as exact for additive models
func TestAdditiveExplainer_MatchesExpected(t *testing.T) {
	// Model: f(x) = x₀ + 2*x₁ + 3*x₂
	linearModel := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] + 2*input[1] + 3*input[2], nil
	}
	fm := model.NewFuncModel(linearModel, 3)

	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	exp, err := New(fm, background)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()

	// Test multiple instances
	testCases := []struct {
		instance     []float64
		expectedSHAP []float64
	}{
		{
			instance:     []float64{1.0, 1.0, 1.0},
			expectedSHAP: []float64{1.0, 2.0, 3.0},
		},
		{
			instance:     []float64{2.0, 3.0, 4.0},
			expectedSHAP: []float64{2.0, 6.0, 12.0},
		},
		{
			instance:     []float64{0.0, 0.0, 0.0},
			expectedSHAP: []float64{0.0, 0.0, 0.0},
		},
	}

	for _, tc := range testCases {
		result, err := exp.Explain(ctx, tc.instance)
		if err != nil {
			t.Fatalf("Explain(%v) error = %v", tc.instance, err)
		}

		tolerance := 1e-10
		shapValues := []float64{
			result.Values["feature_0"],
			result.Values["feature_1"],
			result.Values["feature_2"],
		}

		for i, expected := range tc.expectedSHAP {
			if math.Abs(shapValues[i]-expected) > tolerance {
				t.Errorf("Instance %v: SHAP[%d] = %f, want %f",
					tc.instance, i, shapValues[i], expected)
			}
		}

		// Verify local accuracy
		verifyResult := result.Verify(1e-10)
		if !verifyResult.Valid {
			t.Errorf("Instance %v: local accuracy failed", tc.instance)
		}
	}
}

func BenchmarkAdditiveExplainer(b *testing.B) {
	fm := model.NewFuncModel(additiveLinearModel, 3)
	background := [][]float64{{0.0, 0.0, 0.0}}
	exp, _ := New(fm, background)
	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkAdditiveExplainer_10Features(b *testing.B) {
	// 10-feature additive model
	model10 := func(ctx context.Context, input []float64) (float64, error) {
		sum := 0.0
		for i, v := range input {
			sum += float64(i+1) * v
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
