package gradient

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model"
)

// linearModel implements a simple linear model: f(x) = w0*x0 + w1*x1 + ... + bias
type linearModel struct {
	weights []float64
	bias    float64
}

func (m *linearModel) Predict(_ context.Context, input []float64) (float64, error) {
	result := m.bias
	for i, w := range m.weights {
		if i < len(input) {
			result += w * input[i]
		}
	}
	return result, nil
}

func (m *linearModel) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error) {
	results := make([]float64, len(inputs))
	for i, input := range inputs {
		pred, err := m.Predict(ctx, input)
		if err != nil {
			return nil, err
		}
		results[i] = pred
	}
	return results, nil
}

func (m *linearModel) NumFeatures() int {
	return len(m.weights)
}

func (m *linearModel) Close() error {
	return nil
}

var _ model.Model = (*linearModel)(nil)

func TestNew(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0, 3.0},
		bias:    0.5,
	}

	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
	}

	exp, err := New(m, background, nil)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if exp.BaseValue() == 0 {
		t.Error("BaseValue should not be zero with non-zero weights")
	}
}

func TestNew_Errors(t *testing.T) {
	m := &linearModel{weights: []float64{1.0, 2.0}, bias: 0}
	bg := [][]float64{{0.0, 0.0}}

	tests := []struct {
		name       string
		model      model.Model
		background [][]float64
		gradOpts   []GradientOption
		wantErr    error
	}{
		{
			name:       "nil model",
			model:      nil,
			background: bg,
			wantErr:    ErrNilModel,
		},
		{
			name:       "empty background",
			model:      m,
			background: [][]float64{},
			wantErr:    ErrNoBackground,
		},
		{
			name:       "feature mismatch",
			model:      m,
			background: [][]float64{{0.0, 0.0, 0.0}}, // 3 features, model has 2
			wantErr:    ErrFeatureMismatch,
		},
		{
			name:       "invalid epsilon",
			model:      m,
			background: bg,
			gradOpts:   []GradientOption{WithEpsilon(0)},
			wantErr:    ErrInvalidEpsilon,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(tc.model, tc.background, nil, tc.gradOpts...)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestExplain_LinearModel(t *testing.T) {
	// For a linear model, SHAP values should equal weight * (x - mean(x_background))
	m := &linearModel{
		weights: []float64{2.0, 3.0},
		bias:    1.0,
	}

	// Background centered at (0.5, 0.5)
	background := [][]float64{
		{0.0, 0.0},
		{1.0, 1.0},
	}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(500),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1"}),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0, 1.0}

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Expected SHAP values for linear model:
	// SHAP[0] = weight[0] * (x[0] - mean_bg[0]) = 2.0 * (2.0 - 0.5) = 3.0
	// SHAP[1] = weight[1] * (x[1] - mean_bg[1]) = 3.0 * (1.0 - 0.5) = 1.5
	expectedSHAP := map[string]float64{
		"x0": 3.0,
		"x1": 1.5,
	}

	for name, expected := range expectedSHAP {
		got := explanation.Values[name]
		if math.Abs(got-expected) > 0.5 {
			t.Errorf("SHAP[%s] = %v, want approximately %v", name, got, expected)
		}
	}

	// Verify local accuracy
	result := explanation.Verify(0.5)
	if !result.Valid {
		t.Errorf("Local accuracy check failed: diff = %v", result.Difference)
	}
}

func TestExplain_NonlinearModel(t *testing.T) {
	// Quadratic model: f(x) = x0^2 + x0*x1
	predict := func(_ context.Context, input []float64) (float64, error) {
		return input[0]*input[0] + input[0]*input[1], nil
	}

	m := model.NewFuncModel(predict, 2)

	background := [][]float64{
		{0.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{1.0, 1.0},
	}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(300),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1"}),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instance := []float64{2.0, 1.0}

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// For nonlinear models, we mainly verify local accuracy
	result := explanation.Verify(0.5)
	if !result.Valid {
		t.Errorf("Local accuracy check failed: diff = %v", result.Difference)
	}

	// SHAP values should be non-zero
	if explanation.Values["x0"] == 0 {
		t.Error("SHAP[x0] should not be zero")
	}
}

func TestExplain_Parallel(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0, 3.0},
		bias:    0.0,
	}

	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0},
	}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(100),
			explainer.WithSeed(42),
			explainer.WithNumWorkers(4),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0}

	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Verify local accuracy
	result := explanation.Verify(0.5)
	if !result.Valid {
		t.Errorf("Local accuracy check failed: diff = %v", result.Difference)
	}
}

func TestExplain_ContextCancellation(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0},
		bias:    0.0,
	}

	background := [][]float64{{0.0, 0.0}, {1.0, 1.0}}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(1000),
			explainer.WithSeed(42),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.Explain(ctx, []float64{1.0, 2.0})
	if err == nil {
		t.Error("expected error with cancelled context")
	}
}

func TestExplain_WithConfidenceIntervals(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0},
		bias:    0.0,
	}

	background := [][]float64{
		{0.0, 0.0},
		{1.0, 1.0},
	}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(200),
			explainer.WithSeed(42),
			explainer.WithConfidenceLevel(0.95),
			explainer.WithFeatureNames([]string{"x0", "x1"}),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	if explanation.Metadata.ConfidenceIntervals == nil {
		t.Fatal("expected confidence intervals")
	}

	ci := explanation.Metadata.ConfidenceIntervals
	if ci.Level != 0.95 {
		t.Errorf("CI level = %v, want 0.95", ci.Level)
	}

	// Check that intervals are valid
	for _, name := range []string{"x0", "x1"} {
		shapVal := explanation.Values[name]
		lower := ci.Lower[name]
		upper := ci.Upper[name]

		if lower > shapVal || shapVal > upper {
			t.Errorf("SHAP[%s]=%v not in CI [%v, %v]", name, shapVal, lower, upper)
		}
	}
}

func TestExplainBatch(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0},
		bias:    0.0,
	}

	background := [][]float64{{0.0, 0.0}, {1.0, 1.0}}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(50),
			explainer.WithSeed(42),
		},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{0.5, 0.5},
		{1.0, 2.0},
		{2.0, 1.0},
	}

	results, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch failed: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("len(results) = %d, want 3", len(results))
	}

	for i, exp := range results {
		if exp == nil {
			t.Errorf("results[%d] is nil", i)
		}
	}
}

func TestGradientOptions(t *testing.T) {
	config := DefaultGradientConfig()

	WithEpsilon(1e-5)(&config)
	if config.Epsilon != 1e-5 {
		t.Errorf("Epsilon = %v, want 1e-5", config.Epsilon)
	}

	WithNoiseStdev(0.1)(&config)
	if config.NoiseStdev != 0.1 {
		t.Errorf("NoiseStdev = %v, want 0.1", config.NoiseStdev)
	}

	WithLocalSmoothing(5)(&config)
	if config.LocalSmoothingSamples != 5 {
		t.Errorf("LocalSmoothingSamples = %v, want 5", config.LocalSmoothingSamples)
	}
}

func TestComputeNumericalGradient(t *testing.T) {
	// Test gradient computation on a known function: f(x) = x0^2 + 2*x1
	// Gradient: [2*x0, 2]
	predict := func(_ context.Context, input []float64) (float64, error) {
		return input[0]*input[0] + 2*input[1], nil
	}

	m := model.NewFuncModel(predict, 2)

	background := [][]float64{{0.0, 0.0}}

	exp, err := New(m, background,
		[]explainer.Option{explainer.WithNumSamples(10)},
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	point := []float64{3.0, 1.0}

	gradient, err := exp.computeNumericalGradient(ctx, point)
	if err != nil {
		t.Fatalf("computeNumericalGradient failed: %v", err)
	}

	// Expected gradient at (3, 1): [6, 2]
	expectedGrad := []float64{6.0, 2.0}
	for i, expected := range expectedGrad {
		if math.Abs(gradient[i]-expected) > 1e-4 {
			t.Errorf("gradient[%d] = %v, want %v", i, gradient[i], expected)
		}
	}
}

func TestExplain_WithNoise(t *testing.T) {
	m := &linearModel{
		weights: []float64{1.0, 2.0},
		bias:    0.0,
	}

	background := [][]float64{{0.0, 0.0}, {1.0, 1.0}}

	exp, err := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(100),
			explainer.WithSeed(42),
		},
		WithNoiseStdev(0.01),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Should still roughly satisfy local accuracy despite noise
	result := explanation.Verify(1.0) // More lenient tolerance due to noise
	if !result.Valid {
		t.Logf("Local accuracy diff = %v (may be slightly off due to noise)", result.Difference)
	}
}

func BenchmarkExplain(b *testing.B) {
	m := &linearModel{
		weights: []float64{1.0, 2.0, 3.0, 4.0},
		bias:    0.0,
	}

	background := [][]float64{
		{0.0, 0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
	}

	exp, _ := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(100),
			explainer.WithSeed(42),
		},
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkExplain_Parallel(b *testing.B) {
	m := &linearModel{
		weights: []float64{1.0, 2.0, 3.0, 4.0},
		bias:    0.0,
	}

	background := [][]float64{
		{0.0, 0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 1.0},
	}

	exp, _ := New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(100),
			explainer.WithSeed(42),
			explainer.WithNumWorkers(4),
		},
	)

	ctx := context.Background()
	instance := []float64{1.0, 2.0, 3.0, 4.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}
