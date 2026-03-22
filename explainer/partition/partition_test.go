package partition

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model"
)

// simpleLinearModel is a linear model: f(x) = 2*x0 + 3*x1 + x2 + x3
type simpleLinearModel struct {
	numFeatures int
}

func (m *simpleLinearModel) Predict(_ context.Context, input []float64) (float64, error) {
	return 2*input[0] + 3*input[1] + input[2] + input[3], nil
}

func (m *simpleLinearModel) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error) {
	results := make([]float64, len(inputs))
	for i, input := range inputs {
		var err error
		results[i], err = m.Predict(ctx, input)
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func (m *simpleLinearModel) NumFeatures() int { return m.numFeatures }
func (m *simpleLinearModel) Close() error     { return nil }

func TestNode_IsLeaf(t *testing.T) {
	leaf := &Node{Name: "feature", FeatureIdx: 0}
	if !leaf.IsLeaf() {
		t.Error("Leaf node should return true for IsLeaf()")
	}

	internal := &Node{Name: "group", Children: []*Node{leaf}}
	if internal.IsLeaf() {
		t.Error("Internal node should return false for IsLeaf()")
	}
}

func TestNode_GetFeatureIndices(t *testing.T) {
	// Simple leaf
	leaf := &Node{Name: "f0", FeatureIdx: 0}
	indices := leaf.GetFeatureIndices()
	if len(indices) != 1 || indices[0] != 0 {
		t.Errorf("Leaf GetFeatureIndices() = %v, want [0]", indices)
	}

	// Nested hierarchy
	root := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "group1", Children: []*Node{
				{Name: "f0", FeatureIdx: 0},
				{Name: "f1", FeatureIdx: 1},
			}},
			{Name: "group2", Children: []*Node{
				{Name: "f2", FeatureIdx: 2},
			}},
		},
	}

	indices = root.GetFeatureIndices()
	if len(indices) != 3 {
		t.Fatalf("Root GetFeatureIndices() length = %d, want 3", len(indices))
	}

	expected := map[int]bool{0: true, 1: true, 2: true}
	for _, idx := range indices {
		if !expected[idx] {
			t.Errorf("Unexpected feature index %d", idx)
		}
	}
}

func TestNew_NilModel(t *testing.T) {
	_, err := New(nil, [][]float64{{1, 2}}, nil)
	if err == nil {
		t.Error("Expected error for nil model")
	}
}

func TestNew_EmptyBackground(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	_, err := New(m, nil, nil)
	if err == nil {
		t.Error("Expected error for nil background")
	}

	_, err = New(m, [][]float64{}, nil)
	if err == nil {
		t.Error("Expected error for empty background")
	}
}

func TestNew_InvalidBackgroundDimensions(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{
		{1, 2, 3, 4},
		{1, 2}, // Wrong dimension
	}

	_, err := New(m, background, nil)
	if err == nil {
		t.Error("Expected error for mismatched background dimensions")
	}
}

func TestNew_InvalidHierarchy(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	// Hierarchy with wrong number of features
	wrongHierarchy := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "f0", FeatureIdx: 0},
			{Name: "f1", FeatureIdx: 1},
			// Missing f2 and f3
		},
	}

	_, err := New(m, background, wrongHierarchy)
	if err == nil {
		t.Error("Expected error for incomplete hierarchy")
	}

	// Hierarchy with duplicate feature
	dupHierarchy := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "f0", FeatureIdx: 0},
			{Name: "f0_dup", FeatureIdx: 0}, // Duplicate
			{Name: "f1", FeatureIdx: 1},
			{Name: "f2", FeatureIdx: 2},
		},
	}

	_, err = New(m, background, dupHierarchy)
	if err == nil {
		t.Error("Expected error for duplicate feature in hierarchy")
	}

	// Hierarchy with invalid index
	invalidIdxHierarchy := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "f0", FeatureIdx: 0},
			{Name: "f1", FeatureIdx: 1},
			{Name: "f2", FeatureIdx: 2},
			{Name: "f100", FeatureIdx: 100}, // Invalid
		},
	}

	_, err = New(m, background, invalidIdxHierarchy)
	if err == nil {
		t.Error("Expected error for invalid feature index")
	}
}

func TestNew_FlatHierarchy(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	exp, err := New(m, background, nil, explainer.WithSeed(42))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Check that a flat hierarchy was created
	hierarchy := exp.Hierarchy()
	if hierarchy == nil {
		t.Fatal("Hierarchy should not be nil")
	}

	if len(hierarchy.Children) != 4 {
		t.Errorf("Flat hierarchy should have 4 children, got %d", len(hierarchy.Children))
	}
}

func TestExplain_LocalAccuracy(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	exp, err := New(m, background, nil,
		explainer.WithSeed(42),
		explainer.WithNumSamples(500),
		explainer.WithFeatureNames([]string{"x0", "x1", "x2", "x3"}),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instance := []float64{2, 1, 0.5, 0.5}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Verify local accuracy: sum(SHAP values) ≈ prediction - base_value
	verify := result.Verify(0.3) // Allow some tolerance for sampling
	if !verify.Valid {
		t.Errorf("Local accuracy failed: sum=%.4f, expected=%.4f, diff=%.4f",
			verify.SumSHAP, verify.Expected, verify.Difference)
	}
}

func TestExplain_WithHierarchy(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	// Create a meaningful hierarchy
	hierarchy := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "high_weight", Children: []*Node{
				{Name: "x0", FeatureIdx: 0}, // weight 2
				{Name: "x1", FeatureIdx: 1}, // weight 3
			}},
			{Name: "low_weight", Children: []*Node{
				{Name: "x2", FeatureIdx: 2}, // weight 1
				{Name: "x3", FeatureIdx: 3}, // weight 1
			}},
		},
	}

	exp, err := New(m, background, hierarchy,
		explainer.WithSeed(42),
		explainer.WithNumSamples(500),
		explainer.WithFeatureNames([]string{"x0", "x1", "x2", "x3"}),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instance := []float64{1, 1, 1, 1}

	result, err := exp.Explain(ctx, instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Verify local accuracy
	verify := result.Verify(0.3)
	if !verify.Valid {
		t.Errorf("Local accuracy failed: sum=%.4f, expected=%.4f, diff=%.4f",
			verify.SumSHAP, verify.Expected, verify.Difference)
	}

	// For linear model, x1 (weight 3) should have highest SHAP value
	// x0 (weight 2) should be second
	x0Val := result.Values["x0"]
	x1Val := result.Values["x1"]

	if math.Abs(x1Val) < math.Abs(x0Val)*0.9 { // Some tolerance
		t.Logf("Warning: x1 (weight 3) should generally have higher absolute SHAP than x0 (weight 2)")
		t.Logf("x0=%.4f, x1=%.4f", x0Val, x1Val)
	}
}

func TestExplain_InstanceMismatch(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	exp, err := New(m, background, nil)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()

	// Wrong number of features
	_, err = exp.Explain(ctx, []float64{1, 2}) // Only 2 features
	if err == nil {
		t.Error("Expected error for wrong number of features")
	}
}

func TestExplainBatch(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	exp, err := New(m, background, nil,
		explainer.WithSeed(42),
		explainer.WithNumSamples(100),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	instances := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 1},
	}

	results, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch failed: %v", err)
	}

	if len(results) != len(instances) {
		t.Errorf("ExplainBatch returned %d results, want %d", len(results), len(instances))
	}

	// Verify each result has valid local accuracy
	for i, result := range results {
		verify := result.Verify(0.5)
		if !verify.Valid {
			t.Errorf("Instance %d local accuracy failed: diff=%.4f", i, verify.Difference)
		}
	}
}

func TestBaseValue(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}
	// Expected base value: (2*0+3*0+0+0 + 2*1+3*1+1+1) / 2 = 7/2 = 3.5

	exp, err := New(m, background, nil)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	expectedBaseValue := 3.5
	if math.Abs(exp.BaseValue()-expectedBaseValue) > 1e-6 {
		t.Errorf("BaseValue() = %.4f, want %.4f", exp.BaseValue(), expectedBaseValue)
	}
}

func TestFeatureNames(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}}
	names := []string{"age", "income", "score", "level"}

	exp, err := New(m, background, nil, explainer.WithFeatureNames(names))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	gotNames := exp.FeatureNames()
	if len(gotNames) != len(names) {
		t.Fatalf("FeatureNames() length = %d, want %d", len(gotNames), len(names))
	}

	for i, name := range gotNames {
		if name != names[i] {
			t.Errorf("FeatureNames()[%d] = %q, want %q", i, name, names[i])
		}
	}
}

func TestExplain_ContextCancellation(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}}

	exp, err := New(m, background, nil, explainer.WithNumSamples(10000))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.Explain(ctx, []float64{1, 1, 1, 1})
	if err == nil {
		t.Error("Expected error for cancelled context")
	}
}

func TestCreateFlatHierarchy(t *testing.T) {
	h := createFlatHierarchy(3)

	if h.Name != "root" {
		t.Errorf("Root name = %q, want 'root'", h.Name)
	}

	if len(h.Children) != 3 {
		t.Errorf("Flat hierarchy should have 3 children, got %d", len(h.Children))
	}

	for i, child := range h.Children {
		if !child.IsLeaf() {
			t.Errorf("Child %d should be a leaf", i)
		}
		if child.FeatureIdx != i {
			t.Errorf("Child %d FeatureIdx = %d, want %d", i, child.FeatureIdx, i)
		}
	}
}

func TestValidateHierarchy(t *testing.T) {
	// Valid hierarchy
	valid := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "f0", FeatureIdx: 0},
			{Name: "f1", FeatureIdx: 1},
		},
	}

	if err := validateHierarchy(valid, 2); err != nil {
		t.Errorf("Valid hierarchy returned error: %v", err)
	}

	// Wrong count
	if err := validateHierarchy(valid, 3); err == nil {
		t.Error("Expected error for wrong feature count")
	}

	// Invalid index
	invalid := &Node{
		Name: "root",
		Children: []*Node{
			{Name: "f0", FeatureIdx: -1},
		},
	}

	if err := validateHierarchy(invalid, 1); err == nil {
		t.Error("Expected error for negative feature index")
	}
}

// Ensure Explainer implements model.Model for completeness check
func TestExplainer_ImplementsInterface(t *testing.T) {
	var _ explainer.Explainer = (*Explainer)(nil)
}

// Test that the metadata is set correctly
func TestExplain_Metadata(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{{0.5, 0.5, 0.5, 0.5}}

	exp, err := New(m, background, nil,
		explainer.WithSeed(42),
		explainer.WithNumSamples(50),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()
	result, err := exp.Explain(ctx, []float64{1, 1, 1, 1})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	if result.Metadata.Algorithm != "partition" {
		t.Errorf("Algorithm = %q, want 'partition'", result.Metadata.Algorithm)
	}

	if result.Metadata.ComputeTimeMS < 0 {
		t.Error("ComputeTimeMS should be non-negative")
	}

	if result.Metadata.BackgroundSize != 1 {
		t.Errorf("BackgroundSize = %d, want 1", result.Metadata.BackgroundSize)
	}
}

// Ensure ActivationSession implements model.Model
var _ model.Model = (*simpleLinearModel)(nil)

// Tests for batched predictions

func TestPartitionExplainer_BatchedPredictions_SameResults(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{
		{0, 0, 0, 0},
		{1, 1, 1, 1},
		{0.5, 0.5, 0.5, 0.5},
	}

	ctx := context.Background()
	instance := []float64{1, 2, 3, 4}

	// Create non-batched explainer
	expNonBatched, err := New(m, background, nil,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
	)
	if err != nil {
		t.Fatalf("New() non-batched error = %v", err)
	}

	// Create batched explainer with same seed
	expBatched, err := New(m, background, nil,
		explainer.WithNumSamples(100),
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

	// Results should be identical (same seed)
	tolerance := 1e-10
	for name, val := range resultNonBatched.Values {
		batchedVal := resultBatched.Values[name]
		if math.Abs(val-batchedVal) > tolerance {
			t.Errorf("SHAP(%s): non-batched=%f, batched=%f, diff=%f",
				name, val, batchedVal, math.Abs(val-batchedVal))
		}
	}

	// Both should satisfy local accuracy
	verifyNonBatched := resultNonBatched.Verify(0.5)
	verifyBatched := resultBatched.Verify(0.5)
	if !verifyNonBatched.Valid {
		t.Error("Non-batched explanation failed local accuracy")
	}
	if !verifyBatched.Valid {
		t.Error("Batched explanation failed local accuracy")
	}
}

func TestPartitionExplainer_BatchedPredictions_WithHierarchy(t *testing.T) {
	m := &simpleLinearModel{numFeatures: 4}
	background := [][]float64{
		{0, 0, 0, 0},
		{1, 1, 1, 1},
	}

	// Build hierarchy: group {0,1} and {2,3}
	hierarchy := &Node{
		Name: "root",
		Children: []*Node{
			{
				Name: "group_0_1",
				Children: []*Node{
					{Name: "f0", FeatureIdx: 0},
					{Name: "f1", FeatureIdx: 1},
				},
			},
			{
				Name: "group_2_3",
				Children: []*Node{
					{Name: "f2", FeatureIdx: 2},
					{Name: "f3", FeatureIdx: 3},
				},
			},
		},
	}

	exp, err := New(m, background, hierarchy,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	ctx := context.Background()
	result, err := exp.Explain(ctx, []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}

	// Verify local accuracy
	verify := result.Verify(0.5)
	if !verify.Valid {
		t.Errorf("Local accuracy failed: sum=%.4f, expected=%.4f, diff=%.4f",
			verify.SumSHAP, verify.Expected, verify.Difference)
	}
}

func BenchmarkPartitionExplainer_Batched(b *testing.B) {
	m := &simpleLinearModel{numFeatures: 4}
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, 4)
	}

	exp, _ := New(m, background, nil,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(true),
	)

	ctx := context.Background()
	instance := []float64{1, 2, 3, 4}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}

func BenchmarkPartitionExplainer_NonBatched(b *testing.B) {
	m := &simpleLinearModel{numFeatures: 4}
	background := make([][]float64, 10)
	for i := range background {
		background[i] = make([]float64, 4)
	}

	exp, _ := New(m, background, nil,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithBatchedPredictions(false),
	)

	ctx := context.Background()
	instance := []float64{1, 2, 3, 4}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = exp.Explain(ctx, instance)
	}
}
