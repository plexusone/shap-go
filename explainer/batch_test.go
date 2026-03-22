package explainer

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/plexusone/shap-go/explanation"
)

// mockExplainer is a simple mock for testing batch operations.
type mockExplainer struct {
	delay        time.Duration
	failOnIndex  int // -1 means no failure
	callCount    atomic.Int32
	featureNames []string
	baseValue    float64
}

func newMockExplainer() *mockExplainer {
	return &mockExplainer{
		delay:        0,
		failOnIndex:  -1,
		featureNames: []string{"f0", "f1"},
		baseValue:    0.5,
	}
}

func (m *mockExplainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	m.callCount.Add(1)

	// Check context
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Simulate delay
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Simulate failure on specific index
	idx := int(m.callCount.Load()) - 1
	if m.failOnIndex >= 0 && idx == m.failOnIndex {
		return nil, errors.New("simulated error")
	}

	// Return mock explanation
	var sum float64
	for _, v := range instance {
		sum += v
	}

	values := make(map[string]float64)
	for i, name := range m.featureNames {
		if i < len(instance) {
			values[name] = instance[i]
		}
	}

	return &explanation.Explanation{
		Prediction:   sum,
		BaseValue:    m.baseValue,
		Values:       values,
		FeatureNames: m.featureNames,
	}, nil
}

func (m *mockExplainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, inst := range instances {
		exp, err := m.Explain(ctx, inst)
		if err != nil {
			return nil, err
		}
		results[i] = exp
	}
	return results, nil
}

func (m *mockExplainer) BaseValue() float64 {
	return m.baseValue
}

func (m *mockExplainer) FeatureNames() []string {
	return m.featureNames
}

func TestExplainBatchParallel_Basic(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}

	ctx := context.Background()
	results, err := ExplainBatchParallel(ctx, exp, instances, WithWorkers(2))
	if err != nil {
		t.Fatalf("ExplainBatchParallel failed: %v", err)
	}

	if len(results) != len(instances) {
		t.Errorf("len(results) = %d, want %d", len(results), len(instances))
	}

	// Check results are in correct order
	for i, result := range results {
		expectedSum := instances[i][0] + instances[i][1]
		if result.Prediction != expectedSum {
			t.Errorf("results[%d].Prediction = %v, want %v", i, result.Prediction, expectedSum)
		}
	}
}

func TestExplainBatchParallel_EmptyInput(t *testing.T) {
	exp := newMockExplainer()
	ctx := context.Background()

	results, err := ExplainBatchParallel(ctx, exp, [][]float64{})
	if err != nil {
		t.Fatalf("ExplainBatchParallel failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("len(results) = %d, want 0", len(results))
	}
}

func TestExplainBatchParallel_SingleInstance(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{{1.0, 2.0}}

	ctx := context.Background()
	results, err := ExplainBatchParallel(ctx, exp, instances, WithWorkers(4))
	if err != nil {
		t.Fatalf("ExplainBatchParallel failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("len(results) = %d, want 1", len(results))
	}
}

func TestExplainBatchParallel_MoreWorkersThanInstances(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	ctx := context.Background()
	results, err := ExplainBatchParallel(ctx, exp, instances, WithWorkers(10))
	if err != nil {
		t.Fatalf("ExplainBatchParallel failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}
}

func TestExplainBatchParallel_ContextCancellation(t *testing.T) {
	exp := newMockExplainer()
	exp.delay = 100 * time.Millisecond

	instances := make([][]float64, 10)
	for i := range instances {
		instances[i] = []float64{float64(i), float64(i)}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := ExplainBatchParallel(ctx, exp, instances, WithWorkers(4))
	if err == nil {
		t.Error("ExplainBatchParallel should fail with cancelled context")
	}
}

func TestExplainBatchParallel_StopOnError(t *testing.T) {
	exp := newMockExplainer()
	exp.failOnIndex = 2 // Fail on third instance

	instances := make([][]float64, 10)
	for i := range instances {
		instances[i] = []float64{float64(i), float64(i)}
	}

	ctx := context.Background()
	_, err := ExplainBatchParallel(ctx, exp, instances,
		WithWorkers(1), // Single worker for deterministic order
		WithStopOnError(true))
	if err == nil {
		t.Error("ExplainBatchParallel should fail when StopOnError is true")
	}
}

func TestExplainBatchParallel_ContinueOnError(t *testing.T) {
	exp := newMockExplainer()
	exp.failOnIndex = 2 // Fail on third call

	instances := make([][]float64, 5)
	for i := range instances {
		instances[i] = []float64{float64(i), float64(i)}
	}

	ctx := context.Background()
	// Use 2 workers to ensure we go through explainParallel (not the mock's ExplainBatch)
	results, err := ExplainBatchParallel(ctx, exp, instances,
		WithWorkers(2),
		WithStopOnError(false))

	// Should return partial results with error
	if err == nil {
		t.Error("ExplainBatchParallel should return error even with ContinueOnError")
	}

	// Results should not be nil when ContinueOnError is set
	if results == nil {
		t.Error("Results should not be nil when ContinueOnError is set")
	}

	// Count non-nil results - should have at least some successful ones
	successCount := 0
	for _, r := range results {
		if r != nil {
			successCount++
		}
	}
	if successCount == 0 {
		t.Error("Expected some successful results")
	}
}

func TestExplainBatchParallel_DefaultWorkers(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	ctx := context.Background()
	// Don't specify workers - should use default (NumCPU)
	results, err := ExplainBatchParallel(ctx, exp, instances)
	if err != nil {
		t.Fatalf("ExplainBatchParallel failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("len(results) = %d, want 2", len(results))
	}
}

func TestExplainBatchWithProgress(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}

	var progressCalls int
	var lastDone, lastTotal int

	ctx := context.Background()
	results, err := ExplainBatchWithProgress(ctx, exp, instances,
		func(done, total int) {
			progressCalls++
			lastDone = done
			lastTotal = total
		},
		WithWorkers(2))

	if err != nil {
		t.Fatalf("ExplainBatchWithProgress failed: %v", err)
	}

	if len(results) != len(instances) {
		t.Errorf("len(results) = %d, want %d", len(results), len(instances))
	}

	// Progress should have been called for each instance
	if progressCalls != len(instances) {
		t.Errorf("progressCalls = %d, want %d", progressCalls, len(instances))
	}

	// Final progress should show all done
	if lastDone != len(instances) || lastTotal != len(instances) {
		t.Errorf("final progress = %d/%d, want %d/%d", lastDone, lastTotal, len(instances), len(instances))
	}
}

func TestExplainBatchWithProgress_NilCallback(t *testing.T) {
	exp := newMockExplainer()
	instances := [][]float64{{1.0, 2.0}}

	ctx := context.Background()
	// Should work with nil progress callback
	results, err := ExplainBatchWithProgress(ctx, exp, instances, nil, WithWorkers(1))
	if err != nil {
		t.Fatalf("ExplainBatchWithProgress failed: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("len(results) = %d, want 1", len(results))
	}
}

func TestDefaultBatchConfig(t *testing.T) {
	config := DefaultBatchConfig()

	if config.Workers <= 0 {
		t.Errorf("Workers = %d, want > 0", config.Workers)
	}
	if !config.StopOnError {
		t.Error("StopOnError should default to true")
	}
}

func TestBatchOptions(t *testing.T) {
	config := BatchConfig{}

	WithWorkers(8)(&config)
	if config.Workers != 8 {
		t.Errorf("Workers = %d, want 8", config.Workers)
	}

	WithStopOnError(false)(&config)
	if config.StopOnError {
		t.Error("StopOnError should be false")
	}
}

// Ensure mockExplainer implements Explainer interface
var _ Explainer = (*mockExplainer)(nil)
