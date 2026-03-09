package model

import (
	"context"
	"errors"
	"testing"
)

func TestFuncModel_Predict(t *testing.T) {
	// Simple linear model: y = x0 + 2*x1
	fn := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] + 2*input[1], nil
	}

	m := NewFuncModel(fn, 2)

	ctx := context.Background()
	result, err := m.Predict(ctx, []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	expected := 5.0 // 1 + 2*2
	if result != expected {
		t.Errorf("Predict() = %f, want %f", result, expected)
	}
}

func TestFuncModel_Predict_WrongFeatures(t *testing.T) {
	fn := func(ctx context.Context, input []float64) (float64, error) {
		return input[0], nil
	}

	m := NewFuncModel(fn, 2)

	ctx := context.Background()
	_, err := m.Predict(ctx, []float64{1.0})
	if err == nil {
		t.Error("Predict() should error with wrong number of features")
	}
}

func TestFuncModel_PredictBatch(t *testing.T) {
	fn := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * input[1], nil
	}

	m := NewFuncModel(fn, 2)

	ctx := context.Background()
	inputs := [][]float64{
		{2.0, 3.0},
		{4.0, 5.0},
		{1.0, 1.0},
	}

	results, err := m.PredictBatch(ctx, inputs)
	if err != nil {
		t.Fatalf("PredictBatch() error = %v", err)
	}

	expected := []float64{6.0, 20.0, 1.0}
	if len(results) != len(expected) {
		t.Fatalf("PredictBatch() returned %d results, want %d", len(results), len(expected))
	}

	for i, r := range results {
		if r != expected[i] {
			t.Errorf("PredictBatch()[%d] = %f, want %f", i, r, expected[i])
		}
	}
}

func TestFuncModel_PredictBatch_Error(t *testing.T) {
	fn := func(ctx context.Context, input []float64) (float64, error) {
		if input[0] < 0 {
			return 0, errors.New("negative input not allowed")
		}
		return input[0], nil
	}

	m := NewFuncModel(fn, 1)

	ctx := context.Background()
	inputs := [][]float64{
		{1.0},
		{-1.0}, // This will error
		{2.0},
	}

	_, err := m.PredictBatch(ctx, inputs)
	if err == nil {
		t.Error("PredictBatch() should error when a prediction fails")
	}
}

func TestFuncModel_NumFeatures(t *testing.T) {
	fn := func(ctx context.Context, input []float64) (float64, error) {
		return 0, nil
	}

	m := NewFuncModel(fn, 5)

	if m.NumFeatures() != 5 {
		t.Errorf("NumFeatures() = %d, want 5", m.NumFeatures())
	}
}

func TestFuncModel_Close(t *testing.T) {
	fn := func(ctx context.Context, input []float64) (float64, error) {
		return 0, nil
	}

	m := NewFuncModel(fn, 1)

	if err := m.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestFuncModel_ImplementsModel(t *testing.T) {
	var _ Model = (*FuncModel)(nil)
}
