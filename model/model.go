// Package model defines the model interface for SHAP explainability.
package model

import (
	"context"
	"fmt"
)

// Model represents a predictive model that can be explained with SHAP.
type Model interface {
	// Predict returns the model's prediction for a single input.
	Predict(ctx context.Context, input []float64) (float64, error)

	// PredictBatch returns predictions for multiple inputs.
	// This can be more efficient than calling Predict multiple times.
	PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error)

	// NumFeatures returns the number of input features the model expects.
	NumFeatures() int

	// Close releases any resources held by the model.
	Close() error
}

// PredictFunc is a function type for making predictions.
type PredictFunc func(ctx context.Context, input []float64) (float64, error)

// FuncModel wraps a prediction function as a Model.
// This is useful for testing or wrapping simple models.
type FuncModel struct {
	fn          PredictFunc
	numFeatures int
}

// NewFuncModel creates a new FuncModel from a prediction function.
func NewFuncModel(fn PredictFunc, numFeatures int) *FuncModel {
	return &FuncModel{
		fn:          fn,
		numFeatures: numFeatures,
	}
}

// Predict implements Model.Predict.
func (m *FuncModel) Predict(ctx context.Context, input []float64) (float64, error) {
	if len(input) != m.numFeatures {
		return 0, fmt.Errorf("expected %d features, got %d", m.numFeatures, len(input))
	}
	return m.fn(ctx, input)
}

// PredictBatch implements Model.PredictBatch.
func (m *FuncModel) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error) {
	results := make([]float64, len(inputs))
	for i, input := range inputs {
		pred, err := m.Predict(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("prediction failed for input %d: %w", i, err)
		}
		results[i] = pred
	}
	return results, nil
}

// NumFeatures implements Model.NumFeatures.
func (m *FuncModel) NumFeatures() int {
	return m.numFeatures
}

// Close implements Model.Close.
func (m *FuncModel) Close() error {
	return nil
}

// Ensure FuncModel implements Model.
var _ Model = (*FuncModel)(nil)
