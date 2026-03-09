// Package explainer provides interfaces and implementations for SHAP explainers.
package explainer

import (
	"context"

	"github.com/plexusone/shap-go/explanation"
)

// Explainer is the interface for computing SHAP explanations.
type Explainer interface {
	// Explain computes a SHAP explanation for a single instance.
	Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

	// ExplainBatch computes SHAP explanations for multiple instances.
	ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

	// BaseValue returns the expected model output on the background dataset.
	BaseValue() float64

	// FeatureNames returns the names of the features.
	FeatureNames() []string
}
