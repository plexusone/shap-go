// Package render provides visualization outputs for SHAP explanations.
// It supports multiple output formats:
//   - ChartIR (for echartify/ECharts web rendering)
//   - Pandoc Markdown (for PDF generation)
//   - Dashboard widgets (for dashforge integration)
package render

import (
	"github.com/plexusone/shap-go/explanation"
)

// ChartType represents the type of SHAP visualization.
type ChartType string

const (
	// ChartWaterfall shows how features push prediction from baseline to final value.
	// Best for explaining a single prediction to stakeholders.
	ChartWaterfall ChartType = "waterfall"

	// ChartFeatureImportance shows mean absolute SHAP values across predictions.
	// Best for understanding which features drive the model globally.
	ChartFeatureImportance ChartType = "feature_importance"

	// ChartSummary shows SHAP value distribution for all features (beeswarm plot).
	// Best for understanding both importance and direction of feature effects.
	ChartSummary ChartType = "summary"

	// ChartDependence shows how a single feature's value affects SHAP contribution.
	// Best for understanding non-linear effects and thresholds.
	ChartDependence ChartType = "dependence"
)

// Renderer provides methods to render SHAP explanations in various formats.
type Renderer struct {
	// Theme controls colors and styling.
	Theme Theme
}

// Theme defines colors and styling for SHAP visualizations.
type Theme struct {
	// PositiveColor is the color for positive SHAP contributions.
	PositiveColor string `json:"positive_color"`

	// NegativeColor is the color for negative SHAP contributions.
	NegativeColor string `json:"negative_color"`

	// BaselineColor is the color for the baseline value.
	BaselineColor string `json:"baseline_color"`

	// FontFamily is the font for labels and text.
	FontFamily string `json:"font_family"`
}

// DefaultTheme returns the default SHAP visualization theme.
func DefaultTheme() Theme {
	return Theme{
		PositiveColor: "#ef4444", // Red (increases prediction)
		NegativeColor: "#3b82f6", // Blue (decreases prediction)
		BaselineColor: "#6b7280", // Gray
		FontFamily:    "system-ui, -apple-system, sans-serif",
	}
}

// NewRenderer creates a new Renderer with the default theme.
func NewRenderer() *Renderer {
	return &Renderer{
		Theme: DefaultTheme(),
	}
}

// NewRendererWithTheme creates a new Renderer with a custom theme.
func NewRendererWithTheme(theme Theme) *Renderer {
	return &Renderer{
		Theme: theme,
	}
}

// ExplanationSet represents multiple explanations for batch visualization.
type ExplanationSet struct {
	// Explanations is the list of individual explanations.
	Explanations []*explanation.Explanation

	// FeatureNames is the ordered list of feature names.
	FeatureNames []string

	// ModelID identifies the model.
	ModelID string
}

// NewExplanationSet creates an ExplanationSet from a slice of explanations.
func NewExplanationSet(explanations []*explanation.Explanation) *ExplanationSet {
	if len(explanations) == 0 {
		return &ExplanationSet{}
	}

	// Use feature names from first explanation
	var featureNames []string
	var modelID string
	if len(explanations) > 0 {
		featureNames = explanations[0].FeatureNames
		modelID = explanations[0].ModelID
	}

	return &ExplanationSet{
		Explanations: explanations,
		FeatureNames: featureNames,
		ModelID:      modelID,
	}
}

// MeanAbsoluteSHAP computes mean |SHAP| for each feature across all explanations.
func (es *ExplanationSet) MeanAbsoluteSHAP() map[string]float64 {
	if len(es.Explanations) == 0 {
		return nil
	}

	sums := make(map[string]float64)
	counts := make(map[string]int)

	for _, exp := range es.Explanations {
		for name, value := range exp.Values {
			if value < 0 {
				sums[name] += -value
			} else {
				sums[name] += value
			}
			counts[name]++
		}
	}

	means := make(map[string]float64)
	for name, sum := range sums {
		means[name] = sum / float64(counts[name])
	}

	return means
}
