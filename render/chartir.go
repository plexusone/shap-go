package render

import (
	"fmt"
	"sort"

	"github.com/grokify/gocharts/v2/charts/chartir"
	"github.com/plexusone/shap-go/explanation"
)

// WaterfallChartIR generates a ChartIR for a waterfall plot showing
// how features push the prediction from baseline to final value.
func (r *Renderer) WaterfallChartIR(exp *explanation.Explanation, title string) *chartir.ChartIR {
	if title == "" {
		title = "SHAP Waterfall Plot"
	}

	// Sort features by absolute SHAP value
	sorted := exp.SortedFeatures()

	// Build dataset rows: feature, start, contribution, end
	// For waterfall, we need cumulative positions
	rows := make([][]string, 0, len(sorted)+2)

	// Start with baseline
	rows = append(rows, []string{"Baseline", fmt.Sprintf("%.6f", exp.BaseValue), "0", fmt.Sprintf("%.6f", exp.BaseValue)})

	cumulative := exp.BaseValue
	for _, name := range sorted {
		shap := exp.Values[name]
		start := cumulative
		cumulative += shap

		// Format: feature name, start position, contribution, end position
		rows = append(rows, []string{
			name,
			fmt.Sprintf("%.6f", start),
			fmt.Sprintf("%.6f", shap),
			fmt.Sprintf("%.6f", cumulative),
		})
	}

	// End with prediction
	rows = append(rows, []string{"Prediction", fmt.Sprintf("%.6f", exp.Prediction), "0", fmt.Sprintf("%.6f", exp.Prediction)})

	return &chartir.ChartIR{
		Title: title,
		Datasets: []chartir.Dataset{
			{
				ID: "waterfall",
				Columns: []chartir.Column{
					{Name: "feature", Type: chartir.ColumnTypeString},
					{Name: "start", Type: chartir.ColumnTypeNumber},
					{Name: "contribution", Type: chartir.ColumnTypeNumber},
					{Name: "end", Type: chartir.ColumnTypeNumber},
				},
				Rows: rows,
			},
		},
		Marks: []chartir.Mark{
			{
				ID:        "positive",
				DatasetID: "waterfall",
				Geometry:  chartir.GeometryBar,
				Encode:    chartir.Encode{X: "contribution", Y: "feature"},
				Style:     &chartir.Style{Color: r.Theme.PositiveColor},
				Name:      "Positive",
			},
		},
		Axes: []chartir.Axis{
			{ID: "x", Type: chartir.AxisTypeValue, Position: chartir.AxisPositionBottom, Name: "SHAP Value"},
			{ID: "y", Type: chartir.AxisTypeCategory, Position: chartir.AxisPositionLeft, Name: "Feature"},
		},
		Tooltip: &chartir.Tooltip{Show: true, Trigger: chartir.TooltipTriggerAxis},
		Grid:    &chartir.Grid{Left: "120", Right: "40", Top: "60", Bottom: "40"},
	}
}

// FeatureImportanceChartIR generates a ChartIR for a feature importance bar chart
// showing mean absolute SHAP values.
func (r *Renderer) FeatureImportanceChartIR(es *ExplanationSet, title string, topN int) *chartir.ChartIR {
	if title == "" {
		title = "SHAP Feature Importance"
	}

	// Compute mean absolute SHAP values
	meanAbs := es.MeanAbsoluteSHAP()

	// Sort by importance
	type featureImportance struct {
		name       string
		importance float64
	}
	features := make([]featureImportance, 0, len(meanAbs))
	for name, imp := range meanAbs {
		features = append(features, featureImportance{name, imp})
	}
	sort.Slice(features, func(i, j int) bool {
		return features[i].importance > features[j].importance
	})

	// Limit to topN
	if topN > 0 && topN < len(features) {
		features = features[:topN]
	}

	// Build dataset rows (reversed for horizontal bar chart)
	rows := make([][]string, len(features))
	for i := range features {
		// Reverse order so most important is at top
		f := features[len(features)-1-i]
		rows[i] = []string{f.name, fmt.Sprintf("%.6f", f.importance)}
	}

	return &chartir.ChartIR{
		Title: title,
		Datasets: []chartir.Dataset{
			{
				ID: "importance",
				Columns: []chartir.Column{
					{Name: "feature", Type: chartir.ColumnTypeString},
					{Name: "importance", Type: chartir.ColumnTypeNumber},
				},
				Rows: rows,
			},
		},
		Marks: []chartir.Mark{
			{
				ID:        "bars",
				DatasetID: "importance",
				Geometry:  chartir.GeometryBar,
				Encode:    chartir.Encode{X: "importance", Y: "feature"},
				Style:     &chartir.Style{Color: r.Theme.PositiveColor},
			},
		},
		Axes: []chartir.Axis{
			{ID: "x", Type: chartir.AxisTypeValue, Position: chartir.AxisPositionBottom, Name: "mean(|SHAP value|)"},
			{ID: "y", Type: chartir.AxisTypeCategory, Position: chartir.AxisPositionLeft},
		},
		Tooltip: &chartir.Tooltip{Show: true, Trigger: chartir.TooltipTriggerAxis},
		Grid:    &chartir.Grid{Left: "120", Right: "40", Top: "60", Bottom: "40"},
	}
}

// SummaryChartIR generates a ChartIR for a SHAP summary scatter plot
// showing SHAP value distribution across all predictions.
func (r *Renderer) SummaryChartIR(es *ExplanationSet, title string) *chartir.ChartIR {
	if title == "" {
		title = "SHAP Summary Plot"
	}

	if len(es.Explanations) == 0 {
		return &chartir.ChartIR{Title: title}
	}

	// Compute mean absolute SHAP for ordering
	meanAbs := es.MeanAbsoluteSHAP()

	// Sort features by importance
	type featureImportance struct {
		name       string
		importance float64
	}
	features := make([]featureImportance, 0, len(meanAbs))
	for name, imp := range meanAbs {
		features = append(features, featureImportance{name, imp})
	}
	sort.Slice(features, func(i, j int) bool {
		return features[i].importance > features[j].importance
	})

	// Build scatter data: feature index, SHAP value, feature value (for color)
	rows := make([][]string, 0)
	for _, exp := range es.Explanations {
		for i, f := range features {
			shapVal := exp.Values[f.name]
			featureVal := 0.0
			if exp.FeatureValues != nil {
				featureVal = exp.FeatureValues[f.name]
			}
			// Row: feature_index, shap_value, feature_value
			rows = append(rows, []string{
				fmt.Sprintf("%d", i),
				fmt.Sprintf("%.6f", shapVal),
				fmt.Sprintf("%.6f", featureVal),
			})
		}
	}

	// Build feature name mapping for y-axis
	featureNames := make([]string, len(features))
	for i, f := range features {
		featureNames[i] = f.name
	}

	return &chartir.ChartIR{
		Title: title,
		Datasets: []chartir.Dataset{
			{
				ID: "summary",
				Columns: []chartir.Column{
					{Name: "feature_idx", Type: chartir.ColumnTypeNumber},
					{Name: "shap_value", Type: chartir.ColumnTypeNumber},
					{Name: "feature_value", Type: chartir.ColumnTypeNumber},
				},
				Rows: rows,
			},
		},
		Marks: []chartir.Mark{
			{
				ID:        "scatter",
				DatasetID: "summary",
				Geometry:  chartir.GeometryScatter,
				Encode: chartir.Encode{
					X:     "shap_value",
					Y:     "feature_idx",
					Color: "feature_value",
				},
			},
		},
		Axes: []chartir.Axis{
			{ID: "x", Type: chartir.AxisTypeValue, Position: chartir.AxisPositionBottom, Name: "SHAP value"},
			{ID: "y", Type: chartir.AxisTypeCategory, Position: chartir.AxisPositionLeft},
		},
		Tooltip: &chartir.Tooltip{Show: true, Trigger: chartir.TooltipTriggerItem},
	}
}

// DependenceChartIR generates a ChartIR for a SHAP dependence plot
// showing how a feature's value relates to its SHAP contribution.
func (r *Renderer) DependenceChartIR(es *ExplanationSet, featureName string, title string) *chartir.ChartIR {
	if title == "" {
		title = fmt.Sprintf("SHAP Dependence: %s", featureName)
	}

	if len(es.Explanations) == 0 {
		return &chartir.ChartIR{Title: title}
	}

	// Build scatter data: feature value, SHAP value
	rows := make([][]string, 0, len(es.Explanations))
	for _, exp := range es.Explanations {
		shapVal, ok := exp.Values[featureName]
		if !ok {
			continue
		}
		featureVal := 0.0
		if exp.FeatureValues != nil {
			featureVal = exp.FeatureValues[featureName]
		}
		rows = append(rows, []string{
			fmt.Sprintf("%.6f", featureVal),
			fmt.Sprintf("%.6f", shapVal),
		})
	}

	opacity := 0.6

	return &chartir.ChartIR{
		Title: title,
		Datasets: []chartir.Dataset{
			{
				ID: "dependence",
				Columns: []chartir.Column{
					{Name: "feature_value", Type: chartir.ColumnTypeNumber},
					{Name: "shap_value", Type: chartir.ColumnTypeNumber},
				},
				Rows: rows,
			},
		},
		Marks: []chartir.Mark{
			{
				ID:        "scatter",
				DatasetID: "dependence",
				Geometry:  chartir.GeometryScatter,
				Encode: chartir.Encode{
					X: "feature_value",
					Y: "shap_value",
				},
				Style: &chartir.Style{Color: r.Theme.PositiveColor, Opacity: &opacity},
			},
		},
		Axes: []chartir.Axis{
			{ID: "x", Type: chartir.AxisTypeValue, Position: chartir.AxisPositionBottom, Name: featureName},
			{ID: "y", Type: chartir.AxisTypeValue, Position: chartir.AxisPositionLeft, Name: "SHAP value"},
		},
		Tooltip: &chartir.Tooltip{Show: true, Trigger: chartir.TooltipTriggerItem},
	}
}

// ForceChartData generates data for a force plot (horizontal stacked bar).
// This is a simplified representation suitable for custom rendering.
type ForceChartData struct {
	BaseValue  float64        `json:"base_value"`
	Prediction float64        `json:"prediction"`
	Features   []ForceFeature `json:"features"`
}

// ForceFeature represents a feature in a force plot.
type ForceFeature struct {
	Name      string  `json:"name"`
	Value     float64 `json:"value,omitempty"`
	SHAPValue float64 `json:"shap_value"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
}

// ForceChartData generates data for rendering a force plot.
func (r *Renderer) ForceChartData(exp *explanation.Explanation) *ForceChartData {
	sorted := exp.SortedFeatures()

	features := make([]ForceFeature, 0, len(sorted))
	cumulative := exp.BaseValue

	for _, name := range sorted {
		shap := exp.Values[name]
		start := cumulative
		cumulative += shap

		f := ForceFeature{
			Name:      name,
			SHAPValue: shap,
			Start:     start,
			End:       cumulative,
		}
		if exp.FeatureValues != nil {
			f.Value = exp.FeatureValues[name]
		}
		features = append(features, f)
	}

	return &ForceChartData{
		BaseValue:  exp.BaseValue,
		Prediction: exp.Prediction,
		Features:   features,
	}
}
