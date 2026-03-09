package render

import (
	"fmt"
	"sort"

	"github.com/plexusone/shap-go/explanation"
)

// ChartIR represents the intermediate representation for echartify.
// This is a simplified subset focused on SHAP visualizations.
type ChartIR struct {
	Title    string    `json:"title,omitempty"`
	Datasets []Dataset `json:"datasets"`
	Marks    []Mark    `json:"marks"`
	Axes     []Axis    `json:"axes,omitempty"`
	Legend   *Legend   `json:"legend,omitempty"`
	Tooltip  *Tooltip  `json:"tooltip,omitempty"`
	Grid     *Grid     `json:"grid,omitempty"`
}

// Dataset represents tabular data for the chart.
type Dataset struct {
	ID      string     `json:"id"`
	Columns []Column   `json:"columns"`
	Rows    [][]string `json:"rows"`
}

// Column defines a column in the dataset.
type Column struct {
	Name string `json:"name"`
	Type string `json:"type"` // "string" or "number"
}

// Mark represents a visual mark (series) in the chart.
type Mark struct {
	ID        string `json:"id"`
	DatasetID string `json:"datasetId"`
	Geometry  string `json:"geometry"` // "line", "bar", "scatter", "area"
	Encode    Encode `json:"encode"`
	Stack     string `json:"stack,omitempty"`
	Style     *Style `json:"style,omitempty"`
	Name      string `json:"name,omitempty"`
	Smooth    bool   `json:"smooth,omitempty"`
}

// Encode defines how data columns map to visual properties.
type Encode struct {
	X     string `json:"x,omitempty"`
	Y     string `json:"y,omitempty"`
	Value string `json:"value,omitempty"`
	Name  string `json:"name,omitempty"`
	Size  string `json:"size,omitempty"`
	Color string `json:"color,omitempty"`
}

// Style defines visual styling for marks.
type Style struct {
	Color       string  `json:"color,omitempty"`
	Opacity     float64 `json:"opacity,omitempty"`
	BorderColor string  `json:"borderColor,omitempty"`
	BorderWidth float64 `json:"borderWidth,omitempty"`
}

// Axis defines a chart axis.
type Axis struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`     // "category", "value", "time", "log"
	Position string   `json:"position"` // "bottom", "top", "left", "right"
	Name     string   `json:"name,omitempty"`
	Min      *float64 `json:"min,omitempty"`
	Max      *float64 `json:"max,omitempty"`
}

// Legend defines the chart legend.
type Legend struct {
	Show     bool   `json:"show"`
	Position string `json:"position,omitempty"` // "top", "bottom", "left", "right"
}

// Tooltip defines tooltip behavior.
type Tooltip struct {
	Show    bool   `json:"show"`
	Trigger string `json:"trigger,omitempty"` // "item", "axis"
}

// Grid defines chart grid/padding.
type Grid struct {
	Left   int `json:"left,omitempty"`
	Right  int `json:"right,omitempty"`
	Top    int `json:"top,omitempty"`
	Bottom int `json:"bottom,omitempty"`
}

// WaterfallChartIR generates a ChartIR for a waterfall plot showing
// how features push the prediction from baseline to final value.
func (r *Renderer) WaterfallChartIR(exp *explanation.Explanation, title string) *ChartIR {
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

	return &ChartIR{
		Title: title,
		Datasets: []Dataset{
			{
				ID: "waterfall",
				Columns: []Column{
					{Name: "feature", Type: "string"},
					{Name: "start", Type: "number"},
					{Name: "contribution", Type: "number"},
					{Name: "end", Type: "number"},
				},
				Rows: rows,
			},
		},
		Marks: []Mark{
			{
				ID:        "positive",
				DatasetID: "waterfall",
				Geometry:  "bar",
				Encode:    Encode{X: "contribution", Y: "feature"},
				Style:     &Style{Color: r.Theme.PositiveColor},
				Name:      "Positive",
			},
		},
		Axes: []Axis{
			{ID: "x", Type: "value", Position: "bottom", Name: "SHAP Value"},
			{ID: "y", Type: "category", Position: "left", Name: "Feature"},
		},
		Tooltip: &Tooltip{Show: true, Trigger: "axis"},
		Grid:    &Grid{Left: 120, Right: 40, Top: 60, Bottom: 40},
	}
}

// FeatureImportanceChartIR generates a ChartIR for a feature importance bar chart
// showing mean absolute SHAP values.
func (r *Renderer) FeatureImportanceChartIR(es *ExplanationSet, title string, topN int) *ChartIR {
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

	return &ChartIR{
		Title: title,
		Datasets: []Dataset{
			{
				ID: "importance",
				Columns: []Column{
					{Name: "feature", Type: "string"},
					{Name: "importance", Type: "number"},
				},
				Rows: rows,
			},
		},
		Marks: []Mark{
			{
				ID:        "bars",
				DatasetID: "importance",
				Geometry:  "bar",
				Encode:    Encode{X: "importance", Y: "feature"},
				Style:     &Style{Color: r.Theme.PositiveColor},
			},
		},
		Axes: []Axis{
			{ID: "x", Type: "value", Position: "bottom", Name: "mean(|SHAP value|)"},
			{ID: "y", Type: "category", Position: "left"},
		},
		Tooltip: &Tooltip{Show: true, Trigger: "axis"},
		Grid:    &Grid{Left: 120, Right: 40, Top: 60, Bottom: 40},
	}
}

// SummaryChartIR generates a ChartIR for a SHAP summary scatter plot
// showing SHAP value distribution across all predictions.
func (r *Renderer) SummaryChartIR(es *ExplanationSet, title string) *ChartIR {
	if title == "" {
		title = "SHAP Summary Plot"
	}

	if len(es.Explanations) == 0 {
		return &ChartIR{Title: title}
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

	return &ChartIR{
		Title: title,
		Datasets: []Dataset{
			{
				ID: "summary",
				Columns: []Column{
					{Name: "feature_idx", Type: "number"},
					{Name: "shap_value", Type: "number"},
					{Name: "feature_value", Type: "number"},
				},
				Rows: rows,
			},
		},
		Marks: []Mark{
			{
				ID:        "scatter",
				DatasetID: "summary",
				Geometry:  "scatter",
				Encode: Encode{
					X:     "shap_value",
					Y:     "feature_idx",
					Color: "feature_value",
				},
			},
		},
		Axes: []Axis{
			{ID: "x", Type: "value", Position: "bottom", Name: "SHAP value"},
			{ID: "y", Type: "category", Position: "left"},
		},
		Tooltip: &Tooltip{Show: true, Trigger: "item"},
	}
}

// DependenceChartIR generates a ChartIR for a SHAP dependence plot
// showing how a feature's value relates to its SHAP contribution.
func (r *Renderer) DependenceChartIR(es *ExplanationSet, featureName string, title string) *ChartIR {
	if title == "" {
		title = fmt.Sprintf("SHAP Dependence: %s", featureName)
	}

	if len(es.Explanations) == 0 {
		return &ChartIR{Title: title}
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

	return &ChartIR{
		Title: title,
		Datasets: []Dataset{
			{
				ID: "dependence",
				Columns: []Column{
					{Name: "feature_value", Type: "number"},
					{Name: "shap_value", Type: "number"},
				},
				Rows: rows,
			},
		},
		Marks: []Mark{
			{
				ID:        "scatter",
				DatasetID: "dependence",
				Geometry:  "scatter",
				Encode: Encode{
					X: "feature_value",
					Y: "shap_value",
				},
				Style: &Style{Color: r.Theme.PositiveColor, Opacity: 0.6},
			},
		},
		Axes: []Axis{
			{ID: "x", Type: "value", Position: "bottom", Name: featureName},
			{ID: "y", Type: "value", Position: "left", Name: "SHAP value"},
		},
		Tooltip: &Tooltip{Show: true, Trigger: "item"},
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
