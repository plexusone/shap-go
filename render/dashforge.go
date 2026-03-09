package render

import (
	"encoding/json"
	"fmt"

	"github.com/plexusone/shap-go/explanation"
)

// DashboardWidget represents a widget in a dashforge dashboard.
type DashboardWidget struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Title        string          `json:"title,omitempty"`
	Position     WidgetPosition  `json:"position"`
	DataSourceID string          `json:"dataSourceId,omitempty"`
	Config       json.RawMessage `json:"config"`
}

// WidgetPosition defines the position and size of a widget in the grid.
type WidgetPosition struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

// DashboardDataSource represents an inline data source for dashforge.
type DashboardDataSource struct {
	ID   string      `json:"id"`
	Type string      `json:"type"`
	Data interface{} `json:"data,omitempty"`
}

// SHAPDashboard represents a complete SHAP explanation dashboard.
type SHAPDashboard struct {
	ID          string                `json:"id"`
	Title       string                `json:"title"`
	Description string                `json:"description,omitempty"`
	Version     string                `json:"version,omitempty"`
	DataSources []DashboardDataSource `json:"dataSources"`
	Widgets     []DashboardWidget     `json:"widgets"`
}

// LocalExplanationDashboard generates a dashforge dashboard for a single explanation.
func (r *Renderer) LocalExplanationDashboard(exp *explanation.Explanation, title string) (*SHAPDashboard, error) {
	if title == "" {
		title = "SHAP Local Explanation"
	}

	// Generate chart data
	waterfallIR := r.WaterfallChartIR(exp, "Feature Contributions")
	forceData := r.ForceChartData(exp)

	// Serialize configs
	waterfallConfig, err := json.Marshal(waterfallIR)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal waterfall config: %w", err)
	}

	// Metric config for prediction
	predictionMetric := map[string]interface{}{
		"valueField": "prediction",
		"format":     "number",
		"label":      "Prediction",
	}
	predMetricConfig, _ := json.Marshal(predictionMetric)

	// Metric config for base value
	baseMetric := map[string]interface{}{
		"valueField": "base_value",
		"format":     "number",
		"label":      "Base Value",
	}
	baseMetricConfig, _ := json.Marshal(baseMetric)

	// Text config for interpretation
	topFeatures := exp.TopFeatures(3)
	interpretationText := "**Top Contributors:**\n\n"
	for i, f := range topFeatures {
		direction := "increases"
		if f.SHAPValue < 0 {
			direction = "decreases"
		}
		interpretationText += fmt.Sprintf("%d. **%s** %s prediction by %.4f\n", i+1, f.Name, direction, f.SHAPValue)
	}
	textConfig := map[string]interface{}{
		"content": interpretationText,
		"format":  "markdown",
	}
	textConfigJSON, _ := json.Marshal(textConfig)

	// Table config for all contributions
	tableConfig := map[string]interface{}{
		"columns": []map[string]interface{}{
			{"field": "name", "header": "Feature"},
			{"field": "shap_value", "header": "SHAP Value", "format": "number"},
		},
	}
	tableConfigJSON, _ := json.Marshal(tableConfig)

	// Build data sources
	dataSources := []DashboardDataSource{
		{
			ID:   "prediction-data",
			Type: "inline",
			Data: map[string]interface{}{
				"prediction": exp.Prediction,
				"base_value": exp.BaseValue,
			},
		},
		{
			ID:   "shap-values",
			Type: "inline",
			Data: forceData.Features,
		},
	}

	// Build widgets
	widgets := []DashboardWidget{
		// Prediction metric
		{
			ID:           "prediction-metric",
			Type:         "metric",
			Title:        "Prediction",
			Position:     WidgetPosition{X: 0, Y: 0, W: 3, H: 2},
			DataSourceID: "prediction-data",
			Config:       predMetricConfig,
		},
		// Base value metric
		{
			ID:           "base-metric",
			Type:         "metric",
			Title:        "Base Value",
			Position:     WidgetPosition{X: 3, Y: 0, W: 3, H: 2},
			DataSourceID: "prediction-data",
			Config:       baseMetricConfig,
		},
		// Interpretation text
		{
			ID:       "interpretation",
			Type:     "text",
			Title:    "Interpretation",
			Position: WidgetPosition{X: 6, Y: 0, W: 6, H: 2},
			Config:   textConfigJSON,
		},
		// Waterfall chart
		{
			ID:           "waterfall-chart",
			Type:         "chart",
			Title:        "Feature Contributions (Waterfall)",
			Position:     WidgetPosition{X: 0, Y: 2, W: 8, H: 5},
			DataSourceID: "shap-values",
			Config:       waterfallConfig,
		},
		// Feature table
		{
			ID:           "feature-table",
			Type:         "table",
			Title:        "All Features",
			Position:     WidgetPosition{X: 8, Y: 2, W: 4, H: 5},
			DataSourceID: "shap-values",
			Config:       tableConfigJSON,
		},
	}

	return &SHAPDashboard{
		ID:          fmt.Sprintf("shap-local-%s", exp.ID),
		Title:       title,
		Description: "SHAP explanation for a single prediction",
		Version:     "1.0.0",
		DataSources: dataSources,
		Widgets:     widgets,
	}, nil
}

// GlobalExplanationDashboard generates a dashforge dashboard for multiple explanations.
func (r *Renderer) GlobalExplanationDashboard(es *ExplanationSet, title string) (*SHAPDashboard, error) {
	if title == "" {
		title = "SHAP Global Explanation"
	}

	// Generate chart data
	importanceIR := r.FeatureImportanceChartIR(es, "Feature Importance", 10)
	summaryIR := r.SummaryChartIR(es, "SHAP Summary")

	importanceConfig, _ := json.Marshal(importanceIR)
	summaryConfig, _ := json.Marshal(summaryIR)

	// Compute statistics
	meanAbs := es.MeanAbsoluteSHAP()
	importanceData := make([]map[string]interface{}, 0, len(meanAbs))
	for name, imp := range meanAbs {
		importanceData = append(importanceData, map[string]interface{}{
			"feature":    name,
			"importance": imp,
		})
	}

	// Build data sources
	dataSources := []DashboardDataSource{
		{
			ID:   "importance-data",
			Type: "inline",
			Data: importanceData,
		},
		{
			ID:   "summary-data",
			Type: "inline",
			Data: map[string]interface{}{
				"num_explanations": len(es.Explanations),
				"num_features":     len(es.FeatureNames),
				"model_id":         es.ModelID,
			},
		},
	}

	// Metric configs
	numExplConfig, _ := json.Marshal(map[string]interface{}{
		"valueField": "num_explanations",
		"format":     "number",
		"label":      "Predictions Explained",
	})

	numFeatConfig, _ := json.Marshal(map[string]interface{}{
		"valueField": "num_features",
		"format":     "number",
		"label":      "Features",
	})

	// Table config
	tableConfig, _ := json.Marshal(map[string]interface{}{
		"columns": []map[string]interface{}{
			{"field": "feature", "header": "Feature"},
			{"field": "importance", "header": "Mean |SHAP|", "format": "number"},
		},
		"sortable":   true,
		"pagination": map[string]interface{}{"enabled": true, "pageSize": 10},
	})

	// Build widgets
	widgets := []DashboardWidget{
		// Number of explanations metric
		{
			ID:           "num-explanations",
			Type:         "metric",
			Title:        "Predictions",
			Position:     WidgetPosition{X: 0, Y: 0, W: 3, H: 2},
			DataSourceID: "summary-data",
			Config:       numExplConfig,
		},
		// Number of features metric
		{
			ID:           "num-features",
			Type:         "metric",
			Title:        "Features",
			Position:     WidgetPosition{X: 3, Y: 0, W: 3, H: 2},
			DataSourceID: "summary-data",
			Config:       numFeatConfig,
		},
		// Feature importance chart
		{
			ID:           "importance-chart",
			Type:         "chart",
			Title:        "Feature Importance",
			Position:     WidgetPosition{X: 0, Y: 2, W: 6, H: 5},
			DataSourceID: "importance-data",
			Config:       importanceConfig,
		},
		// Summary chart
		{
			ID:           "summary-chart",
			Type:         "chart",
			Title:        "SHAP Summary",
			Position:     WidgetPosition{X: 6, Y: 2, W: 6, H: 5},
			DataSourceID: "importance-data",
			Config:       summaryConfig,
		},
		// Importance table
		{
			ID:           "importance-table",
			Type:         "table",
			Title:        "Feature Rankings",
			Position:     WidgetPosition{X: 0, Y: 7, W: 12, H: 4},
			DataSourceID: "importance-data",
			Config:       tableConfig,
		},
	}

	return &SHAPDashboard{
		ID:          fmt.Sprintf("shap-global-%s", es.ModelID),
		Title:       title,
		Description: "SHAP global explanation across multiple predictions",
		Version:     "1.0.0",
		DataSources: dataSources,
		Widgets:     widgets,
	}, nil
}

// ToJSON serializes a SHAPDashboard to JSON.
func (d *SHAPDashboard) ToJSON() ([]byte, error) {
	return json.Marshal(d)
}

// ToJSONPretty serializes a SHAPDashboard to indented JSON.
func (d *SHAPDashboard) ToJSONPretty() ([]byte, error) {
	return json.MarshalIndent(d, "", "  ")
}
