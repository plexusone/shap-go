package render

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/plexusone/shap-go/explanation"
)

func createTestExplanation() *explanation.Explanation {
	return &explanation.Explanation{
		ID:         "test-123",
		ModelID:    "test-model",
		Prediction: 0.75,
		BaseValue:  0.50,
		Values: map[string]float64{
			"income":       0.15,
			"age":          0.08,
			"credit_score": 0.05,
			"tenure":       -0.03,
		},
		FeatureNames: []string{"income", "age", "credit_score", "tenure"},
		FeatureValues: map[string]float64{
			"income":       75000,
			"age":          35,
			"credit_score": 720,
			"tenure":       5,
		},
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "permutation",
			NumSamples:     100,
			BackgroundSize: 50,
		},
	}
}

func createTestExplanationSet() *ExplanationSet {
	exp1 := createTestExplanation()
	exp2 := &explanation.Explanation{
		ID:         "test-456",
		ModelID:    "test-model",
		Prediction: 0.65,
		BaseValue:  0.50,
		Values: map[string]float64{
			"income":       0.10,
			"age":          0.05,
			"credit_score": 0.02,
			"tenure":       -0.02,
		},
		FeatureNames: []string{"income", "age", "credit_score", "tenure"},
	}

	return NewExplanationSet([]*explanation.Explanation{exp1, exp2})
}

func TestDefaultTheme(t *testing.T) {
	theme := DefaultTheme()

	if theme.PositiveColor == "" {
		t.Error("PositiveColor should not be empty")
	}
	if theme.NegativeColor == "" {
		t.Error("NegativeColor should not be empty")
	}
}

func TestNewRenderer(t *testing.T) {
	r := NewRenderer()

	if r.Theme.PositiveColor == "" {
		t.Error("Renderer should have default theme")
	}
}

func TestExplanationSet_MeanAbsoluteSHAP(t *testing.T) {
	es := createTestExplanationSet()

	meanAbs := es.MeanAbsoluteSHAP()

	// income: (|0.15| + |0.10|) / 2 = 0.125
	expectedIncome := 0.125
	if diff := meanAbs["income"] - expectedIncome; diff > 0.001 || diff < -0.001 {
		t.Errorf("Mean |SHAP| for income = %f, want %f", meanAbs["income"], expectedIncome)
	}

	// tenure: (|-0.03| + |-0.02|) / 2 = 0.025
	expectedTenure := 0.025
	if diff := meanAbs["tenure"] - expectedTenure; diff > 0.001 || diff < -0.001 {
		t.Errorf("Mean |SHAP| for tenure = %f, want %f", meanAbs["tenure"], expectedTenure)
	}
}

func TestWaterfallChartIR(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	chartIR := r.WaterfallChartIR(exp, "Test Waterfall")

	if chartIR.Title != "Test Waterfall" {
		t.Errorf("Title = %s, want Test Waterfall", chartIR.Title)
	}

	if len(chartIR.Datasets) != 1 {
		t.Fatalf("Expected 1 dataset, got %d", len(chartIR.Datasets))
	}

	if chartIR.Datasets[0].ID != "waterfall" {
		t.Errorf("Dataset ID = %s, want waterfall", chartIR.Datasets[0].ID)
	}

	// Should have baseline + features + prediction rows
	expectedRows := len(exp.Values) + 2
	if len(chartIR.Datasets[0].Rows) != expectedRows {
		t.Errorf("Expected %d rows, got %d", expectedRows, len(chartIR.Datasets[0].Rows))
	}
}

func TestFeatureImportanceChartIR(t *testing.T) {
	r := NewRenderer()
	es := createTestExplanationSet()

	chartIR := r.FeatureImportanceChartIR(es, "Test Importance", 3)

	if chartIR.Title != "Test Importance" {
		t.Errorf("Title = %s, want Test Importance", chartIR.Title)
	}

	// Should have at most 3 rows (topN=3)
	if len(chartIR.Datasets[0].Rows) > 3 {
		t.Errorf("Expected at most 3 rows (topN=3), got %d", len(chartIR.Datasets[0].Rows))
	}
}

func TestDependenceChartIR(t *testing.T) {
	r := NewRenderer()
	es := createTestExplanationSet()

	chartIR := r.DependenceChartIR(es, "income", "")

	if !strings.Contains(chartIR.Title, "income") {
		t.Errorf("Title should contain feature name, got %s", chartIR.Title)
	}

	// Should have one point per explanation
	if len(chartIR.Datasets[0].Rows) != len(es.Explanations) {
		t.Errorf("Expected %d rows, got %d", len(es.Explanations), len(chartIR.Datasets[0].Rows))
	}
}

func TestForceChartData(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	data := r.ForceChartData(exp)

	if data.BaseValue != exp.BaseValue {
		t.Errorf("BaseValue = %f, want %f", data.BaseValue, exp.BaseValue)
	}

	if data.Prediction != exp.Prediction {
		t.Errorf("Prediction = %f, want %f", data.Prediction, exp.Prediction)
	}

	if len(data.Features) != len(exp.Values) {
		t.Errorf("Expected %d features, got %d", len(exp.Values), len(data.Features))
	}

	// Verify cumulative: last end should equal prediction
	lastEnd := data.Features[len(data.Features)-1].End
	if diff := lastEnd - exp.Prediction; diff > 0.0001 || diff < -0.0001 {
		t.Errorf("Last feature end = %f, should equal prediction %f", lastEnd, exp.Prediction)
	}
}

func TestExplanationMarkdown(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	opts := DefaultMarkdownOptions()
	opts.Title = "Test Report"

	md := r.ExplanationMarkdown(exp, opts)

	// Check for key sections
	if !strings.Contains(md, "---") {
		t.Error("Markdown should contain YAML frontmatter")
	}
	if !strings.Contains(md, "Test Report") {
		t.Error("Markdown should contain title")
	}
	if !strings.Contains(md, "Prediction Summary") {
		t.Error("Markdown should contain Prediction Summary section")
	}
	if !strings.Contains(md, "Feature Contributions") {
		t.Error("Markdown should contain Feature Contributions section")
	}
	if !strings.Contains(md, "Local Accuracy Verification") {
		t.Error("Markdown should contain verification section")
	}
	if !strings.Contains(md, "income") {
		t.Error("Markdown should contain feature names")
	}
}

func TestExplanationSetMarkdown(t *testing.T) {
	r := NewRenderer()
	es := createTestExplanationSet()

	opts := DefaultMarkdownOptions()
	opts.Title = "Global Report"

	md := r.ExplanationSetMarkdown(es, opts)

	if !strings.Contains(md, "Global Report") {
		t.Error("Markdown should contain title")
	}
	if !strings.Contains(md, "Global Feature Importance") {
		t.Error("Markdown should contain Global Feature Importance section")
	}
	if !strings.Contains(md, "Mean |SHAP|") {
		t.Error("Markdown should contain mean SHAP column")
	}
}

func TestWaterfallASCII(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	ascii := r.WaterfallASCII(exp, 40)

	if !strings.Contains(ascii, "Baseline") {
		t.Error("ASCII should contain Baseline")
	}
	if !strings.Contains(ascii, "Prediction") {
		t.Error("ASCII should contain Prediction")
	}
	if !strings.Contains(ascii, "income") {
		t.Error("ASCII should contain feature names")
	}
}

func TestLocalExplanationDashboard(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	dashboard, err := r.LocalExplanationDashboard(exp, "Test Dashboard")
	if err != nil {
		t.Fatalf("LocalExplanationDashboard error: %v", err)
	}

	if dashboard.Title != "Test Dashboard" {
		t.Errorf("Title = %s, want Test Dashboard", dashboard.Title)
	}

	if len(dashboard.DataSources) == 0 {
		t.Error("Dashboard should have data sources")
	}

	if len(dashboard.Widgets) == 0 {
		t.Error("Dashboard should have widgets")
	}

	// Check widget types
	hasChart := false
	hasMetric := false
	for _, w := range dashboard.Widgets {
		if w.Type == "chart" {
			hasChart = true
		}
		if w.Type == "metric" {
			hasMetric = true
		}
	}

	if !hasChart {
		t.Error("Dashboard should have chart widgets")
	}
	if !hasMetric {
		t.Error("Dashboard should have metric widgets")
	}
}

func TestGlobalExplanationDashboard(t *testing.T) {
	r := NewRenderer()
	es := createTestExplanationSet()

	dashboard, err := r.GlobalExplanationDashboard(es, "Global Dashboard")
	if err != nil {
		t.Fatalf("GlobalExplanationDashboard error: %v", err)
	}

	if dashboard.Title != "Global Dashboard" {
		t.Errorf("Title = %s, want Global Dashboard", dashboard.Title)
	}

	if len(dashboard.Widgets) < 3 {
		t.Errorf("Expected at least 3 widgets, got %d", len(dashboard.Widgets))
	}
}

func TestSHAPDashboard_ToJSON(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	dashboard, _ := r.LocalExplanationDashboard(exp, "Test")

	jsonData, err := dashboard.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON error: %v", err)
	}

	// Verify it's valid JSON
	var m map[string]interface{}
	if err := json.Unmarshal(jsonData, &m); err != nil {
		t.Fatalf("Produced invalid JSON: %v", err)
	}

	if m["title"] != "Test" {
		t.Errorf("JSON title = %v, want Test", m["title"])
	}
}

func TestChartIR_ValidStructure(t *testing.T) {
	r := NewRenderer()
	exp := createTestExplanation()

	chartIR := r.WaterfallChartIR(exp, "Test")

	// Verify all required fields are present
	if len(chartIR.Datasets) == 0 {
		t.Error("ChartIR should have datasets")
	}

	if len(chartIR.Marks) == 0 {
		t.Error("ChartIR should have marks")
	}

	// Verify marks reference valid datasets
	datasetIDs := make(map[string]bool)
	for _, ds := range chartIR.Datasets {
		datasetIDs[ds.ID] = true
	}

	for _, mark := range chartIR.Marks {
		if !datasetIDs[mark.DatasetID] {
			t.Errorf("Mark %s references unknown dataset %s", mark.ID, mark.DatasetID)
		}
	}

	// Verify columns match row widths
	for _, ds := range chartIR.Datasets {
		numCols := len(ds.Columns)
		for i, row := range ds.Rows {
			if len(row) != numCols {
				t.Errorf("Dataset %s row %d has %d values, expected %d",
					ds.ID, i, len(row), numCols)
			}
		}
	}
}
