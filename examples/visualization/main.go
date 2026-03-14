// Example: Generating SHAP visualizations
//
// This example demonstrates how to create visualization data from SHAP
// explanations using the render package. The output is ChartIR format
// which can be converted to ECharts, Chart.js, or other visualization libraries.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/permutation"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
	"github.com/plexusone/shap-go/render"
)

func main() {
	fmt.Println("SHAP Visualization Example")
	fmt.Println("==========================")
	fmt.Println()

	// Create a model and explainer
	explanations := generateExplanations()

	// Create renderer with default theme
	renderer := render.NewRenderer()

	// 1. Waterfall Plot - for single prediction explanation
	fmt.Println("1. Waterfall Plot (Single Prediction)")
	fmt.Println("--------------------------------------")
	waterfallChart := renderer.WaterfallChartIR(explanations[0], "Loan Approval Explanation")
	printChartSummary(waterfallChart)
	fmt.Println()

	// 2. Feature Importance - global view across all predictions
	fmt.Println("2. Feature Importance (Global)")
	fmt.Println("------------------------------")
	expSet := render.NewExplanationSet(explanations)
	importanceChart := renderer.FeatureImportanceChartIR(expSet, "Feature Importance", 10)
	printChartSummary(importanceChart)
	fmt.Println()

	// 3. Summary Plot - SHAP distribution across all predictions
	fmt.Println("3. Summary Plot (Distribution)")
	fmt.Println("------------------------------")
	summaryChart := renderer.SummaryChartIR(expSet, "SHAP Summary")
	printChartSummary(summaryChart)
	fmt.Println()

	// 4. Dependence Plot - single feature's effect
	fmt.Println("4. Dependence Plot (Single Feature)")
	fmt.Println("------------------------------------")
	dependenceChart := renderer.DependenceChartIR(expSet, "income", "Income vs SHAP")
	printChartSummary(dependenceChart)
	fmt.Println()

	// 5. Force Plot Data - for interactive visualization
	fmt.Println("5. Force Plot Data")
	fmt.Println("------------------")
	forceData := renderer.ForceChartData(explanations[0])
	fmt.Printf("Base Value: %.4f\n", forceData.BaseValue)
	fmt.Printf("Prediction: %.4f\n", forceData.Prediction)
	fmt.Println("Features (sorted by contribution):")
	for _, f := range forceData.Features {
		direction := "+"
		if f.SHAPValue < 0 {
			direction = "-"
		}
		fmt.Printf("  %s %s: %.4f (value=%.2f, range=[%.4f, %.4f])\n",
			direction, f.Name, f.SHAPValue, f.Value, f.Start, f.End)
	}
	fmt.Println()

	// 6. Custom Theme
	fmt.Println("6. Custom Theme")
	fmt.Println("---------------")
	customTheme := render.Theme{
		PositiveColor: "#22c55e", // Green for positive
		NegativeColor: "#f97316", // Orange for negative
		BaselineColor: "#64748b", // Slate
		FontFamily:    "Inter, system-ui, sans-serif",
	}
	customRenderer := render.NewRendererWithTheme(customTheme)
	customChart := customRenderer.WaterfallChartIR(explanations[0], "Custom Themed Chart")
	fmt.Printf("Positive color: %s\n", customChart.Marks[0].Style.Color)
	fmt.Println()

	// 7. Export full ChartIR as JSON (for use with echartify)
	fmt.Println("7. Full ChartIR JSON Export")
	fmt.Println("---------------------------")
	jsonData, err := json.MarshalIndent(waterfallChart, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal: %v", err)
	}
	// Show truncated output
	output := string(jsonData)
	if len(output) > 500 {
		output = output[:500] + "\n  ... (truncated)"
	}
	fmt.Println(output)
}

func printChartSummary(chart *render.ChartIR) {
	fmt.Printf("Title: %s\n", chart.Title)
	fmt.Printf("Datasets: %d\n", len(chart.Datasets))
	if len(chart.Datasets) > 0 {
		ds := chart.Datasets[0]
		fmt.Printf("  - %s: %d columns, %d rows\n", ds.ID, len(ds.Columns), len(ds.Rows))
	}
	fmt.Printf("Marks: %d\n", len(chart.Marks))
	for _, m := range chart.Marks {
		fmt.Printf("  - %s (%s)\n", m.ID, m.Geometry)
	}
	fmt.Printf("Axes: %d\n", len(chart.Axes))
}

func generateExplanations() []*explanation.Explanation {
	// Create a simple model
	predict := func(ctx context.Context, input []float64) (float64, error) {
		// Loan approval score based on income, age, credit_score
		income, age, creditScore := input[0], input[1], input[2]
		score := 0.5 + (income-50000)/100000 + (age-30)/100 + (creditScore-650)/500
		return score, nil
	}

	m := model.NewFuncModel(predict, 3)

	background := [][]float64{
		{50000, 30, 650},
		{60000, 35, 700},
		{40000, 25, 600},
		{80000, 45, 750},
		{55000, 32, 680},
	}

	exp, err := permutation.New(m, background,
		explainer.WithNumSamples(50),
		explainer.WithSeed(42),
		explainer.WithFeatureNames([]string{"income", "age", "credit_score"}),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	// Generate explanations for various instances
	instances := [][]float64{
		{75000, 35, 720}, // High approval
		{35000, 22, 580}, // Low approval
		{60000, 40, 700}, // Medium approval
		{90000, 50, 800}, // Very high approval
		{45000, 28, 620}, // Below average
	}

	ctx := context.Background()
	results := make([]*explanation.Explanation, len(instances))

	for i, inst := range instances {
		result, err := exp.Explain(ctx, inst)
		if err != nil {
			log.Fatalf("Failed to explain: %v", err)
		}
		results[i] = result
	}

	return results
}
