// Example: Generating Pandoc Markdown SHAP reports
//
// This example generates a complete SHAP explanation report in Pandoc Markdown
// format, suitable for conversion to PDF, HTML, or other formats.
//
// To convert to PDF:
//   go run main.go > report.md
//   pandoc report.md -o report.pdf
//
// To convert to HTML:
//   pandoc report.md -o report.html --standalone
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/kernel"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
	"github.com/plexusone/shap-go/render"
)

func main() {
	// Generate explanations
	explanations := generateExplanations()
	renderer := render.NewRenderer()

	// Output to stdout (can be redirected to file)
	out := os.Stdout

	// === YAML Frontmatter ===
	fmt.Fprintln(out, "---")
	fmt.Fprintln(out, "title: \"SHAP Explanation Report\"")
	fmt.Fprintln(out, "subtitle: \"Loan Approval Model Analysis\"")
	fmt.Fprintln(out, "author: \"SHAP-Go\"")
	fmt.Fprintf(out, "date: \"%s\"\n", time.Now().Format("January 2, 2006"))
	fmt.Fprintln(out, "geometry: margin=1in")
	fmt.Fprintln(out, "colorlinks: true")
	fmt.Fprintln(out, "---")
	fmt.Fprintln(out)

	// === Executive Summary ===
	fmt.Fprintln(out, "# Executive Summary")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "This report provides SHAP (SHapley Additive exPlanations) analysis for a loan approval model.")
	fmt.Fprintln(out, "SHAP values explain how each feature contributes to individual predictions, enabling transparent")
	fmt.Fprintln(out, "and interpretable machine learning decisions.")
	fmt.Fprintln(out)

	// === Single Prediction Explanation ===
	fmt.Fprintln(out, "# Individual Prediction Analysis")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "## Applicant Profile")
	fmt.Fprintln(out)
	exp := explanations[0]
	fmt.Fprintln(out, "| Feature | Value |")
	fmt.Fprintln(out, "|---------|-------|")
	for _, name := range exp.FeatureNames {
		fmt.Fprintf(out, "| %s | %.2f |\n", name, exp.FeatureValues[name])
	}
	fmt.Fprintln(out)

	// Prediction summary
	fmt.Fprintln(out, "## Prediction Breakdown")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "| Metric | Value |")
	fmt.Fprintln(out, "|--------|-------|")
	fmt.Fprintf(out, "| Base Value (Average) | %.4f |\n", exp.BaseValue)
	fmt.Fprintf(out, "| Final Prediction | %.4f |\n", exp.Prediction)
	fmt.Fprintf(out, "| Net Effect | %+.4f |\n", exp.Prediction-exp.BaseValue)
	fmt.Fprintln(out)

	// Feature contributions with visual bars
	fmt.Fprintln(out, "## Feature Contributions")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Each feature's SHAP value shows its contribution to pushing the prediction above or below the baseline.")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "| Feature | Value | SHAP | Direction | Impact |")
	fmt.Fprintln(out, "|---------|-------|------|-----------|--------|")

	sorted := exp.SortedFeatures()
	maxAbs := 0.0
	for _, name := range sorted {
		abs := exp.Values[name]
		if abs < 0 {
			abs = -abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	for _, name := range sorted {
		shap := exp.Values[name]
		val := exp.FeatureValues[name]
		direction := "(+) increases"
		if shap < 0 {
			direction = "(-) decreases"
		}
		// Create simple text bar (ASCII-safe for PDF)
		barLen := int((shap / maxAbs) * 10)
		if barLen < 0 {
			barLen = -barLen
		}
		bar := strings.Repeat("#", barLen)
		if shap < 0 {
			bar = strings.Repeat("-", barLen)
		}
		fmt.Fprintf(out, "| %s | %.2f | %+.4f | %s | `%s` |\n", name, val, shap, direction, bar)
	}
	fmt.Fprintln(out)

	// ASCII Waterfall
	fmt.Fprintln(out, "## Waterfall Visualization")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "```")
	fmt.Fprint(out, renderer.WaterfallASCII(exp, 50))
	fmt.Fprintln(out, "```")
	fmt.Fprintln(out)

	// === Global Analysis ===
	fmt.Fprintln(out, "# Global Feature Importance")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Aggregated across all predictions, feature importance is measured as the mean absolute SHAP value.")
	fmt.Fprintln(out)

	expSet := render.NewExplanationSet(explanations)
	meanAbs := expSet.MeanAbsoluteSHAP()

	fmt.Fprintln(out, "| Rank | Feature | Mean |SHAP| | Importance |")
	fmt.Fprintln(out, "|------|---------|-------------|------------|")

	// Sort by importance
	type fi struct {
		name string
		imp  float64
	}
	features := make([]fi, 0)
	maxImp := 0.0
	for name, imp := range meanAbs {
		features = append(features, fi{name, imp})
		if imp > maxImp {
			maxImp = imp
		}
	}
	// Sort
	for i := 0; i < len(features); i++ {
		for j := i + 1; j < len(features); j++ {
			if features[j].imp > features[i].imp {
				features[i], features[j] = features[j], features[i]
			}
		}
	}

	for i, f := range features {
		barLen := int((f.imp / maxImp) * 20)
		bar := strings.Repeat("#", barLen)
		fmt.Fprintf(out, "| %d | %s | %.4f | `%s` |\n", i+1, f.name, f.imp, bar)
	}
	fmt.Fprintln(out)

	// === Multiple Predictions Summary ===
	fmt.Fprintln(out, "# Prediction Summary")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "| # | Prediction | Top Positive | Top Negative |")
	fmt.Fprintln(out, "|---|------------|--------------|--------------|")

	for i, e := range explanations {
		topPos, topNeg := "", ""
		maxPos, minNeg := 0.0, 0.0
		for name, shap := range e.Values {
			if shap > maxPos {
				maxPos = shap
				topPos = fmt.Sprintf("%s (%+.2f)", name, shap)
			}
			if shap < minNeg {
				minNeg = shap
				topNeg = fmt.Sprintf("%s (%+.2f)", name, shap)
			}
		}
		if topNeg == "" {
			topNeg = "none"
		}
		fmt.Fprintf(out, "| %d | %.4f | %s | %s |\n", i+1, e.Prediction, topPos, topNeg)
	}
	fmt.Fprintln(out)

	// === Verification ===
	fmt.Fprintln(out, "# Local Accuracy Verification")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "SHAP values satisfy the local accuracy property: the sum of SHAP values equals the difference")
	fmt.Fprintln(out, "between the prediction and baseline.")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "| # | Sum(SHAP) | Pred - Base | Difference | Status |")
	fmt.Fprintln(out, "|---|-----------|-------------|------------|--------|")

	for i, e := range explanations {
		result := e.Verify(1e-6)
		status := "PASS"
		if !result.Valid {
			status = "FAIL"
		}
		fmt.Fprintf(out, "| %d | %.6f | %.6f | %.2e | %s |\n",
			i+1, result.SumSHAP, result.Expected, result.Difference, status)
	}
	fmt.Fprintln(out)

	// === Methodology ===
	fmt.Fprintln(out, "# Methodology")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "- **Algorithm:** KernelSHAP (model-agnostic)")
	fmt.Fprintln(out, "- **Samples:** 200 coalition samples per explanation")
	fmt.Fprintln(out, "- **Background:** 5 reference instances")
	fmt.Fprintln(out, "- **Library:** SHAP-Go (github.com/plexusone/shap-go)")
	fmt.Fprintln(out)

	fmt.Fprintln(out, "---")
	fmt.Fprintln(out)
	fmt.Fprintf(out, "*Report generated on %s*\n", time.Now().Format("2006-01-02 15:04:05"))
}

func generateExplanations() []*explanation.Explanation {
	// Loan approval model
	predict := func(ctx context.Context, input []float64) (float64, error) {
		income, age, creditScore, debtRatio := input[0], input[1], input[2], input[3]
		score := 0.3 + (income/100000)*0.4 + (age/100)*0.15 + (creditScore/850)*0.3 - (debtRatio)*0.2
		return score, nil
	}

	m := model.NewFuncModel(predict, 4)

	background := [][]float64{
		{50000, 30, 650, 0.3},
		{60000, 35, 700, 0.25},
		{70000, 40, 720, 0.2},
		{55000, 32, 680, 0.35},
		{65000, 38, 710, 0.28},
	}

	exp, _ := kernel.New(m, background,
		explainer.WithNumSamples(200),
		explainer.WithSeed(42),
		explainer.WithFeatureNames([]string{"income", "age", "credit_score", "debt_ratio"}),
	)

	instances := [][]float64{
		{75000, 35, 750, 0.15}, // Strong applicant
		{40000, 25, 620, 0.45}, // Weak applicant
		{60000, 42, 700, 0.30}, // Average applicant
		{90000, 50, 800, 0.10}, // Excellent applicant
		{45000, 28, 650, 0.40}, // Below average
	}

	ctx := context.Background()
	results := make([]*explanation.Explanation, len(instances))

	for i, inst := range instances {
		result, _ := exp.Explain(ctx, inst)
		// Store feature values for reporting
		result.FeatureValues = map[string]float64{
			"income":       inst[0],
			"age":          inst[1],
			"credit_score": inst[2],
			"debt_ratio":   inst[3],
		}
		results[i] = result
	}

	return results
}
