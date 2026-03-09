package render

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/plexusone/shap-go/explanation"
)

// MarkdownOptions configures Markdown output generation.
type MarkdownOptions struct {
	// Title is the document title.
	Title string

	// IncludeMetadata adds YAML frontmatter for Pandoc.
	IncludeMetadata bool

	// Author is included in frontmatter if IncludeMetadata is true.
	Author string

	// Date is included in frontmatter. If empty, uses current date.
	Date string

	// TopN limits feature tables to top N features by importance.
	TopN int

	// DecimalPlaces controls numeric precision in tables.
	DecimalPlaces int

	// IncludeVerification adds local accuracy verification section.
	IncludeVerification bool
}

// DefaultMarkdownOptions returns default options for Markdown generation.
func DefaultMarkdownOptions() MarkdownOptions {
	return MarkdownOptions{
		Title:               "SHAP Explanation Report",
		IncludeMetadata:     true,
		TopN:                10,
		DecimalPlaces:       4,
		IncludeVerification: true,
	}
}

// ExplanationMarkdown generates Pandoc Markdown for a single explanation.
func (r *Renderer) ExplanationMarkdown(exp *explanation.Explanation, opts MarkdownOptions) string {
	var sb strings.Builder

	// YAML frontmatter for Pandoc
	if opts.IncludeMetadata {
		sb.WriteString("---\n")
		fmt.Fprintf(&sb, "title: \"%s\"\n", opts.Title)
		if opts.Author != "" {
			fmt.Fprintf(&sb, "author: \"%s\"\n", opts.Author)
		}
		date := opts.Date
		if date == "" {
			date = time.Now().Format("2006-01-02")
		}
		fmt.Fprintf(&sb, "date: \"%s\"\n", date)
		sb.WriteString("---\n\n")
	}

	// Header
	fmt.Fprintf(&sb, "# %s\n\n", opts.Title)

	// Model info
	if exp.ModelID != "" {
		fmt.Fprintf(&sb, "**Model:** %s\n\n", exp.ModelID)
	}

	// Prediction summary
	sb.WriteString("## Prediction Summary\n\n")
	sb.WriteString("| Metric | Value |\n")
	sb.WriteString("|--------|-------|\n")
	fmt.Fprintf(&sb, "| Prediction | %.*f |\n", opts.DecimalPlaces, exp.Prediction)
	fmt.Fprintf(&sb, "| Base Value | %.*f |\n", opts.DecimalPlaces, exp.BaseValue)
	fmt.Fprintf(&sb, "| Difference | %.*f |\n", opts.DecimalPlaces, exp.Prediction-exp.BaseValue)
	sb.WriteString("\n")

	// Feature contributions table
	sb.WriteString("## Feature Contributions\n\n")
	sb.WriteString(r.featureContributionTable(exp, opts))
	sb.WriteString("\n")

	// Top positive contributors
	sb.WriteString("### Top Positive Contributors\n\n")
	sb.WriteString(r.topContributorsTable(exp, opts, true))
	sb.WriteString("\n")

	// Top negative contributors
	sb.WriteString("### Top Negative Contributors\n\n")
	sb.WriteString(r.topContributorsTable(exp, opts, false))
	sb.WriteString("\n")

	// Verification
	if opts.IncludeVerification {
		sb.WriteString("## Local Accuracy Verification\n\n")
		result := exp.Verify(1e-6)
		sb.WriteString("| Check | Value |\n")
		sb.WriteString("|-------|-------|\n")
		fmt.Fprintf(&sb, "| Sum of SHAP values | %.*f |\n", opts.DecimalPlaces, result.SumSHAP)
		fmt.Fprintf(&sb, "| Expected (pred - base) | %.*f |\n", opts.DecimalPlaces, result.Expected)
		fmt.Fprintf(&sb, "| Difference | %.2e |\n", result.Difference)
		status := "✅ PASSED"
		if !result.Valid {
			status = "❌ FAILED"
		}
		fmt.Fprintf(&sb, "| Status | %s |\n", status)
		sb.WriteString("\n")
	}

	// Metadata
	if exp.Metadata.Algorithm != "" {
		sb.WriteString("## Computation Details\n\n")
		sb.WriteString("| Parameter | Value |\n")
		sb.WriteString("|-----------|-------|\n")
		fmt.Fprintf(&sb, "| Algorithm | %s |\n", exp.Metadata.Algorithm)
		if exp.Metadata.NumSamples > 0 {
			fmt.Fprintf(&sb, "| Samples | %d |\n", exp.Metadata.NumSamples)
		}
		if exp.Metadata.BackgroundSize > 0 {
			fmt.Fprintf(&sb, "| Background Size | %d |\n", exp.Metadata.BackgroundSize)
		}
		if exp.Metadata.ComputeTimeMS > 0 {
			fmt.Fprintf(&sb, "| Compute Time | %d ms |\n", exp.Metadata.ComputeTimeMS)
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// featureContributionTable generates a Markdown table of all feature contributions.
func (r *Renderer) featureContributionTable(exp *explanation.Explanation, opts MarkdownOptions) string {
	var sb strings.Builder

	// Header
	hasValues := len(exp.FeatureValues) > 0
	if hasValues {
		sb.WriteString("| Feature | Value | SHAP Contribution | Direction |\n")
		sb.WriteString("|---------|-------|-------------------|----------|\n")
	} else {
		sb.WriteString("| Feature | SHAP Contribution | Direction |\n")
		sb.WriteString("|---------|-------------------|----------|\n")
	}

	// Sort by absolute SHAP value
	sorted := exp.SortedFeatures()

	// Limit to topN
	if opts.TopN > 0 && opts.TopN < len(sorted) {
		sorted = sorted[:opts.TopN]
	}

	for _, name := range sorted {
		shap := exp.Values[name]
		direction := "↑"
		if shap < 0 {
			direction = "↓"
		}

		if hasValues {
			val := exp.FeatureValues[name]
			fmt.Fprintf(&sb, "| %s | %.*f | %+.*f | %s |\n",
				name, opts.DecimalPlaces, val, opts.DecimalPlaces, shap, direction)
		} else {
			fmt.Fprintf(&sb, "| %s | %+.*f | %s |\n",
				name, opts.DecimalPlaces, shap, direction)
		}
	}

	return sb.String()
}

// topContributorsTable generates a table of top positive or negative contributors.
func (r *Renderer) topContributorsTable(exp *explanation.Explanation, opts MarkdownOptions, positive bool) string {
	var sb strings.Builder

	// Filter and sort features
	type contrib struct {
		name string
		shap float64
	}
	contribs := make([]contrib, 0)
	for name, shap := range exp.Values {
		if (positive && shap > 0) || (!positive && shap < 0) {
			contribs = append(contribs, contrib{name, shap})
		}
	}

	// Sort by absolute value descending
	sort.Slice(contribs, func(i, j int) bool {
		ai := contribs[i].shap
		aj := contribs[j].shap
		if ai < 0 {
			ai = -ai
		}
		if aj < 0 {
			aj = -aj
		}
		return ai > aj
	})

	// Limit
	limit := 5
	if opts.TopN > 0 && opts.TopN < limit {
		limit = opts.TopN
	}
	if limit > len(contribs) {
		limit = len(contribs)
	}

	if limit == 0 {
		sb.WriteString("*No contributors in this category.*\n")
		return sb.String()
	}

	sb.WriteString("| Rank | Feature | SHAP Value |\n")
	sb.WriteString("|------|---------|------------|\n")

	for i := 0; i < limit; i++ {
		c := contribs[i]
		fmt.Fprintf(&sb, "| %d | %s | %+.*f |\n", i+1, c.name, opts.DecimalPlaces, c.shap)
	}

	return sb.String()
}

// ExplanationSetMarkdown generates Pandoc Markdown for multiple explanations (global view).
func (r *Renderer) ExplanationSetMarkdown(es *ExplanationSet, opts MarkdownOptions) string {
	var sb strings.Builder

	// YAML frontmatter
	if opts.IncludeMetadata {
		sb.WriteString("---\n")
		fmt.Fprintf(&sb, "title: \"%s\"\n", opts.Title)
		if opts.Author != "" {
			fmt.Fprintf(&sb, "author: \"%s\"\n", opts.Author)
		}
		date := opts.Date
		if date == "" {
			date = time.Now().Format("2006-01-02")
		}
		fmt.Fprintf(&sb, "date: \"%s\"\n", date)
		sb.WriteString("---\n\n")
	}

	// Header
	fmt.Fprintf(&sb, "# %s\n\n", opts.Title)

	if es.ModelID != "" {
		fmt.Fprintf(&sb, "**Model:** %s\n\n", es.ModelID)
	}

	fmt.Fprintf(&sb, "**Number of Explanations:** %d\n\n", len(es.Explanations))

	// Feature importance table
	sb.WriteString("## Global Feature Importance\n\n")
	sb.WriteString("Feature importance is measured as the mean absolute SHAP value across all predictions.\n\n")

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

	// Limit
	if opts.TopN > 0 && opts.TopN < len(features) {
		features = features[:opts.TopN]
	}

	sb.WriteString("| Rank | Feature | Mean |SHAP| |\n")
	sb.WriteString("|------|---------|---------------|\n")
	for i, f := range features {
		fmt.Fprintf(&sb, "| %d | %s | %.*f |\n", i+1, f.name, opts.DecimalPlaces, f.importance)
	}
	sb.WriteString("\n")

	// Statistics summary
	sb.WriteString("## Summary Statistics\n\n")
	if len(es.Explanations) > 0 {
		var sumPred, sumBase float64
		for _, exp := range es.Explanations {
			sumPred += exp.Prediction
			sumBase += exp.BaseValue
		}
		n := float64(len(es.Explanations))

		sb.WriteString("| Metric | Value |\n")
		sb.WriteString("|--------|-------|\n")
		fmt.Fprintf(&sb, "| Mean Prediction | %.*f |\n", opts.DecimalPlaces, sumPred/n)
		fmt.Fprintf(&sb, "| Mean Base Value | %.*f |\n", opts.DecimalPlaces, sumBase/n)
		fmt.Fprintf(&sb, "| Number of Features | %d |\n", len(es.FeatureNames))
	}

	return sb.String()
}

// WaterfallASCII generates a simple ASCII waterfall representation.
func (r *Renderer) WaterfallASCII(exp *explanation.Explanation, width int) string {
	var sb strings.Builder

	sorted := exp.SortedFeatures()

	fmt.Fprintf(&sb, "Baseline: %.4f\n", exp.BaseValue)
	sb.WriteString(strings.Repeat("-", width) + "\n")

	for _, name := range sorted {
		shap := exp.Values[name]
		sign := "+"
		if shap < 0 {
			sign = ""
		}
		fmt.Fprintf(&sb, "%-20s %s%.4f\n", name, sign, shap)
	}

	sb.WriteString(strings.Repeat("-", width) + "\n")
	fmt.Fprintf(&sb, "Prediction: %.4f\n", exp.Prediction)

	return sb.String()
}
