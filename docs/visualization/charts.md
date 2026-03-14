# Visualization

SHAP-Go provides a `render` package that generates chart specifications in a portable JSON format called ChartIR (Chart Intermediate Representation).

## Overview

The render package doesn't directly create images. Instead, it produces JSON specifications that can be rendered by:

- **JavaScript libraries** (D3.js, Chart.js, ECharts)
- **Python libraries** (Matplotlib, Plotly)
- **Custom renderers** (any tool that can read JSON)

This approach keeps the Go code lightweight while enabling rich visualizations.

## Chart Types

### Waterfall Plot

Shows how each feature contributes to a single prediction:

```go
import (
    "encoding/json"
    "github.com/plexusone/shap-go/render"
)

// After getting an explanation
chart := render.Waterfall(explanation, render.WaterfallOptions{
    Title:       "Loan Default Prediction",
    MaxFeatures: 10,
    ShowValues:  true,
})

// Export to JSON
jsonData, _ := json.MarshalIndent(chart, "", "  ")
```

Output structure:

```json
{
  "type": "waterfall",
  "title": "Loan Default Prediction",
  "baseValue": 0.5,
  "prediction": 0.73,
  "features": [
    {"name": "income", "value": 0.15, "cumulative": 0.65},
    {"name": "debt_ratio", "value": 0.08, "cumulative": 0.73}
  ]
}
```

### Feature Importance (Bar Chart)

Shows average absolute SHAP values across multiple predictions:

```go
// Collect explanations for multiple instances
var explanations []*explainer.Explanation
for _, instance := range testData {
    exp, _ := exp.Explain(ctx, instance)
    explanations = append(explanations, exp)
}

// Generate importance chart
chart := render.FeatureImportance(explanations, render.ImportanceOptions{
    Title:       "Global Feature Importance",
    MaxFeatures: 15,
    SortBy:      "mean_abs",  // or "max_abs"
})
```

Output structure:

```json
{
  "type": "bar",
  "title": "Global Feature Importance",
  "orientation": "horizontal",
  "features": [
    {"name": "income", "importance": 0.234},
    {"name": "age", "importance": 0.189}
  ]
}
```

### Summary Plot (Beeswarm)

Shows SHAP value distribution for each feature:

```go
chart := render.Summary(explanations, featureValues, render.SummaryOptions{
    Title:       "SHAP Summary",
    MaxFeatures: 20,
    ColorScale:  "bluered",  // Feature value coloring
})
```

Output structure:

```json
{
  "type": "beeswarm",
  "title": "SHAP Summary",
  "colorScale": "bluered",
  "features": [
    {
      "name": "income",
      "points": [
        {"shap": 0.15, "featureValue": 75000, "normalized": 0.8},
        {"shap": -0.12, "featureValue": 35000, "normalized": 0.3}
      ]
    }
  ]
}
```

### Dependence Plot

Shows relationship between a feature's value and its SHAP value:

```go
chart := render.Dependence(explanations, featureValues, render.DependenceOptions{
    Feature:      "income",
    ColorFeature: "age",  // Optional: color by another feature
    Title:        "Income Dependence",
})
```

Output structure:

```json
{
  "type": "scatter",
  "title": "Income Dependence",
  "xAxis": {"name": "income", "label": "Feature Value"},
  "yAxis": {"name": "shap", "label": "SHAP Value"},
  "colorAxis": {"name": "age", "label": "Age"},
  "points": [
    {"x": 75000, "y": 0.15, "color": 35},
    {"x": 35000, "y": -0.12, "color": 62}
  ]
}
```

## Rendering ChartIR

### With D3.js (JavaScript)

```javascript
// Load the ChartIR JSON
const chart = await fetch('/api/explanation/chart').then(r => r.json());

if (chart.type === 'waterfall') {
    renderWaterfall(chart);
} else if (chart.type === 'bar') {
    renderBarChart(chart);
}

function renderWaterfall(chart) {
    const svg = d3.select('#chart').append('svg');
    // ... D3 rendering code
}
```

### With Plotly (Python)

```python
import json
import plotly.graph_objects as go

# Load ChartIR
with open('chart.json') as f:
    chart = json.load(f)

if chart['type'] == 'waterfall':
    fig = go.Figure(go.Waterfall(
        x=[f['name'] for f in chart['features']],
        y=[f['value'] for f in chart['features']],
        base=chart['baseValue']
    ))
    fig.show()
```

### With ECharts

```javascript
const chart = loadChartIR();

if (chart.type === 'bar') {
    const option = {
        title: { text: chart.title },
        xAxis: { type: 'value' },
        yAxis: {
            type: 'category',
            data: chart.features.map(f => f.name)
        },
        series: [{
            type: 'bar',
            data: chart.features.map(f => f.importance)
        }]
    };
    echarts.init(document.getElementById('chart')).setOption(option);
}
```

## Complete Example

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "os"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/tree"
    "github.com/plexusone/shap-go/render"
)

func main() {
    // Load model
    ensemble, _ := tree.LoadXGBoostModel("model.json")
    exp, _ := tree.New(ensemble)

    ctx := context.Background()

    // Explain multiple instances
    testData := loadTestData()
    var explanations []*explainer.Explanation
    var featureValues [][]float64

    for _, instance := range testData {
        explanation, _ := exp.Explain(ctx, instance)
        explanations = append(explanations, explanation)
        featureValues = append(featureValues, instance)
    }

    // Generate all chart types
    charts := map[string]interface{}{
        "waterfall": render.Waterfall(explanations[0], render.WaterfallOptions{
            Title:       "Individual Prediction",
            MaxFeatures: 10,
        }),
        "importance": render.FeatureImportance(explanations, render.ImportanceOptions{
            Title:       "Feature Importance",
            MaxFeatures: 15,
        }),
        "summary": render.Summary(explanations, featureValues, render.SummaryOptions{
            Title:       "SHAP Summary",
            MaxFeatures: 20,
        }),
        "dependence": render.Dependence(explanations, featureValues, render.DependenceOptions{
            Feature: "income",
            Title:   "Income Dependence",
        }),
    }

    // Save each chart
    for name, chart := range charts {
        data, _ := json.MarshalIndent(chart, "", "  ")
        os.WriteFile(fmt.Sprintf("%s.json", name), data, 0644)
    }
}
```

## ChartIR Specification

ChartIR is a simple JSON format designed for SHAP visualizations:

### Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Chart type: "waterfall", "bar", "beeswarm", "scatter" |
| `title` | string | Chart title |
| `subtitle` | string | Optional subtitle |

### Waterfall-Specific

| Field | Type | Description |
|-------|------|-------------|
| `baseValue` | float | Expected value (E[f(x)]) |
| `prediction` | float | Model output for this instance |
| `features` | array | Feature contributions |

### Bar-Specific

| Field | Type | Description |
|-------|------|-------------|
| `orientation` | string | "horizontal" or "vertical" |
| `features` | array | Feature importance values |

### Scatter-Specific

| Field | Type | Description |
|-------|------|-------------|
| `xAxis` | object | X-axis configuration |
| `yAxis` | object | Y-axis configuration |
| `colorAxis` | object | Optional color axis |
| `points` | array | Data points |

## Customization

### Color Scales

```go
render.Summary(explanations, featureValues, render.SummaryOptions{
    ColorScale: "bluered",    // Blue (low) to Red (high)
    // or "viridis", "plasma", "coolwarm"
})
```

### Feature Selection

```go
// Show specific features
render.Waterfall(explanation, render.WaterfallOptions{
    Features: []string{"income", "age", "credit_score"},
})

// Exclude features
render.FeatureImportance(explanations, render.ImportanceOptions{
    ExcludeFeatures: []string{"id", "timestamp"},
})
```

### Sorting

```go
render.FeatureImportance(explanations, render.ImportanceOptions{
    SortBy: "mean_abs",  // Default: mean absolute SHAP
    // or "max_abs", "variance"
})
```

## Integration Examples

### REST API

```go
func handleExplanationChart(w http.ResponseWriter, r *http.Request) {
    instance := parseInstance(r)
    explanation, _ := exp.Explain(r.Context(), instance)

    chart := render.Waterfall(explanation, render.WaterfallOptions{
        MaxFeatures: 10,
    })

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(chart)
}
```

### File Export

```go
// JSON
data, _ := json.MarshalIndent(chart, "", "  ")
os.WriteFile("chart.json", data, 0644)

// For web rendering, embed in HTML template
tmpl := `<script>const chartData = {{.}}</script>`
```

## Next Steps

- [Benchmarks](../benchmarks.md) - Performance characteristics
- [API Reference](../api/reference.md) - Full API documentation
