# Visualization Expected Output

Run with: `go run main.go`

## Output

```
SHAP Visualization Example
==========================

1. Waterfall Plot (Single Prediction)
--------------------------------------
Title: Loan Approval Explanation
Datasets: 1
  - waterfall: 4 columns, 5 rows
Marks: 1
  - positive (bar)
Axes: 2

2. Feature Importance (Global)
------------------------------
Title: Feature Importance
Datasets: 1
  - importance: 2 columns, 3 rows
Marks: 1
  - bars (bar)
Axes: 2

3. Summary Plot (Distribution)
------------------------------
Title: SHAP Summary
Datasets: 1
  - summary: 3 columns, 15 rows
Marks: 1
  - scatter (scatter)
Axes: 2

4. Dependence Plot (Single Feature)
------------------------------------
Title: Income vs SHAP
Datasets: 1
  - dependence: 2 columns, 5 rows
Marks: 1
  - scatter (scatter)
Axes: 2

5. Force Plot Data
------------------
Base Value: 0.6560
Prediction: 0.9400
Features (sorted by contribution):
  + income: 0.1830 (value=75000.00, range=[0.6560, 0.8390])
  + credit_score: 0.0932 (value=720.00, range=[0.8390, 0.9322])
  + age: 0.0174 (value=35.00, range=[0.9322, 0.9496])

6. Custom Theme
---------------
Positive color: #22c55e

7. Full ChartIR JSON Export
---------------------------
{
  "title": "Loan Approval Explanation",
  "datasets": [
    {
      "id": "waterfall",
      "columns": [
        {
          "name": "feature",
          "type": "string"
        },
        ...
      ],
      "rows": [...]
    }
  ],
  "marks": [...],
  "axes": [...]
}
```

## Key Points Demonstrated

1. **ChartIR Output** - Universal chart representation format
2. **Multiple Chart Types** - Waterfall, bar, scatter, dependence plots
3. **ExplanationSet** - Aggregate multiple explanations for global views
4. **Custom Themes** - Customize colors and fonts
5. **Force Plot Data** - Structured data for interactive visualizations

## Chart Types

### 1. Waterfall Plot
Shows how each feature pushes the prediction from baseline to final value.

### 2. Feature Importance
Bar chart of mean absolute SHAP values across all predictions.

### 3. Summary Plot
Scatter plot showing SHAP value distribution for each feature.

### 4. Dependence Plot
Shows relationship between feature value and its SHAP value.

### 5. Force Plot
Interactive visualization showing feature contributions.

## Using ChartIR Output

ChartIR is a universal chart intermediate representation that can be converted to:

- **ECharts** - via echartify
- **Chart.js** - via chartjs adapter
- **D3.js** - via d3 adapter
- **Vega-Lite** - via vega adapter

Example conversion (pseudo-code):
```go
chart := renderer.WaterfallChartIR(explanation, "Title")
echartOptions := echartify.Convert(chart)
```

## Custom Themes

```go
theme := render.Theme{
    PositiveColor: "#22c55e",  // Green
    NegativeColor: "#f97316",  // Orange
    BaselineColor: "#64748b",  // Slate
    FontFamily:    "Inter, system-ui, sans-serif",
}
renderer := render.NewRendererWithTheme(theme)
```
