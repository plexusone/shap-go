# LinearSHAP

LinearSHAP computes **exact** SHAP values for linear models using a closed-form solution. No sampling or approximation is needed.

## When to Use

- **Linear regression** models
- **Logistic regression** (explains log-odds)
- **Ridge/Lasso regression** (any linear model)
- When you need **exact** SHAP values with **no variance**

## How It Works

For a linear model:

```
f(x) = bias + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

The SHAP value for feature i has a closed-form solution:

```
SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
```

Where `E[xᵢ]` is the mean of feature i in the background data.

This is:

- **Exact** (not an approximation)
- **O(n)** complexity (where n = number of features)
- **Deterministic** (same input always gives same output)

## Basic Usage

```go
package main

import (
    "context"
    "fmt"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/linear"
)

func main() {
    // Model: f(x) = 10 + 2*x₀ + 3*x₁
    weights := []float64{2.0, 3.0}
    bias := 10.0

    // Background data (representative samples)
    background := [][]float64{
        {1.0, 2.0},
        {3.0, 4.0},
        {2.0, 3.0},
    }

    // Create explainer
    exp, err := linear.New(weights, bias, background,
        explainer.WithFeatureNames([]string{"age", "income"}),
    )
    if err != nil {
        panic(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{5.0, 6.0}
    explanation, _ := exp.Explain(ctx, instance)

    fmt.Printf("Prediction: %.2f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.2f\n", explanation.BaseValue)

    for _, f := range explanation.TopFeatures(2) {
        fmt.Printf("SHAP[%s]: %.2f\n", f.Name, f.SHAPValue)
    }

    // Verify: sum(SHAP) = prediction - base_value
    result := explanation.Verify(1e-10)
    fmt.Printf("Exact: %v\n", result.Valid)  // Always true for LinearSHAP
}
```

## From scikit-learn

Export your model coefficients from Python:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
import json

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

export = {
    "weights": model.coef_.tolist(),
    "bias": float(model.intercept_),
    "feature_names": feature_names
}
with open("linear_model.json", "w") as f:
    json.dump(export, f)

# Logistic Regression (binary)
model = LogisticRegression()
model.fit(X_train, y_train)

export = {
    "weights": model.coef_[0].tolist(),  # First class coefficients
    "bias": float(model.intercept_[0]),
    "feature_names": feature_names
}
with open("logistic_model.json", "w") as f:
    json.dump(export, f)
```

Load in Go:

```go
import (
    "encoding/json"
    "os"
)

type ModelExport struct {
    Weights      []float64 `json:"weights"`
    Bias         float64   `json:"bias"`
    FeatureNames []string  `json:"feature_names"`
}

func loadModel(path string) (*linear.Explainer, error) {
    data, _ := os.ReadFile(path)

    var model ModelExport
    json.Unmarshal(data, &model)

    return linear.New(model.Weights, model.Bias, background,
        explainer.WithFeatureNames(model.FeatureNames),
    )
}
```

## Logistic Regression

For logistic regression, LinearSHAP explains the **log-odds** (before sigmoid):

```go
// Logistic regression: P(y=1) = sigmoid(bias + w·x)
// LinearSHAP explains the linear part (log-odds)

weights := []float64{0.5, -0.3, 0.8}  // Log-odds coefficients
bias := -1.2

exp, _ := linear.New(weights, bias, background)
explanation, _ := exp.Explain(ctx, instance)

// explanation.Prediction is the log-odds
// To get probability: prob = 1 / (1 + exp(-prediction))
logOdds := explanation.Prediction
probability := 1.0 / (1.0 + math.Exp(-logOdds))
```

## Comparison with Other Explainers

| Property | LinearSHAP | TreeSHAP | PermutationSHAP |
|----------|------------|----------|-----------------|
| **Accuracy** | Exact | Exact | Approximate |
| **Complexity** | O(n) | O(TLD²) | O(S×M×P) |
| **Model type** | Linear only | Trees only | Any model |
| **Variance** | None | None | Some |

## Configuration

```go
exp, _ := linear.New(weights, bias, background,
    // Feature names for interpretability
    explainer.WithFeatureNames([]string{"age", "income", "score"}),

    // Model identifier for tracking
    explainer.WithModelID("credit-risk-v1"),
)
```

## API Reference

### New

```go
func New(weights []float64, bias float64, background [][]float64, opts ...explainer.Option) (*Explainer, error)
```

Creates a LinearSHAP explainer.

**Parameters:**

- `weights`: Model coefficients (one per feature)
- `bias`: Model intercept
- `background`: Representative samples for computing E[xᵢ]
- `opts`: Optional configuration

### Explainer Methods

```go
// Explain computes exact SHAP values for a single instance
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// BaseValue returns E[f(x)]
func (e *Explainer) BaseValue() float64

// Weights returns the model coefficients
func (e *Explainer) Weights() []float64

// Bias returns the model intercept
func (e *Explainer) Bias() float64

// FeatureMeans returns E[xᵢ] for each feature
func (e *Explainer) FeatureMeans() []float64
```

## Mathematical Details

### Base Value

The base value (expected prediction) is:

```
E[f(x)] = bias + Σᵢ wᵢ × E[xᵢ]
```

### SHAP Values

For each feature i:

```
SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
```

### Local Accuracy

LinearSHAP always satisfies local accuracy exactly:

```
Σᵢ SHAP[i] = f(x) - E[f(x)]
```

Proof:
```
Σᵢ SHAP[i] = Σᵢ wᵢ × (xᵢ - E[xᵢ])
           = Σᵢ wᵢxᵢ - Σᵢ wᵢE[xᵢ]
           = (bias + Σᵢ wᵢxᵢ) - (bias + Σᵢ wᵢE[xᵢ])
           = f(x) - E[f(x)]
```

## Next Steps

- [PermutationSHAP](permutation.md) - For non-linear models
- [TreeSHAP](treeshap.md) - For tree ensembles
- [Visualization](../visualization/charts.md) - Create charts
