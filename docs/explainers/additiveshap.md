# AdditiveExplainer

AdditiveExplainer computes exact SHAP values for **Generalized Additive Models (GAMs)** - models where the prediction is a sum of individual feature effects with no interactions.

## When to Use

Use AdditiveExplainer when your model has **no feature interactions**:

- Generalized Additive Models (GAMs)
- Spline-based models (e.g., `pygam`, `interpret-ml`)
- Linear models (though [LinearSHAP](linearshap.md) is more efficient)
- Any model of the form: `f(x) = f₀(x₀) + f₁(x₁) + ... + fₙ(xₙ)`

## Mathematical Background

For an additive model:

$$f(x) = \sum_{i=1}^{n} f_i(x_i) + \text{intercept}$$

SHAP values have a closed-form solution:

$$\phi_i = f_i(x_i) - \mathbb{E}[f_i(X_i)]$$

This means:

- Each feature's SHAP value equals its contribution minus its expected contribution
- No sampling or approximation is needed
- Results are mathematically exact

## Complexity

| Aspect | Complexity |
|--------|------------|
| Time | O(n × b) |
| Space | O(n + b) |

Where:

- n = number of features
- b = number of background samples

This is much faster than model-agnostic methods like KernelSHAP (O(2ⁿ × b)) or PermutationSHAP (O(n! × b)).

## Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/additive"
    "github.com/plexusone/shap-go/model"
)

func main() {
    // Define an additive model: f(x) = x₀² + 2*x₁ + sin(x₂)
    gamModel := func(ctx context.Context, input []float64) (float64, error) {
        return input[0]*input[0] + 2*input[1] + math.Sin(input[2]), nil
    }
    m := model.NewFuncModel(gamModel, 3)

    // Background data for computing expected effects
    background := [][]float64{
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    }

    // Create explainer
    exp, err := additive.New(m, background,
        explainer.WithFeatureNames([]string{"age", "income", "score"}),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Explain an instance
    ctx := context.Background()
    instance := []float64{1.5, 2.0, 0.5}

    result, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    // Print SHAP values
    fmt.Printf("Prediction: %.4f\n", result.Prediction)
    fmt.Printf("Base Value: %.4f\n", result.BaseValue)
    for _, name := range result.FeatureNames {
        fmt.Printf("SHAP(%s): %.4f\n", name, result.Values[name])
    }

    // Verify local accuracy
    verify := result.Verify(1e-10)
    fmt.Printf("Local accuracy: %v\n", verify.Valid)
}
```

## How It Works

1. **Compute reference point**: Mean of background data
2. **Compute base value**: `E[f(X)]` - mean prediction over background
3. **Precompute expected effects**: For each feature i, compute `E[fᵢ(Xᵢ)]` from background
4. **For each instance**:
   - Compute effect of each feature: `effect_i = f(ref, xᵢ, ref) - f(ref)`
   - SHAP value: `φᵢ = effect_i - expected_effect_i`

## Comparison with Other Methods

| Aspect | AdditiveExplainer | LinearSHAP | KernelSHAP |
|--------|-------------------|------------|------------|
| Model type | GAMs (no interactions) | Linear only | Any model |
| Accuracy | Exact | Exact | Approximate |
| Speed | Very fast | Fastest | Slow |
| Complexity | O(n × b) | O(n) | O(2ⁿ × b) |

## Important Notes

### Model Assumptions

AdditiveExplainer assumes your model has **no feature interactions**. If your model has interactions (e.g., `x₀ * x₁`), the SHAP values will not be correct. For models with interactions, use:

- [KernelSHAP](kernelshap.md) for model-agnostic explanations
- [TreeSHAP](treeshap.md) for tree ensemble models
- [DeepSHAP](deepshap.md) for neural networks

### Verifying Additivity

You can test if your model is truly additive by checking that interaction effects are zero:

```go
// For an additive model, f(x₀, x₁) - f(x₀, 0) - f(0, x₁) + f(0, 0) ≈ 0
// If this is non-zero, your model has interactions
```

### Background Data Selection

The background data should be representative of your training distribution. See the [Background Data Selection Guide](../guides/background-data.md) for best practices.

## API Reference

### Constructor

```go
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)
```

Creates a new AdditiveExplainer. During construction, it:

- Computes the reference point (mean of background)
- Computes base value `E[f(X)]`
- Precomputes expected effects for each feature

### Methods

```go
// Explain computes SHAP values for a single instance
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch explains multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// BaseValue returns E[f(X)]
func (e *Explainer) BaseValue() float64

// FeatureNames returns feature names
func (e *Explainer) FeatureNames() []string

// Reference returns the reference point (mean of background)
func (e *Explainer) Reference() []float64

// ExpectedEffects returns precomputed E[fᵢ(Xᵢ)] for each feature
func (e *Explainer) ExpectedEffects() []float64
```

### Configuration Options

```go
// Set feature names
explainer.WithFeatureNames([]string{"age", "income", "score"})

// Set model ID for tracking
explainer.WithModelID("my-gam-model")
```

## See Also

- [LinearSHAP](linearshap.md) - For linear models (faster)
- [KernelSHAP](kernelshap.md) - For models with interactions
- [Background Data Selection Guide](../guides/background-data.md)
