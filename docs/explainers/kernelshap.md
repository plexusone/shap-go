# KernelSHAP

KernelSHAP computes **approximate** SHAP values for any model using weighted linear regression. It is a model-agnostic method that treats the model as a black-box, requiring only the ability to make predictions.

## When to Use

- **Any black-box model** (ONNX, custom functions, APIs)
- When you need **lower variance** than SamplingSHAP
- When you want a **principled approximation** based on Shapley kernel weights
- As a **model-agnostic baseline** for comparison

## How It Works

KernelSHAP approximates SHAP values by:

1. **Sampling coalitions**: Randomly selecting subsets of features
2. **Weighting samples**: Using Shapley kernel weights that emphasize informative coalitions
3. **Linear regression**: Solving a weighted least squares problem with the constraint that SHAP values sum to (prediction - baseline)

The Shapley kernel weight for coalition size |S| is:

```
w(|S|) = (M - 1) / (|S| × (M - |S|))
```

This weight is highest for intermediate-sized coalitions, which provide the most information about individual feature contributions.

## Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/kernel"
    "github.com/plexusone/shap-go/model"
)

func main() {
    // Define your model as a function
    predictFn := func(ctx context.Context, input []float64) (float64, error) {
        // Your model prediction logic here
        return input[0]*2 + input[1]*3 + input[2], nil
    }

    // Wrap as a Model
    m := model.NewFuncModel(predictFn, 3)

    // Background data (representative samples)
    background := [][]float64{
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0},
        {1.5, 2.5, 3.5},
    }

    // Create KernelSHAP explainer
    exp, err := kernel.New(m, background,
        explainer.WithNumSamples(200),
        explainer.WithSeed(42),
        explainer.WithFeatureNames([]string{"age", "income", "score"}),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{5.0, 4.0, 3.0}
    explanation, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    fmt.Println("\nFeature contributions:")
    for _, feat := range explanation.TopFeatures(3) {
        sign := "+"
        if feat.SHAPValue < 0 {
            sign = "-"
        }
        fmt.Printf("  %s %s: %.4f\n", sign, feat.Name, feat.SHAPValue)
    }

    // Verify local accuracy
    result := explanation.Verify(1e-6)
    fmt.Printf("\nLocal accuracy: %v (diff=%.2e)\n", result.Valid, result.Difference)
}
```

## Algorithm Details

### Two-Phase Sampling

KernelSHAP uses a two-phase sampling strategy:

1. **Complete enumeration**: For small coalition sizes, enumerate all possible combinations
2. **Random sampling**: For remaining budget, sample coalitions weighted by kernel weights

### Coalition + Complement

For each sampled coalition, KernelSHAP also evaluates its complement. This variance reduction technique ensures both "with feature" and "without feature" scenarios are equally represented.

### Constrained Regression

The SHAP values are computed by solving:

```
minimize Σᵢ wᵢ × (yᵢ - φ₀ - Σⱼ zᵢⱼ × φⱼ)²
subject to: Σⱼ φⱼ = f(x) - E[f(x)]
```

The constraint ensures **local accuracy**: SHAP values always sum exactly to (prediction - baseline).

## Comparison with Other Explainers

| Property | KernelSHAP | PermutationSHAP | SamplingSHAP | TreeSHAP |
|----------|------------|-----------------|--------------|----------|
| **Model type** | Any | Any | Any | Trees only |
| **Accuracy** | Approximate | Exact | Approximate | Exact |
| **Local accuracy** | Guaranteed | Guaranteed | Not guaranteed | Guaranteed |
| **Variance** | Medium | Low | Higher | None |
| **Speed** | Medium | Slow | Fast | Very fast |

## Configuration Options

```go
exp, err := kernel.New(model, background,
    // Number of coalition samples (more = better accuracy)
    explainer.WithNumSamples(200),

    // Random seed for reproducibility
    explainer.WithSeed(42),

    // Parallel workers for faster computation
    explainer.WithNumWorkers(4),

    // Human-readable feature names
    explainer.WithFeatureNames([]string{"age", "income", "score"}),

    // Model identifier for tracking
    explainer.WithModelID("credit-risk-v1"),
)
```

### Sample Size Guidelines

| Features | Recommended Samples |
|----------|---------------------|
| 2-5 | 50-100 |
| 5-10 | 100-200 |
| 10-20 | 200-500 |
| 20+ | 500+ |

More samples improve accuracy but increase computation time.

## Working with ONNX Models

KernelSHAP works well with ONNX models:

```go
import (
    "github.com/plexusone/shap-go/explainer/kernel"
    "github.com/plexusone/shap-go/model/onnx"
)

// Load ONNX model
session, err := onnx.NewSession("model.onnx", "input", "output")
if err != nil {
    log.Fatal(err)
}
defer session.Close()

// Create KernelSHAP explainer
exp, err := kernel.New(session, background,
    explainer.WithNumSamples(200),
)
```

## Background Data Selection

The background data significantly affects SHAP values:

- **Representative**: Should represent typical inputs
- **Size**: 50-200 samples is usually sufficient
- **Diversity**: Include varied examples from your dataset

```go
// Good: diverse background data
background := selectRandomSamples(trainingData, 100)

// Bad: single or very few samples
background := [][]float64{{0.0, 0.0, 0.0}}
```

!!! tip "Use k-means summarization"
    For large datasets, use `background.KMeansSummarize()` to create a compact representative set.

## Performance Tips

1. **Use parallel workers** for large sample counts:
   ```go
   explainer.WithNumWorkers(4)
   ```

2. **Batch predictions** if your model supports it (ONNX does)

3. **Cache the explainer** - don't recreate for each instance:
   ```go
   // Good: create once, use many times
   exp, _ := kernel.New(model, background)
   for _, instance := range instances {
       explanation, _ := exp.Explain(ctx, instance)
   }
   ```

4. **Start with fewer samples** for prototyping, increase for production

## Troubleshooting

### High variance in SHAP values

Increase the number of samples:
```go
explainer.WithNumSamples(500)
```

### Slow computation

- Reduce background data size (use k-means summarization)
- Use parallel workers
- Consider SamplingSHAP for initial exploration

### Different results each run

Set a fixed seed for reproducibility:
```go
explainer.WithSeed(42)
```

## API Reference

### New

```go
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error)
```

Creates a new KernelSHAP explainer.

**Parameters:**

- `m`: Model implementing the `model.Model` interface
- `background`: Representative samples for baseline computation
- `opts`: Configuration options

### Explainer Methods

```go
// Explain computes SHAP values for a single instance
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error)

// ExplainBatch computes SHAP values for multiple instances
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error)

// BaseValue returns E[f(x)] computed on background data
func (e *Explainer) BaseValue() float64

// FeatureNames returns the feature names
func (e *Explainer) FeatureNames() []string
```

## Next Steps

- [PermutationSHAP](permutation.md) - For guaranteed local accuracy with any model
- [TreeSHAP](treeshap.md) - For exact SHAP values with tree models
- [Visualization](../visualization/charts.md) - Create charts from explanations
- [ONNX Integration](../models/onnx.md) - Using KernelSHAP with ONNX models
