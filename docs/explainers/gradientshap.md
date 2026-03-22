# GradientSHAP

GradientSHAP (Expected Gradients) computes SHAP values by combining ideas from Integrated Gradients and SHAP sampling. It computes gradients at interpolated points between the input and background samples, providing theoretically grounded feature attributions.

## Overview

GradientSHAP works by:

1. Sampling a background reference x' from the background dataset
2. Sampling α uniformly from [0, 1]
3. Computing the interpolated point: z = x' + α(x - x')
4. Computing the gradient ∂f/∂z at the interpolated point
5. Computing SHAP contribution: (x_i - x'_i) × ∂f/∂z_i
6. Averaging over many (background, α) pairs

## Key Properties

| Property | Value |
|----------|-------|
| **Accuracy** | Monte Carlo approximation |
| **Complexity** | O(samples × features × 2) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |
| **Gradient method** | Numerical (finite differences) |

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/gradient"
    "github.com/plexusone/shap-go/model"
)

func main() {
    // Create a model (any model implementing model.Model)
    predict := func(ctx context.Context, input []float64) (float64, error) {
        x0, x1, x2 := input[0], input[1], input[2]
        return x0*x0 + 2*x0*x1 + x2, nil
    }
    m := model.NewFuncModel(predict, 3)

    // Background data for SHAP computation
    background := [][]float64{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.5, 0.5, 0.5},
        {1.0, 1.0, 1.0},
    }

    // Create GradientSHAP explainer
    exp, err := gradient.New(m, background,
        []explainer.Option{
            explainer.WithNumSamples(300),
            explainer.WithSeed(42),
            explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
        },
    )
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{2.0, 1.0, 0.5}
    explanation, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    for _, name := range explanation.FeatureNames {
        fmt.Printf("  %s: %+.4f\n", name, explanation.Values[name])
    }

    // Verify local accuracy
    result := explanation.Verify(0.5)
    fmt.Printf("Local Accuracy: %v (diff=%.2e)\n", result.Valid, result.Difference)
}
```

## Configuration Options

### Standard Options

```go
exp, err := gradient.New(model, background,
    []explainer.Option{
        explainer.WithNumSamples(300),      // Number of (background, alpha) pairs
        explainer.WithSeed(42),             // Random seed for reproducibility
        explainer.WithNumWorkers(4),        // Parallel workers
        explainer.WithFeatureNames(names),  // Human-readable names
        explainer.WithConfidenceLevel(0.95),// 95% confidence intervals
        explainer.WithModelID("my-model"),  // Model identifier
    },
)
```

### GradientSHAP-Specific Options

```go
exp, err := gradient.New(model, background,
    opts,
    gradient.WithEpsilon(1e-7),      // Step size for numerical gradients
    gradient.WithNoiseStdev(0.01),   // Add Gaussian noise for smoothing
    gradient.WithLocalSmoothing(5),  // Number of local smoothing samples
)
```

## Confidence Intervals

GradientSHAP supports computing confidence intervals for SHAP estimates:

```go
exp, err := gradient.New(model, background,
    []explainer.Option{
        explainer.WithNumSamples(500),
        explainer.WithConfidenceLevel(0.95),
    },
)

explanation, _ := exp.Explain(ctx, instance)

if explanation.HasConfidenceIntervals() {
    for _, name := range explanation.FeatureNames {
        low, high, _ := explanation.GetConfidenceInterval(name)
        fmt.Printf("%s: %.4f [%.4f, %.4f]\n",
            name, explanation.Values[name], low, high)
    }
}
```

## Parallel Computation

GradientSHAP supports parallel computation for better performance:

```go
exp, err := gradient.New(model, background,
    []explainer.Option{
        explainer.WithNumSamples(1000),
        explainer.WithNumWorkers(8),  // Use 8 parallel workers
    },
)
```

## Numerical Gradient Computation

GradientSHAP uses central finite differences to compute gradients:

```
∂f/∂x_i ≈ (f(x + ε·e_i) - f(x - ε·e_i)) / (2ε)
```

Where `ε` is the step size (default: 1e-7) and `e_i` is the unit vector in direction i.

The default epsilon provides a good balance between accuracy and numerical stability. For models with different scales, you may need to adjust it:

```go
// For models with small outputs
gradient.WithEpsilon(1e-9)

// For models with large outputs
gradient.WithEpsilon(1e-5)
```

## When to Use GradientSHAP

**Use GradientSHAP when:**

- You have a differentiable model (neural networks, polynomial models, etc.)
- The model is complex but gradient-based attribution is meaningful
- You want lower variance than pure sampling methods
- You need confidence intervals for SHAP values

**Don't use GradientSHAP when:**

- Your model is a tree ensemble (use TreeSHAP instead)
- Your model is linear (use LinearSHAP for exact values)
- You have a small feature set (use ExactSHAP for exact values)
- Your model is not differentiable (use KernelSHAP or PermutationSHAP)

## Comparison with Other Methods

| Method | Complexity | Accuracy | Best For |
|--------|------------|----------|----------|
| **GradientSHAP** | O(samples × features) | Approximate | Differentiable models |
| **DeepSHAP** | O(layers × neurons) | Approximate | Neural networks |
| **KernelSHAP** | O(samples × features²) | Approximate | Any model |
| **PermutationSHAP** | O(samples × features²) | Approximate | Any model |
| **ExactSHAP** | O(2^features) | Exact | Small feature sets |

## Background Dataset

The background dataset determines the baseline for attribution:

- Use representative samples from your training data
- 100-1000 samples typically provides good results
- More samples improve accuracy but increase computation time
- For linear models, the background mean determines the baseline

```go
// Create background from training data
bgDataset := background.NewDataset(trainingData, featureNames)
summary := bgDataset.KMeansSummary(100, 10, rng)
```

## Technical Details

### Expected Gradients Formula

GradientSHAP implements the Expected Gradients method:

```
SHAP_i = E_{x'~D, α~U(0,1)}[(x_i - x'_i) × ∂f(x' + α(x - x'))/∂x_i]
```

This is an approximation to the SHAP values that:

1. Samples reference points from the background distribution
2. Interpolates between reference and input
3. Computes gradients at interpolated points
4. Weights by the input difference

### Local Accuracy

The sum of SHAP values approximately equals the difference between prediction and baseline:

```
Σ SHAP_i ≈ f(x) - E[f(x')]
```

GradientSHAP satisfies this property in expectation, with variance decreasing as sample size increases.

### Noise Smoothing

For models with non-smooth gradients, adding Gaussian noise can improve stability:

```go
gradient.WithNoiseStdev(0.01)  // Add N(0, 0.01) noise to interpolated points
```

This creates a smoothed gradient estimate that is less sensitive to local irregularities.

## Example: Multi-class Classification

For multi-class models, create a wrapper for each class:

```go
// Wrapper that returns probability for a specific class
type ClassWrapper struct {
    model    *MultiClassModel
    classIdx int
}

func (w *ClassWrapper) Predict(ctx context.Context, input []float64) (float64, error) {
    probs := w.model.PredictProba(input)
    return probs[w.classIdx], nil
}

// Explain each class separately
for classIdx := range classes {
    wrapper := &ClassWrapper{model, classIdx}
    shapModel := model.NewFuncModel(wrapper.Predict, numFeatures)

    exp, _ := gradient.New(shapModel, background, opts)
    explanation, _ := exp.Explain(ctx, instance)

    fmt.Printf("Class %d SHAP values: %v\n", classIdx, explanation.Values)
}
```

## References

- [Expected Gradients paper](https://arxiv.org/abs/1906.10670): Explaining Models by Propagating Shapley Values
- [Integrated Gradients paper](https://arxiv.org/abs/1703.01365): Axiomatic Attribution for Deep Networks
- [SHAP paper](https://arxiv.org/abs/1705.07874): A Unified Approach to Interpreting Model Predictions
