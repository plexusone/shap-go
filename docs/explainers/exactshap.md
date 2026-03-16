# ExactSHAP

ExactSHAP computes mathematically exact Shapley values by enumerating all possible feature coalitions. This is the brute-force reference implementation that guarantees correct values, but is only practical for small feature sets.

## When to Use ExactSHAP

ExactSHAP is ideal for:

- **Validation**: Verify that other SHAP implementations produce correct values
- **Small feature sets**: Problems with ≤15 features where exact values are critical
- **Reference/education**: Understanding Shapley values without approximation
- **High-stakes decisions**: When approximation error is unacceptable

## Algorithm

For each feature i, the Shapley value is computed as:

```
φᵢ = Σ_{S ⊆ N\{i}} [|S|! * (n-|S|-1)! / n!] * [f(S ∪ {i}) - f(S)]
```

where:

- N is the set of all features
- S ranges over all subsets not containing feature i
- f(S) is the model prediction with features in S from the instance and others from background

This enumerates all 2^(n-1) coalitions for each feature, giving O(n * 2^n) total complexity.

## Quick Start

```go
import (
    "context"
    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/exact"
    "github.com/plexusone/shap-go/model"
)

// Define your model
predict := func(ctx context.Context, input []float64) (float64, error) {
    return input[0]*input[1] + input[2], nil  // Nonlinear model
}
m := model.NewFuncModel(predict, 3)

// Background data
background := [][]float64{
    {0.0, 0.0, 0.0},
}

// Create ExactSHAP explainer
exp, err := exact.New(m, background,
    explainer.WithFeatureNames([]string{"x", "y", "z"}),
)
if err != nil {
    log.Fatal(err)
}

// Explain a prediction
ctx := context.Background()
explanation, err := exp.Explain(ctx, []float64{2.0, 3.0, 1.0})
if err != nil {
    log.Fatal(err)
}

// SHAP values are mathematically exact
for _, feat := range explanation.TopFeatures(3) {
    fmt.Printf("%s: %.6f\n", feat.Name, feat.Value)
}
```

## Complexity Analysis

| Features | Coalitions | Predictions |
|----------|------------|-------------|
| 3 | 8 | 24 |
| 5 | 32 | 160 |
| 10 | 1,024 | 10,240 |
| 15 | 32,768 | 491,520 |
| 20 | 1,048,576 | 20,971,520 |

The maximum supported feature count is 20, after which the exponential complexity becomes impractical.

## Comparison with Other Methods

| Method | Complexity | Type | Use Case |
|--------|------------|------|----------|
| **ExactSHAP** | O(n * 2^n) | Exact | Validation, small problems |
| TreeSHAP | O(TLD²) | Exact | Tree models |
| LinearSHAP | O(n) | Exact | Linear models |
| KernelSHAP | O(samples) | Approximate | Model-agnostic |
| PermutationSHAP | O(samples) | Approximate | Local accuracy |

## Configuration Options

```go
exp, err := exact.New(model, background,
    explainer.WithFeatureNames(names),  // Custom feature names
    explainer.WithModelID("model-v1"),  // Model identifier
)
```

Note: ExactSHAP doesn't use `NumSamples` or `Seed` since it's deterministic and enumerates all coalitions.

## Verifying Local Accuracy

ExactSHAP always satisfies local accuracy exactly:

```go
result := explanation.Verify(1e-10)  // Tight tolerance for exact values
if !result.Valid {
    // Should never happen with ExactSHAP
    log.Printf("Unexpected: diff=%f", result.Difference)
}
```

## Best Practices

1. **Check feature count**: Ensure n ≤ 15 for practical runtimes
2. **Use for validation**: Compare ExactSHAP results against KernelSHAP or PermutationSHAP
3. **Background size**: Keep background small since each coalition evaluates over all background samples
4. **Cache predictions**: ExactSHAP internally caches coalition predictions to avoid redundant model calls

## Error Handling

```go
exp, err := exact.New(model, background)
if err != nil {
    switch {
    case errors.Is(err, exact.ErrTooManyFeatures):
        // Use a different method for large feature sets
        log.Printf("Too many features, use KernelSHAP instead")
    case errors.Is(err, exact.ErrNilModel):
        log.Fatal("Model cannot be nil")
    case errors.Is(err, exact.ErrNoBackground):
        log.Fatal("Background data required")
    default:
        log.Fatal(err)
    }
}
```
