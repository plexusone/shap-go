# SamplingSHAP

SamplingSHAP uses simple Monte Carlo sampling to approximate SHAP values. It's faster than PermutationSHAP but doesn't guarantee local accuracy.

## When to Use

- **Prototyping**: Quick exploration of feature importance
- **Speed over accuracy**: When approximate values are acceptable
- **Large models**: When PermutationSHAP is too slow

## Basic Usage

```go
package main

import (
    "context"
    "fmt"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/sampling"
    "github.com/plexusone/shap-go/model"
)

func main() {
    // Define your model
    predict := func(ctx context.Context, input []float64) (float64, error) {
        return input[0]*input[0] + 2*input[1] + input[0]*input[2], nil
    }
    m := model.NewFuncModel(predict, 3)

    // Background data
    background := [][]float64{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.5, 0.5, 0.5},
    }

    // Create sampling explainer
    exp, _ := sampling.New(m, background,
        explainer.WithNumSamples(100),
        explainer.WithSeed(42),
        explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
    )

    // Explain
    ctx := context.Background()
    explanation, _ := exp.Explain(ctx, []float64{1.0, 2.0, 0.5})

    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    for _, name := range explanation.FeatureNames {
        fmt.Printf("SHAP[%s]: %.4f\n", name, explanation.Values[name])
    }

    // Check accuracy (may not be exact)
    result := explanation.Verify(0.5)  // Use larger tolerance
    fmt.Printf("\nWithin tolerance: %v (diff=%.4f)\n", result.Valid, result.Difference)
}
```

## How It Works

SamplingSHAP uses simple Monte Carlo estimation:

1. Generate random permutations of features
2. For each permutation, compute marginal contribution of each feature
3. Average contributions across all samples

```
For each sample:
    permutation = random_shuffle([0, 1, 2, ..., n-1])
    for each feature i in permutation:
        contribution[i] = f(S ∪ {i}) - f(S)
        where S = features before i in permutation
```

## Sample Count Trade-offs

| Samples | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| 10 | Very fast | Low | Quick check |
| 50 | Fast | Medium | Exploration |
| 100 | Medium | Good | General use |
| 500 | Slow | High | Important decisions |

### Effect on Accuracy

```go
ctx := context.Background()
instance := []float64{1.0, 2.0, 0.5}

for _, samples := range []int{10, 50, 100, 500} {
    exp, _ := sampling.New(m, background,
        explainer.WithNumSamples(samples),
        explainer.WithSeed(42),
    )
    explanation, _ := exp.Explain(ctx, instance)
    result := explanation.Verify(1.0)

    fmt.Printf("Samples=%3d: diff=%.4f\n", samples, result.Difference)
}
```

Output:
```
Samples= 10: diff=0.4521
Samples= 50: diff=0.1823
Samples=100: diff=0.0891
Samples=500: diff=0.0234
```

## Configuration

```go
exp, _ := sampling.New(m, background,
    // Number of Monte Carlo samples
    explainer.WithNumSamples(100),

    // Seed for reproducibility
    explainer.WithSeed(42),

    // Feature names
    explainer.WithFeatureNames([]string{"age", "income", "score"}),

    // Model identifier
    explainer.WithModelID("quick-model"),
)
```

## Comparison with PermutationSHAP

| Property | SamplingSHAP | PermutationSHAP |
|----------|--------------|-----------------|
| **Local accuracy** | Not guaranteed | Guaranteed |
| **Variance** | Higher | Lower |
| **Speed** | Faster | Slower |
| **Algorithm** | Simple MC | Antithetic sampling |

### When to Choose SamplingSHAP

- Exploratory analysis
- Model is very slow to evaluate
- Approximate values are acceptable
- You'll validate important predictions separately

### When to Choose PermutationSHAP

- Production deployments
- Regulatory/compliance requirements
- Need guaranteed accuracy
- Model is fast enough

## Improving Accuracy

### More Samples

The most direct way to improve accuracy:

```go
exp, _ := sampling.New(m, background,
    explainer.WithNumSamples(500),
)
```

### Better Background Data

Representative background data improves both accuracy and reliability:

```go
// Use stratified sampling from your dataset
background := stratifiedSample(trainingData, 20)
```

### Multiple Runs

For critical decisions, run multiple times and check consistency:

```go
results := make([]float64, 5)
for i := 0; i < 5; i++ {
    exp, _ := sampling.New(m, background,
        explainer.WithNumSamples(100),
        explainer.WithSeed(int64(i)),  // Different seed each time
    )
    explanation, _ := exp.Explain(ctx, instance)
    results[i] = explanation.Values["important_feature"]
}
// Check if results are consistent
```

## Use Cases

### Quick Feature Importance Check

```go
// Fast exploration - which features matter?
exp, _ := sampling.New(m, background,
    explainer.WithNumSamples(50),
)
explanation, _ := exp.Explain(ctx, instance)

fmt.Println("Top features (approximate):")
for _, f := range explanation.TopFeatures(5) {
    fmt.Printf("  %s: %.4f\n", f.Name, f.SHAPValue)
}
```

### Batch Screening

```go
// Quickly identify interesting instances for detailed analysis
for _, instance := range candidates {
    explanation, _ := exp.Explain(ctx, instance)

    // Flag instances where key feature has high impact
    if abs(explanation.Values["risk_factor"]) > 0.5 {
        flagForReview(instance)
    }
}
```

### Development/Testing

```go
// Fast iteration during development
func TestModelExplanations(t *testing.T) {
    exp, _ := sampling.New(m, background,
        explainer.WithNumSamples(20),  // Fast for tests
        explainer.WithSeed(42),        // Reproducible
    )

    explanation, _ := exp.Explain(ctx, testInstance)

    // Check approximate behavior
    if explanation.Values["x0"] < 0 {
        t.Error("Expected positive contribution from x0")
    }
}
```

## Limitations

1. **No local accuracy guarantee**: SHAP values may not sum exactly to `prediction - base_value`
2. **Higher variance**: Results vary more between runs
3. **Less suitable for compliance**: Use PermutationSHAP for auditable explanations

## Next Steps

- [PermutationSHAP](permutation.md) - When you need guaranteed accuracy
- [TreeSHAP](treeshap.md) - For tree models (exact and fast)
- [Visualization](../visualization/charts.md) - Create charts
