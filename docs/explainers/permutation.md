# PermutationSHAP

PermutationSHAP computes SHAP values for **any model** by measuring how predictions change when features are permuted. It uses antithetic sampling for variance reduction.

## When to Use

- You have a black-box model (ONNX, custom function, API)
- You need **guaranteed local accuracy**
- TreeSHAP isn't applicable (non-tree model)

## Basic Usage

```go
package main

import (
    "context"
    "fmt"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/permutation"
    "github.com/plexusone/shap-go/model"
)

func main() {
    // Define your model as a prediction function
    predict := func(ctx context.Context, input []float64) (float64, error) {
        // Your model logic here
        x0, x1, x2 := input[0], input[1], input[2]
        return 2*x0 + 3*x1 + x0*x2, nil
    }
    m := model.NewFuncModel(predict, 3)

    // Background data: representative samples from your data
    background := [][]float64{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.5, 0.5, 0.5},
    }

    // Create explainer
    exp, err := permutation.New(m, background,
        explainer.WithNumSamples(100),
        explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
    )
    if err != nil {
        panic(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{1.0, 2.0, 0.5}
    explanation, _ := exp.Explain(ctx, instance)

    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    for _, name := range explanation.FeatureNames {
        fmt.Printf("SHAP[%s]: %.4f\n", name, explanation.Values[name])
    }

    // Verify local accuracy
    result := explanation.Verify(1e-10)
    fmt.Printf("\nLocal accuracy: %v\n", result.Valid)
}
```

## How It Works

PermutationSHAP uses **antithetic sampling** for variance reduction:

1. **Forward pass**: Start with background, add features one by one
2. **Reverse pass**: Start with instance, remove features one by one
3. **Average**: Combine both passes for lower variance

```
For each permutation:
    Forward:  background → +f1 → +f2 → +f3 → instance
    Reverse:  instance → -f3 → -f2 → -f1 → background

SHAP[fi] = average of (contribution when adding fi) and (contribution when removing fi)
```

This guarantees that SHAP values sum exactly to `prediction - base_value`.

## Configuration

### Number of Samples

More samples = more accurate, but slower:

```go
// Fast but noisy
exp, _ := permutation.New(m, background, explainer.WithNumSamples(10))

// Balanced
exp, _ := permutation.New(m, background, explainer.WithNumSamples(100))

// Very accurate
exp, _ := permutation.New(m, background, explainer.WithNumSamples(500))
```

### Reproducibility

Set a seed for deterministic results:

```go
exp, _ := permutation.New(m, background,
    explainer.WithNumSamples(100),
    explainer.WithSeed(42),
)
```

### Parallel Processing

Speed up computation with multiple workers:

```go
exp, _ := permutation.New(m, background,
    explainer.WithNumSamples(100),
    explainer.WithNumWorkers(4),
)
```

## Background Data

The background dataset is crucial for accurate explanations.

### Guidelines

- **Representative**: Should reflect your data distribution
- **Size**: 10-100 samples typically sufficient
- **Diversity**: Cover the range of feature values

### Good Background Data

```go
// From your training/validation data
background := selectRepresentativeSamples(trainingData, 50)

// Or use k-means clustering
background := kMeansSummarize(trainingData, 20)
```

### Bad Background Data

```go
// Single point - loses information about data distribution
background := [][]float64{{0.0, 0.0, 0.0}}

// All zeros - not representative of real data
background := make([][]float64, 10)
for i := range background {
    background[i] = make([]float64, numFeatures)
}
```

## With ONNX Models

```go
import (
    "github.com/plexusone/shap-go/model/onnx"
    "github.com/plexusone/shap-go/explainer/permutation"
)

// Initialize ONNX Runtime
onnx.InitializeRuntime("/path/to/libonnxruntime.so")
defer onnx.DestroyRuntime()

// Create ONNX session
session, _ := onnx.NewSession(onnx.Config{
    ModelPath:   "model.onnx",
    InputName:   "float_input",
    OutputName:  "output",
    NumFeatures: 10,
})
defer session.Close()

// Create explainer
exp, _ := permutation.New(session, background,
    explainer.WithNumSamples(100),
)
```

## Performance Considerations

PermutationSHAP requires many model evaluations:

```
Evaluations per instance = 2 × (num_features + 1) × num_samples
```

For 10 features and 100 samples: **2,200 model calls per explanation**.

### Tips

1. **Reduce samples** if model is slow
2. **Use parallel workers** to utilize multiple cores
3. **Consider TreeSHAP** if you have a tree model
4. **Cache model** if initialization is expensive

## Comparison with SamplingSHAP

| Property | PermutationSHAP | SamplingSHAP |
|----------|-----------------|--------------|
| Local accuracy | Guaranteed | Not guaranteed |
| Variance | Lower (antithetic) | Higher |
| Speed | Slower | Faster |
| Use case | Production | Prototyping |

## Troubleshooting

### High Variance in SHAP Values

Increase the number of samples:

```go
exp, _ := permutation.New(m, background,
    explainer.WithNumSamples(500),  // More samples
)
```

### Slow Computation

1. Reduce samples: `WithNumSamples(50)`
2. Add workers: `WithNumWorkers(4)`
3. Reduce background size
4. Consider SamplingSHAP for quick estimates

### Memory Issues

Large background datasets consume memory. Use summarization:

```go
// Instead of 10,000 background samples, use 50 representative ones
background := kMeansSummarize(fullData, 50)
```

## Next Steps

- [SamplingSHAP](sampling.md) - Faster alternative
- [Visualization](../visualization/charts.md) - Create charts
- [ONNX Integration](../models/onnx.md) - Use ONNX models
