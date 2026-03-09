# SHAP-Go

A Go library for computing SHAP (SHapley Additive exPlanations) values for ML model explainability.

[![Go Reference](https://pkg.go.dev/badge/github.com/plexusone/shap-go.svg)](https://pkg.go.dev/github.com/plexusone/shap-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/plexusone/shap-go)](https://goreportcard.com/report/github.com/plexusone/shap-go)

## Overview

SHAP-Go provides a Go-native implementation of SHAP value computation for explaining machine learning model predictions. It supports:

- **ONNX models** via ONNX Runtime bindings
- **Custom models** via a simple function interface
- **Permutation SHAP** with antithetic sampling for variance reduction
- **Sampling SHAP** using Monte Carlo estimation
- **JSON-serializable explanations** for audit/compliance

## Explainer Types

| Status | Explainer | Model Type | Notes |
|:------:|-----------|------------|-------|
| ✅ | **PermutationSHAP** | Any | Black-box, antithetic sampling for variance reduction, guarantees local accuracy |
| ✅ | **SamplingSHAP** | Any | Monte Carlo approximation, fast, good for quick estimates |
| ⬜ | **KernelSHAP** | Any | Black-box, weighted linear regression, model-agnostic baseline |
| ⬜ | **LinearSHAP** | Linear | Exact closed-form solution for linear/logistic regression |
| ⬜ | **TreeSHAP** | Trees | Exact & fast (O(TLD²)) for XGBoost, LightGBM, CatBoost, scikit-learn trees |
| ⬜ | **DeepSHAP** | Neural Nets | Combines DeepLIFT with Shapley values, efficient for deep networks |
| ⬜ | **GradientSHAP** | Neural Nets | Expected gradients + noise, connects SHAP to integrated gradients |
| ⬜ | **PartitionSHAP** | Structured | Hierarchical clustering of features, faster for correlated features |
| ⬜ | **ExactSHAP** | Any | Brute-force exact computation, O(2ⁿ) - only for small feature sets |

### Legend

- ✅ Implemented
- ⬜ Not yet implemented

### Choosing an Explainer

| Use Case | Recommended Explainer |
|----------|----------------------|
| Any model, need guaranteed accuracy | PermutationSHAP |
| Any model, quick estimates | SamplingSHAP |
| Tree-based models (XGBoost, etc.) | TreeSHAP (when available) |
| Linear/logistic regression | LinearSHAP (when available) |
| Deep learning models | DeepSHAP or GradientSHAP (when available) |
| Highly correlated features | PartitionSHAP (when available) |
| Small feature sets (≤15 features) | ExactSHAP (when available) |

## Installation

```bash
go get github.com/plexusone/shap-go
```

## Quick Start

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
    // Define a simple model
    predict := func(ctx context.Context, input []float64) (float64, error) {
        return input[0] + 2*input[1], nil
    }
    m := model.NewFuncModel(predict, 2)

    // Background data for SHAP computation
    background := [][]float64{
        {0.0, 0.0},
    }

    // Create explainer
    exp, _ := permutation.New(m, background,
        explainer.WithNumSamples(100),
        explainer.WithFeatureNames([]string{"x1", "x2"}),
    )

    // Explain a prediction
    ctx := context.Background()
    explanation, _ := exp.Explain(ctx, []float64{1.0, 2.0})

    fmt.Printf("Prediction: %.2f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.2f\n", explanation.BaseValue)
    for name, shap := range explanation.Values {
        fmt.Printf("SHAP(%s): %.4f\n", name, shap)
    }

    // Verify local accuracy
    result := explanation.Verify(1e-10)
    fmt.Printf("Local accuracy valid: %v\n", result.Valid)
}
```

## Packages

### `explanation`

Core types for SHAP explanations:

- `Explanation` - Contains prediction, base value, SHAP values, and metadata
- `Verify()` - Checks local accuracy (sum of SHAP values = prediction - base)
- `TopFeatures()` - Returns features sorted by absolute SHAP value
- JSON serialization with `ToJSON()` and `FromJSON()`

### `model`

Model interface for SHAP computation:

- `Model` interface with `Predict()`, `PredictBatch()`, and `NumFeatures()`
- `FuncModel` - Wraps a prediction function as a Model

### `model/onnx`

ONNX Runtime wrapper:

- `Session` - Wraps an ONNX Runtime session
- Supports batch predictions
- Requires ONNX Runtime shared library

### `explainer/permutation`

Permutation SHAP with antithetic sampling:

- Guarantees local accuracy
- Supports parallel computation
- Lower variance than pure Monte Carlo

### `explainer/sampling`

Monte Carlo sampling SHAP:

- Simple implementation
- Good for quick estimates

### `masker`

Feature masking strategies:

- `IndependentMasker` - Marginal/independent masking using background samples

### `background`

Background dataset management:

- Dataset loading and statistics
- Random sampling and k-means summarization

## Algorithms

### Permutation SHAP with Antithetic Sampling

The permutation explainer uses antithetic sampling for variance reduction:

1. For each permutation sample:
   - **Forward pass**: Start with background, add features one by one
   - **Reverse pass**: Start with instance, remove features one by one
   - Average contributions from both passes

2. Average over all permutation samples

This guarantees that SHAP values sum exactly to (prediction - base value).

### Sampling SHAP

The sampling explainer uses simple Monte Carlo:

1. Generate random permutations
2. For each permutation, compute marginal contributions
3. Average over all samples

## Configuration Options

```go
exp, err := permutation.New(model, background,
    explainer.WithNumSamples(100),     // Number of permutation samples
    explainer.WithSeed(42),            // Random seed for reproducibility
    explainer.WithNumWorkers(4),       // Parallel workers
    explainer.WithFeatureNames(names), // Feature names
    explainer.WithModelID("my-model"), // Model identifier
)
```

## ONNX Runtime Usage

```go
import "github.com/plexusone/shap-go/model/onnx"

// Initialize ONNX Runtime
onnx.InitializeRuntime("/path/to/libonnxruntime.so")
defer onnx.DestroyRuntime()

// Create session
session, err := onnx.NewSession(onnx.Config{
    ModelPath:   "model.onnx",
    InputName:   "float_input",
    OutputName:  "probabilities",
    NumFeatures: 10,
})
defer session.Close()

// Use with explainer
exp, err := permutation.New(session, background)
```

## Local Accuracy Verification

Every SHAP explanation should satisfy local accuracy:

```
sum(SHAP values) = prediction - base_value
```

You can verify this with:

```go
result := explanation.Verify(tolerance)
if !result.Valid {
    fmt.Printf("Local accuracy failed: difference = %f\n", result.Difference)
}
```

## License

MIT License
