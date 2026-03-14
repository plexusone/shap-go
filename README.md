# SHAP-Go

[![Go CI][go-ci-svg]][go-ci-url]
[![Go Lint][go-lint-svg]][go-lint-url]
[![Go SAST][go-sast-svg]][go-sast-url]
[![Go Report Card][goreport-svg]][goreport-url]
[![Docs][docs-godoc-svg]][docs-godoc-url]
[![Visualization][viz-svg]][viz-url]
[![License][license-svg]][license-url]

 [go-ci-svg]: https://github.com/plexusone/shap-go/actions/workflows/go-ci.yaml/badge.svg?branch=main
 [go-ci-url]: https://github.com/plexusone/shap-go/actions/workflows/go-ci.yaml
 [go-lint-svg]: https://github.com/plexusone/shap-go/actions/workflows/go-lint.yaml/badge.svg?branch=main
 [go-lint-url]: https://github.com/plexusone/shap-go/actions/workflows/go-lint.yaml
 [go-sast-svg]: https://github.com/plexusone/shap-go/actions/workflows/go-sast-codeql.yaml/badge.svg?branch=main
 [go-sast-url]: https://github.com/plexusone/shap-go/actions/workflows/go-sast-codeql.yaml
 [goreport-svg]: https://goreportcard.com/badge/github.com/plexusone/shap-go
 [goreport-url]: https://goreportcard.com/report/github.com/plexusone/shap-go
 [docs-godoc-svg]: https://pkg.go.dev/badge/github.com/plexusone/shap-go
 [docs-godoc-url]: https://pkg.go.dev/github.com/plexusone/shap-go
 [viz-svg]: https://img.shields.io/badge/visualizaton-Go-blue.svg
 [viz-url]: https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=plexusone%2Fshap-go
 [loc-svg]: https://tokei.rs/b1/github/plexusone/shap-go
 [repo-url]: https://github.com/plexusone/shap-go
 [license-svg]: https://img.shields.io/badge/license-MIT-blue.svg
 [license-url]: https://github.com/plexusone/shap-go/blob/master/LICENSE

A Go library for computing SHAP (SHapley Additive exPlanations) values for ML model explainability.

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
| ✅ | **TreeSHAP** | Trees | Exact & fast (O(TLD²)) for XGBoost, LightGBM; 40-100x faster than permutation |
| ✅ | **LinearSHAP** | Linear | Exact closed-form solution for linear/logistic regression |
| ⬜ | **KernelSHAP** | Any | Black-box, weighted linear regression, model-agnostic baseline |
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
| **Tree-based models (XGBoost, LightGBM)** | **TreeSHAP** ✅ |
| **Linear/logistic regression** | **LinearSHAP** ✅ |
| Any model, need guaranteed accuracy | PermutationSHAP |
| Any model, quick estimates | SamplingSHAP |
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

## TreeSHAP for XGBoost/LightGBM

TreeSHAP computes **exact** SHAP values in O(TLD²) time, where T=trees, L=leaves, D=depth. This is 40-100x faster than permutation-based methods for typical tree ensembles.

### XGBoost Example

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    // Load XGBoost model (saved with model.save_model("model.json"))
    ensemble, err := tree.LoadXGBoostModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    // Create TreeSHAP explainer
    explainer, err := tree.New(ensemble)
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{0.5, 0.3, 0.8}
    explanation, err := explainer.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    for _, feat := range explanation.TopFeatures(10) {
        fmt.Printf("  %s: %.4f\n", feat.Name, feat.Value)
    }
}
```

### LightGBM Example

```go
// Load LightGBM model (saved with booster.dump_model())
ensemble, err := tree.LoadLightGBMModel("model.json")
if err != nil {
    log.Fatal(err)
}

explainer, err := tree.New(ensemble)
// ... same as XGBoost
```

### Python: Export Models for Go

**XGBoost:**
```python
import xgboost as xgb

model = xgb.Booster()
model.load_model("model.bin")
model.save_model("model.json")  # JSON format for Go
```

**LightGBM:**
```python
import lightgbm as lgb
import json

model = lgb.Booster(model_file="model.txt")
with open("model.json", "w") as f:
    json.dump(model.dump_model(), f)
```

### Batch Processing

```go
// Explain multiple instances in parallel
instances := [][]float64{
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9},
}
explanations, err := explainer.ExplainBatch(ctx, instances)
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

### `explainer/tree`

TreeSHAP for tree-based models:

- **Exact** SHAP values (not approximations)
- O(TLD²) complexity - 40-100x faster than permutation
- XGBoost JSON model support
- LightGBM JSON model support
- Parallel batch processing

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

## Benchmarks

Performance benchmarks on Apple M1 Max (arm64):

### TreeSHAP Scaling

| Configuration | Time/op | Allocs/op |
|---------------|---------|-----------|
| 10 trees, depth 4, 10 features | 20μs | 372 |
| 100 trees, depth 4, 10 features | 194μs | 3,612 |
| 1000 trees, depth 4, 10 features | 1.9ms | 36,012 |

| Tree Depth | Time/op | Notes |
|------------|---------|-------|
| Depth 3 | 39μs | Shallow trees |
| Depth 6 | 598μs | Typical production depth |
| Depth 10 | 13.2ms | Very deep trees |

### TreeSHAP vs PermutationSHAP

| Method | Time/op | Type |
|--------|---------|------|
| **TreeSHAP** | **8.8μs** | **Exact** |
| PermutationSHAP (10 samples) | 16μs | Approximate |
| PermutationSHAP (50 samples) | 77μs | Approximate |
| PermutationSHAP (100 samples) | 153μs | Approximate |

TreeSHAP is **~17x faster** than PermutationSHAP with 100 samples while providing **exact** values.

### Realistic Model Sizes

| Model Size | Trees | Depth | Features | Time/op |
|------------|-------|-------|----------|---------|
| Small | 50 | 4 | 10 | 106μs |
| Medium | 200 | 6 | 30 | 2.7ms |
| Large | 500 | 8 | 50 | 31.7ms |

### Batch Processing

| Workers | 100 instances | Speedup |
|---------|---------------|---------|
| 1 (sequential) | 10.2ms | 1.0x |
| 4 (parallel) | 8.0ms | 1.3x |
| 8 (parallel) | 8.1ms | 1.3x |

Run benchmarks with:
```bash
go test -bench=. -benchmem ./explainer/tree/...
```

## Examples

The `examples/` directory contains working examples:

| Example | Description |
|---------|-------------|
| [`examples/linear`](examples/linear/) | PermutationSHAP with a simple linear model |
| [`examples/linearshap`](examples/linearshap/) | LinearSHAP for linear/logistic regression |
| [`examples/treeshap`](examples/treeshap/) | TreeSHAP with manually constructed tree ensembles |
| [`examples/sampling`](examples/sampling/) | SamplingSHAP Monte Carlo approximation |
| [`examples/batch`](examples/batch/) | Batch processing with parallel workers |
| [`examples/visualization`](examples/visualization/) | Generating chart data for visualizations |

Run an example:
```bash
go run ./examples/linear
go run ./examples/linearshap
go run ./examples/treeshap
go run ./examples/sampling
go run ./examples/batch
go run ./examples/visualization
```

## License

MIT License
