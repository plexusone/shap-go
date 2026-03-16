# SHAP-Go

A Go library for computing SHAP (SHapley Additive exPlanations) values for ML model explainability.

[![Go CI](https://github.com/plexusone/shap-go/actions/workflows/go-ci.yaml/badge.svg?branch=main)](https://github.com/plexusone/shap-go/actions/workflows/go-ci.yaml)
[![Go Reference](https://pkg.go.dev/badge/github.com/plexusone/shap-go.svg)](https://pkg.go.dev/github.com/plexusone/shap-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/plexusone/shap-go)](https://goreportcard.com/report/github.com/plexusone/shap-go)

## Overview

SHAP-Go provides a Go-native implementation of SHAP value computation for explaining machine learning model predictions. It supports:

- **ONNX models** via ONNX Runtime bindings
- **XGBoost/LightGBM** via TreeSHAP (exact, fast)
- **Custom models** via a simple function interface
- **Multiple explainer algorithms** with different trade-offs
- **JSON-serializable explanations** for audit/compliance

## Why SHAP?

SHAP (SHapley Additive exPlanations) values provide a unified approach to explaining machine learning predictions. They answer: "How much did each feature contribute to this prediction?"

Key properties:

- **Local accuracy**: SHAP values sum to the difference between prediction and baseline
- **Consistency**: If a feature's contribution increases, its SHAP value won't decrease
- **Missingness**: Features with no impact have SHAP value of 0

## Why Go?

- **Performance**: Native code, no Python overhead
- **Deployment**: Single binary, easy containerization
- **Concurrency**: Built-in parallel processing
- **Type safety**: Catch errors at compile time

## Quick Example

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
    background := [][]float64{{0.0, 0.0}}

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
}
```

## Explainer Types

| Status | Explainer | Model Type | Notes |
|:------:|-----------|------------|-------|
| :white_check_mark: | **TreeSHAP** | Trees | Exact & fast (O(TLD²)) for XGBoost, LightGBM |
| :white_check_mark: | **LinearSHAP** | Linear | Exact closed-form for linear/logistic regression |
| :white_check_mark: | **PermutationSHAP** | Any | Black-box, guarantees local accuracy |
| :white_check_mark: | **SamplingSHAP** | Any | Monte Carlo approximation, fast estimates |
| :white_check_mark: | **KernelSHAP** | Any | Model-agnostic weighted linear regression |
| :white_check_mark: | **ExactSHAP** | Any | Brute-force exact, O(2^n) for small feature sets |
| :white_large_square: | DeepSHAP | Neural Nets | For deep learning (planned) |

## Getting Started

Ready to start explaining your models?

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install SHAP-Go in your project

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quick start](getting-started/quickstart.md)

-   :material-tree:{ .lg .middle } **TreeSHAP**

    ---

    Exact SHAP for XGBoost/LightGBM

    [:octicons-arrow-right-24: TreeSHAP guide](explainers/treeshap.md)

-   :material-function:{ .lg .middle } **LinearSHAP**

    ---

    Exact SHAP for linear/logistic regression

    [:octicons-arrow-right-24: LinearSHAP guide](explainers/linearshap.md)

-   :material-cube-outline:{ .lg .middle } **KernelSHAP**

    ---

    Model-agnostic weighted linear regression

    [:octicons-arrow-right-24: KernelSHAP guide](explainers/kernelshap.md)

-   :material-chart-bar:{ .lg .middle } **Benchmarks**

    ---

    Performance characteristics

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

</div>
