# DeepSHAP

DeepSHAP is an efficient algorithm for computing SHAP values for deep neural networks. It combines the DeepLIFT algorithm with Shapley values to provide theoretically grounded feature attributions.

## Overview

DeepSHAP works by:

1. Running a forward pass to capture activations at each layer
2. Computing reference activations from a background dataset
3. Propagating attribution multipliers backward using the DeepLIFT rescale rule
4. Averaging attributions over multiple background samples

## Key Properties

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (DeepLIFT-based) |
| **Complexity** | O(layers × neurons × background samples) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/deepshap"
    "github.com/plexusone/shap-go/model/onnx"
)

func main() {
    // Initialize ONNX Runtime
    if err := onnx.InitializeRuntime("/path/to/libonnxruntime.so"); err != nil {
        log.Fatal(err)
    }
    defer onnx.DestroyRuntime()

    // Parse the ONNX model graph
    graphInfo, err := onnx.ParseGraph("model.onnx")
    if err != nil {
        log.Fatal(err)
    }

    // Create activation session with intermediate outputs
    config := onnx.ActivationConfig{
        Config: onnx.Config{
            ModelPath:   "model.onnx",
            InputName:   "input",
            OutputName:  "output",
            NumFeatures: 10,
        },
        IntermediateOutputs: graphInfo.GetAllLayerOutputs(),
    }
    session, err := onnx.NewActivationSession(config)
    if err != nil {
        log.Fatal(err)
    }
    defer session.Close()

    // Background data for SHAP computation
    background := [][]float64{
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        // ... more background samples
    }

    // Create DeepSHAP explainer
    exp, err := deepshap.New(session, graphInfo, background,
        explainer.WithFeatureNames([]string{"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"}),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5}
    explanation, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    for _, feat := range explanation.TopFeatures(5) {
        fmt.Printf("  %s: %.4f\n", feat.Name, feat.Value)
    }
}
```

## Simplified Usage

If you don't need full graph structure, use `NewSimple`:

```go
exp, err := deepshap.NewSimple(session, background,
    explainer.WithFeatureNames(names),
)
```

This uses a simpler attribution method that doesn't require graph parsing.

## Supported Layer Types

DeepSHAP supports common neural network layer types:

| Layer Type | ONNX Op Types | Attribution Rule |
|------------|---------------|------------------|
| Dense | Gemm, MatMul | Linear backprop through weights |
| ReLU | Relu | DeepLIFT rescale rule |
| Sigmoid | Sigmoid | DeepLIFT rescale rule |
| Tanh | Tanh | DeepLIFT rescale rule |
| Softmax | Softmax | DeepLIFT rescale rule |
| Add | Add | Proportional split |
| Identity | Identity, Dropout, Flatten | Pass-through |

## Attribution Rules

DeepSHAP uses the DeepLIFT rescale rule for propagating attributions:

```
mult_in = mult_out × (x - x_ref) / (y - y_ref)
```

Where:

- `mult_in` / `mult_out` are input/output multipliers
- `x`, `x_ref` are input activations for instance/reference
- `y`, `y_ref` are output activations for instance/reference

When the denominator is near zero, the rule falls back to gradient computation.

## Background Dataset

The background dataset serves as the baseline for attribution:

- Use representative samples from your training data
- 100-1000 samples typically provides good results
- More samples improve accuracy but increase computation time

```go
// Use k-means to summarize large datasets
bgDataset := background.NewDataset(trainingData, featureNames)
summary := bgDataset.KMeansSummary(100, 10, rng)
```

## Configuration Options

```go
exp, err := deepshap.New(session, graphInfo, background,
    explainer.WithFeatureNames(names),  // Human-readable names
    explainer.WithModelID("my-nn"),     // Model identifier
)
```

## When to Use DeepSHAP

**Use DeepSHAP when:**

- You have a neural network model in ONNX format
- The network uses supported layer types (Dense, ReLU, Sigmoid, etc.)
- You need efficient explanations for deep networks
- Local accuracy (sum of SHAP ≈ prediction - baseline) is important

**Don't use DeepSHAP when:**

- Your model is a tree ensemble (use TreeSHAP)
- Your model is linear (use LinearSHAP)
- You need convolutional layer support (not yet implemented)
- You need guaranteed exact values (use PermutationSHAP or ExactSHAP)

## Limitations

Current limitations of the DeepSHAP implementation:

- **Sequential networks only**: Residual connections have limited support
- **No convolutional layers**: Conv2D, MaxPool, BatchNorm not yet supported
- **Single output**: Multi-output models not yet supported
- **Dense networks**: Best suited for fully-connected architectures

## Technical Details

### ONNX Graph Parsing

DeepSHAP parses the ONNX model structure to understand layer connectivity:

```go
graphInfo, err := onnx.ParseGraph("model.onnx")

// Get all layer outputs for activation capture
outputs := graphInfo.GetAllLayerOutputs()

// Get nodes in reverse topological order for backprop
reversed := graphInfo.ReverseTopologicalOrder()
```

### Activation Capture

The `ActivationSession` captures intermediate layer outputs:

```go
config := onnx.ActivationConfig{
    Config: onnx.Config{...},
    IntermediateOutputs: []string{"dense1_out", "relu1_out", "dense2_out"},
}
session, _ := onnx.NewActivationSession(config)

result, _ := session.PredictWithActivations(ctx, input)
// result.Prediction contains final output
// result.Activations contains intermediate values
```

### Backward Propagation

The propagation engine traverses the graph in reverse order:

```go
engine := deepshap.NewPropagationEngine(graphInfo)
result, _ := engine.Propagate(instanceAct, referenceAct, 1.0)
// result.Attributions contains SHAP values
```

## References

- [SHAP paper](https://arxiv.org/abs/1705.07874): A Unified Approach to Interpreting Model Predictions
- [DeepLIFT paper](https://arxiv.org/abs/1704.02685): Learning Important Features Through Propagating Activation Differences
- [Python SHAP library](https://github.com/slundberg/shap): Original implementation
