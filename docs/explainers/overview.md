# Choosing an Explainer

SHAP-Go provides multiple explainer algorithms, each with different trade-offs between speed, accuracy, and model compatibility. This guide helps you choose the right explainer for your use case.

## Quick Reference

| Explainer | Model Type | Accuracy | Speed | Status |
|-----------|-----------|----------|-------|--------|
| **TreeSHAP** | XGBoost, LightGBM | Exact | Very Fast | Implemented |
| **LinearSHAP** | Linear, Logistic | Exact | Fastest | Implemented |
| **PermutationSHAP** | Any | Exact | Slow | Implemented |
| **SamplingSHAP** | Any | Approximate | Fast | Implemented |
| **KernelSHAP** | Any | Approximate | Medium | Implemented |
| **ExactSHAP** | Any (≤15 features) | Exact | Very Slow | Planned |
| **DeepSHAP** | Neural Networks | Approximate | Fast | Planned |
| **GradientSHAP** | Neural Networks | Approximate | Fast | Planned |
| **PartitionSHAP** | Structured Data | Approximate | Fast | Planned |

## Decision Guide

```mermaid
graph TD
    A[What type of model?] --> B{Tree-based?}
    B -->|Yes| C[TreeSHAP]
    B -->|No| D{Linear model?}
    D -->|Yes| E[LinearSHAP]
    D -->|No| F{≤15 features?}
    F -->|Yes| G{Need exact values?}
    G -->|Yes| H[ExactSHAP]
    G -->|No| I[KernelSHAP]
    F -->|No| J{Need local accuracy guarantee?}
    J -->|Yes| K[PermutationSHAP]
    J -->|No| L{Need lower variance?}
    L -->|Yes| M[KernelSHAP]
    L -->|No| N[SamplingSHAP]

    C --> O[Exact, O(TLD²)]
    E --> P[Exact, O(n)]
    H --> Q[Exact, O(2ⁿ)]
    K --> R[Exact, O(n! × model calls)]
    M --> S[Approximate, weighted regression]
    N --> T[Approximate, fast sampling]
```

---

## Implemented Explainers

### TreeSHAP

**Best for:** XGBoost, LightGBM, and other tree ensemble models.

```go
ensemble, _ := tree.LoadXGBoostModel("model.json")
exp, _ := tree.New(ensemble)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Exact |
| **Complexity** | O(TLD²) where T=trees, L=leaves, D=depth |
| **Background data** | Not needed |
| **Local accuracy** | Guaranteed |

**When to use:**

- You have an XGBoost or LightGBM model
- You need exact values for regulatory/compliance
- You're explaining many predictions (batch processing)
- Speed is critical

**When NOT to use:**

- Your model is not a tree ensemble
- You need interaction values (not yet implemented)

[TreeSHAP Guide →](treeshap.md)

---

### LinearSHAP

**Best for:** Linear regression, logistic regression, and other linear models.

```go
exp, _ := linear.New(weights, bias, background)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Exact (closed-form solution) |
| **Complexity** | O(n) where n=features |
| **Background data** | Required (for feature means) |
| **Local accuracy** | Guaranteed |

**When to use:**

- Linear regression, logistic regression
- Ridge/Lasso regression
- Any model with linear coefficients
- You need zero variance in estimates

**When NOT to use:**

- Your model is non-linear
- You don't have access to model weights

[LinearSHAP Guide →](linearshap.md)

---

### PermutationSHAP

**Best for:** Any model where you need guaranteed local accuracy.

```go
exp, _ := permutation.New(model, background,
    explainer.WithNumSamples(100),
)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Exact (with sufficient samples) |
| **Complexity** | O(samples × features × model calls) |
| **Background data** | Required |
| **Local accuracy** | Guaranteed (antithetic sampling) |

**When to use:**

- Black-box models (ONNX, custom functions)
- Accuracy is more important than speed
- You need local accuracy guarantee (sum of SHAP = prediction - baseline)
- Regulatory/audit requirements

**When NOT to use:**

- Model inference is expensive
- You need real-time explanations
- You have a specialized model type (use TreeSHAP/LinearSHAP)

[PermutationSHAP Guide →](permutation.md)

---

### SamplingSHAP

**Best for:** Quick estimates when exact values aren't critical.

```go
exp, _ := sampling.New(model, background,
    explainer.WithNumSamples(50),
)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate |
| **Complexity** | O(samples × features × model calls) |
| **Background data** | Required |
| **Local accuracy** | Not guaranteed |

**When to use:**

- Prototyping and exploration
- Speed is more important than precision
- You'll validate important predictions manually
- Initial feature importance screening

**When NOT to use:**

- Regulatory/compliance requirements
- You need exact SHAP values
- Results will be used for automated decisions

[SamplingSHAP Guide →](sampling.md)

---

### KernelSHAP

**Best for:** Model-agnostic explanations with lower variance than sampling.

```go
exp, _ := kernel.New(model, background,
    explainer.WithNumSamples(100),
)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (weighted linear regression) |
| **Complexity** | O(samples × model calls + regression) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |

**When to use:**

- Any black-box model
- You want lower variance than SamplingSHAP
- Model calls are not extremely expensive
- You need a principled model-agnostic baseline

**When NOT to use:**

- You have a tree model (use TreeSHAP)
- You have a linear model (use LinearSHAP)
- Model inference is very expensive
- You need guaranteed local accuracy (use PermutationSHAP)

[KernelSHAP Guide →](kernelshap.md)

---

## Planned Explainers

### ExactSHAP

**For:** Any model with a small number of features (≤15).

| Property | Value |
|----------|-------|
| **Accuracy** | Exact (brute-force enumeration) |
| **Complexity** | O(2ⁿ × model calls) |
| **Use case** | Small feature sets, validation |

ExactSHAP computes true Shapley values by enumerating all 2ⁿ feature coalitions. This is only practical for small feature sets due to exponential complexity, but provides ground truth for validating other methods.

**When to use:**

- Very few features (≤12-15)
- You need mathematically exact values
- Validating other approximate methods
- Educational/research purposes

---

### DeepSHAP

**For:** Deep neural networks (TensorFlow, PyTorch, ONNX).

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (DeepLIFT-based) |
| **Complexity** | O(layers × neurons) |
| **Use case** | Neural network explanations |

DeepSHAP combines SHAP with DeepLIFT to efficiently explain neural network predictions. It propagates contributions through the network layers using modified backpropagation rules.

**When to use:**

- Deep neural networks
- Need efficient explanations for large networks
- Image/text model explanations

---

### GradientSHAP

**For:** Differentiable models where gradients are available.

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (expected gradients) |
| **Complexity** | O(samples × backward pass) |
| **Use case** | Neural networks, differentiable models |

GradientSHAP uses expected gradients (integrated gradients + noise) to estimate SHAP values. It connects SHAP to gradient-based attribution methods.

**When to use:**

- Differentiable models
- You have access to gradients
- Image classification explanations

---

### PartitionSHAP

**For:** Structured data with feature hierarchies or correlations.

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate |
| **Complexity** | O(partitions × model calls) |
| **Use case** | Correlated features, feature groups |

PartitionSHAP uses hierarchical clustering to group correlated features, computing SHAP values for feature groups first, then attributing within groups. This is faster for high-dimensional data with known structure.

**When to use:**

- Many correlated features
- Natural feature groupings (e.g., one-hot encoded categories)
- High-dimensional data (>100 features)

---

## Comparison Matrix

### Accuracy vs Speed

| Explainer | Accuracy | Speed |
|-----------|----------|-------|
| TreeSHAP | ★★★★★ Exact | ★★★★★ Very Fast |
| LinearSHAP | ★★★★★ Exact | ★★★★★ Fastest |
| PermutationSHAP | ★★★★★ Exact | ★★☆☆☆ Slow |
| KernelSHAP | ★★★★☆ Good | ★★★☆☆ Medium |
| SamplingSHAP | ★★★☆☆ Approximate | ★★★★☆ Fast |
| ExactSHAP | ★★★★★ Exact | ★☆☆☆☆ Very Slow |

### Model Compatibility

| Explainer | Trees | Linear | Neural Nets | Black-box |
|-----------|-------|--------|-------------|-----------|
| TreeSHAP | ✅ | ❌ | ❌ | ❌ |
| LinearSHAP | ❌ | ✅ | ❌ | ❌ |
| PermutationSHAP | ✅ | ✅ | ✅ | ✅ |
| KernelSHAP | ✅ | ✅ | ✅ | ✅ |
| SamplingSHAP | ✅ | ✅ | ✅ | ✅ |
| DeepSHAP | ❌ | ❌ | ✅ | ❌ |
| GradientSHAP | ❌ | ❌ | ✅ | ❌ |

---

## Performance Comparison

On Apple M1 Max with a simple model (5 features):

| Method | Time | Type |
|--------|------|------|
| LinearSHAP | ~1μs | Exact |
| TreeSHAP | 8.8μs | Exact |
| SamplingSHAP (50 samples) | ~40μs | Approximate |
| KernelSHAP (100 samples) | ~100μs | Approximate |
| PermutationSHAP (10 samples) | 16μs | Exact |
| PermutationSHAP (100 samples) | 153μs | Exact |

!!! note "Use the right tool for the job"
    TreeSHAP is 17x faster than PermutationSHAP for tree models while providing exact values. Always prefer specialized explainers when available.

---

## Common Configuration Options

All explainers support these options:

```go
exp, _ := explainer.New(model, background,
    explainer.WithNumSamples(100),      // Sampling-based only
    explainer.WithSeed(42),             // Reproducibility
    explainer.WithNumWorkers(4),        // Parallel processing
    explainer.WithFeatureNames(names),  // Human-readable names
    explainer.WithModelID("my-model"),  // For tracking
)
```

---

## Decision Flowchart by Use Case

### Production/Compliance

1. **Tree model?** → TreeSHAP (exact, fast)
2. **Linear model?** → LinearSHAP (exact, fastest)
3. **Other model + need audit trail?** → PermutationSHAP (guaranteed local accuracy)

### Exploration/Prototyping

1. **Quick iteration?** → SamplingSHAP (fast, approximate)
2. **Lower variance needed?** → KernelSHAP (principled approximation)

### Research/Validation

1. **Few features (≤15)?** → ExactSHAP (ground truth)
2. **Validating new method?** → Compare against ExactSHAP or PermutationSHAP
