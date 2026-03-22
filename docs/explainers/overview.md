# Choosing an Explainer

SHAP-Go provides multiple explainer algorithms, each with different trade-offs between speed, accuracy, and model compatibility. This guide helps you choose the right explainer for your use case.

## Quick Reference

| Explainer | Model Type | Accuracy | Speed | Status |
|-----------|-----------|----------|-------|--------|
| **TreeSHAP** | XGBoost, LightGBM, CatBoost | Exact | Very Fast | Implemented |
| **LinearSHAP** | Linear, Logistic | Exact | Fastest | Implemented |
| **AdditiveSHAP** | GAMs (no interactions) | Exact | Very Fast | Implemented |
| **PermutationSHAP** | Any | Exact | Slow | Implemented |
| **SamplingSHAP** | Any | Approximate | Fast | Implemented |
| **KernelSHAP** | Any | Approximate | Medium | Implemented |
| **ExactSHAP** | Any (≤15 features) | Exact | Very Slow | Implemented |
| **DeepSHAP** | Neural Networks | Approximate | Fast | Implemented |
| **GradientSHAP** | Any Differentiable | Approximate | Fast | Implemented |
| **PartitionSHAP** | Structured/Grouped | Approximate | Fast | Implemented |

## Decision Guide

```mermaid
graph TD
    A[What type of model?] --> B{Tree-based?}
    B -->|Yes| C[TreeSHAP]
    B -->|No| D{Linear model?}
    D -->|Yes| E[LinearSHAP]
    D -->|No| D2{Additive/GAM?}
    D2 -->|Yes| E2[AdditiveSHAP]
    D2 -->|No| F{≤15 features?}
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
    E2 --> P2[Exact, O(n×b)]
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

### AdditiveSHAP

**Best for:** Generalized Additive Models (GAMs) with no feature interactions.

```go
exp, _ := additive.New(model, background)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Exact (closed-form solution) |
| **Complexity** | O(n × b) where n=features, b=background |
| **Background data** | Required |
| **Local accuracy** | Guaranteed |

**When to use:**

- Generalized Additive Models (GAMs)
- Spline-based models (pygam, interpret-ml)
- Any model of the form: f(x) = Σ fᵢ(xᵢ)
- Your model has no feature interactions

**When NOT to use:**

- Your model has feature interactions
- Tree models (use TreeSHAP)
- Linear models (use LinearSHAP - more efficient)
- You're unsure if your model is truly additive

[AdditiveSHAP Guide →](additiveshap.md)

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

### ExactSHAP

**Best for:** Any model with a small number of features (≤15) where exact values are required.

```go
exp, _ := exact.New(model, background,
    explainer.WithFeatureNames(names),
)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Exact (brute-force enumeration) |
| **Complexity** | O(n × 2ⁿ × model calls) |
| **Background data** | Required |
| **Local accuracy** | Guaranteed |

**When to use:**

- Very few features (≤15)
- You need mathematically exact SHAP values
- Validating other approximate methods (KernelSHAP, SamplingSHAP)
- Educational/research purposes
- Ground truth for testing

**When NOT to use:**

- More than 15 features (exponential complexity)
- Real-time applications
- Tree models (use TreeSHAP)
- Linear models (use LinearSHAP)

[ExactSHAP Guide →](exactshap.md)

---

### DeepSHAP

**Best for:** Neural networks in ONNX format.

```go
graphInfo, _ := onnx.ParseGraph("model.onnx")
session, _ := onnx.NewActivationSession(config)
exp, _ := deepshap.New(session, graphInfo, background)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (DeepLIFT-based) |
| **Complexity** | O(layers × neurons × background samples) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |

**When to use:**

- Deep neural networks in ONNX format
- Need efficient explanations for dense networks
- Model uses Dense, ReLU, Sigmoid, Tanh, or Softmax layers

**When NOT to use:**

- Convolutional networks (not yet supported)
- Tree models (use TreeSHAP)
- Linear models (use LinearSHAP)
- Need exact SHAP values (use ExactSHAP or PermutationSHAP)

[DeepSHAP Guide →](deepshap.md)

---

### GradientSHAP

**Best for:** Differentiable models using numerical gradients.

```go
exp, _ := gradient.New(model, background,
    []explainer.Option{
        explainer.WithNumSamples(300),
    },
)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (expected gradients) |
| **Complexity** | O(samples × features × 2) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |

**When to use:**

- Differentiable models (neural networks, polynomial models)
- You want lower variance than pure sampling methods
- Confidence intervals are needed
- Model-agnostic explanations with gradient awareness

**When NOT to use:**

- Tree models (use TreeSHAP)
- Linear models (use LinearSHAP)
- Non-differentiable models (use KernelSHAP)
- Need guaranteed exact values (use ExactSHAP or PermutationSHAP)

[GradientSHAP Guide →](gradientshap.md)

---

### PartitionSHAP

**Best for:** Structured data with feature hierarchies or natural groupings.

```go
hierarchy := &partition.Node{
    Name: "root",
    Children: []*partition.Node{
        {Name: "demographics", Children: []*partition.Node{
            {Name: "age", FeatureIdx: 0},
            {Name: "gender", FeatureIdx: 1},
        }},
        {Name: "financials", Children: []*partition.Node{
            {Name: "income", FeatureIdx: 2},
        }},
    },
}
exp, _ := partition.New(model, background, hierarchy)
```

| Property | Value |
|----------|-------|
| **Accuracy** | Approximate (Owen values) |
| **Complexity** | O(k! × samples × depth) |
| **Background data** | Required |
| **Local accuracy** | Approximately satisfied |

**When to use:**

- Features have natural groupings (demographics, financials, etc.)
- Feature correlations within groups are stronger than between groups
- You want hierarchical explanations
- Domain knowledge suggests feature organization

**When NOT to use:**

- No clear feature groupings exist
- Tree models (use TreeSHAP)
- Linear models (use LinearSHAP)
- Need exact values (use ExactSHAP)

[PartitionSHAP Guide →](partitionshap.md)

---

## Comparison Matrix

### Accuracy vs Speed

| Explainer | Accuracy | Speed |
|-----------|----------|-------|
| TreeSHAP | ★★★★★ Exact | ★★★★★ Very Fast |
| LinearSHAP | ★★★★★ Exact | ★★★★★ Fastest |
| AdditiveSHAP | ★★★★★ Exact | ★★★★★ Very Fast |
| PermutationSHAP | ★★★★★ Exact | ★★☆☆☆ Slow |
| KernelSHAP | ★★★★☆ Good | ★★★☆☆ Medium |
| SamplingSHAP | ★★★☆☆ Approximate | ★★★★☆ Fast |
| ExactSHAP | ★★★★★ Exact | ★☆☆☆☆ Very Slow |
| DeepSHAP | ★★★★☆ Good | ★★★★☆ Fast |
| GradientSHAP | ★★★★☆ Good | ★★★★☆ Fast |
| PartitionSHAP | ★★★★☆ Good | ★★★★☆ Fast |

### Model Compatibility

| Explainer | Trees | Linear | GAMs | Neural Nets | Black-box |
|-----------|-------|--------|------|-------------|-----------|
| TreeSHAP | ✅ | ❌ | ❌ | ❌ | ❌ |
| LinearSHAP | ❌ | ✅ | ❌ | ❌ | ❌ |
| AdditiveSHAP | ❌ | ✅ | ✅ | ❌ | ❌ |
| PermutationSHAP | ✅ | ✅ | ✅ | ✅ | ✅ |
| KernelSHAP | ✅ | ✅ | ✅ | ✅ | ✅ |
| SamplingSHAP | ✅ | ✅ | ✅ | ✅ | ✅ |
| ExactSHAP | ✅* | ✅* | ✅* | ✅* | ✅* |
| DeepSHAP | ❌ | ❌ | ❌ | ✅ | ❌ |
| GradientSHAP | ❌ | ✅ | ✅ | ✅ | ❌ |
| PartitionSHAP | ✅ | ✅ | ✅ | ✅ | ✅ |

*ExactSHAP: Limited to ≤15 features due to O(2^n) complexity

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
3. **Additive/GAM model?** → AdditiveSHAP (exact, fast)
4. **Other model + need audit trail?** → PermutationSHAP (guaranteed local accuracy)

### Exploration/Prototyping

1. **Quick iteration?** → SamplingSHAP (fast, approximate)
2. **Lower variance needed?** → KernelSHAP (principled approximation)

### Research/Validation

1. **Few features (≤15)?** → ExactSHAP (ground truth)
2. **Validating new method?** → Compare against ExactSHAP or PermutationSHAP
