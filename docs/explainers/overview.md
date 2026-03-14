# Choosing an Explainer

SHAP-Go provides multiple explainer algorithms, each with different trade-offs between speed, accuracy, and model compatibility.

## Decision Guide

```mermaid
graph TD
    A[What type of model?] --> B{Tree-based?}
    B -->|Yes| C[TreeSHAP]
    B -->|No| D{Linear model?}
    D -->|Yes| E[LinearSHAP]
    D -->|No| F{Need exact values?}
    F -->|Yes| G[PermutationSHAP]
    F -->|No| H[SamplingSHAP]

    C --> I[Exact, O(TLD²)]
    E --> J[Exact, O(n)]
    G --> K[Exact, O(n! × model calls)]
    H --> L[Approximate, fast]
```

## Comparison Table

| Explainer | Model Type | Accuracy | Speed | Use Case |
|-----------|-----------|----------|-------|----------|
| **TreeSHAP** | XGBoost, LightGBM | Exact | Fast | Production tree models |
| **LinearSHAP** | Linear, Logistic | Exact | Fastest | Linear/logistic regression |
| **PermutationSHAP** | Any | Exact | Slow | When accuracy is critical |
| **SamplingSHAP** | Any | Approximate | Fast | Quick estimates, prototyping |

## TreeSHAP

**Best for:** XGBoost, LightGBM, and other tree ensemble models.

```go
ensemble, _ := tree.LoadXGBoostModel("model.json")
exp, _ := tree.New(ensemble)
```

**Characteristics:**

- :white_check_mark: Exact SHAP values (not approximations)
- :white_check_mark: O(TLD²) complexity - very fast
- :white_check_mark: Handles missing values correctly
- :white_check_mark: No background data needed
- :x: Only works with tree models

**When to use:**

- You have an XGBoost or LightGBM model
- You need exact values for regulatory/compliance
- You're explaining many predictions (batch processing)

[TreeSHAP Guide →](treeshap.md)

## LinearSHAP

**Best for:** Linear regression, logistic regression, and other linear models.

```go
exp, _ := linear.New(weights, bias, background)
```

**Characteristics:**

- :white_check_mark: Exact SHAP values (closed-form solution)
- :white_check_mark: O(n) complexity - fastest possible
- :white_check_mark: No sampling variance
- :white_check_mark: Deterministic
- :x: Only works with linear models

**When to use:**

- Linear regression, logistic regression
- Ridge/Lasso regression
- Any model with linear coefficients

[LinearSHAP Guide →](linearshap.md)

## PermutationSHAP

**Best for:** Any model where you need guaranteed accuracy.

```go
exp, _ := permutation.New(model, background,
    explainer.WithNumSamples(100),
)
```

**Characteristics:**

- :white_check_mark: Works with any model
- :white_check_mark: Guarantees local accuracy property
- :white_check_mark: Uses antithetic sampling for variance reduction
- :x: Slower - requires many model calls
- :x: Needs representative background data

**When to use:**

- Black-box models (ONNX, custom functions)
- Accuracy is more important than speed
- You have representative background samples

[PermutationSHAP Guide →](permutation.md)

## SamplingSHAP

**Best for:** Quick estimates when exact values aren't critical.

```go
exp, _ := sampling.New(model, background,
    explainer.WithNumSamples(50),
)
```

**Characteristics:**

- :white_check_mark: Works with any model
- :white_check_mark: Faster than PermutationSHAP
- :white_check_mark: Simple implementation
- :x: Approximate (higher variance)
- :x: Doesn't guarantee local accuracy

**When to use:**

- Prototyping and exploration
- Speed is more important than precision
- You'll validate important predictions manually

[SamplingSHAP Guide →](sampling.md)

## Performance Comparison

On Apple M1 Max with a simple model (5 features):

| Method | Time | Type |
|--------|------|------|
| TreeSHAP | 8.8μs | Exact |
| PermutationSHAP (10 samples) | 16μs | Exact |
| PermutationSHAP (100 samples) | 153μs | Exact |
| SamplingSHAP (50 samples) | ~40μs | Approximate |

!!! note "TreeSHAP is 17x faster"
    For tree models, TreeSHAP provides exact values in a fraction of the time.

## Common Configuration Options

All explainers support these options:

```go
exp, _ := explainer.New(model, background,
    explainer.WithNumSamples(100),      // Permutation/Sampling only
    explainer.WithSeed(42),             // Reproducibility
    explainer.WithNumWorkers(4),        // Parallel processing
    explainer.WithFeatureNames(names),  // Human-readable names
    explainer.WithModelID("my-model"),  // For tracking
)
```

## Future Explainers

These are planned but not yet implemented:

| Explainer | Model Type | Status |
|-----------|-----------|--------|
| KernelSHAP | Any | Planned |
| DeepSHAP | Neural Networks | Planned |
| GradientSHAP | Neural Networks | Planned |
| ExactSHAP | Any (≤15 features) | Planned |
