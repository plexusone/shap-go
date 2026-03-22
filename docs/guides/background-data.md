# Background Data Selection Guide

Background data (also called reference data or baseline data) is fundamental to SHAP explanations. This guide covers best practices for selecting background data that produces meaningful and stable explanations.

## What is Background Data?

SHAP values explain a prediction by comparing it to a baseline expectation. The background dataset defines this baseline - it represents what the model would predict "on average" without knowing specific feature values.

```go
// Background data passed to explainer constructors
background := [][]float64{
    {0.5, 1.2, 3.4},
    {0.8, 1.5, 2.9},
    {0.3, 0.9, 3.1},
    // ... more samples
}

exp, err := kernel.New(model, background, opts)
```

## Key Principles

### 1. Representativeness

Background data should represent the typical distribution of your input data.

**Good practice:**

```go
// Use a random sample from your training/validation data
background := selectRandomSample(trainingData, 100)
```

**Avoid:**

```go
// Don't use only extreme values
background := [][]float64{
    {0.0, 0.0, 0.0},  // All minimum values
    {1.0, 1.0, 1.0},  // All maximum values
}
```

### 2. Sample Size

The optimal background size depends on your explainer and computational budget.

| Explainer | Recommended Size | Notes |
|-----------|-----------------|-------|
| TreeSHAP | 100-1000 | More samples improve accuracy |
| KernelSHAP | 50-200 | Larger increases computation quadratically |
| DeepSHAP | 50-200 | Average over background is computed |
| GradientSHAP | 50-500 | Interpolation performed with each sample |
| PermutationSHAP | 100-500 | Marginal expectation estimation |
| SamplingSHAP | 100-500 | Monte Carlo sampling |
| LinearSHAP | 10-100 | Only needs mean estimation |
| ExactSHAP | 10-50 | Computational cost is O(n*2^d) |

### 3. Diversity

Include diverse samples that cover the feature space.

```go
// Stratified sampling for classification tasks
background := make([][]float64, 0)
for _, class := range classes {
    samples := selectFromClass(data, class, samplesPerClass)
    background = append(background, samples...)
}
```

### 4. Feature Independence Assumption

Many SHAP methods assume features are independent given the background. If features are correlated:

- Consider larger background samples to capture correlations
- Use PartitionSHAP (future) for hierarchical feature grouping
- Be aware that explanations may misattribute between correlated features

## Explainer-Specific Guidance

### TreeSHAP

TreeSHAP uses background data for marginal expectation in tree paths.

```go
import "github.com/plexusone/shap-go/explainer/tree"

// TreeSHAP benefits from more background samples
// 100-1000 samples recommended for stable estimates
background := selectRandomSample(trainingData, 500)

exp, err := tree.New(ensemble, background,
    explainer.WithFeatureNames(featureNames),
)
```

**Tips:**

- Include edge cases if they're part of normal operation
- More samples improve accuracy but increase memory usage
- Background influences base value calculation

### KernelSHAP

KernelSHAP's computation scales with background size. Choose carefully.

```go
import "github.com/plexusone/shap-go/explainer/kernel"

// KernelSHAP: smaller background for speed
// Each sample adds a row to the weighted regression
background := selectRandomSample(trainingData, 100)

exp, err := kernel.New(model, background,
    explainer.WithNumSamples(2048),  // Coalition samples
)
```

**Tips:**

- Start with ~100 samples
- Use k-means clustering to select diverse representatives
- Background affects base value: E[f(x)] over background

### DeepSHAP

DeepSHAP computes attributions relative to averaged reference activations.

```go
import "github.com/plexusone/shap-go/explainer/deepshap"

// DeepSHAP: moderate background size
// Activations are averaged over all background samples
background := selectRandomSample(trainingData, 100)

exp, err := deepshap.New(activationSession, background)
```

**Tips:**

- Include representative examples from each class
- Avoid using only zeros (can cause numerical issues)
- Background activations are cached - larger samples increase memory

### GradientSHAP

GradientSHAP interpolates between instance and background samples.

```go
import "github.com/plexusone/shap-go/explainer/gradient"

// GradientSHAP: uses background for Expected Gradients
background := selectRandomSample(trainingData, 200)

exp, err := gradient.New(model, background,
    []explainer.Option{explainer.WithNumSamples(500)},
    gradient.WithEpsilon(1e-4),  // For numerical gradients
)
```

**Tips:**

- Each sample generates interpolated points
- More background samples reduce variance
- Include both typical and boundary cases

### LinearSHAP

LinearSHAP only needs feature means from background.

```go
import "github.com/plexusone/shap-go/explainer/linear"

// LinearSHAP: minimal background needed
// Only uses mean of background features
background := selectRandomSample(trainingData, 50)

exp, err := linear.New(weights, bias, background)
```

**Tips:**

- Even small samples give stable means
- Exact closed-form: no sampling variance
- Larger samples only marginally improve base value accuracy

## Selection Strategies

### Random Sampling

Simplest approach - randomly sample from training data.

```go
func selectRandomSample(data [][]float64, n int) [][]float64 {
    if n >= len(data) {
        return data
    }

    indices := make([]int, len(data))
    for i := range indices {
        indices[i] = i
    }

    rand.Shuffle(len(indices), func(i, j int) {
        indices[i], indices[j] = indices[j], indices[i]
    })

    result := make([][]float64, n)
    for i := 0; i < n; i++ {
        result[i] = data[indices[i]]
    }
    return result
}
```

### Stratified Sampling

For classification, sample proportionally from each class.

```go
func selectStratifiedSample(data [][]float64, labels []int, n int) [][]float64 {
    // Group by label
    byLabel := make(map[int][]int)
    for i, label := range labels {
        byLabel[label] = append(byLabel[label], i)
    }

    // Sample proportionally
    result := make([][]float64, 0, n)
    for _, indices := range byLabel {
        count := int(float64(n) * float64(len(indices)) / float64(len(data)))
        if count < 1 {
            count = 1
        }
        rand.Shuffle(len(indices), func(i, j int) {
            indices[i], indices[j] = indices[j], indices[i]
        })
        for i := 0; i < count && i < len(indices); i++ {
            result = append(result, data[indices[i]])
        }
    }
    return result
}
```

### K-Means Clustering

Select cluster centroids for maximum diversity.

```go
// Using a k-means library
func selectKMeansCentroids(data [][]float64, k int) [][]float64 {
    // Run k-means clustering
    clusters := kmeans.Cluster(data, k)

    // Return centroids
    return clusters.Centroids()
}
```

### Prototype Selection

Select prototypical examples that represent data regions.

```go
// Select samples closest to their cluster centers
func selectPrototypes(data [][]float64, k int) [][]float64 {
    clusters := kmeans.Cluster(data, k)

    prototypes := make([][]float64, k)
    for i, centroid := range clusters.Centroids() {
        prototypes[i] = findClosestPoint(data, centroid)
    }
    return prototypes
}
```

## Common Pitfalls

### 1. Using All Zeros

```go
// AVOID: Zero background can cause numerical issues
background := [][]float64{{0, 0, 0, 0}}
```

Zero backgrounds can cause:

- Division by zero in some attribution rules
- Undefined gradients at discontinuities
- Explanations that don't generalize

### 2. Too Few Samples

```go
// AVOID: Single sample gives unstable estimates
background := [][]float64{{0.5, 0.5, 0.5}}
```

Use at least 10-50 samples for stable explanations.

### 3. Outliers Only

```go
// AVOID: Extreme values don't represent typical behavior
background := [][]float64{
    extremeMin,
    extremeMax,
}
```

Include typical values, not just boundary cases.

### 4. Ignoring Data Types

```go
// For categorical features encoded as integers:
// Background should include all category values
background := [][]float64{
    {0.0, 1.5, 2.3},  // category 0
    {1.0, 1.8, 2.1},  // category 1
    {2.0, 1.2, 2.5},  // category 2
}
```

### 5. Train vs. Test Distribution Shift

If your test data differs from training:

```go
// If explaining production data, sample from production distribution
background := selectFromProductionData(n)

// Not just training data if distributions differ
// background := selectFromTrainingData(n)  // May be inappropriate
```

## Validation

### Check Base Value Stability

```go
// Base value should be stable across similar backgrounds
bg1 := selectRandomSample(data, 100)
bg2 := selectRandomSample(data, 100)

exp1, _ := kernel.New(model, bg1, opts)
exp2, _ := kernel.New(model, bg2, opts)

// Base values should be similar
diff := math.Abs(exp1.BaseValue() - exp2.BaseValue())
if diff > tolerance {
    // Background may be too small or non-representative
}
```

### Verify Local Accuracy

```go
// SHAP values should sum to prediction - base_value
result, _ := exp.Explain(ctx, instance)
verify := result.Verify(tolerance)
if !verify.Valid {
    // May indicate background issues
}
```

### Check Explanation Stability

```go
// Run multiple times with different seeds
explanations := make([]*explanation.Explanation, 10)
for i := 0; i < 10; i++ {
    exp, _ := sampling.New(model, background,
        explainer.WithSeed(int64(i)),
    )
    explanations[i], _ = exp.Explain(ctx, instance)
}

// Check variance in SHAP values
variance := computeVariance(explanations)
if variance > threshold {
    // Consider larger background or more samples
}
```

## Summary

| Aspect | Recommendation |
|--------|----------------|
| Size | 50-500 samples depending on explainer |
| Selection | Random or stratified sampling from training data |
| Diversity | Cover feature space, include all categories |
| Validation | Check base value stability and local accuracy |
| Avoid | All zeros, single sample, outliers only |

Good background data selection is essential for meaningful SHAP explanations. When in doubt, use more samples from a representative subset of your data.
