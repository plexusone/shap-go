# TreeSHAP

TreeSHAP computes **exact** SHAP values for tree ensemble models (XGBoost, LightGBM) using a polynomial-time algorithm that exploits the tree structure.

## Why TreeSHAP?

| Property | TreeSHAP | Permutation SHAP |
|----------|----------|------------------|
| **Accuracy** | Exact | Exact (with enough samples) |
| **Speed** | O(TLD²) | O(samples × features × model calls) |
| **Typical time** | 8-200μs | 15-150ms |
| **Background data** | Not needed | Required |

For a model with 100 trees, TreeSHAP is typically **100-1000x faster** than sampling-based methods.

## Basic Usage

### From XGBoost

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    // Load XGBoost model (exported from Python as JSON)
    ensemble, err := tree.LoadXGBoostModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    // Create explainer
    exp, err := tree.New(ensemble)
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{0.5, 0.3, 0.8, 0.2, 0.9}
    explanation, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
    fmt.Println("\nTop contributing features:")
    for _, feat := range explanation.TopFeatures(5) {
        sign := "+"
        if feat.SHAPValue < 0 {
            sign = "-"
        }
        fmt.Printf("  %s %s: %.4f\n", sign, feat.Name, feat.SHAPValue)
    }
}
```

### From LightGBM

```go
// Load LightGBM model (exported with dump_model())
ensemble, err := tree.LoadLightGBMModel("model.json")
if err != nil {
    log.Fatal(err)
}

exp, err := tree.New(ensemble)
// ... same as above
```

## Batch Processing

Explain multiple instances efficiently with parallel workers:

```go
// Create explainer with parallel processing
exp, _ := tree.New(ensemble,
    explainer.WithNumWorkers(4),  // Use 4 CPU cores
)

// Prepare batch
instances := [][]float64{
    {0.1, 0.2, 0.3, 0.4, 0.5},
    {0.5, 0.4, 0.3, 0.2, 0.1},
    // ... more instances
}

// Explain all at once
ctx := context.Background()
explanations, err := exp.ExplainBatch(ctx, instances)
if err != nil {
    log.Fatal(err)
}

// Process results
for i, exp := range explanations {
    fmt.Printf("Instance %d: prediction=%.4f\n", i, exp.Prediction)
}
```

## Understanding TreeSHAP Output

### Base Value

The base value is the **expected prediction** across all possible feature coalitions, weighted by the training data distribution (cover). This is computed directly from the tree structure.

```go
baseValue := exp.BaseValue()
fmt.Printf("Expected prediction: %.4f\n", baseValue)
```

### SHAP Values

Each feature gets a SHAP value indicating how much it pushed the prediction away from the base value:

- **Positive SHAP**: Feature increased the prediction
- **Negative SHAP**: Feature decreased the prediction
- **Zero SHAP**: Feature had no effect

### Local Accuracy

TreeSHAP guarantees:

```
sum(SHAP values) = prediction - base_value
```

Verify this:

```go
result := explanation.Verify(1e-9)
if !result.Valid {
    log.Printf("Warning: accuracy issue, diff=%.2e\n", result.Difference)
}
```

## Configuration Options

```go
exp, err := tree.New(ensemble,
    // Feature names (optional - uses model's if available)
    explainer.WithFeatureNames([]string{"age", "income", "score"}),

    // Model identifier for tracking
    explainer.WithModelID("loan-approval-v2"),

    // Parallel workers for batch processing
    explainer.WithNumWorkers(4),
)
```

## Working with Tree Ensembles

### Inspecting the Model

```go
ensemble, _ := tree.LoadXGBoostModel("model.json")

fmt.Printf("Trees: %d\n", ensemble.NumTrees)
fmt.Printf("Features: %d\n", ensemble.NumFeatures)
fmt.Printf("Max depth: %d\n", ensemble.MaxDepth())
fmt.Printf("Base score: %.4f\n", ensemble.BaseScore)
```

### Feature Names

If your model has feature names:

```go
for i, name := range ensemble.FeatureNames {
    fmt.Printf("Feature %d: %s\n", i, name)
}
```

### Serialization

Save/load the unified ensemble format:

```go
// Save
jsonData, _ := ensemble.ToJSON()
os.WriteFile("ensemble.json", jsonData, 0644)

// Load
data, _ := os.ReadFile("ensemble.json")
loaded, _ := tree.EnsembleFromJSON(data)
```

## Interaction Values

TreeSHAP can compute pairwise feature interactions, which reveal how features work together to influence predictions. The interaction matrix is symmetric and satisfies key SHAP properties.

### Computing Interactions

```go
// Compute SHAP interaction values
result, err := exp.ExplainInteractions(ctx, instance)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Prediction: %.4f\n", result.Prediction)
fmt.Printf("Base Value: %.4f\n", result.BaseValue)
```

### Interaction Matrix Properties

The interaction matrix `Φ` has these properties:

- **Diagonal** `Φ[i][i]`: Main effect of feature i
- **Off-diagonal** `Φ[i][j]`: Interaction between features i and j
- **Symmetric**: `Φ[i][j] == Φ[j][i]`
- **Rows sum to SHAP**: `sum(Φ[i][:]) == SHAP[i]`
- **Total sum**: `sum(all Φ) == prediction - base_value`

### Working with Interactions

```go
// Get interaction between two features
interaction := result.GetInteraction(0, 1)

// Get main effect (diagonal)
mainEffect := result.GetMainEffect(0)

// Get derived SHAP value (row sum)
shapValue := result.GetSHAPValue(0)

// Get top k strongest interactions (by absolute value)
topK := result.TopInteractions(5)
for _, inter := range topK {
    fmt.Printf("%s <-> %s: %.4f\n", inter.Name1, inter.Name2, inter.Value)
}
```

### When to Use Interactions

Interaction values are useful when:

- **Understanding feature synergies**: Which features amplify each other's effects?
- **Model debugging**: Identify unexpected feature dependencies
- **Feature engineering**: Discover features that should be combined
- **Regulatory compliance**: Explain complex model behavior

### Complexity

Computing interactions is more expensive than regular SHAP values:

- **Regular TreeSHAP**: O(TLD²)
- **TreeSHAP Interactions**: O(TLD² × D) where D is the number of features

For large feature sets, consider computing interactions only for instances of interest.

## Algorithm Details

TreeSHAP uses a path-based algorithm that:

1. Traverses each tree from root to leaf
2. Tracks which features are "on path" vs "off path"
3. Computes contributions using combinatorial weights
4. Accumulates SHAP values across all trees

### Complexity

- **Time**: O(T × L × D²) where T=trees, L=leaves per tree, D=depth
- **Space**: O(D) for the path tracking

### Handling Missing Values

TreeSHAP respects the model's missing value handling. If a feature is NaN:

```go
instance := []float64{0.5, math.NaN(), 0.8}  // Feature 1 is missing
explanation, _ := exp.Explain(ctx, instance)
// Uses the tree's default direction for missing values
```

## Performance Tips

1. **Use batch processing** for multiple instances
2. **Set appropriate workers** based on CPU cores
3. **Reuse the explainer** - don't recreate for each instance

```go
// Good: Create once, use many times
exp, _ := tree.New(ensemble, explainer.WithNumWorkers(4))
for _, instance := range instances {
    explanation, _ := exp.Explain(ctx, instance)
    // process...
}

// Bad: Recreating explainer each time
for _, instance := range instances {
    exp, _ := tree.New(ensemble)  // Wasteful!
    explanation, _ := exp.Explain(ctx, instance)
}
```

## Troubleshooting

### "model has no trees"

The JSON file doesn't contain tree data. Ensure you're using the correct export format:

- XGBoost: `model.save_model("model.json")`
- LightGBM: `json.dump(model.dump_model(), f)`

### Feature count mismatch

```
instance has 10 features, expected 15
```

Your input doesn't match the model's expected features. Check `ensemble.NumFeatures`.

### Local accuracy violation

TreeSHAP should always satisfy local accuracy. If you see violations, it might indicate:

- Floating point precision issues (use tolerance like `1e-9`)
- Bug in model loading (please report!)

## Next Steps

- [XGBoost Integration](../models/xgboost.md) - Export models from Python
- [LightGBM Integration](../models/lightgbm.md) - Export models from Python
- [Visualization](../visualization/charts.md) - Create charts from explanations
- [Benchmarks](../benchmarks.md) - Performance data
