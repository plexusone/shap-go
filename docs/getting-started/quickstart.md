# Quick Start

This guide will get you explaining model predictions in 5 minutes.

## Choose Your Path

=== "Tree Models (XGBoost/LightGBM)"

    If you have XGBoost or LightGBM models, use **TreeSHAP** for exact, fast explanations.

    ```go
    package main

    import (
        "context"
        "fmt"
        "log"

        "github.com/plexusone/shap-go/explainer/tree"
    )

    func main() {
        // Load your model (exported from Python)
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
        instance := []float64{0.5, 0.3, 0.8}
        explanation, err := exp.Explain(ctx, instance)
        if err != nil {
            log.Fatal(err)
        }

        // Print results
        fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
        fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)
        for _, feat := range explanation.TopFeatures(5) {
            fmt.Printf("  %s: %.4f\n", feat.Name, feat.SHAPValue)
        }
    }
    ```

=== "Any Model (Black-box)"

    For any model you can call, use **PermutationSHAP**.

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
        // Wrap your prediction function
        predict := func(ctx context.Context, input []float64) (float64, error) {
            // Your model prediction logic here
            return input[0]*2 + input[1]*3, nil
        }
        m := model.NewFuncModel(predict, 2)

        // Background data (representative samples)
        background := [][]float64{
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
        }

        // Create explainer
        exp, _ := permutation.New(m, background,
            explainer.WithNumSamples(100),
            explainer.WithFeatureNames([]string{"feature_a", "feature_b"}),
        )

        // Explain
        ctx := context.Background()
        explanation, _ := exp.Explain(ctx, []float64{1.0, 2.0})

        fmt.Printf("Prediction: %.2f\n", explanation.Prediction)
        for name, shap := range explanation.Values {
            fmt.Printf("SHAP(%s): %.4f\n", name, shap)
        }
    }
    ```

## Understanding the Output

Every SHAP explanation contains:

| Field | Description |
|-------|-------------|
| `Prediction` | The model's output for this instance |
| `BaseValue` | Expected prediction (average over background) |
| `Values` | Map of feature name → SHAP contribution |
| `FeatureNames` | Ordered list of feature names |
| `FeatureValues` | Map of feature name → input value |

### The Local Accuracy Property

SHAP values always satisfy:

```
sum(SHAP values) = Prediction - BaseValue
```

You can verify this:

```go
result := explanation.Verify(1e-6)
if !result.Valid {
    fmt.Printf("Warning: local accuracy violated, diff=%.2e\n", result.Difference)
}
```

## Batch Processing

Explain multiple instances efficiently:

```go
instances := [][]float64{
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9},
}

// Use parallel workers for speed
exp, _ := tree.New(ensemble, explainer.WithNumWorkers(4))
explanations, err := exp.ExplainBatch(ctx, instances)
```

## JSON Serialization

Export explanations for logging or APIs:

```go
// To JSON
jsonData, _ := explanation.ToJSON()

// Pretty-printed
jsonPretty, _ := explanation.ToJSONPretty()

// From JSON
loaded, _ := explanation.FromJSON(jsonData)
```

## Next Steps

- [TreeSHAP Guide](../explainers/treeshap.md) - Deep dive into tree explanations
- [PermutationSHAP Guide](../explainers/permutation.md) - Black-box explanations
- [Visualization](../visualization/charts.md) - Generate charts
- [XGBoost Integration](../models/xgboost.md) - Export models from Python
