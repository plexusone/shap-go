# XGBoost Integration

This guide shows how to export XGBoost models from Python and use them with SHAP-Go.

## Export from Python

### Basic Export

```python
import xgboost as xgb

# Train your model
model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Export to JSON (required format for SHAP-Go)
model.save_model("model.json")
```

### From Existing Model

```python
import xgboost as xgb

# Load existing model
booster = xgb.Booster()
booster.load_model("model.bin")  # or .ubj, .xgb

# Export to JSON
booster.save_model("model.json")
```

### With Feature Names

```python
# Feature names are automatically included if set during training
model = xgb.XGBRegressor()
model.fit(X_train, y_train, feature_names=["age", "income", "score"])
model.save_model("model.json")  # Feature names included
```

## Load in Go

### Basic Loading

```go
package main

import (
    "log"
    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    // Load the model
    ensemble, err := tree.LoadXGBoostModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    // Inspect the model
    log.Printf("Trees: %d", ensemble.NumTrees)
    log.Printf("Features: %d", ensemble.NumFeatures)
    log.Printf("Feature names: %v", ensemble.FeatureNames)
}
```

### From io.Reader

```go
import (
    "os"
    "github.com/plexusone/shap-go/explainer/tree"
)

// From file
f, _ := os.Open("model.json")
defer f.Close()
ensemble, _ := tree.LoadXGBoostModelFromReader(f)

// From bytes
data := []byte(`{"learner": ...}`)
ensemble, _ := tree.ParseXGBoostJSON(data)
```

## Create Explainer

```go
import (
    "context"
    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/tree"
)

// Load model
ensemble, _ := tree.LoadXGBoostModel("model.json")

// Create explainer with options
exp, _ := tree.New(ensemble,
    explainer.WithNumWorkers(4),
    explainer.WithModelID("xgboost-loan-model-v2"),
)

// Explain predictions
ctx := context.Background()
instance := []float64{35, 75000, 720}
explanation, _ := exp.Explain(ctx, instance)
```

## Model Requirements

### Supported Features

| Feature | Supported |
|---------|-----------|
| Binary classification | ✅ |
| Multi-class classification | ✅ |
| Regression | ✅ |
| Ranking | ✅ |
| Missing values | ✅ |
| Categorical features | ⚠️ Partial |

### JSON Format Requirements

SHAP-Go expects the standard XGBoost JSON format:

```json
{
  "learner": {
    "gradient_booster": {
      "model": {
        "trees": [...]
      }
    },
    "learner_model_param": {
      "base_score": "0.5",
      "num_feature": "10"
    }
  }
}
```

!!! warning "Binary formats not supported"
    SHAP-Go requires JSON format. Binary formats (`.bin`, `.ubj`) must be converted first.

## Handling Different Objectives

### Regression

```python
# Python
model = xgb.XGBRegressor(objective='reg:squarederror')
```

SHAP values represent contribution to the raw prediction.

### Binary Classification

```python
# Python
model = xgb.XGBClassifier(objective='binary:logistic')
```

SHAP values represent contribution to the log-odds (before sigmoid).

### Multi-class Classification

```python
# Python
model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
```

!!! note
    For multi-class, SHAP-Go currently explains the sum of tree outputs. Per-class explanations are on the roadmap.

## Common Issues

### "failed to parse XGBoost JSON"

The file isn't valid XGBoost JSON. Check:

1. File was exported with `model.save_model("model.json")`
2. File extension is `.json`, not `.bin` or `.ubj`
3. File isn't corrupted

### "could not determine number of features"

The model JSON doesn't specify feature count. This can happen with very old XGBoost versions. Workaround:

```go
// Manually set feature count after loading
ensemble.NumFeatures = 10
```

### Feature Names Missing

If feature names weren't set during training:

```go
// Set them manually
exp, _ := tree.New(ensemble,
    explainer.WithFeatureNames([]string{"age", "income", "score", ...}),
)
```

## Complete Example

```python
# train_and_export.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
)
model.fit(X_train, y_train)

# Export
model.save_model("model.json")
print(f"Exported model with {len(model.get_booster().get_dump())} trees")
print(f"Features: {X.columns.tolist()}")
```

```go
// explain.go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "os"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    // Load model
    ensemble, err := tree.LoadXGBoostModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    // Create explainer
    exp, _ := tree.New(ensemble,
        explainer.WithNumWorkers(4),
    )

    // Example instance
    instance := []float64{35, 75000, 720, 2, 150000}

    // Explain
    ctx := context.Background()
    explanation, _ := exp.Explain(ctx, instance)

    // Output
    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    fmt.Printf("Base Value: %.4f\n\n", explanation.BaseValue)

    fmt.Println("Feature Contributions:")
    for _, f := range explanation.TopFeatures(10) {
        sign := "+"
        if f.SHAPValue < 0 {
            sign = "-"
        }
        fmt.Printf("  %s %s: %.4f\n", sign, f.Name, f.SHAPValue)
    }

    // Save explanation
    jsonData, _ := explanation.ToJSONPretty()
    os.WriteFile("explanation.json", jsonData, 0644)
}
```

## Next Steps

- [TreeSHAP Guide](../explainers/treeshap.md) - Deep dive into TreeSHAP
- [LightGBM Integration](lightgbm.md) - Alternative tree framework
- [Visualization](../visualization/charts.md) - Create charts
