# LightGBM Integration

This guide shows how to export LightGBM models from Python and use them with SHAP-Go.

## Export from Python

### Using dump_model()

LightGBM requires `dump_model()` for the full tree structure:

```python
import lightgbm as lgb
import json

# Train your model
model = lgb.LGBMRegressor(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Export using dump_model() - required for SHAP-Go
model_json = model.booster_.dump_model()
with open("model.json", "w") as f:
    json.dump(model_json, f)
```

### From Booster Directly

```python
import lightgbm as lgb
import json

# Load existing model
booster = lgb.Booster(model_file="model.txt")

# Export to JSON
model_json = booster.dump_model()
with open("model.json", "w") as f:
    json.dump(model_json, f)
```

!!! warning "Don't use save_model()"
    `save_model()` produces a text format that SHAP-Go doesn't support.
    Always use `dump_model()` with `json.dump()`.

### With Feature Names

```python
# Feature names are included automatically if set during training
train_data = lgb.Dataset(X_train, label=y_train, feature_name=["age", "income", "score"])
model = lgb.train(params, train_data)

model_json = model.dump_model()
# Feature names will be in model_json["feature_names"]
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
    ensemble, err := tree.LoadLightGBMModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    // Inspect the model
    log.Printf("Trees: %d", ensemble.NumTrees)
    log.Printf("Features: %d", ensemble.NumFeatures)
    log.Printf("Feature names: %v", ensemble.FeatureNames)
    log.Printf("Objective: %s", ensemble.Objective)
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
ensemble, _ := tree.LoadLightGBMModelFromReader(f)

// From bytes
data := []byte(`{"name": "tree", ...}`)
ensemble, _ := tree.ParseLightGBMJSON(data)
```

## Create Explainer

```go
import (
    "context"
    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/tree"
)

// Load model
ensemble, _ := tree.LoadLightGBMModel("model.json")

// Create explainer with options
exp, _ := tree.New(ensemble,
    explainer.WithNumWorkers(4),
    explainer.WithModelID("lightgbm-fraud-model-v1"),
)

// Explain predictions
ctx := context.Background()
instance := []float64{35, 75000, 720}
explanation, _ := exp.Explain(ctx, instance)
```

## LightGBM vs XGBoost

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| Export method | `dump_model()` + `json.dump()` | `save_model("model.json")` |
| Split condition | `<=` (less-than-or-equal) | `<` (less-than) |
| Default direction | `default_left` field | `default_left` field |
| Feature names | In `feature_names` array | In `learner.feature_names` |

SHAP-Go handles these differences automatically.

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

### JSON Format

SHAP-Go expects the `dump_model()` format:

```json
{
  "name": "tree",
  "version": "v3",
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "max_feature_idx": 9,
  "objective": "regression",
  "feature_names": ["f0", "f1", ...],
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 31,
      "tree_structure": {...}
    }
  ]
}
```

## Common Issues

### "model has no trees"

The JSON doesn't contain tree data. Check:

1. Used `dump_model()` not `save_model()`
2. Wrapped with `json.dump()`, not just `dump_model()` alone
3. Model actually has trees (not an empty model)

### "could not determine number of features"

```go
// Check what's in the model
log.Printf("max_feature_idx: %d", modelJson["max_feature_idx"])
log.Printf("feature_names: %v", modelJson["feature_names"])
```

### Split Direction Differences

LightGBM uses `<=` by default, XGBoost uses `<`. SHAP-Go handles this automatically, but be aware when comparing raw predictions.

## Complete Example

```python
# train_and_export.py
import lightgbm as lgb
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create dataset with feature names
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=X.columns.tolist()
)

# Train model
params = {
    "objective": "regression",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "n_estimators": 100,
}
model = lgb.train(params, train_data, num_boost_round=100)

# Export - IMPORTANT: use dump_model() with json.dump()
model_json = model.dump_model()
with open("model.json", "w") as f:
    json.dump(model_json, f, indent=2)

print(f"Exported model with {model_json['tree_info'].__len__()} trees")
print(f"Features: {model_json['feature_names']}")
```

```go
// explain.go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    // Load model
    ensemble, err := tree.LoadLightGBMModel("model.json")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Loaded LightGBM model: %d trees, %d features\n",
        ensemble.NumTrees, ensemble.NumFeatures)
    fmt.Printf("Objective: %s\n\n", ensemble.Objective)

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

    // Verify local accuracy
    result := explanation.Verify(1e-6)
    fmt.Printf("\nLocal accuracy: %v\n", result.Valid)
}
```

## Next Steps

- [TreeSHAP Guide](../explainers/treeshap.md) - Deep dive into TreeSHAP
- [XGBoost Integration](xgboost.md) - Alternative tree framework
- [Visualization](../visualization/charts.md) - Create charts
