# TreeSHAP Expected Output

Run with: `go run main.go`

## Output

```
TreeSHAP Example
================

Example 1: Simple Decision Tree
--------------------------------
Tree: x0 < 0.5 -> 1.0, else -> 2.0
Base Value (expected value): 1.5000

Instance: x0=0.3
  Prediction: 1.0000
  SHAP[x0]: -0.5000
  Local accuracy: true (diff=0.00e+00)
Instance: x0=0.7
  Prediction: 2.0000
  SHAP[x0]: 0.5000
  Local accuracy: true (diff=0.00e+00)

Example 2: Two-Feature Tree
----------------------------
Tree: x0<0.5 -> (x1<0.5 -> 1.0, else -> 2.0), else -> 4.0
Base Value: 2.7500

Instance: x0<0.5, x1<0.5 (x0=0.3, x1=0.3)
  Prediction: 1.0000
  SHAP[x0]: -1.3750
  SHAP[x1]: -0.3750
  Local accuracy: true
Instance: x0<0.5, x1>=0.5 (x0=0.3, x1=0.7)
  Prediction: 2.0000
  SHAP[x0]: -1.1250
  SHAP[x1]: 0.3750
  Local accuracy: true
Instance: x0>=0.5 (x0=0.7, x1=0.3)
  Prediction: 4.0000
  SHAP[x0]: 1.3750
  SHAP[x1]: -0.1250
  Local accuracy: true

Example 3: Batch Processing
----------------------------
Processed 6 instances

Instance 1: x0=0.1 -> pred=1.0, SHAP[x0]=-0.5000
Instance 2: x0=0.2 -> pred=1.0, SHAP[x0]=-0.5000
Instance 3: x0=0.3 -> pred=1.0, SHAP[x0]=-0.5000
Instance 4: x0=0.6 -> pred=2.0, SHAP[x0]=0.5000
Instance 5: x0=0.7 -> pred=2.0, SHAP[x0]=0.5000
Instance 6: x0=0.8 -> pred=2.0, SHAP[x0]=0.5000

Example 4: Loading Models from JSON
------------------------------------
// XGBoost: tree.LoadXGBoostModel("model.json")
// LightGBM: tree.LoadLightGBMModel("model.json")
// See README.md for Python export instructions
```

## Key Points Demonstrated

1. **Exact SHAP Values** - TreeSHAP computes mathematically exact Shapley values
2. **Local Accuracy** - SHAP values always sum to (prediction - base value)
3. **Base Value** - The expected prediction over the training data distribution
4. **Batch Processing** - Efficiently explain multiple instances

## Understanding the Results

### Example 1: Single Feature Tree

- Tree: `x0 < 0.5 → 1.0, else → 2.0`
- Base value = (1.0 + 2.0) / 2 = 1.5 (weighted by cover)
- For x0=0.3: prediction=1.0, SHAP = 1.0 - 1.5 = -0.5
- For x0=0.7: prediction=2.0, SHAP = 2.0 - 1.5 = +0.5

### Example 2: Two Feature Tree

Shows how SHAP values are distributed when multiple features contribute to the prediction. The algorithm correctly attributes contributions based on marginal contributions across all possible feature orderings.
