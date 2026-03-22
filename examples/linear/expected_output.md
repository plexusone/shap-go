# Linear Model Expected Output

Run with: `go run main.go`

## Output

```
SHAP Explanation for Linear Model
==================================
Model: y = 2*x0 + 3*x1 + 1*x2
Instance: [1 2 3]

Prediction: 11.0000
Base Value: 0.0000

SHAP Values:
  feature_a: value=1.00, SHAP=2.0000
  feature_b: value=2.00, SHAP=6.0000
  feature_c: value=3.00, SHAP=3.0000

Verification:
  Sum of SHAP values: 11.0000
  Expected (pred - base): 11.0000
  Difference: 0.0000000000
  Valid: true

Top Features by |SHAP|:
  1. feature_b: 6.0000
  2. feature_c: 3.0000
  3. feature_a: 2.0000

JSON Output:
{
  "model_id": "linear-model",
  "prediction": 11,
  "base_value": 0,
  "shap_values": {
    "feature_a": 2,
    "feature_b": 6,
    "feature_c": 3
  },
  "feature_names": [
    "feature_a",
    "feature_b",
    "feature_c"
  ],
  "feature_values": {
    "feature_a": 1,
    "feature_b": 2,
    "feature_c": 3
  },
  "timestamp": "2026-03-20T03:40:33.866168-07:00",
  "metadata": {
    "algorithm": "permutation",
    "num_samples": 100,
    "background_size": 1
  }
}
```

Note: Timestamp will vary between runs.

## Key Points Demonstrated

1. **Exact SHAP Values** - For linear models with zero background, SHAP = weight × value
2. **Perfect Local Accuracy** - SHAP values sum exactly to (prediction - base)
3. **TopFeatures()** - Rank features by absolute SHAP contribution
4. **JSON Export** - Serialize explanations for storage or API responses

## Understanding the Results

For model `y = 2×x0 + 3×x1 + 1×x2` with input `[1, 2, 3]`:

| Feature | Weight | Value | SHAP = Weight × Value |
|---------|--------|-------|------------------------|
| x0 (feature_a) | 2 | 1 | 2 × 1 = 2 |
| x1 (feature_b) | 3 | 2 | 3 × 2 = 6 |
| x2 (feature_c) | 1 | 3 | 1 × 3 = 3 |
| **Total** | | | **11** |

With a zero background (all zeros), the base value is 0, so SHAP values equal weighted feature values exactly.

## Why This Works

For linear models `f(x) = Σ wᵢxᵢ`:

- SHAP value for feature i = wᵢ × (xᵢ - E[xᵢ])
- With background = [0, 0, 0], E[xᵢ] = 0
- Therefore SHAP[i] = wᵢ × xᵢ

This example uses PermutationSHAP to demonstrate the algorithm works correctly, but for linear models you should use **LinearSHAP** for O(n) complexity instead.
