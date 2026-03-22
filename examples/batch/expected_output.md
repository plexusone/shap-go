# Batch Processing Expected Output

Run with: `go run main.go`

## Output

```
Batch Processing Example with TreeSHAP
======================================

Model: 3 trees, 3 features
Base Value: 0.8800

Explaining 100 instances...

Performance Comparison
----------------------
Sequential (1 worker):  125.625µs
Parallel (4 workers):   179.916µs
Speedup:                0.70x

Sample Explanations (first 3)
-----------------------------
Instance 1: prediction=0.0500
  feature_0: value=30000.00, SHAP=-0.3000
  feature_1: value=20.00, SHAP=-0.2100
  feature_2: value=550.00, SHAP=-0.3200

Instance 2: prediction=0.0500
  feature_0: value=31000.00, SHAP=-0.3000
  feature_1: value=27.00, SHAP=-0.2100
  feature_2: value=561.00, SHAP=-0.3200

Instance 3: prediction=0.3500
  feature_0: value=32000.00, SHAP=-0.3000
  feature_1: value=34.00, SHAP=0.0900
  feature_2: value=572.00, SHAP=-0.3200

Global Feature Importance
-------------------------
  income: 0.0000
  age: 0.0000
  credit_score: 0.0000

Local Accuracy Verification
---------------------------
Valid explanations: 100/100 (100.0%)
Max difference: 1.11e-16
```

Note: Performance numbers vary between runs. On small workloads, parallelization overhead may exceed benefits.

## Key Points Demonstrated

1. **Batch Processing** - Explain multiple instances efficiently with `ExplainBatch()`
2. **Parallel Workers** - Use `WithNumWorkers(n)` for multi-core processing
3. **Global Feature Importance** - Aggregate SHAP values across all explanations
4. **100% Local Accuracy** - All explanations satisfy the local accuracy property

## Performance Notes

For this small example (100 instances, simple trees), parallelization overhead exceeds benefits. Parallelization shines when:

- Processing thousands of instances
- Model inference is expensive (ONNX, large trees)
- Computing SHAP for complex models

## Use Cases

- **Batch Inference** - Explain entire test datasets
- **Feature Selection** - Identify globally important features
- **Model Monitoring** - Track feature importance over time
- **Audit Trails** - Generate explanations for regulatory compliance
