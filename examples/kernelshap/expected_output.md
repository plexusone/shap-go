# KernelSHAP Expected Output

Run with: `go run main.go`

## Output

```
KernelSHAP Example
==================

Base Value: 1.4167

Instance: Origin = [0 0 0]
  Prediction: 0.0000
  SHAP Values:
    x0: -0.4792
    x1: -0.8333
    x2: -0.1042
  Local Accuracy: true (diff=0.00e+00)

Instance: Unit x0 = [1 0 0]
  Prediction: 1.0000
  SHAP Values:
    x0: +0.7292
    x1: -0.8333
    x2: -0.3125
  Local Accuracy: true (diff=0.00e+00)

Instance: Unit x1 = [0 1 0]
  Prediction: 2.0000
  SHAP Values:
    x0: -0.4792
    x1: +1.1667
    x2: -0.1042
  Local Accuracy: true (diff=0.00e+00)

Instance: Mixed = [1 1 1]
  Prediction: 4.0000
  SHAP Values:
    x0: +1.0208
    x1: +1.1667
    x2: +0.3958
  Local Accuracy: true (diff=0.00e+00)

Instance: Large x0 = [2 0.5 0.5]
  Prediction: 6.0000
  SHAP Values:
    x0: +4.3333
    x1: +0.1667
    x2: +0.0833
  Local Accuracy: true (diff=0.00e+00)

Effect of Sample Count on SHAP Estimates
-----------------------------------------

Instance: [2.0, 1.0, 0.5]

Samples    SHAP(x0)     SHAP(x1)     SHAP(x2)     Sum
--------------------------------------------------------------
20         +4.3333      +1.1667      +0.0833      5.5833
50         +4.3333      +1.1667      +0.0833      5.5833
100        +4.3333      +1.1667      +0.0833      5.5833
200        +4.3333      +1.1667      +0.0833      5.5833
500        +4.3333      +1.1667      +0.0833      5.5833

Note: Local accuracy is always guaranteed regardless of sample count.
More samples improve the accuracy of individual SHAP values.
```

## Key Points Demonstrated

1. **Model-Agnostic** - Works with any model (uses function f(x) = x0² + 2x1 + x0×x2)
2. **Local Accuracy Guaranteed** - Constrained regression ensures SHAP values sum correctly
3. **Sample Count** - More samples improve individual SHAP value accuracy
4. **Non-Linear Models** - Can explain complex interactions (x0×x2 term)

## Understanding the Model

The example uses: `f(x) = x0² + 2×x1 + x0×x2`

- **x0** has quadratic effect (x0²) and interaction with x2
- **x1** has linear effect (coefficient = 2)
- **x2** only contributes through interaction with x0

## How KernelSHAP Works

1. **Sample coalitions** - Generate random subsets of features
2. **Compute kernel weights** - Weight coalitions by Shapley kernel
3. **Evaluate model** - For each coalition, mask absent features with background
4. **Solve regression** - Fit weighted linear model to get SHAP values
5. **Enforce constraints** - Ensure SHAP values sum to (prediction - baseline)
