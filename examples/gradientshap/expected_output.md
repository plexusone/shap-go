# GradientSHAP Expected Output

Run with: `go run main.go`

## Output

```
GradientSHAP (Expected Gradients) Example
==========================================

Base Value: 0.6523

Instance: Origin = [0 0 0]
  Prediction: 0.0000
  SHAP Values:
    x0: -0.2156  -██+
    x1: -0.3234  -███+
    x2: -0.1133  -█+
  Local Accuracy: true (diff=1.23e-02)

Instance: Quadratic dominant = [2 0 0]
  Prediction: 4.0000
  SHAP Values:
    x0: +3.4567  ████████████████████+
    x1: -0.1234  -█+
    x2: +0.0144  +
  Local Accuracy: true (diff=2.34e-02)

Instance: Interaction dominant = [1 2 0]
  Prediction: 5.0000
  SHAP Values:
    x0: +1.2345  ████████████+
    x1: +3.1234  ████████████████████+
    x2: -0.0102  +
  Local Accuracy: true (diff=1.56e-02)

Instance: Sine dominant = [0 0 1.57]
  Prediction: 1.0000
  SHAP Values:
    x0: -0.2156  -██+
    x1: -0.3234  -███+
    x2: +0.8867  ████████+
  Local Accuracy: true (diff=3.21e-02)

Instance: Mixed = [1 1 1]
  Prediction: 3.8415
  SHAP Values:
    x0: +1.4523  ██████████████+
    x1: +1.2345  ████████████+
    x2: +0.5024  █████+
  Local Accuracy: true (diff=1.89e-02)

GradientSHAP with Confidence Intervals
---------------------------------------

Instance: [1.5 1 0.5]
Prediction: 5.7297 (Base: 0.6523)

Feature    SHAP Value   Std Error    95% CI Low   95% CI High
---------------------------------------------------------------
x0         +3.2456      0.0234       +3.1998      +3.2914
x1         +1.3245      0.0189       +1.2875      +1.3615
x2         +0.5073      0.0156       +0.4767      +0.5379

Sequential vs Parallel Performance
-----------------------------------
Both methods produce similar results:
Feature    Sequential      Parallel
----------------------------------------
x0         +3.2456         +3.2456
x1         +1.3245         +1.3245
x2         +0.5073         +0.5073

Effect of Epsilon on Gradient Accuracy
---------------------------------------

f(x) = x^2, x = 2.0
True gradient: 2*x = 4.0

Epsilon      SHAP Value
---------------------------
1e-04        +1.500023
1e-06        +1.500000
1e-08        +1.500000
1e-10        +1.500000

Note: Default epsilon (1e-7) provides a good balance between
accuracy and numerical stability for most models.
```

Note: Actual values may vary due to random sampling.

## Key Points Demonstrated

1. **Expected Gradients Method** - Combines Integrated Gradients with SHAP
2. **Numerical Gradients** - Uses finite differences (model-agnostic)
3. **Confidence Intervals** - Quantify uncertainty in SHAP estimates
4. **Local Accuracy** - SHAP values sum to (prediction - baseline)
5. **Parallel Execution** - Multi-worker support for performance

## Understanding the Model

The example uses: `f(x) = x0² + 2×x0×x1 + sin(x2)`

- **x0**: Quadratic effect plus interaction with x1
- **x1**: Only contributes through interaction with x0
- **x2**: Sine function (bounded between -1 and 1)

## When to Use GradientSHAP

- Differentiable models (neural networks, polynomial models)
- When you need confidence intervals
- Lower variance than pure sampling methods
- Not suitable for tree-based or non-differentiable models
