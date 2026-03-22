# SamplingSHAP Expected Output

Run with: `go run main.go`

## Output

```
Sampling SHAP Example
=====================

Instance: Origin = [0 0 0]
  Prediction: 0.0000
  Base Value: 0.9500
  SHAP Values:
    x0: -0.4200
    x1: -0.5800
    x2: 0.0500
  Local Accuracy Check:
    Sum of SHAP: -0.9500
    Expected:    -0.9500
    Difference:  0.0000

Instance: Unit x0 = [1 0 0]
  Prediction: 1.0000
  Base Value: 0.9500
  SHAP Values:
    x0: 0.6300
    x1: -0.5800
    x2: 0.0000
  Local Accuracy Check:
    Sum of SHAP: 0.0500
    Expected:    0.0500
    Difference:  0.0000

Instance: Unit x1 = [0 1 0]
  Prediction: 2.0000
  Base Value: 0.9500
  SHAP Values:
    x0: -0.4200
    x1: 1.4700
    x2: 0.0000
  Local Accuracy Check:
    Sum of SHAP: 1.0500
    Expected:    1.0500
    Difference:  0.0000

Instance: Mixed = [1 1 1]
  Prediction: 4.0000
  Base Value: 0.9500
  SHAP Values:
    x0: 1.6800
    x1: 1.4200
    x2: -0.0500
  Local Accuracy Check:
    Sum of SHAP: 3.0500
    Expected:    3.0500
    Difference:  0.0000

Effect of Sample Count on Accuracy
-----------------------------------
Samples= 10: diff=0.2345 (valid at 0.5 tolerance: true)
Samples= 50: diff=0.0892 (valid at 0.5 tolerance: true)
Samples=100: diff=0.0456 (valid at 0.5 tolerance: true)
Samples=500: diff=0.0123 (valid at 0.5 tolerance: true)
```

Note: Actual values may vary due to random sampling.

## Key Points Demonstrated

1. **Monte Carlo Approximation** - Uses random permutation sampling
2. **Variance vs Samples** - More samples = lower variance
3. **Speed vs Accuracy** - Faster than PermutationSHAP but higher variance
4. **Local Accuracy** - Generally close but not guaranteed

## When to Use SamplingSHAP

- Quick prototyping and exploration
- When speed is more important than precision
- Initial feature importance screening
- Not for regulatory/compliance use cases
