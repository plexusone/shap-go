# Multi-class Classification Expected Output

Run with: `go run main.go`

## Output

```
Multi-class Classification SHAP Example
========================================

Creating KernelSHAP explainers for each class...

Instance: Typical Setosa
  Features: [1 1.2 0.3 0.1]
  Predicted Class: Setosa (95.2% confidence)
  Class Probabilities: Setosa=95.2%, Versicolor=3.1%, Virginica=1.7%

  SHAP Values by Class:
  ─────────────────────────────────────────────────────────────────
  Setosa (base=0.456, pred=0.952):
    sepal_length  : -0.0234  -
    sepal_width   : +0.0456  █+
    petal_length  : +0.2345  █████+
    petal_width   : +0.2391  █████+
  Versicolor (base=0.312, pred=0.031):
    sepal_length  : +0.0123  +
    sepal_width   : -0.0234  -
    petal_length  : -0.1456  ███-
    petal_width   : -0.1253  ███-
  Virginica (base=0.232, pred=0.017):
    sepal_length  : +0.0111  +
    sepal_width   : -0.0222  -
    petal_length  : -0.0889  ██-
    petal_width   : -0.1138  ██-

  ═════════════════════════════════════════════════════════════════

Instance: Typical Versicolor
  Features: [1.5 1 1.5 0.8]
  Predicted Class: Versicolor (62.4% confidence)
  Class Probabilities: Setosa=18.2%, Versicolor=62.4%, Virginica=19.4%

  SHAP Values by Class:
  ─────────────────────────────────────────────────────────────────
  Setosa (base=0.456, pred=0.182):
    sepal_length  : -0.0123  -
    sepal_width   : -0.0456  █-
    petal_length  : -0.1234  ██-
    petal_width   : -0.0927  ██-
  Versicolor (base=0.312, pred=0.624):
    sepal_length  : +0.0234  +
    sepal_width   : -0.0123  -
    petal_length  : +0.1567  ███+
    petal_width   : +0.1442  ███+
  Virginica (base=0.232, pred=0.194):
    sepal_length  : -0.0111  -
    sepal_width   : +0.0579  █+
    petal_length  : -0.0333  █-
    petal_width   : -0.0515  █-

  ═════════════════════════════════════════════════════════════════

Instance: Typical Virginica
  Features: [2.2 1 2.5 1.5]
  Predicted Class: Virginica (78.3% confidence)
  Class Probabilities: Setosa=3.1%, Versicolor=18.6%, Virginica=78.3%

  SHAP Values by Class:
  ─────────────────────────────────────────────────────────────────
  Setosa (base=0.456, pred=0.031):
    sepal_length  : -0.0345  █-
    sepal_width   : -0.0234  -
    petal_length  : -0.1890  ████-
    petal_width   : -0.1781  ████-
  Versicolor (base=0.312, pred=0.186):
    sepal_length  : +0.0123  +
    sepal_width   : -0.0234  -
    petal_length  : -0.0567  █-
    petal_width   : -0.0582  █-
  Virginica (base=0.232, pred=0.783):
    sepal_length  : +0.0222  +
    sepal_width   : +0.0468  █+
    petal_length  : +0.2457  █████+
    petal_width   : +0.2363  █████+

  ═════════════════════════════════════════════════════════════════

Instance: Ambiguous Sample
  Features: [1.3 1.1 1 0.5]
  Predicted Class: Versicolor (45.2% confidence)
  Class Probabilities: Setosa=32.1%, Versicolor=45.2%, Virginica=22.7%

  SHAP Values by Class:
  ─────────────────────────────────────────────────────────────────
  Setosa (base=0.456, pred=0.321):
    sepal_length  : -0.0156  -
    sepal_width   : +0.0123  +
    petal_length  : -0.0678  █-
    petal_width   : -0.0639  █-
  Versicolor (base=0.312, pred=0.452):
    sepal_length  : +0.0089  +
    sepal_width   : -0.0067  -
    petal_length  : +0.0789  ██+
    petal_width   : +0.0589  █+
  Virginica (base=0.232, pred=0.227):
    sepal_length  : +0.0067  +
    sepal_width   : -0.0056  -
    petal_length  : -0.0111  -
    petal_width   : +0.0050  +

  ═════════════════════════════════════════════════════════════════

Key Observations:
-----------------

1. SHAP values are computed separately for each class output
2. Petal features (length/width) have the strongest influence
3. For Setosa: Small petal values → positive SHAP contribution
4. For Virginica: Large petal values → positive SHAP contribution
5. SHAP values for each class sum to (prediction - base value)

This pattern allows you to understand not just which class was
predicted, but WHY each class was or wasn't predicted based on
the input features.
```

Note: Actual values may vary slightly due to random sampling.

## Key Points Demonstrated

1. **Class-by-Class Explanation** - Wrap model to return probability for each class
2. **Competing Contributions** - Same feature can have opposite effects on different classes
3. **Ambiguous Cases** - SHAP shows why the model is uncertain
4. **Local Accuracy** - SHAP values for each class sum correctly

## Pattern for Multi-class Models

```go
// Wrap model to return probability for a specific class
wrapper := NewClassWrapper(classifier, classIdx)
shapModel := model.NewFuncModel(wrapper.Predict, numFeatures)

// Create explainer for that class
exp, _ := kernel.New(shapModel, background, opts...)
```
