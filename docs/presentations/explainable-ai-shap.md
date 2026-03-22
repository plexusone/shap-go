---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #2563eb;
  }
  h2 {
    color: #1e40af;
  }
  code {
    background-color: #f1f5f9;
  }
  table {
    font-size: 0.8em;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .section-title {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: white;
  }
  .section-title h1 {
    color: white;
  }
---

# Explainable AI & SHAP
## Understanding Model Predictions with SHAP-Go

---

# Course Overview

## 7 Sections | Self-Paced Learning

| Section | Topic | Duration |
|---------|-------|----------|
| 1 | Introduction to Explainable AI | 10 min |
| 2 | SHAP Fundamentals | 15 min |
| 3 | Python SHAP vs SHAP-Go | 10 min |
| 4 | Explainer Deep Dive | 25 min |
| 5 | Choosing the Right Explainer | 10 min |
| 6 | Hands-On Examples | 30 min |
| 7 | Summary & Resources | 5 min |

---

<!-- _class: section-title -->

# Section 1
## Introduction to Explainable AI

---

# Section 1: Learning Objectives

By the end of this section, you will understand:

- What Explainable AI (XAI) is and why it matters
- The "black box" problem in machine learning
- Key use cases: compliance, trust, debugging, fairness
- Types of explainability (global vs local)

---

# What is Explainable AI?

**Explainable AI (XAI)** refers to methods and techniques that make AI/ML model decisions understandable to humans.

## The Goal
Transform "black box" models into transparent, interpretable systems where we can answer:

- **What** features influenced this prediction?
- **How much** did each feature contribute?
- **Why** did the model make this specific decision?

---

# The Black Box Problem

## Traditional ML Workflow
```
Input Data → [Black Box Model] → Prediction
                    ↓
              "Trust me" 🤷
```

## With Explainability
```
Input Data → [Model] → Prediction
                ↓
         Feature Contributions
         "Age: +15%, Income: -8%, ..."
```

---

# Why is XAI Important?

## 1. Regulatory Compliance
- **GDPR** (EU): Right to explanation for automated decisions
- **ECOA** (US): Fair lending requires explainable credit decisions
- **Healthcare**: FDA requires interpretability for clinical AI

## 2. Trust & Adoption
- Stakeholders need to understand model behavior
- Users are more likely to trust explainable systems
- Easier to identify and fix model errors

---

# Why is XAI Important? (continued)

## 3. Debugging & Improvement
- Identify which features drive predictions
- Detect data leakage or spurious correlations
- Validate model logic matches domain knowledge

## 4. Fairness & Bias Detection
- Understand if protected attributes influence decisions
- Ensure equitable treatment across groups
- Document model behavior for audits

---

# Types of Explainability

| Type | Description | Example |
|------|-------------|---------|
| **Global** | Overall model behavior | Feature importance across all predictions |
| **Local** | Single prediction explanation | Why did THIS customer get denied? |
| **Model-specific** | Exploits model structure | TreeSHAP for decision trees |
| **Model-agnostic** | Works with any model | LIME, KernelSHAP |

**SHAP provides both global AND local explanations!**

---

# Section 1: Key Takeaways

✅ XAI makes ML decisions understandable to humans

✅ Critical for compliance, trust, debugging, and fairness

✅ Two types: Global (overall) and Local (per-prediction)

✅ SHAP is a leading XAI method with strong theoretical foundation

**Next: Section 2 - SHAP Fundamentals**

---

<!-- _class: section-title -->

# Section 2
## SHAP Fundamentals

---

# Section 2: Learning Objectives

By the end of this section, you will understand:

- What SHAP values are and where they come from
- The game theory foundation (Shapley values)
- How to interpret SHAP values
- The mathematical properties that make SHAP trustworthy

---

# What is SHAP?

**SHAP** = **SH**apley **A**dditive ex**P**lanations

A unified approach to explain individual predictions based on **game theory**.

## Key Idea
Treat each feature as a "player" in a cooperative game, where the "payout" is the prediction. SHAP fairly distributes credit among features.

> "How much did each feature contribute to moving the prediction from the baseline to the actual output?"

---

# Shapley Values: Game Theory Foundation

From cooperative game theory (Lloyd Shapley, 1953 - Nobel Prize 2012):

## The Problem
How to fairly distribute the total gain among players who cooperated?

## Shapley's Solution
Average a player's marginal contribution across ALL possible orderings of players.

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [f(S \cup \{i\}) - f(S)]$$

---

# SHAP Values Explained Simply

## Example: Loan Approval Model

| Feature | Value | SHAP Value |
|---------|-------|------------|
| Income | $80,000 | +0.15 |
| Credit Score | 720 | +0.25 |
| Debt Ratio | 0.4 | -0.10 |
| Employment | 5 years | +0.08 |

**Base value**: 0.50 (average approval rate)
**Prediction**: 0.88 (88% approval probability)

**Sum of SHAP**: 0.15 + 0.25 - 0.10 + 0.08 = **0.38**
**Verification**: 0.50 + 0.38 = **0.88** ✓

---

# Reading SHAP Values

## Positive SHAP = Pushes prediction UP
- Income = $80,000 → SHAP = **+0.15**
- Higher than average income increases approval probability

## Negative SHAP = Pushes prediction DOWN
- Debt Ratio = 0.4 → SHAP = **-0.10**
- Higher than average debt decreases approval probability

## Magnitude = Importance
- Credit Score has the largest |SHAP| = **0.25**
- Most influential feature for this prediction

---

# SHAP Properties (Guarantees)

SHAP values satisfy important mathematical properties:

| Property | Meaning |
|----------|---------|
| **Local Accuracy** | SHAP values sum to prediction - baseline |
| **Missingness** | Missing features contribute zero |
| **Consistency** | If a feature's contribution increases, its SHAP value increases |
| **Efficiency** | All credit is distributed (no leftover) |

These properties make SHAP theoretically grounded and trustworthy.

---

# Why SHAP is Important

## Compared to Other Methods

| Method | Local | Global | Consistent | Model-agnostic |
|--------|-------|--------|------------|----------------|
| SHAP | ✅ | ✅ | ✅ | ✅ |
| LIME | ✅ | ❌ | ❌ | ✅ |
| Feature Importance | ❌ | ✅ | ❌ | ❌ |
| Partial Dependence | ❌ | ✅ | ❌ | ✅ |

**SHAP is the only method with strong theoretical guarantees AND practical applicability.**

---

# Section 2: Key Takeaways

✅ SHAP = Shapley Additive exPlanations (from game theory)

✅ Each feature gets a "fair share" of credit for the prediction

✅ **Local accuracy**: SHAP values always sum to (prediction - baseline)

✅ Positive SHAP = pushes up, Negative SHAP = pushes down

✅ Magnitude indicates importance

**Next: Section 3 - Python SHAP vs SHAP-Go**

---

<!-- _class: section-title -->

# Section 3
## Python SHAP vs SHAP-Go

---

# Section 3: Learning Objectives

By the end of this section, you will understand:

- The Python SHAP library and its strengths
- Why Go is advantageous for production ML
- What SHAP-Go offers
- When to use each library

---

# Python SHAP Library

The original SHAP implementation by Scott Lundberg (Microsoft Research).

## Features
- Comprehensive explainer suite
- Rich visualizations (waterfall, beeswarm, dependence plots)
- Deep integration with scikit-learn, XGBoost, TensorFlow
- Active community and extensive documentation

## Limitations
- Python-only (GIL, deployment challenges)
- Memory-intensive for large datasets
- Not ideal for production microservices

---

# Why SHAP-Go?

## Go Advantages for Production ML

| Aspect | Python | Go |
|--------|--------|-----|
| **Deployment** | Complex (venv, dependencies) | Single binary |
| **Concurrency** | GIL limitations | Native goroutines |
| **Memory** | Higher overhead | Efficient |
| **Startup** | Slow (interpreter) | Instant |
| **Type Safety** | Runtime errors | Compile-time |

**SHAP-Go**: Production-ready SHAP for Go microservices and APIs.

---

# SHAP-Go Features

## Complete Explainer Suite
- 10 explainer algorithms (matching Python SHAP)
- Model-agnostic and model-specific options

## Production Ready
- Thread-safe concurrent explanations
- Context support for timeouts/cancellation
- JSON-serializable for APIs

## Integrations
- ONNX Runtime for model inference
- XGBoost, LightGBM, CatBoost model loading
- ChartIR visualization output

---

# When to Use Which?

| Scenario | Recommendation |
|----------|----------------|
| Research & exploration | Python SHAP |
| Jupyter notebooks | Python SHAP |
| Production API | **SHAP-Go** |
| Microservices | **SHAP-Go** |
| Batch processing at scale | **SHAP-Go** |
| Rich interactive visualizations | Python SHAP |
| CI/CD pipelines | **SHAP-Go** |

---

# Section 3: Key Takeaways

✅ Python SHAP: Great for research, notebooks, visualizations

✅ SHAP-Go: Ideal for production, APIs, microservices

✅ Both implement the same algorithms with same accuracy

✅ SHAP-Go: Single binary, fast startup, native concurrency

**Next: Section 4 - Explainer Deep Dive**

---

<!-- _class: section-title -->

# Section 4
## Explainer Deep Dive

---

# Section 4: Learning Objectives

By the end of this section, you will understand:

- All 10 SHAP-Go explainer algorithms
- When to use each explainer
- Time complexity and accuracy trade-offs
- Code patterns for each explainer

---

# Explainer Overview

| Explainer | Model Type | Accuracy | Speed |
|-----------|------------|----------|-------|
| **TreeSHAP** | Tree ensembles | Exact | ⚡⚡⚡⚡⚡ |
| **LinearSHAP** | Linear models | Exact | ⚡⚡⚡⚡⚡ |
| **AdditiveSHAP** | GAMs | Exact | ⚡⚡⚡⚡⚡ |
| **ExactSHAP** | Any (≤15 features) | Exact | ⚡ |
| **KernelSHAP** | Any | Approximate | ⚡⚡⚡ |
| **PermutationSHAP** | Any | Exact* | ⚡⚡ |
| **SamplingSHAP** | Any | Approximate | ⚡⚡⚡⚡ |
| **DeepSHAP** | Neural networks | Approximate | ⚡⚡⚡⚡ |
| **GradientSHAP** | Differentiable | Approximate | ⚡⚡⚡⚡ |
| **PartitionSHAP** | Grouped features | Approximate | ⚡⚡⚡⚡ |

---

# TreeSHAP

## Best For: XGBoost, LightGBM, CatBoost

**Algorithm**: Exploits tree structure for O(TLD²) exact computation.

```go
ensemble, _ := tree.LoadXGBoostModel("model.json")
exp, _ := tree.New(ensemble)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | **Exact** |
| Complexity | O(TLD²) - Trees × Leaves × Depth² |
| Background Data | Not required |
| Interactions | ✅ Supported |

---

# TreeSHAP - How It Works

## Key Insight
Instead of enumerating all 2ⁿ coalitions, TreeSHAP tracks feature contributions through tree paths.

```
        [Age < 30?]
        /         \
   [Income>50k?]   [Score>700?]
    /     \         /      \
  Approve  Deny   Approve  Deny
```

**Path tracking**: As we traverse, accumulate each feature's contribution to the final leaf value.

**Supported Models**: XGBoost JSON, LightGBM JSON/text, CatBoost JSON, ONNX-ML TreeEnsemble

---

# LinearSHAP

## Best For: Linear Regression, Logistic Regression

**Algorithm**: Closed-form solution using model coefficients.

```go
exp, _ := linear.New(weights, bias, background)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | **Exact** |
| Complexity | O(n) - linear in features |
| Background Data | Required (for feature means) |

## Formula
$$\phi_i = w_i \cdot (x_i - \mathbb{E}[X_i])$$

---

# AdditiveSHAP

## Best For: Generalized Additive Models (GAMs)

**Algorithm**: Exact computation for models with no interactions.

```go
exp, _ := additive.New(model, background)
result, _ := exp.Explain(ctx, instance)
```

For models: $f(x) = f_1(x_1) + f_2(x_2) + ... + f_n(x_n)$

| Property | Value |
|----------|-------|
| Accuracy | **Exact** |
| Complexity | O(n × b) |
| Use Case | Spline models, pygam, interpret-ml |

---

# KernelSHAP

## Best For: Any Model (Model-Agnostic Baseline)

**Algorithm**: Weighted linear regression with SHAP kernel weights.

```go
exp, _ := kernel.New(model, background,
    explainer.WithNumSamples(100),
)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | Approximate (converges to exact) |
| Complexity | O(samples × background) |
| Local Accuracy | Enforced via constraint |

---

# KernelSHAP - How It Works

## The SHAP Kernel

1. Sample random feature coalitions (subsets)
2. Weight each sample using Shapley kernel:
   $$w(|S|) = \frac{n-1}{|S| \cdot (n-|S|)}$$
3. Fit weighted linear regression
4. Coefficients = SHAP values

**Key Insight**: Samples near empty and full coalitions get highest weight (most informative).

---

# ExactSHAP

## Best For: Ground Truth / Validation (≤15 features)

**Algorithm**: Brute-force enumeration of all 2ⁿ coalitions.

```go
exp, _ := exact.New(model, background)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | **Exact** (mathematically perfect) |
| Complexity | O(n × 2ⁿ × b) |
| Max Features | 15-20 (exponential growth) |

**Use Case**: Validating other methods, research, small feature sets.

---

# PermutationSHAP

## Best For: Guaranteed Local Accuracy

**Algorithm**: Average marginal contributions over random permutations.

```go
exp, _ := permutation.New(model, background,
    explainer.WithNumSamples(100),
)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | Exact (with sufficient samples) |
| Local Accuracy | **Guaranteed** (antithetic sampling) |
| Complexity | O(samples × n × b) |

---

# PermutationSHAP - How It Works

## Antithetic Sampling for Variance Reduction

For each permutation π:
1. Start with all features masked (background)
2. Add features one by one in permutation order
3. Record marginal contribution of each feature

**Antithetic**: For permutation [A, B, C], also use [C, B, A]
→ Reduces variance, guarantees sum = prediction - baseline

---

# SamplingSHAP

## Best For: Quick Estimates

**Algorithm**: Simple Monte Carlo sampling of coalitions.

```go
exp, _ := sampling.New(model, background,
    explainer.WithNumSamples(50),
)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | Approximate |
| Complexity | O(samples × n × b) |
| Speed | Fastest model-agnostic |

**Trade-off**: Speed vs accuracy. Good for exploration, not compliance.

---

# DeepSHAP

## Best For: Neural Networks (ONNX)

**Algorithm**: DeepLIFT attribution rules with Shapley averaging.

```go
session, _ := onnx.NewActivationSession(config)
exp, _ := deepshap.New(session, graphInfo, background)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Supported Layers | Dense, ReLU, Sigmoid, Tanh, Softmax |
| Complexity | O(layers × neurons × b) |
| Accuracy | Approximate (DeepLIFT-based) |

---

# DeepSHAP - How It Works

## DeepLIFT Rescale Rule

For each neuron, compute contribution relative to reference:

$$\text{contribution}_i = \frac{x_i - x_i^{ref}}{y - y^{ref}} \cdot \text{incoming\_attribution}$$

**Backward Propagation**: Start from output, propagate attributions back to inputs layer by layer.

**Averaging**: Average over multiple background samples for SHAP values.

---

# GradientSHAP

## Best For: Differentiable Models

**Algorithm**: Expected gradients using numerical differentiation.

```go
exp, _ := gradient.New(model, background,
    gradient.WithEpsilon(1e-7),
    explainer.WithNumSamples(200),
)
result, _ := exp.Explain(ctx, instance)
```

| Property | Value |
|----------|-------|
| Accuracy | Approximate |
| Confidence Intervals | ✅ Supported |
| Complexity | O(samples × n × 2) |

---

# GradientSHAP - How It Works

## Expected Gradients

1. Sample interpolation points between background and instance
2. Compute numerical gradient at each point
3. Average gradients weighted by input difference

$$\phi_i \approx \mathbb{E}_{x' \sim \text{background}} [(x_i - x'_i) \cdot \nabla_i f(\alpha x + (1-\alpha) x')]$$

**Options**: Local smoothing (SmoothGrad) for noise robustness.

---

# PartitionSHAP

## Best For: Hierarchical Feature Groupings

**Algorithm**: Owen values respecting feature hierarchy.

```go
hierarchy := &partition.Node{
    Name: "root",
    Children: []*partition.Node{
        {Name: "demographics", Children: []*partition.Node{
            {Name: "age", FeatureIdx: 0},
            {Name: "gender", FeatureIdx: 1},
        }},
        {Name: "financials", Children: []*partition.Node{
            {Name: "income", FeatureIdx: 2},
        }},
    },
}
exp, _ := partition.New(model, background, hierarchy)
```

---

# PartitionSHAP - Use Cases

## When Features Have Natural Groupings

| Domain | Groups |
|--------|--------|
| Healthcare | Vitals, Lab Results, Demographics |
| Finance | Income, Assets, Credit History |
| E-commerce | User Profile, Session, Product |
| NLP | Word Groups, Sentence Parts |

**Benefit**: Respects domain structure, handles correlated features better.

---

# Section 4: Key Takeaways

✅ **TreeSHAP**: Use for XGBoost/LightGBM (exact, fast)

✅ **LinearSHAP**: Use for linear models (exact, O(n))

✅ **KernelSHAP**: Model-agnostic baseline

✅ **DeepSHAP**: Neural networks via ONNX

✅ **PermutationSHAP**: When you need guaranteed local accuracy

✅ **ExactSHAP**: Ground truth for small feature sets

**Next: Section 5 - Choosing the Right Explainer**

---

<!-- _class: section-title -->

# Section 5
## Choosing the Right Explainer

---

# Section 5: Learning Objectives

By the end of this section, you will be able to:

- Select the optimal explainer for any model type
- Understand accuracy vs speed trade-offs
- Match explainers to use cases (compliance, exploration, production)

---

# Decision Flowchart

```
Is it a tree model (XGBoost/LightGBM/CatBoost)?
  └─ YES → TreeSHAP ⭐

Is it a linear model?
  └─ YES → LinearSHAP ⭐

Is it an additive model (GAM)?
  └─ YES → AdditiveSHAP ⭐

Is it a neural network?
  └─ YES → DeepSHAP or GradientSHAP

Do you have ≤15 features and need exact values?
  └─ YES → ExactSHAP

Do you need guaranteed local accuracy?
  └─ YES → PermutationSHAP

Otherwise → KernelSHAP (model-agnostic baseline)
```

---

# Explainer Selection Matrix

| Use Case | Recommended | Why |
|----------|-------------|-----|
| XGBoost model | TreeSHAP | Exact, O(TLD²) |
| Logistic regression | LinearSHAP | Exact, O(n) |
| GAM/Spline model | AdditiveSHAP | Exact, O(n×b) |
| Deep learning | DeepSHAP | Efficient for networks |
| Regulatory/Audit | PermutationSHAP | Guaranteed accuracy |
| Quick exploration | SamplingSHAP | Fast approximate |
| General black-box | KernelSHAP | Principled baseline |
| Grouped features | PartitionSHAP | Respects structure |

---

# By Use Case

## Regulatory Compliance / Audit
- **PermutationSHAP** or **ExactSHAP** (guaranteed accuracy)
- Document all parameters and background data

## Real-time API
- **TreeSHAP** for tree models (microseconds)
- **LinearSHAP** for linear models (nanoseconds)

## Batch Processing
- Any explainer with `WithNumWorkers(n)` for parallelism

## Exploration / Prototyping
- **SamplingSHAP** or **KernelSHAP** (fast iteration)

---

# Section 5: Key Takeaways

✅ Always prefer model-specific explainers when available

✅ TreeSHAP + LinearSHAP cover most production models

✅ KernelSHAP is the universal fallback

✅ PermutationSHAP for regulatory/audit requirements

✅ Consider speed vs accuracy based on your use case

**Next: Section 6 - Hands-On Examples**

---

<!-- _class: section-title -->

# Section 6
## Hands-On Examples

---

# Section 6: Learning Objectives

This section covers practical, runnable examples:

1. **Linear Model Basics** - Understanding SHAP fundamentals
2. **TreeSHAP** - Working with tree ensembles
3. **KernelSHAP** - Model-agnostic explanations
4. **Batch Processing** - Efficient parallel explanations
5. **Multi-class Classification** - Per-class explanations
6. **Visualization** - ChartIR output generation
7. **Markdown Reports** - Documentation for compliance

---

# Example 1: Linear Model Basics

## The Simplest Case

```go
// Model: y = 2*x0 + 3*x1 + 1*x2
predict := func(ctx context.Context, input []float64) (float64, error) {
    return 2*input[0] + 3*input[1] + 1*input[2], nil
}

m := model.NewFuncModel(predict, 3)
```

With background = [0, 0, 0], SHAP values equal weighted feature values:
- Input: [1, 2, 3]
- SHAP: [2, 6, 3] (weights × values)
- Sum: 2 + 6 + 3 = 11 ✓

---

# Example 1: Expected Output

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
  Valid: true
```

---

# Example 2: TreeSHAP

## Working with Tree Ensembles

```go
// Create a simple tree ensemble
ensemble := &tree.TreeEnsemble{
    NumTrees:    1,
    NumFeatures: 2,
    Nodes: []tree.Node{
        // Root: x0 < 0.5?
        {Feature: 0, Threshold: 0.5, Yes: 1, No: 2},
        // Left leaf: 1.0
        {IsLeaf: true, Prediction: 1.0},
        // Right leaf: 2.0
        {IsLeaf: true, Prediction: 2.0},
    },
}

exp, _ := tree.New(ensemble)
```

---

# Example 2: Expected Output

```
TreeSHAP Example
================

Tree: x0 < 0.5 -> 1.0, else -> 2.0
Base Value: 1.5000

Instance: x0=0.3
  Prediction: 1.0000
  SHAP[x0]: -0.5000
  Local accuracy: true

Instance: x0=0.7
  Prediction: 2.0000
  SHAP[x0]: 0.5000
  Local accuracy: true
```

**Key Insight**: Base value (1.5) is weighted average of leaf values.

---

# Example 3: KernelSHAP

## Model-Agnostic Explanations

```go
// Non-linear model: f(x) = x0² + 2*x1 + x0*x2
predict := func(ctx context.Context, input []float64) (float64, error) {
    return input[0]*input[0] + 2*input[1] + input[0]*input[2], nil
}

background := [][]float64{
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}, {0.5, 0.5, 0.5},
}

exp, _ := kernel.New(m, background,
    explainer.WithNumSamples(200),
)
```

---

# Example 3: Expected Output

```
KernelSHAP Example
==================

Base Value: 1.4167

Instance: Mixed = [1 1 1]
  Prediction: 4.0000
  SHAP Values:
    x0: +1.0208
    x1: +1.1667
    x2: +0.3958
  Local Accuracy: true

Note: SHAP values always sum to (prediction - baseline)
      1.0208 + 1.1667 + 0.3958 = 2.5833
      4.0000 - 1.4167 = 2.5833 ✓
```

---

# Example 4: Batch Processing

## Efficient Parallel Explanations

```go
// Create explainer with parallel workers
exp, _ := tree.New(ensemble,
    explainer.WithNumWorkers(4),  // 4 parallel workers
)

// Generate 100 instances
instances := generateTestInstances(100)

// Batch explain
explanations, _ := exp.ExplainBatch(ctx, instances)

// Compute global feature importance
expSet := render.NewExplanationSet(explanations)
meanAbs := expSet.MeanAbsoluteSHAP()
```

---

# Example 4: Expected Output

```
Batch Processing Example
========================

Model: 3 trees, 3 features
Explaining 100 instances...

Performance Comparison
----------------------
Sequential (1 worker):  125µs
Parallel (4 workers):   180µs

Global Feature Importance
-------------------------
  income: 0.2345
  age: 0.1567
  credit_score: 0.0892

Local Accuracy Verification
---------------------------
Valid explanations: 100/100 (100.0%)
```

---

# Example 5: Multi-class Classification

## Per-Class Explanations

```go
// Wrap model to return probability for each class
for classIdx := range classNames {
    wrapper := NewClassWrapper(classifier, classIdx)
    shapModel := model.NewFuncModel(wrapper.Predict, numFeatures)

    exp, _ := kernel.New(shapModel, background,
        explainer.WithNumSamples(200),
    )

    explanation, _ := exp.Explain(ctx, instance)
    // SHAP values explain this class's probability
}
```

---

# Example 5: Expected Output

```
Instance: Typical Setosa
  Predicted Class: Setosa (91.4% confidence)

  SHAP Values by Class:
  ─────────────────────────────────────────────
  Setosa (pred=0.914):
    petal_length  : +0.3871  ███████+
    petal_width   : +0.2261  ████+

  Versicolor (pred=0.061):
    petal_length  : -0.0462  -
    petal_width   : -0.0419  -

  Virginica (pred=0.025):
    petal_length  : -0.3409  -██████
    petal_width   : -0.1843  -███
```

---

# Example 6: Visualization

## ChartIR Output Generation

```go
renderer := render.NewRenderer()

// 1. Waterfall Plot (single prediction)
waterfallChart := renderer.WaterfallChartIR(explanation, "Title")

// 2. Feature Importance (global)
expSet := render.NewExplanationSet(explanations)
importanceChart := renderer.FeatureImportanceChartIR(expSet, "Title", 10)

// 3. Summary Plot (distribution)
summaryChart := renderer.SummaryChartIR(expSet, "Title")

// 4. Dependence Plot
dependenceChart := renderer.DependenceChartIR(expSet, "income", "Title")
```

---

# Example 6: Visualization Output

```
1. Waterfall Plot
-----------------
Title: Loan Approval Explanation
Datasets: 1 (waterfall: 4 columns, 5 rows)
Marks: 1 (positive: bar)

5. Force Plot Data
------------------
Base Value: 0.6560
Prediction: 0.9400
Features (sorted by contribution):
  + income: 0.1830 (value=75000.00)
  + credit_score: 0.0932 (value=720.00)
  + age: 0.0174 (value=35.00)
```

ChartIR can be converted to ECharts, Chart.js, D3.js, or Vega-Lite.

---

# Example 7: Markdown Reports

## Documentation for Compliance

```go
// Output to file
out, _ := os.Create("report.md")

// YAML frontmatter for Pandoc
fmt.Fprintln(out, "---")
fmt.Fprintln(out, "title: \"SHAP Explanation Report\"")
fmt.Fprintln(out, "---")

// Feature contributions table
fmt.Fprintln(out, "| Feature | Value | SHAP |")
for _, name := range exp.FeatureNames {
    fmt.Fprintf(out, "| %s | %.2f | %+.4f |\n",
        name, exp.FeatureValues[name], exp.Values[name])
}
```

---

# Example 7: Report Output

```markdown
# SHAP Explanation Report

## Feature Contributions

| Feature | Value | SHAP | Direction |
|---------|-------|------|-----------|
| income | 75000.00 | +0.1234 | (+) increases |
| credit_score | 750.00 | +0.0567 | (+) increases |
| debt_ratio | 0.15 | -0.0345 | (-) decreases |

## Local Accuracy Verification

| # | Sum(SHAP) | Pred - Base | Status |
|---|-----------|-------------|--------|
| 1 | 0.1456 | 0.1456 | PASS |
```

Convert with: `pandoc report.md -o report.pdf`

---

# Running the Examples

## Prerequisites
```bash
# Most examples work with just Go
go run examples/treeshap/main.go
go run examples/linearshap/main.go
go run examples/kernelshap/main.go

# ONNX examples require ONNX Runtime
brew install onnxruntime  # macOS
go run examples/onnx_basic/main.go
```

## Expected Output
Each example includes `expected_output.md` for verification.

---

# Section 6: Key Takeaways

✅ Start with simple linear model to understand SHAP basics

✅ Use `ExplainBatch()` with `WithNumWorkers(n)` for efficiency

✅ Multi-class: Create separate explainer for each class

✅ ChartIR enables visualization in any frontend framework

✅ Markdown reports provide audit trails for compliance

**Next: Section 7 - Summary & Resources**

---

<!-- _class: section-title -->

# Section 7
## Summary & Resources

---

# Course Summary

## What We Learned

1. **XAI** makes ML decisions understandable and trustworthy
2. **SHAP** provides theoretically grounded feature attributions
3. **SHAP-Go** brings SHAP to Go for production deployments
4. **10 Explainers** cover every model type
5. **Local accuracy** ensures explanation consistency
6. **Hands-on examples** demonstrate real-world usage

---

# Quick Reference: Explainer Selection

```
Tree model?         → TreeSHAP (exact, fast)
Linear model?       → LinearSHAP (exact, O(n))
Neural network?     → DeepSHAP
GAM?                → AdditiveSHAP
Need exact values?  → ExactSHAP (if ≤15 features)
Need guaranteed accuracy? → PermutationSHAP
General black-box?  → KernelSHAP
Grouped features?   → PartitionSHAP
Quick exploration?  → SamplingSHAP
```

---

# SHAP-Go: 10 Explainers, Production Ready

| Exact | Approximate |
|-------|-------------|
| TreeSHAP | KernelSHAP |
| LinearSHAP | SamplingSHAP |
| AdditiveSHAP | DeepSHAP |
| ExactSHAP | GradientSHAP |
| PermutationSHAP* | PartitionSHAP |

*With sufficient samples

---

# Resources

## SHAP-Go
- **GitHub**: `github.com/plexusone/shap-go`
- **Documentation**: MkDocs site with guides for each explainer
- **Examples**: 12 runnable examples with expected output

## References
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Shapley (1953): "A Value for n-Person Games"
- Python SHAP: `github.com/shap/shap`

---

# Next Steps

1. **Clone SHAP-Go**: `git clone github.com/plexusone/shap-go`

2. **Run examples**:
   ```bash
   cd examples/linearshap && go run main.go
   cd examples/treeshap && go run main.go
   ```

3. **Read the docs**: Explainer guides and API reference

4. **Integrate into your project**: Add SHAP explanations to your ML pipeline

---

# Questions?

## Contact & Resources

- **SHAP-Go Repository**: github.com/plexusone/shap-go
- **Python SHAP**: github.com/shap/shap
- **Original Paper**: NeurIPS 2017

---

# Appendix A: Complexity Reference

| Explainer | Time Complexity | Space |
|-----------|-----------------|-------|
| TreeSHAP | O(TLD²) | O(D) |
| LinearSHAP | O(n) | O(n) |
| AdditiveSHAP | O(n×b) | O(n+b) |
| KernelSHAP | O(s×b) | O(s×n) |
| ExactSHAP | O(n×2ⁿ×b) | O(2ⁿ) |
| PermutationSHAP | O(s×n×b) | O(n) |
| SamplingSHAP | O(s×n×b) | O(n) |
| DeepSHAP | O(L×N×b) | O(N) |
| GradientSHAP | O(s×n×2) | O(n) |
| PartitionSHAP | O(s×g!×b) | O(g) |

n=features, b=background, s=samples, T=trees, L=leaves, D=depth, g=groups

---

# Appendix B: Configuration Options

## Common Options (all explainers)

```go
explainer.WithNumSamples(100)      // Sampling-based only
explainer.WithSeed(42)             // Reproducibility
explainer.WithNumWorkers(4)        // Parallel processing
explainer.WithFeatureNames(names)  // Human-readable names
explainer.WithModelID("my-model")  // For tracking
explainer.WithConfidenceLevel(0.95) // CI (sampling methods)
```

## Explainer-Specific Options

```go
gradient.WithEpsilon(1e-7)         // Numerical gradient step
gradient.WithLocalSmoothing(50)    // SmoothGrad samples
```

---

# Appendix C: Example Files

| Example | Location | Description |
|---------|----------|-------------|
| linear | `examples/linear/` | Basic PermutationSHAP |
| linearshap | `examples/linearshap/` | LinearSHAP for house prices |
| treeshap | `examples/treeshap/` | TreeSHAP fundamentals |
| kernelshap | `examples/kernelshap/` | Model-agnostic baseline |
| batch | `examples/batch/` | Parallel processing |
| multiclass | `examples/multiclass/` | Per-class explanations |
| visualization | `examples/visualization/` | ChartIR output |
| markdown_report | `examples/markdown_report/` | Pandoc reports |
| gradientshap | `examples/gradientshap/` | Expected gradients |
| sampling | `examples/sampling/` | Monte Carlo sampling |
| onnx_basic | `examples/onnx_basic/` | ONNX integration |
| deepshap | `examples/deepshap/` | Neural network SHAP |
