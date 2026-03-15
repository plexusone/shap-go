---
title: "SHAP Explanation Report"
subtitle: "Loan Approval Model Analysis"
author: "SHAP-Go"
date: "March 15, 2026"
geometry: margin=1in
colorlinks: true
---

# Executive Summary

This report provides SHAP (SHapley Additive exPlanations) analysis for a loan approval model.
SHAP values explain how each feature contributes to individual predictions, enabling transparent
and interpretable machine learning decisions.

# Individual Prediction Analysis

## Applicant Profile

| Feature | Value |
|---------|-------|
| income | 75000.00 |
| age | 35.00 |
| credit_score | 750.00 |
| debt_ratio | 0.15 |

## Prediction Breakdown

| Metric | Value |
|--------|-------|
| Base Value (Average) | 0.7815 |
| Final Prediction | 0.8872 |
| Net Effect | +0.1057 |

## Feature Contributions

Each feature's SHAP value shows its contribution to pushing the prediction above or below the baseline.

| Feature | Value | SHAP | Direction | Impact |
|---------|-------|------|-----------|--------|
| income | 75000.00 | +0.0600 | (+) increases | `##########` |
| debt_ratio | 0.15 | +0.0252 | (+) increases | `####` |
| credit_score | 750.00 | +0.0205 | (+) increases | `###` |
| age | 35.00 | +0.0000 | (+) increases | `` |

## Waterfall Visualization

```
Baseline: 0.7815
--------------------------------------------------
income               +0.0600
debt_ratio           +0.0252
credit_score         +0.0205
age                  +0.0000
--------------------------------------------------
Prediction: 0.8872
```

# Global Feature Importance

Aggregated across all predictions, feature importance is measured as the mean absolute SHAP value.

| Rank | Feature | Mean |SHAP| | Importance |
|------|---------|-------------|------------|
| 1 | income | 0.0640 | `####################` |
| 2 | debt_ratio | 0.0250 | `#######` |
| 3 | credit_score | 0.0203 | `######` |
| 4 | age | 0.0117 | `###` |

# Prediction Summary

| # | Prediction | Top Positive | Top Negative |
|---|------------|--------------|--------------|
| 1 | 0.8872 | income (+0.06) | none |
| 2 | 0.6263 |  | income (-0.08) |
| 3 | 0.7901 | age (+0.01) | debt_ratio (-0.00) |
| 4 | 0.9974 | income (+0.12) | none |
| 5 | 0.6714 |  | income (-0.06) |

# Local Accuracy Verification

SHAP values satisfy the local accuracy property: the sum of SHAP values equals the difference
between the prediction and baseline.

| # | Sum(SHAP) | Pred - Base | Difference | Status |
|---|-----------|-------------|------------|--------|
| 1 | 0.105671 | 0.105671 | 0.00e+00 | PASS |
| 2 | -0.155212 | -0.155212 | 0.00e+00 | PASS |
| 3 | 0.008524 | 0.008524 | 0.00e+00 | PASS |
| 4 | 0.215818 | 0.215818 | 0.00e+00 | PASS |
| 5 | -0.110124 | -0.110124 | 1.39e-17 | PASS |

# Methodology

- **Algorithm:** KernelSHAP (model-agnostic)
- **Samples:** 200 coalition samples per explanation
- **Background:** 5 reference instances
- **Library:** SHAP-Go (github.com/plexusone/shap-go)

---

*Report generated on 2026-03-15 13:53:32*
