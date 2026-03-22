# SHAP-Go Tasks

## Completed

### v0.3.0
- [x] TreeSHAP implementation (XGBoost, LightGBM JSON)
- [x] PermutationSHAP with antithetic sampling
- [x] SamplingSHAP (Monte Carlo)
- [x] LinearSHAP - exact closed-form SHAP for linear models
- [x] KernelSHAP - model-agnostic weighted linear regression (validated against Python SHAP)
- [x] ExactSHAP - brute-force exact Shapley values (O(n*2^n), validated against mathematical derivations)
- [x] ONNX Runtime integration
- [x] Render package (ChartIR visualizations)
- [x] MkDocs documentation site
- [x] CI/CD workflows (tests, lint, docs deployment)
- [x] Benchmarks
- [x] Fix masker panics (return errors instead)
- [x] Add ONNX tests
- [x] Python validation documentation - testing methodology docs
- [x] Explainers overview page - comparison and decision guide
- [x] Add internal/rand tests
- [x] Update README to mark KernelSHAP and ExactSHAP as complete

### v0.4.0
- [x] DeepSHAP - neural network explanations using DeepLIFT rescale rule
- [x] GradientSHAP - expected gradients using numerical differentiation
- [x] PartitionSHAP - hierarchical Owen values for feature groupings
- [x] AdditiveSHAP - exact SHAP for Generalized Additive Models (GAMs)
- [x] ONNX graph parsing and ActivationSession for intermediate layer capture
- [x] TreeSHAP interaction values - feature interaction computation
- [x] CatBoost model support
- [x] LightGBM text format parser - parse `.txt` model files
- [x] ONNX-ML TreeEnsemble parser - native ONNX tree model support
- [x] Batch explanation API - generic parallel wrapper for any explainer
- [x] Batched model predictions - `WithBatchedPredictions()` option for KernelSHAP, ExactSHAP, PartitionSHAP
- [x] Confidence intervals - uncertainty bounds for sampling methods
- [x] Improved model/onnx test coverage (15% -> 33%)
- [x] Background data selection guide - best practices for reference data
- [x] API reference improvements - comprehensive docs for all explainers
- [x] Examples (deepshap, gradientshap, multiclass, onnx_basic)
- [x] Expected output documentation for all examples
- [x] Udemy-style LMS presentation
- [x] Coverage badge in README (80% coverage)

## Pending

### Infrastructure
- [ ] Performance profiling and optimization
- [ ] Memory usage optimization for large models

## Coverage Status

| Package | Coverage |
|---------|----------|
| background | 90.2% |
| explainer | 58.6% |
| explainer/additive | 89.7% |
| explainer/deepshap | 60.7% |
| explainer/exact | 90.9% |
| explainer/gradient | 87.9% |
| explainer/kernel | 88.5% |
| explainer/linear | 96.6% |
| explainer/partition | 91.7% |
| explainer/permutation | 91.4% |
| explainer/sampling | 93.3% |
| explainer/tree | 84.4% |
| explanation | 77.8% |
| internal/rand | 100.0% |
| masker | 98.0% |
| model | 100.0% |
| model/onnx | 33.2% |
| render | 90.9% |
| **Total** | **80.6%** |
