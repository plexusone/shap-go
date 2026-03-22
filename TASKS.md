# SHAP-Go Tasks

## Completed

- [x] TreeSHAP implementation (XGBoost, LightGBM JSON)
- [x] PermutationSHAP with antithetic sampling
- [x] SamplingSHAP (Monte Carlo)
- [x] LinearSHAP - exact closed-form SHAP for linear models
- [x] KernelSHAP - model-agnostic weighted linear regression (validated against Python SHAP)
- [x] ExactSHAP - brute-force exact Shapley values (O(n*2^n), validated against mathematical derivations)
- [x] DeepSHAP - neural network explanations using DeepLIFT rescale rule (v0.4.0)
- [x] ONNX Runtime integration
- [x] ONNX graph parsing and ActivationSession for intermediate layer capture (v0.4.0)
- [x] Render package (ChartIR visualizations)
- [x] MkDocs documentation site
- [x] CI/CD workflows (tests, lint, docs deployment)
- [x] Benchmarks
- [x] Examples (linear, treeshap, sampling, batch, visualization, linearshap, kernelshap, markdown_report, multiclass, gradientshap)
- [x] Fix masker panics (return errors instead)
- [x] Add ONNX tests
- [x] Python validation documentation - testing methodology docs
- [x] Explainers overview page - comparison and decision guide
- [x] Add internal/rand tests (v0.3.0)
- [x] Update README to mark KernelSHAP and ExactSHAP as complete (v0.3.0)
- [x] ONNX examples (onnx_basic with KernelSHAP, deepshap with neural networks) (v0.4.0)
- [x] TreeSHAP interaction values - feature interaction computation (v0.4.0)
- [x] LightGBM text format parser - parse `.txt` model files (v0.4.0)
- [x] Additional ONNX types tests (v0.4.0)
- [x] Confidence intervals - uncertainty bounds for sampling methods (v0.5.0)
- [x] CatBoost model support (v0.5.0)
- [x] ONNX-ML TreeEnsemble parser - native ONNX tree model support (v0.5.0)
- [x] Improved model/onnx test coverage (15% -> 33%) (v0.5.0)
- [x] Batched model predictions - `WithBatchedPredictions()` option for KernelSHAP, ExactSHAP, PartitionSHAP (v0.5.0)
- [x] AdditiveSHAP - exact SHAP for Generalized Additive Models (GAMs) (v0.5.0)

## In Progress

## Pending

### High Priority (Next Up)

1. [x] Background data selection guide - best practices for reference data (v0.5.0)
2. [x] Improve model/onnx coverage (15% -> 33%, graph/types fully covered) (v0.5.0)
3. [x] PartitionSHAP - hierarchical Owen values for feature groupings (v0.5.0)

### Feature Enhancements

- [x] Batch explanation API (Option 1) - generic parallel wrapper for any explainer (v0.5.0)
- [x] Batch explanation API (Option 2) - optimized per-explainer with batched model predictions (v0.5.0)

### Documentation & Examples

- [x] Multi-class classification example (v0.5.0)
- [x] GradientSHAP example (v0.5.0)
- [x] Background data selection guide (v0.5.0)
- [x] API reference improvements - comprehensive docs for all explainers (v0.5.0)

### Future Explainers

- [x] GradientSHAP - expected gradients using numerical gradients (v0.5.0)
- [x] PartitionSHAP - hierarchical Owen values for feature groupings (v0.5.0)

### Infrastructure
- [x] Coverage badge in README (80% coverage, excluding examples)
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
