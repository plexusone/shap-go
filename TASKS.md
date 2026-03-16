# SHAP-Go Tasks

## Completed

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
- [x] Examples (linear, treeshap, sampling, batch, visualization, linearshap, kernelshap, markdown_report)
- [x] Fix masker panics (return errors instead)
- [x] Add ONNX tests
- [x] Python validation documentation - testing methodology docs
- [x] Explainers overview page - comparison and decision guide
- [x] Add internal/rand tests (v0.3.0)
- [x] Update README to mark KernelSHAP and ExactSHAP as complete (v0.3.0)

## In Progress

## Pending

### High Priority

- [ ] Improve model/onnx coverage (currently 8%)

### Feature Enhancements

- [ ] TreeSHAP interaction values - feature interaction computation
- [ ] LightGBM text format parser - parse `.txt` model files (currently JSON only)
- [ ] Confidence intervals - uncertainty bounds for sampling methods
- [ ] CatBoost model support
- [ ] Batch explanation API - explain multiple instances efficiently

### Documentation & Examples

- [ ] Multi-class classification example
- [ ] ONNX examples (binary, multi-class, regression)
- [ ] Background data selection guide - best practices for reference data
- [ ] API reference improvements

### Future Explainers

- [ ] DeepSHAP - neural network explanations (combines DeepLIFT with Shapley)
- [ ] GradientSHAP - expected gradients for neural networks
- [ ] PartitionSHAP - hierarchical clustering for correlated features

### Infrastructure

- [ ] ONNX-ML TreeEnsemble parser - native ONNX tree model support
- [ ] Coverage badge in README
- [ ] Performance profiling and optimization
- [ ] Memory usage optimization for large models

## Coverage Status

| Package | Coverage |
|---------|----------|
| background | 90.2% |
| explainer/exact | ~85% |
| explainer/kernel | 86.9% |
| explainer/linear | 96.6% |
| explainer/permutation | 89.3% |
| explainer/sampling | 92.3% |
| explainer/tree | 84.4% |
| explanation | 93.3% |
| internal/rand | ~80% |
| masker | 98.0% |
| model | 100.0% |
| model/onnx | 8.0% |
| render | 90.9% |
