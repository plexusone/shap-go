# Python SHAP Validation

SHAP-Go validates its explainer implementations against the reference [Python SHAP library](https://github.com/shap/shap). This ensures correctness and compatibility with the widely-used Python implementation.

## Testing Methodology

Each explainer follows the same validation pattern:

1. **Generate test cases** - A Python script uses the official SHAP library to compute SHAP values for known models and instances
2. **Commit JSON test data** - The computed values are saved to `testdata/*.json` and committed to the repository
3. **Go tests validate** - Go tests load the JSON and verify the Go implementation produces matching results

This approach ensures:

- **Reproducibility** - Tests run without requiring Python dependencies in CI
- **Determinism** - Pre-computed values eliminate sampling variance in test comparisons
- **Transparency** - Test cases are human-readable JSON files

## Validated Explainers

| Explainer | Python Script | Test Data | Go Test |
|-----------|--------------|-----------|---------|
| TreeSHAP | `testdata/python/generate_treeshap_test_cases.py` | `testdata/treeshap_test_cases.json` | `explainer/tree/python_comparison_test.go` |
| KernelSHAP | `testdata/python/generate_kernelshap_test_cases.py` | `testdata/kernelshap_test_cases.json` | `explainer/kernel/python_comparison_test.go` |

## TreeSHAP Validation

TreeSHAP has exact analytical solutions for tree-based models, so the Go implementation should match Python SHAP exactly (within floating-point tolerance).

### Test Models

- **Simple decision tree** - Basic tree with a few splits
- **XGBoost model** - Gradient boosted trees

### Running the Tests

```bash
# Run TreeSHAP comparison tests
go test -v ./explainer/tree/... -run TestAgainstPythonSHAP

# Verify local accuracy property
go test -v ./explainer/tree/... -run TestLocalAccuracyProperty
```

### Regenerating Test Data

If you need to regenerate the TreeSHAP test cases:

```bash
cd testdata/python
pip install shap xgboost numpy
python generate_treeshap_test_cases.py > ../treeshap_test_cases.json
```

## KernelSHAP Validation

KernelSHAP is a sampling-based method, so SHAP values have inherent variance. The Go tests verify:

1. **Approximate equality** - Values within tolerance (0.3) of Python SHAP
2. **Local accuracy** - Sum of SHAP values equals prediction minus baseline (exact due to constrained regression)
3. **Correlation** - Pearson correlation > 0.9 with Python SHAP values

### Test Models

- **Linear model** - `f(x) = 2*x0 + 3*x1 + 1*x2`
- **Two-feature model** - `f(x) = x0 + x1`
- **Weighted model** - `f(x) = 0.5*x0 + 2.0*x1 + 0.3*x2 + 1.0*x3 + 5.0`
- **Quadratic model** - `f(x) = x0^2 + 2*x1 + x0*x2` (non-linear with interactions)

### Running the Tests

```bash
# Run KernelSHAP comparison tests
go test -v ./explainer/kernel/... -run TestAgainstPythonSHAP

# Verify local accuracy property (should be exact)
go test -v ./explainer/kernel/... -run TestLocalAccuracyProperty

# Check correlation with Python SHAP
go test -v ./explainer/kernel/... -run TestCorrelationWithPythonSHAP
```

### Regenerating Test Data

If you need to regenerate the KernelSHAP test cases:

```bash
cd testdata/python
pip install shap numpy
python generate_kernelshap_test_cases.py > ../kernelshap_test_cases.json
```

## Test Data Format

Both explainers use the same JSON schema for test data:

```json
{
  "version": "1.0",
  "description": "Test cases generated with Python SHAP library",
  "shap_version": "0.51.0",
  "generated_by": "testdata/python/generate_*_test_cases.py",
  "test_suites": [
    {
      "name": "model_name",
      "description": "Human-readable model description",
      "model": {
        "type": "linear|tree|quadratic",
        "weights": [1.0, 2.0],
        "bias": 0.0
      },
      "background": [[0.0, 0.0], [1.0, 1.0]],
      "base_value": 1.0,
      "cases": [
        {
          "name": "test_case_name",
          "instance": [0.5, 0.5],
          "prediction": 1.5,
          "shap_values": [0.25, 0.25]
        }
      ]
    }
  ]
}
```

## CI Integration

The GitHub Actions workflow validates test data and runs comparison tests:

```yaml
# .github/workflows/test.yml
jobs:
  treeshap:
    name: TreeSHAP Tests
    steps:
      - run: go test -v ./explainer/tree/... -run TestAgainstPythonSHAP
      - run: go test -v ./explainer/tree/... -run TestLocalAccuracyProperty

  kernelshap:
    name: KernelSHAP Tests
    steps:
      - run: go test -v ./explainer/kernel/... -run TestAgainstPythonSHAP
      - run: go test -v ./explainer/kernel/... -run TestLocalAccuracyProperty
      - run: go test -v ./explainer/kernel/... -run TestCorrelationWithPythonSHAP

  verify-testdata:
    name: Verify Test Data
    steps:
      - run: python3 -c "import json; json.load(open('testdata/treeshap_test_cases.json'))"
      - run: python3 -c "import json; json.load(open('testdata/kernelshap_test_cases.json'))"
```

## Adding Validation for New Explainers

To add Python validation for a new explainer:

1. **Create Python generator** - Add `testdata/python/generate_<explainer>_test_cases.py`
2. **Generate test data** - Run the script and save output to `testdata/<explainer>_test_cases.json`
3. **Create Go test** - Add `explainer/<name>/python_comparison_test.go` with:
   - `TestAgainstPythonSHAP` - Compare values within tolerance
   - `TestLocalAccuracyProperty` - Verify SHAP property holds
4. **Update CI** - Add the new explainer to `.github/workflows/test.yml`

## Tolerance Guidelines

| Property | Tolerance | Rationale |
|----------|-----------|-----------|
| Prediction | 1e-6 | Model evaluation should be exact |
| Base value | 0.1 | Computed from background samples |
| SHAP values (exact methods) | 1e-6 | TreeSHAP, LinearSHAP are analytical |
| SHAP values (sampling methods) | 0.3 | KernelSHAP, PermutationSHAP have variance |
| Local accuracy | 1e-9 | Constrained methods guarantee this |
| Correlation | > 0.9 | Overall agreement with Python SHAP |
