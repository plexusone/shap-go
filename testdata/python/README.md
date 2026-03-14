# Python SHAP Test Case Generator

This directory contains Python scripts for generating TreeSHAP test cases using the official [SHAP library](https://github.com/shap/shap).

## Purpose

The Python SHAP library serves as the **source of truth** for TreeSHAP calculations. These scripts generate test cases with known-correct SHAP values that verify our Go implementation remains in sync with the canonical Python implementation.

## Setup

### Linux / CI (Ubuntu)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### macOS

On macOS, SHAP's `numba` dependency requires `llvmlite`, which may fail to build. Options:

1. **Use conda** (recommended):
   ```bash
   conda create -n shap-test python=3.11 numpy scikit-learn shap -c conda-forge
   conda activate shap-test
   ```

2. **Install llvmlite via Homebrew first**:
   ```bash
   brew install llvm
   export LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config
   pip install llvmlite numba
   pip install -r requirements.txt
   ```

3. **Rely on CI**: The GitHub Actions workflow runs on Ubuntu where pre-built wheels work correctly.

## Generate Test Cases

```bash
# Generate JSON test cases
python generate_test_cases.py > ../treeshap_test_cases.json
```

Or use the Makefile from the project root:

```bash
make generate-testdata
```

## Test Case Format

The generated JSON has the following structure:

```json
{
  "version": "1.0",
  "description": "TreeSHAP test cases generated with Python SHAP library",
  "shap_version": "0.45.1",
  "test_suites": [
    {
      "name": "suite_name",
      "description": "...",
      "tree": {
        "n_features": 2,
        "nodes": [
          {"feature": 0, "threshold": 0.5, "yes": 1, "no": 2, "cover": 100},
          {"is_leaf": true, "value": 1.0, "cover": 50},
          {"is_leaf": true, "value": 2.0, "cover": 50}
        ]
      },
      "base_value": 1.5,
      "cases": [
        {
          "name": "case_name",
          "instance": [0.3, 0.7],
          "prediction": 2.0,
          "shap_values": [-0.5, 0.5]
        }
      ]
    }
  ]
}
```

## Verifying Go Implementation

The Go test suite automatically reads `../treeshap_test_cases.json` and verifies:

1. **Local accuracy**: `sum(SHAP values) == prediction - base_value`
2. **Value correctness**: SHAP values match Python SHAP within tolerance (1e-6)

Run Go tests:

```bash
go test -v ./explainer/tree/... -run TestAgainstPythonSHAP
```

## CI Workflow

The GitHub Actions workflow:

1. Sets up Python environment with pinned dependencies
2. Generates fresh test cases using Python SHAP
3. Runs Go tests against the generated test cases
4. Fails if Go and Python SHAP values diverge

This ensures the Go implementation stays in sync with Python SHAP across updates.

## Updating Dependencies

When updating SHAP or other dependencies:

1. Update versions in `requirements.txt`
2. Regenerate test cases: `make generate-testdata`
3. Run tests: `make test`
4. Commit both `requirements.txt` and `treeshap_test_cases.json`
