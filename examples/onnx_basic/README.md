# ONNX Basic Example

This example demonstrates using SHAP-Go with ONNX models for binary classification.

## Prerequisites

1. **ONNX Runtime** installed:
   ```bash
   # macOS
   brew install onnxruntime

   # Linux (Ubuntu/Debian)
   # Download from https://github.com/microsoft/onnxruntime/releases
   ```

2. **ONNX Model** (choose one option):

   **Option A: Generate locally (recommended)**
   ```bash
   pip install scikit-learn skl2onnx onnx
   python generate_model.py
   ```

   **Option B: Download from Hugging Face**
   ```bash
   ./download_model.sh
   ```
   Downloads from [Ritual-Net/iris-classification](https://huggingface.co/Ritual-Net/iris-classification).
   Note: May require adjusting input/output names in `main.go`.

## Usage

Run the example:
```bash
go run main.go
```

## What This Example Shows

- Loading an ONNX model with `onnx.NewSession`
- Using KernelSHAP for model-agnostic explanations
- Interpreting SHAP values for classification
- Verifying local accuracy

## Model Details

The example uses a logistic regression model trained on the Iris dataset for binary classification (versicolor vs. non-versicolor).

Features:

- sepal_length
- sepal_width
- petal_length
- petal_width

## Expected Output

See [expected_output.md](expected_output.md) for detailed sample output.

```
ONNX Model SHAP Explanations
============================

Instance: Setosa sample
Features: [5.1 3.5 1.4 0.2]
Prediction (P(class=1)): 0.0234
Base Value: 0.3333
SHAP Values:
  sepal_length  : value= 5.10, SHAP=-0.0521
  sepal_width   : value= 3.50, SHAP=+0.0123
  petal_length  : value= 1.40, SHAP=-0.1543
  petal_width   : value= 0.20, SHAP=-0.1158
Local accuracy: sum=-0.3099, expected=-0.3099, valid=true
```
