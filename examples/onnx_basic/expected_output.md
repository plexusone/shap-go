# ONNX Basic Expected Output

This example demonstrates KernelSHAP with an ONNX model.

## Prerequisites

1. ONNX Runtime installed
2. ONNX model file (generate or download)

### Option 1: Generate Model Locally

```bash
pip install scikit-learn skl2onnx onnx
python generate_model.py
```

### Option 2: Download from Hugging Face

```bash
curl -L -o iris_model.onnx "https://huggingface.co/Ritual-Net/iris-classification/resolve/main/iris.onnx"
```

Note: The Hugging Face model has different input/output names. You may need to adjust the code.

## Run

```bash
go run main.go
```

## Expected Output

```
ONNX Model SHAP Explanations
============================

Instance: Setosa sample
Features: [5.1 3.5 1.4 0.2]
Prediction (P(class=1)): 0.0234
Base Value: 0.3312
SHAP Values:
  sepal_length : value= 5.10, SHAP=-0.0156
  sepal_width  : value= 3.50, SHAP=+0.0089
  petal_length : value= 1.40, SHAP=-0.1678
  petal_width  : value= 0.20, SHAP=-0.1333
Local accuracy: sum=-0.3078, expected=-0.3078, valid=true

Instance: Versicolor sample
Features: [6 2.7 4.5 1.5]
Prediction (P(class=1)): 0.8956
Base Value: 0.3312
SHAP Values:
  sepal_length : value= 6.00, SHAP=+0.0234
  sepal_width  : value= 2.70, SHAP=-0.0156
  petal_length : value= 4.50, SHAP=+0.2789
  petal_width  : value= 1.50, SHAP=+0.2777
Local accuracy: sum=+0.5644, expected=+0.5644, valid=true

Instance: Virginica sample
Features: [7.2 3.2 6 1.8]
Prediction (P(class=1)): 0.4523
Base Value: 0.3312
SHAP Values:
  sepal_length : value= 7.20, SHAP=+0.0456
  sepal_width  : value= 3.20, SHAP=+0.0023
  petal_length : value= 6.00, SHAP=-0.0512
  petal_width  : value= 1.80, SHAP=+0.1244
Local accuracy: sum=+0.1211, expected=+0.1211, valid=true
```

Note: Actual SHAP values may vary slightly due to random sampling and model training.

## Key Points Demonstrated

1. **ONNX Integration** - Load and explain any ONNX model
2. **Model-Agnostic** - KernelSHAP works without knowing model internals
3. **Probability Outputs** - Explains class probability predictions
4. **Local Accuracy** - SHAP values sum to (prediction - base value)

## Interpreting Results

For the **Setosa sample** (versicolor probability = 0.023):

- `petal_length` and `petal_width` strongly push prediction DOWN (negative SHAP)
- These small petal values are characteristic of Setosa, not Versicolor

For the **Versicolor sample** (versicolor probability = 0.896):

- `petal_length` and `petal_width` strongly push prediction UP (positive SHAP)
- These medium petal values are characteristic of Versicolor
