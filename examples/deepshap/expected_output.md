# DeepSHAP Expected Output

This example demonstrates DeepSHAP for neural network explanations.

## Prerequisites

1. ONNX Runtime installed
2. ONNX neural network model (generate locally)

### Generate Model

```bash
pip install scikit-learn torch onnx
python generate_model.py
```

## Run

```bash
go run main.go
```

## Expected Output

```
ONNX Graph Structure:
  Inputs: [input]
  Outputs: [output]
  Nodes: 7
    /fc1/Gemm (Gemm -> dense)
    /relu1/Relu (Relu -> relu)
    /fc2/Gemm (Gemm -> dense)
    /relu2/Relu (Relu -> relu)
    /fc3/Gemm (Gemm -> dense)
    /sigmoid/Sigmoid (Sigmoid -> sigmoid)
    output_squeeze (Squeeze -> identity)

DeepSHAP Neural Network Explanations
=====================================

Instance: Setosa sample
Features: [5.1 3.5 1.4 0.2]
Prediction: 0.0856
Base Value: 0.3245
SHAP Values (DeepLIFT-based):
  sepal_length : value= 5.10, SHAP=-0.0234
  sepal_width  : value= 3.50, SHAP=+0.0156
  petal_length : value= 1.40, SHAP=-0.1234
  petal_width  : value= 0.20, SHAP=-0.1077
Top Contributing Features:
  1. petal_length: -0.1234 ↓
  2. petal_width: -0.1077 ↓
  3. sepal_length: -0.0234 ↓
  4. sepal_width: +0.0156 ↑
Local accuracy: sum=-0.2389, expected=-0.2389, diff=0.0000

Instance: Versicolor sample
Features: [6 2.7 4.5 1.5]
Prediction: 0.7823
Base Value: 0.3245
SHAP Values (DeepLIFT-based):
  sepal_length : value= 6.00, SHAP=+0.0345
  sepal_width  : value= 2.70, SHAP=-0.0123
  petal_length : value= 4.50, SHAP=+0.2234
  petal_width  : value= 1.50, SHAP=+0.2122
Top Contributing Features:
  1. petal_length: +0.2234 ↑
  2. petal_width: +0.2122 ↑
  3. sepal_length: +0.0345 ↑
  4. sepal_width: -0.0123 ↓
Local accuracy: sum=+0.4578, expected=+0.4578, diff=0.0000

Instance: Virginica sample
Features: [7.2 3.2 6 1.8]
Prediction: 0.5234
Base Value: 0.3245
SHAP Values (DeepLIFT-based):
  sepal_length : value= 7.20, SHAP=+0.0567
  sepal_width  : value= 3.20, SHAP=+0.0089
  petal_length : value= 6.00, SHAP=+0.0789
  petal_width  : value= 1.80, SHAP=+0.0544
Top Contributing Features:
  1. petal_length: +0.0789 ↑
  2. sepal_length: +0.0567 ↑
  3. petal_width: +0.0544 ↑
  4. sepal_width: +0.0089 ↑
Local accuracy: sum=+0.1989, expected=+0.1989, diff=0.0000

Activation Analysis
===================
Input: [6 2.7 4.5 1.5]
Output: 0.7823
Layer Activations:
  /fc1/Gemm_output_0: [0.234 0.567 0.123 0.890 0.456 0.012 0.789 0.345]
  /relu1/Relu_output_0: [0.234 0.567 0.123 0.890 0.456 0.000 0.789 0.345]
  /fc2/Gemm_output_0: [0.345 0.678 0.012 0.901 0.567 0.123 0.234 0.456]
  /relu2/Relu_output_0: [0.345 0.678 0.012 0.901 0.567 0.123 0.234 0.456]
  /fc3/Gemm_output_0: [1.234]
  /sigmoid/Sigmoid_output_0: [0.7823]
```

Note: Actual values depend on the trained model and may vary.

## Key Points Demonstrated

1. **Graph Parsing** - ONNX model structure is automatically parsed
2. **DeepLIFT Attribution** - Uses rescale rule for backpropagation
3. **Layer Activations** - Intermediate layer values are captured
4. **Local Accuracy** - SHAP values sum to (prediction - base value)
5. **Feature Importance** - `TopFeatures()` method ranks contributions

## DeepSHAP Algorithm

DeepSHAP combines DeepLIFT with Shapley values:

1. **Forward Pass**: Compute activations for input and reference (background)
2. **Backward Pass**: Propagate attributions using DeepLIFT rescale rule
3. **Average**: Aggregate attributions over multiple background samples

## Supported Layer Types

| ONNX Op | DeepSHAP Rule |
|---------|---------------|
| Gemm/MatMul | Linear (sum of weighted contributions) |
| Relu | Rescale (activation ratio) |
| Sigmoid | Rescale with derivative fallback |
| Tanh | Rescale with derivative fallback |
| Softmax | Logit attribution |
| Add | Proportional split |
| Identity/Flatten | Pass-through |
