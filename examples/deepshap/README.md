# DeepSHAP Example

This example demonstrates using DeepSHAP for neural network explanations with ONNX models.

## Prerequisites

1. **ONNX Runtime** installed:
   ```bash
   # macOS
   brew install onnxruntime

   # Linux (Ubuntu/Debian)
   # Download from https://github.com/microsoft/onnxruntime/releases
   ```

2. **Python dependencies** for model generation:
   ```bash
   pip install torch scikit-learn onnx
   ```

## Usage

1. Generate the ONNX neural network model:
   ```bash
   python generate_model.py
   ```

2. Run the example:
   ```bash
   go run main.go
   ```

## What This Example Shows

- Parsing ONNX graph structure with `onnx.ParseGraph`
- Creating `ActivationSession` for intermediate layer capture
- Using DeepSHAP for neural network explanations
- Analyzing layer activations
- Interpreting DeepLIFT-based SHAP values

## Model Details

The example uses a simple MLP (Multi-Layer Perceptron) trained on the Iris dataset:

```
Input (4 features)
    ↓
Dense (8 neurons) + ReLU
    ↓
Dense (8 neurons) + ReLU
    ↓
Dense (1 neuron) + Sigmoid
    ↓
Output (probability)
```

## DeepSHAP Algorithm

DeepSHAP combines DeepLIFT with Shapley values:

1. **Forward pass**: Capture activations at each layer
2. **Reference activations**: Compute baseline activations from background data
3. **Backward propagation**: Apply DeepLIFT rescale rule to propagate attributions
4. **Average**: Average attributions over multiple background samples

The rescale rule for each neuron:
```
multiplier_in = multiplier_out × (activation - reference) / (output - output_ref)
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

DeepSHAP Neural Network Explanations
=====================================

Instance: Versicolor sample
Features: [6 2.7 4.5 1.5]
Prediction: 0.7234
Base Value: 0.3333
SHAP Values (DeepLIFT-based):
  sepal_length  : value= 6.00, SHAP=+0.0856
  sepal_width   : value= 2.70, SHAP=-0.0234
  petal_length  : value= 4.50, SHAP=+0.2145
  petal_width   : value= 1.50, SHAP=+0.1134
Top Contributing Features:
  1. petal_length: +0.2145 ↑
  2. petal_width: +0.1134 ↑
  3. sepal_length: +0.0856 ↑
  4. sepal_width: -0.0234 ↓
```

## Limitations

Current DeepSHAP implementation supports:

- Dense (Gemm, MatMul) layers
- ReLU, Sigmoid, Tanh activations
- Softmax output layers
- Sequential architectures

Not yet supported:

- Convolutional layers
- Recurrent layers
- Residual connections
- Attention mechanisms
