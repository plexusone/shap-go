# SHAP-Go Examples

This directory contains runnable examples demonstrating SHAP-Go's explainers.

## Quick Start

```bash
# Run any example
cd examples/<example-name>
go run main.go
```

## Examples by Explainer Type

### Exact Explainers

| Example | Explainer | Model Type | Description |
|---------|-----------|------------|-------------|
| [treeshap](treeshap/) | TreeSHAP | Tree ensembles | Exact SHAP for XGBoost/LightGBM |
| [linearshap](linearshap/) | LinearSHAP | Linear models | Closed-form SHAP for linear regression |

### Approximate Explainers

| Example | Explainer | Model Type | Description |
|---------|-----------|------------|-------------|
| [kernelshap](kernelshap/) | KernelSHAP | Any (black-box) | Weighted regression approximation |
| [sampling](sampling/) | SamplingSHAP | Any (black-box) | Monte Carlo approximation |
| [gradientshap](gradientshap/) | GradientSHAP | Differentiable | Expected gradients method |

### Neural Network Explainers

| Example | Explainer | Model Type | Description |
|---------|-----------|------------|-------------|
| [deepshap](deepshap/) | DeepSHAP | Neural networks | DeepLIFT-based attribution |
| [onnx_basic](onnx_basic/) | KernelSHAP | ONNX models | Model-agnostic for ONNX |

### Utilities & Patterns

| Example | Description |
|---------|-------------|
| [linear](linear/) | Basic SHAP computation with FuncModel |
| [batch](batch/) | Parallel batch processing |
| [multiclass](multiclass/) | Multi-class classification |
| [visualization](visualization/) | ChartIR output generation |
| [markdown_report](markdown_report/) | Markdown report generation |

## External Models

Some examples can use pre-trained models from Hugging Face instead of generating locally.

### Available External Models

| Model | Source | Format | Use With |
|-------|--------|--------|----------|
| Iris Classifier | [Ritual-Net/iris-classification](https://huggingface.co/Ritual-Net/iris-classification) | ONNX | onnx_basic, deepshap |

### Downloading External Models

```bash
# Option 1: Using curl
curl -L -o iris.onnx "https://huggingface.co/Ritual-Net/iris-classification/resolve/main/iris.onnx"

# Option 2: Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download Ritual-Net/iris-classification iris.onnx --local-dir .
```

## Model Generation

Examples that require trained models include Python scripts to generate them:

```bash
cd examples/onnx_basic
pip install scikit-learn skl2onnx onnx
python generate_model.py
```

## Expected Output

Each example directory contains an `expected_output.md` file showing the expected output when running the example. Use this to:

- Verify your setup is working correctly
- Understand what the example demonstrates
- Compare results without running the code

## Requirements

### Go Examples (no external dependencies)

Most examples work with just Go:

- treeshap, linearshap, kernelshap, sampling, gradientshap
- linear, batch, multiclass, visualization, markdown_report

### ONNX Examples (require ONNX Runtime)

The following examples require ONNX Runtime:

- onnx_basic
- deepshap

Install ONNX Runtime:

```bash
# macOS
brew install onnxruntime

# Linux
# See https://onnxruntime.ai/docs/install/
```

## Running All Examples

```bash
# Run examples that don't require ONNX
for dir in treeshap linearshap kernelshap sampling gradientshap linear batch multiclass visualization markdown_report; do
    echo "=== $dir ==="
    (cd examples/$dir && go run main.go)
done
```

## Example Output Format

All examples follow a consistent pattern:

1. **Header** - Example name and description
2. **Model Info** - Model type, features, configuration
3. **SHAP Values** - Per-feature attribution values
4. **Local Accuracy** - Verification that SHAP values sum correctly

## Complete Example List

All 12 examples include `expected_output.md` documentation:

| Example | Has expected_output.md | Requires ONNX |
|---------|------------------------|---------------|
| batch | Yes | No |
| deepshap | Yes | Yes |
| gradientshap | Yes | No |
| kernelshap | Yes | No |
| linear | Yes | No |
| linearshap | Yes | No |
| markdown_report | Yes | No |
| multiclass | Yes | No |
| onnx_basic | Yes | Yes |
| sampling | Yes | No |
| treeshap | Yes | No |
| visualization | Yes | No |
