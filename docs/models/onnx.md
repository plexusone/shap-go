# ONNX Runtime Integration

SHAP-Go supports ONNX models through ONNX Runtime bindings, enabling explanations for models from any framework that exports to ONNX.

## Prerequisites

### Install ONNX Runtime

=== "macOS"

    ```bash
    brew install onnxruntime
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Download from GitHub releases
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
    tar -xzf onnxruntime-linux-x64-1.16.0.tgz

    # Set library path
    export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.16.0/lib:$LD_LIBRARY_PATH
    ```

=== "Windows"

    Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) and add the DLL directory to your PATH.

## Export Models to ONNX

### From scikit-learn

```python
import sklearn
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### From PyTorch

```python
import torch

# Your trained model
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, num_features)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['float_input'],
    output_names=['output'],
    dynamic_axes={'float_input': {0: 'batch_size'}}
)
```

### From TensorFlow/Keras

```python
import tf2onnx
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("model.h5")

# Convert to ONNX
spec = (tf.TensorSpec((None, num_features), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

## Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plexusone/shap-go/explainer"
    "github.com/plexusone/shap-go/explainer/permutation"
    "github.com/plexusone/shap-go/model/onnx"
)

func main() {
    // Initialize ONNX Runtime (once per application)
    err := onnx.InitializeRuntime("/usr/local/lib/libonnxruntime.dylib")
    if err != nil {
        log.Fatal(err)
    }
    defer onnx.DestroyRuntime()

    // Create ONNX session
    session, err := onnx.NewSession(onnx.Config{
        ModelPath:   "model.onnx",
        InputName:   "float_input",
        OutputName:  "output",
        NumFeatures: 10,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer session.Close()

    // Background data for SHAP
    background := [][]float64{
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
        {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
        // ... more representative samples
    }

    // Create explainer
    exp, err := permutation.New(session, background,
        explainer.WithNumSamples(100),
        explainer.WithFeatureNames([]string{"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"}),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Explain a prediction
    ctx := context.Background()
    instance := []float64{0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 0.5, 0.5}
    explanation, err := exp.Explain(ctx, instance)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
    for _, f := range explanation.TopFeatures(5) {
        fmt.Printf("  %s: %.4f\n", f.Name, f.SHAPValue)
    }
}
```

## Configuration

### Session Config

```go
session, _ := onnx.NewSession(onnx.Config{
    // Required: path to ONNX model file
    ModelPath: "model.onnx",

    // Required: input tensor name (from model)
    InputName: "float_input",

    // Required: output tensor name (from model)
    OutputName: "output",

    // Required: number of input features
    NumFeatures: 10,

    // Optional: output index for multi-output models
    OutputIndex: 0,
})
```

### Finding Input/Output Names

Use `netron` to visualize your ONNX model:

```bash
pip install netron
netron model.onnx
```

Or use Python:

```python
import onnx

model = onnx.load("model.onnx")
print("Inputs:", [i.name for i in model.graph.input])
print("Outputs:", [o.name for o in model.graph.output])
```

## Batch Predictions

The ONNX session supports efficient batch predictions:

```go
// Single prediction
pred, _ := session.Predict(ctx, instance)

// Batch prediction (more efficient)
instances := [][]float64{
    {0.1, 0.2, ...},
    {0.3, 0.4, ...},
}
predictions, _ := session.PredictBatch(ctx, instances)
```

## Performance Tips

### Reuse Session

Create the session once and reuse it:

```go
// Good: Create once
session, _ := onnx.NewSession(config)
defer session.Close()

for _, instance := range instances {
    pred, _ := session.Predict(ctx, instance)
}

// Bad: Creating session each time
for _, instance := range instances {
    session, _ := onnx.NewSession(config)  // Slow!
    pred, _ := session.Predict(ctx, instance)
    session.Close()
}
```

### Optimize Background Size

ONNX models can be slow, so minimize background data:

```go
// Use k-means to summarize background data
background := kMeansSummarize(fullData, 20)  // 20 centroids

// Reduce sample count for faster explanations
exp, _ := permutation.New(session, background,
    explainer.WithNumSamples(50),  // Fewer samples
)
```

### Use Parallel Workers

```go
exp, _ := permutation.New(session, background,
    explainer.WithNumSamples(100),
    explainer.WithNumWorkers(4),  // Parallel computation
)
```

## Troubleshooting

### "failed to initialize runtime"

The ONNX Runtime library wasn't found:

```go
// Check the path
err := onnx.InitializeRuntime("/correct/path/to/libonnxruntime.so")
```

Common locations:

- macOS (Homebrew): `/usr/local/lib/libonnxruntime.dylib`
- Linux: `/usr/local/lib/libonnxruntime.so`
- Windows: `C:\Program Files\onnxruntime\lib\onnxruntime.dll`

### "input name not found"

The input name doesn't match the model:

```python
# Check your model's input names
import onnx
model = onnx.load("model.onnx")
print([i.name for i in model.graph.input])
```

### "shape mismatch"

Input dimensions don't match:

```go
// Make sure NumFeatures matches model's expected input
session, _ := onnx.NewSession(onnx.Config{
    NumFeatures: 10,  // Must match model's input shape
})
```

## Multi-Output Models

For models with multiple outputs (e.g., probabilities for each class):

```go
session, _ := onnx.NewSession(onnx.Config{
    ModelPath:   "model.onnx",
    InputName:   "float_input",
    OutputName:  "probabilities",
    NumFeatures: 10,
    OutputIndex: 1,  // Use second output (e.g., probability of class 1)
})
```

## Next Steps

- [PermutationSHAP Guide](../explainers/permutation.md) - Deep dive into black-box explanations
- [Visualization](../visualization/charts.md) - Create charts from explanations
- [Benchmarks](../benchmarks.md) - Performance characteristics
