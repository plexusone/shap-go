# Installation

## Requirements

- Go 1.21 or later
- (Optional) ONNX Runtime for ONNX model support

## Install with Go

```bash
go get github.com/plexusone/shap-go
```

## Verify Installation

Create a simple test file:

```go
package main

import (
    "fmt"
    "github.com/plexusone/shap-go/explainer/tree"
)

func main() {
    fmt.Println("SHAP-Go installed successfully!")
    fmt.Printf("TreeSHAP available: %v\n", tree.New != nil)
}
```

Run it:

```bash
go run main.go
```

## Package Structure

SHAP-Go is organized into focused packages:

| Package | Description |
|---------|-------------|
| `explainer/tree` | TreeSHAP for XGBoost/LightGBM |
| `explainer/permutation` | PermutationSHAP with antithetic sampling |
| `explainer/sampling` | Monte Carlo SamplingSHAP |
| `explanation` | Core explanation types |
| `model` | Model interface and wrappers |
| `model/onnx` | ONNX Runtime integration |
| `render` | Visualization outputs |
| `background` | Background dataset management |
| `masker` | Feature masking strategies |

## ONNX Runtime (Optional)

For ONNX model support, you need the ONNX Runtime shared library.

### macOS

```bash
brew install onnxruntime
```

### Linux

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases):

```bash
# Example for Linux x64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.16.0/lib:$LD_LIBRARY_PATH
```

### Windows

Download the Windows release and add the DLL directory to your PATH.

## Next Steps

- [Quick Start](quickstart.md) - Get up and running
- [TreeSHAP Guide](../explainers/treeshap.md) - For tree-based models
- [Benchmarks](../benchmarks.md) - Performance characteristics
