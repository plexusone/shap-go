// Example: DeepSHAP for neural network explanations
//
// This example demonstrates how to use DeepSHAP with ONNX neural networks.
// Before running, you need:
// 1. ONNX Runtime library installed
// 2. An ONNX neural network model (see generate_model.py to create one)
//
// Run:
//
//	python generate_model.py  # Creates mlp_model.onnx
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/deepshap"
	"github.com/plexusone/shap-go/model/onnx"
)

func main() {
	// Find ONNX runtime library
	runtimePath := findONNXRuntime()
	if runtimePath == "" {
		fmt.Println("ONNX Runtime not found. Please install it:")
		fmt.Println("  macOS: brew install onnxruntime")
		fmt.Println("  Linux: See https://onnxruntime.ai/")
		fmt.Println("\nOr set ONNX_RUNTIME_PATH environment variable.")
		os.Exit(1)
	}

	// Check for model file
	modelPath := "mlp_model.onnx"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Model file %s not found.\n", modelPath)
		fmt.Println("Run: python generate_model.py")
		os.Exit(1)
	}

	// Initialize ONNX Runtime
	if err := onnx.InitializeRuntime(runtimePath); err != nil {
		log.Fatalf("Failed to initialize ONNX runtime: %v", err)
	}
	defer func() {
		if err := onnx.DestroyRuntime(); err != nil {
			log.Printf("Warning: failed to destroy runtime: %v", err)
		}
	}()

	// Parse the ONNX graph to get layer information
	graphInfo, err := onnx.ParseGraph(modelPath)
	if err != nil {
		log.Fatalf("Failed to parse ONNX graph: %v", err)
	}

	fmt.Println("ONNX Graph Structure:")
	fmt.Printf("  Inputs: %v\n", graphInfo.InputNames)
	fmt.Printf("  Outputs: %v\n", graphInfo.OutputNames)
	fmt.Printf("  Nodes: %d\n", len(graphInfo.Nodes))
	for _, node := range graphInfo.Nodes {
		fmt.Printf("    %s (%s -> %s)\n", node.Name, node.OpType, node.LayerType)
	}
	fmt.Println()

	// Get intermediate layer outputs for activation capture
	intermediateOutputs := graphInfo.GetAllLayerOutputs()

	// Create activation session that captures intermediate values
	session, err := onnx.NewActivationSession(onnx.ActivationConfig{
		Config: onnx.Config{
			ModelPath:   modelPath,
			InputName:   "input",
			OutputName:  "output",
			NumFeatures: 4,
		},
		IntermediateOutputs: intermediateOutputs,
	})
	if err != nil {
		log.Fatalf("Failed to create activation session: %v", err)
	}
	defer func() { _ = session.Close() }()

	// Background data (representative samples)
	background := [][]float64{
		{5.0, 3.4, 1.5, 0.2}, // Typical setosa
		{5.9, 2.8, 4.3, 1.3}, // Typical versicolor
		{6.6, 3.0, 5.5, 2.0}, // Typical virginica
	}

	// Feature names
	featureNames := []string{
		"sepal_length",
		"sepal_width",
		"petal_length",
		"petal_width",
	}

	// Create DeepSHAP explainer
	exp, err := deepshap.New(session, graphInfo, background,
		explainer.WithFeatureNames(featureNames),
		explainer.WithModelID("iris-mlp"),
	)
	if err != nil {
		log.Fatalf("Failed to create DeepSHAP explainer: %v", err)
	}

	// Instances to explain
	instances := []struct {
		name   string
		values []float64
	}{
		{"Setosa sample", []float64{5.1, 3.5, 1.4, 0.2}},
		{"Versicolor sample", []float64{6.0, 2.7, 4.5, 1.5}},
		{"Virginica sample", []float64{7.2, 3.2, 6.0, 1.8}},
	}

	ctx := context.Background()

	fmt.Println("DeepSHAP Neural Network Explanations")
	fmt.Println("=====================================")
	fmt.Println()

	for _, inst := range instances {
		fmt.Printf("Instance: %s\n", inst.name)
		fmt.Printf("Features: %v\n", inst.values)

		explanation, err := exp.Explain(ctx, inst.values)
		if err != nil {
			log.Printf("Failed to explain %s: %v", inst.name, err)
			continue
		}

		fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
		fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)

		fmt.Println("SHAP Values (DeepLIFT-based):")
		for _, name := range featureNames {
			shap := explanation.Values[name]
			value := explanation.FeatureValues[name]
			fmt.Printf("  %-14s: value=%5.2f, SHAP=%+.4f\n", name, value, shap)
		}

		// Show top features
		fmt.Println("Top Contributing Features:")
		for i, f := range explanation.TopFeatures(4) {
			direction := "↑"
			if f.SHAPValue < 0 {
				direction = "↓"
			}
			fmt.Printf("  %d. %s: %+.4f %s\n", i+1, f.Name, f.SHAPValue, direction)
		}

		// Verify local accuracy
		result := explanation.Verify(0.2) // Neural networks may have larger errors
		fmt.Printf("Local accuracy: sum=%.4f, expected=%.4f, diff=%.4f\n",
			result.SumSHAP, result.Expected, result.Difference)

		fmt.Println()
	}

	// Compare with intermediate activations
	fmt.Println("Activation Analysis")
	fmt.Println("===================")
	testInstance := []float64{6.0, 2.7, 4.5, 1.5}
	actResult, err := session.PredictWithActivations(ctx, testInstance)
	if err != nil {
		log.Printf("Failed to get activations: %v", err)
	} else {
		fmt.Printf("Input: %v\n", testInstance)
		fmt.Printf("Output: %.4f\n", actResult.Prediction)
		fmt.Println("Layer Activations:")
		for name, acts := range actResult.Activations {
			if len(acts) <= 8 {
				fmt.Printf("  %s: %v\n", name, acts)
			} else {
				fmt.Printf("  %s: [%d values]\n", name, len(acts))
			}
		}
	}
}

// findONNXRuntime attempts to locate the ONNX runtime library.
func findONNXRuntime() string {
	paths := []string{
		"/usr/local/lib/libonnxruntime.dylib",    // macOS Intel
		"/opt/homebrew/lib/libonnxruntime.dylib", // macOS ARM
		"/usr/local/lib/libonnxruntime.so",       // Linux
		"/usr/lib/libonnxruntime.so",             // Linux alt
	}

	// Check environment variable first
	if envPath := os.Getenv("ONNX_RUNTIME_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil { //nolint:gosec // User-provided path is intentional
			return envPath
		}
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
