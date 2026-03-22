// Example: Computing SHAP values for an ONNX model
//
// This example demonstrates how to use SHAP-Go with ONNX models.
// Before running, you need:
// 1. ONNX Runtime library installed
// 2. An ONNX model file (see generate_model.py to create one)
//
// Run:
//
//	python generate_model.py  # Creates iris_model.onnx
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/kernel"
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
	modelPath := "iris_model.onnx"
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

	// Create ONNX session
	session, err := onnx.NewSession(onnx.Config{
		ModelPath:   modelPath,
		InputName:   "float_input",
		OutputName:  "probabilities",
		NumFeatures: 4, // Iris dataset has 4 features
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer func() { _ = session.Close() }()

	// Background data (representative samples from training data)
	// Using Iris dataset means/typical values
	background := [][]float64{
		{5.0, 3.4, 1.5, 0.2}, // Typical setosa
		{5.9, 2.8, 4.3, 1.3}, // Typical versicolor
		{6.6, 3.0, 5.5, 2.0}, // Typical virginica
	}

	// Feature names for the Iris dataset
	featureNames := []string{
		"sepal_length",
		"sepal_width",
		"petal_length",
		"petal_width",
	}

	// Create KernelSHAP explainer (works with any model)
	exp, err := kernel.New(session, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithFeatureNames(featureNames),
		explainer.WithModelID("iris-classifier"),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
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

	fmt.Println("ONNX Model SHAP Explanations")
	fmt.Println("============================")
	fmt.Println()

	for _, inst := range instances {
		fmt.Printf("Instance: %s\n", inst.name)
		fmt.Printf("Features: %v\n", inst.values)

		explanation, err := exp.Explain(ctx, inst.values)
		if err != nil {
			log.Printf("Failed to explain %s: %v", inst.name, err)
			continue
		}

		fmt.Printf("Prediction (P(class=1)): %.4f\n", explanation.Prediction)
		fmt.Printf("Base Value: %.4f\n", explanation.BaseValue)

		fmt.Println("SHAP Values:")
		for _, name := range featureNames {
			shap := explanation.Values[name]
			value := explanation.FeatureValues[name]
			fmt.Printf("  %-14s: value=%5.2f, SHAP=%+.4f\n", name, value, shap)
		}

		// Verify local accuracy
		result := explanation.Verify(0.1) // Use larger tolerance for approximation
		fmt.Printf("Local accuracy: sum=%.4f, expected=%.4f, valid=%v\n",
			result.SumSHAP, result.Expected, result.Valid)

		fmt.Println()
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
