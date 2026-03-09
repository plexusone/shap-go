// Package onnx provides an ONNX Runtime model wrapper for SHAP explainability.
package onnx

import (
	"context"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/plexusone/shap-go/model"
)

// Session wraps an ONNX Runtime session for predictions.
type Session struct {
	session     *ort.DynamicAdvancedSession
	inputName   string
	outputName  string
	numFeatures int
	mu          sync.Mutex
}

// Config contains configuration for creating an ONNX session.
type Config struct {
	// ModelPath is the path to the ONNX model file.
	ModelPath string

	// InputName is the name of the input tensor (default: "float_input").
	InputName string

	// OutputName is the name of the output tensor (default: "probabilities" or "output").
	OutputName string

	// NumFeatures is the expected number of input features.
	// If 0, it will be inferred from the model if possible.
	NumFeatures int

	// UseGPU enables GPU execution if available.
	UseGPU bool
}

// DefaultConfig returns default ONNX session configuration.
func DefaultConfig() Config {
	return Config{
		InputName:  "float_input",
		OutputName: "probabilities",
	}
}

// InitializeRuntime initializes the ONNX Runtime library.
// This must be called before creating any sessions.
// The sharedLibraryPath should point to the ONNX Runtime shared library.
func InitializeRuntime(sharedLibraryPath string) error {
	ort.SetSharedLibraryPath(sharedLibraryPath)
	return ort.InitializeEnvironment()
}

// DestroyRuntime cleans up the ONNX Runtime environment.
// This should be called when all sessions are done.
func DestroyRuntime() error {
	return ort.DestroyEnvironment()
}

// NewSession creates a new ONNX Runtime session from a model file.
func NewSession(config Config) (*Session, error) {
	if config.ModelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}

	if config.InputName == "" {
		config.InputName = "float_input"
	}
	if config.OutputName == "" {
		config.OutputName = "probabilities"
	}

	// Create session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	// Create the session with dynamic shapes
	session, err := ort.NewDynamicAdvancedSession(
		config.ModelPath,
		[]string{config.InputName},
		[]string{config.OutputName},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &Session{
		session:     session,
		inputName:   config.InputName,
		outputName:  config.OutputName,
		numFeatures: config.NumFeatures,
	}, nil
}

// NewSessionFromBytes creates an ONNX session from model bytes in memory.
func NewSessionFromBytes(modelData []byte, config Config) (*Session, error) {
	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data cannot be empty")
	}

	if config.InputName == "" {
		config.InputName = "float_input"
	}
	if config.OutputName == "" {
		config.OutputName = "probabilities"
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{config.InputName},
		[]string{config.OutputName},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session from bytes: %w", err)
	}

	return &Session{
		session:     session,
		inputName:   config.InputName,
		outputName:  config.OutputName,
		numFeatures: config.NumFeatures,
	}, nil
}

// Predict returns the model's prediction for a single input.
// For binary classification models, this returns the probability of the positive class.
func (s *Session) Predict(ctx context.Context, input []float64) (float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.numFeatures > 0 && len(input) != s.numFeatures {
		return 0, fmt.Errorf("expected %d features, got %d", s.numFeatures, len(input))
	}

	// Convert to float32 for ONNX
	input32 := make([]float32, len(input))
	for i, v := range input {
		input32[i] = float32(v)
	}

	// Create input tensor with shape [1, numFeatures]
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(input))), input32)
	if err != nil {
		return 0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()

	// Prepare output slice (nil means allocate for us)
	outputs := []ort.Value{nil}

	// Run inference
	err = s.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return 0, fmt.Errorf("failed to run inference: %w", err)
	}

	if outputs[0] == nil {
		return 0, fmt.Errorf("no output returned from model")
	}
	defer func() { _ = outputs[0].Destroy() }()

	// Cast to float32 tensor to get data
	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("unexpected output tensor type")
	}

	outputData := outputTensor.GetData()
	if len(outputData) == 0 {
		return 0, fmt.Errorf("empty output tensor")
	}

	// For binary classification, return probability of positive class (index 1 if available)
	if len(outputData) >= 2 {
		return float64(outputData[1]), nil
	}

	return float64(outputData[0]), nil
}

// PredictBatch returns predictions for multiple inputs.
func (s *Session) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	numFeatures := len(inputs[0])
	if s.numFeatures > 0 && numFeatures != s.numFeatures {
		return nil, fmt.Errorf("expected %d features, got %d", s.numFeatures, numFeatures)
	}

	// Flatten inputs to a single array
	batchSize := len(inputs)
	input32 := make([]float32, batchSize*numFeatures)
	for i, input := range inputs {
		if len(input) != numFeatures {
			return nil, fmt.Errorf("input %d has %d features, expected %d", i, len(input), numFeatures)
		}
		for j, v := range input {
			input32[i*numFeatures+j] = float32(v)
		}
	}

	// Create input tensor with shape [batchSize, numFeatures]
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(numFeatures)), input32)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()

	// Prepare output slice (nil means allocate for us)
	outputs := []ort.Value{nil}

	// Run inference
	err = s.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to run batch inference: %w", err)
	}

	if outputs[0] == nil {
		return nil, fmt.Errorf("no output returned from model")
	}
	defer func() { _ = outputs[0].Destroy() }()

	// Cast to float32 tensor to get data
	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type")
	}

	outputData := outputTensor.GetData()
	shape := outputTensor.GetShape()

	// Determine output shape and extract predictions
	results := make([]float64, batchSize)

	if len(shape) == 2 && shape[1] >= 2 {
		// Output shape is [batch, numClasses] - take class 1 probability
		numClasses := int(shape[1])
		for i := range batchSize {
			results[i] = float64(outputData[i*numClasses+1])
		}
	} else if len(shape) == 2 && shape[1] == 1 {
		// Output shape is [batch, 1]
		for i := range batchSize {
			results[i] = float64(outputData[i])
		}
	} else if len(shape) == 1 {
		// Output shape is [batch]
		for i := range batchSize {
			results[i] = float64(outputData[i])
		}
	} else {
		return nil, fmt.Errorf("unexpected output shape: %v", shape)
	}

	return results, nil
}

// NumFeatures returns the number of input features.
func (s *Session) NumFeatures() int {
	return s.numFeatures
}

// SetNumFeatures sets the expected number of input features.
func (s *Session) SetNumFeatures(n int) {
	s.numFeatures = n
}

// Close releases resources held by the session.
func (s *Session) Close() error {
	if s.session != nil {
		return s.session.Destroy()
	}
	return nil
}

// Ensure Session implements model.Model.
var _ model.Model = (*Session)(nil)
