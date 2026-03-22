package onnx

import (
	"context"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/plexusone/shap-go/model"
)

// ActivationConfig contains configuration for creating an ActivationSession.
type ActivationConfig struct {
	Config

	// IntermediateOutputs are the names of intermediate layer outputs to capture.
	// These should be valid tensor names in the ONNX graph.
	IntermediateOutputs []string
}

// ActivationResult contains the prediction and intermediate activations.
type ActivationResult struct {
	// Prediction is the final model output.
	Prediction float64

	// Activations maps layer output names to their activation values.
	// Values are stored as float32 (native ONNX type).
	Activations map[string][]float32
}

// ActivationSession wraps an ONNX Runtime session to capture intermediate activations.
type ActivationSession struct {
	session           *ort.DynamicAdvancedSession
	inputName         string
	outputName        string   // final output
	intermediateNames []string // intermediate outputs to capture
	allOutputNames    []string // outputName + intermediateNames
	numFeatures       int
	mu                sync.Mutex
}

// NewActivationSession creates a new ONNX session that captures intermediate activations.
func NewActivationSession(config ActivationConfig) (*ActivationSession, error) {
	if config.ModelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}

	if config.InputName == "" {
		config.InputName = "float_input"
	}
	if config.OutputName == "" {
		config.OutputName = "probabilities"
	}

	// Combine final output with intermediate outputs
	allOutputNames := make([]string, 0, 1+len(config.IntermediateOutputs))
	allOutputNames = append(allOutputNames, config.OutputName)
	allOutputNames = append(allOutputNames, config.IntermediateOutputs...)

	// Create session options
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	// Create the session with multiple outputs
	session, err := ort.NewDynamicAdvancedSession(
		config.ModelPath,
		[]string{config.InputName},
		allOutputNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ActivationSession{
		session:           session,
		inputName:         config.InputName,
		outputName:        config.OutputName,
		intermediateNames: config.IntermediateOutputs,
		allOutputNames:    allOutputNames,
		numFeatures:       config.NumFeatures,
	}, nil
}

// NewActivationSessionFromBytes creates an ActivationSession from model bytes.
func NewActivationSessionFromBytes(modelData []byte, config ActivationConfig) (*ActivationSession, error) {
	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data cannot be empty")
	}

	if config.InputName == "" {
		config.InputName = "float_input"
	}
	if config.OutputName == "" {
		config.OutputName = "probabilities"
	}

	// Combine final output with intermediate outputs
	allOutputNames := make([]string, 0, 1+len(config.IntermediateOutputs))
	allOutputNames = append(allOutputNames, config.OutputName)
	allOutputNames = append(allOutputNames, config.IntermediateOutputs...)

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		modelData,
		[]string{config.InputName},
		allOutputNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session from bytes: %w", err)
	}

	return &ActivationSession{
		session:           session,
		inputName:         config.InputName,
		outputName:        config.OutputName,
		intermediateNames: config.IntermediateOutputs,
		allOutputNames:    allOutputNames,
		numFeatures:       config.NumFeatures,
	}, nil
}

// PredictWithActivations returns both the prediction and intermediate activations.
func (s *ActivationSession) PredictWithActivations(ctx context.Context, input []float64) (*ActivationResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.numFeatures > 0 && len(input) != s.numFeatures {
		return nil, fmt.Errorf("expected %d features, got %d", s.numFeatures, len(input))
	}

	// Convert to float32 for ONNX
	input32 := make([]float32, len(input))
	for i, v := range input {
		input32[i] = float32(v)
	}

	// Create input tensor with shape [1, numFeatures]
	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(input))), input32)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()

	// Prepare output slices
	outputs := make([]ort.Value, len(s.allOutputNames))

	// Run inference
	err = s.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Ensure cleanup
	defer func() {
		for _, out := range outputs {
			if out != nil {
				_ = out.Destroy()
			}
		}
	}()

	// Extract prediction from first output
	if outputs[0] == nil {
		return nil, fmt.Errorf("no output returned from model")
	}

	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type")
	}

	outputData := outputTensor.GetData()
	if len(outputData) == 0 {
		return nil, fmt.Errorf("empty output tensor")
	}

	// For binary classification, return probability of positive class
	var prediction float64
	if len(outputData) >= 2 {
		prediction = float64(outputData[1])
	} else {
		prediction = float64(outputData[0])
	}

	// Extract intermediate activations
	activations := make(map[string][]float32)
	for i, name := range s.intermediateNames {
		outputIdx := i + 1 // +1 because first output is the final prediction
		if outputs[outputIdx] == nil {
			continue
		}

		tensor, ok := outputs[outputIdx].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected tensor type for %s", name)
		}

		// Copy the data (since we'll destroy the tensor)
		data := tensor.GetData()
		copied := make([]float32, len(data))
		copy(copied, data)
		activations[name] = copied
	}

	return &ActivationResult{
		Prediction:  prediction,
		Activations: activations,
	}, nil
}

// PredictBatchWithActivations returns predictions and activations for multiple inputs.
func (s *ActivationSession) PredictBatchWithActivations(ctx context.Context, inputs [][]float64) ([]*ActivationResult, error) {
	results := make([]*ActivationResult, len(inputs))
	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := s.PredictWithActivations(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("failed to predict input %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// Predict returns only the model prediction (for Model interface compatibility).
func (s *ActivationSession) Predict(ctx context.Context, input []float64) (float64, error) {
	result, err := s.PredictWithActivations(ctx, input)
	if err != nil {
		return 0, err
	}
	return result.Prediction, nil
}

// PredictBatch returns predictions for multiple inputs.
func (s *ActivationSession) PredictBatch(ctx context.Context, inputs [][]float64) ([]float64, error) {
	results, err := s.PredictBatchWithActivations(ctx, inputs)
	if err != nil {
		return nil, err
	}

	predictions := make([]float64, len(results))
	for i, r := range results {
		predictions[i] = r.Prediction
	}
	return predictions, nil
}

// NumFeatures returns the number of input features.
func (s *ActivationSession) NumFeatures() int {
	return s.numFeatures
}

// SetNumFeatures sets the expected number of input features.
func (s *ActivationSession) SetNumFeatures(n int) {
	s.numFeatures = n
}

// IntermediateOutputs returns the names of intermediate outputs being captured.
func (s *ActivationSession) IntermediateOutputs() []string {
	return s.intermediateNames
}

// Close releases resources held by the session.
func (s *ActivationSession) Close() error {
	if s.session != nil {
		return s.session.Destroy()
	}
	return nil
}

// Ensure ActivationSession implements model.Model.
var _ model.Model = (*ActivationSession)(nil)
