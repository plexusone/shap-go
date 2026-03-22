package onnx

import (
	"testing"

	"github.com/plexusone/shap-go/model"
)

func TestActivationSession_ImplementsModel(t *testing.T) {
	// This test verifies that ActivationSession implements model.Model
	var _ model.Model = (*ActivationSession)(nil)
}

func TestActivationConfig_Defaults(t *testing.T) {
	config := ActivationConfig{}

	if config.InputName != "" {
		t.Errorf("Default InputName should be empty, got %q", config.InputName)
	}

	if config.OutputName != "" {
		t.Errorf("Default OutputName should be empty, got %q", config.OutputName)
	}

	if len(config.IntermediateOutputs) != 0 {
		t.Errorf("Default IntermediateOutputs should be empty, got %v", config.IntermediateOutputs)
	}
}

func TestActivationResult_Empty(t *testing.T) {
	result := &ActivationResult{
		Prediction:  0.5,
		Activations: make(map[string][]float32),
	}

	if result.Prediction != 0.5 {
		t.Errorf("Expected prediction 0.5, got %v", result.Prediction)
	}

	if len(result.Activations) != 0 {
		t.Errorf("Expected empty activations, got %d entries", len(result.Activations))
	}
}

func TestNewActivationSession_EmptyPath(t *testing.T) {
	config := ActivationConfig{}
	_, err := NewActivationSession(config)
	if err == nil {
		t.Error("Expected error for empty model path")
	}
}

func TestNewActivationSessionFromBytes_EmptyData(t *testing.T) {
	config := ActivationConfig{}
	_, err := NewActivationSessionFromBytes(nil, config)
	if err == nil {
		t.Error("Expected error for empty model data")
	}

	// Also test with empty slice
	_, err = NewActivationSessionFromBytes([]byte{}, config)
	if err == nil {
		t.Error("Expected error for empty model data slice")
	}
}

func TestActivationSession_NumFeatures(t *testing.T) {
	// Create a mock session without ONNX (just testing the methods)
	s := &ActivationSession{
		numFeatures: 10,
	}

	if s.NumFeatures() != 10 {
		t.Errorf("NumFeatures() = %d, want 10", s.NumFeatures())
	}

	s.SetNumFeatures(20)
	if s.NumFeatures() != 20 {
		t.Errorf("NumFeatures() after SetNumFeatures(20) = %d, want 20", s.NumFeatures())
	}
}

func TestActivationSession_IntermediateOutputs(t *testing.T) {
	s := &ActivationSession{
		intermediateNames: []string{"layer1_out", "layer2_out", "layer3_out"},
	}

	outputs := s.IntermediateOutputs()
	if len(outputs) != 3 {
		t.Fatalf("IntermediateOutputs() returned %d names, want 3", len(outputs))
	}

	expected := []string{"layer1_out", "layer2_out", "layer3_out"}
	for i, name := range outputs {
		if name != expected[i] {
			t.Errorf("IntermediateOutputs()[%d] = %q, want %q", i, name, expected[i])
		}
	}
}

func TestActivationSession_Close_NilSession(t *testing.T) {
	s := &ActivationSession{session: nil}
	err := s.Close()
	if err != nil {
		t.Errorf("Close() on nil session returned error: %v", err)
	}
}

func TestActivationConfig_WithIntermediateOutputs(t *testing.T) {
	config := ActivationConfig{
		Config: Config{
			ModelPath:   "/path/to/model.onnx",
			InputName:   "input",
			OutputName:  "output",
			NumFeatures: 10,
		},
		IntermediateOutputs: []string{"relu1", "relu2"},
	}

	if config.ModelPath != "/path/to/model.onnx" {
		t.Errorf("ModelPath = %q, want %q", config.ModelPath, "/path/to/model.onnx")
	}
	if config.InputName != "input" {
		t.Errorf("InputName = %q, want %q", config.InputName, "input")
	}
	if config.OutputName != "output" {
		t.Errorf("OutputName = %q, want %q", config.OutputName, "output")
	}
	if config.NumFeatures != 10 {
		t.Errorf("NumFeatures = %d, want 10", config.NumFeatures)
	}
	if len(config.IntermediateOutputs) != 2 {
		t.Errorf("IntermediateOutputs length = %d, want 2", len(config.IntermediateOutputs))
	}
}

func TestActivationResult_WithActivations(t *testing.T) {
	result := &ActivationResult{
		Prediction: 0.85,
		Activations: map[string][]float32{
			"layer1": {0.1, 0.2, 0.3},
			"layer2": {0.4, 0.5},
		},
	}

	if result.Prediction != 0.85 {
		t.Errorf("Prediction = %v, want 0.85", result.Prediction)
	}

	if len(result.Activations) != 2 {
		t.Errorf("Activations length = %d, want 2", len(result.Activations))
	}

	layer1 := result.Activations["layer1"]
	if len(layer1) != 3 {
		t.Errorf("layer1 activations length = %d, want 3", len(layer1))
	}

	layer2 := result.Activations["layer2"]
	if len(layer2) != 2 {
		t.Errorf("layer2 activations length = %d, want 2", len(layer2))
	}
}

func TestActivationSession_EmptyIntermediateOutputs(t *testing.T) {
	s := &ActivationSession{
		intermediateNames: nil,
	}

	outputs := s.IntermediateOutputs()
	if outputs != nil {
		t.Errorf("IntermediateOutputs() should return nil for empty list, got %v", outputs)
	}
}
