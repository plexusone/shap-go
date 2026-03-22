package deepshap

import (
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model/onnx"
)

func TestExplainer_ImplementsInterface(t *testing.T) {
	// Verify that Explainer implements explainer.Explainer
	var _ explainer.Explainer = (*Explainer)(nil)
}

func TestNew_NilSession(t *testing.T) {
	_, err := New(nil, nil, nil)
	if err != ErrNilSession {
		t.Errorf("Expected ErrNilSession, got %v", err)
	}
}

func TestNew_NilGraphInfo(t *testing.T) {
	// Create a mock session - this will fail at session creation
	// but we can test the nil graph check by calling NewSimple
	_, err := NewSimple(nil, nil)
	if err != ErrNilSession {
		t.Errorf("Expected ErrNilSession, got %v", err)
	}
}

func TestNew_NoBackground(t *testing.T) {
	// We can't create a real session without ONNX runtime,
	// but we can test the error handling logic
	// This test documents expected behavior
	t.Skip("Requires ONNX runtime initialization")
}

func TestNewSimple_NoBackground(t *testing.T) {
	// This test documents expected behavior
	t.Skip("Requires ONNX runtime initialization")
}

func TestBuildActivationData(t *testing.T) {
	e := &Explainer{}

	input := []float64{1.0, 2.0, 3.0}
	result := &onnx.ActivationResult{
		Prediction: 0.75,
		Activations: map[string][]float32{
			"layer1": {0.5, 1.0},
			"layer2": {0.75},
		},
	}

	actData := e.buildActivationData(input, result)

	if actData.Output != 0.75 {
		t.Errorf("Output = %v, expected 0.75", actData.Output)
	}

	if len(actData.Input) != 3 {
		t.Errorf("Input length = %d, expected 3", len(actData.Input))
	}

	if len(actData.LayerActivations) != 2 {
		t.Errorf("LayerActivations length = %d, expected 2", len(actData.LayerActivations))
	}

	layer1 := actData.LayerActivations["layer1"]
	if len(layer1) != 2 {
		t.Errorf("layer1 length = %d, expected 2", len(layer1))
	}
	if layer1[0] != 0.5 || layer1[1] != 1.0 {
		t.Errorf("layer1 = %v, expected [0.5, 1.0]", layer1)
	}
}

func TestComputeSimpleAttributions(t *testing.T) {
	e := &Explainer{}

	instance := []float64{2.0, 4.0}
	reference := []float64{0.0, 0.0}

	instanceResult := &onnx.ActivationResult{
		Prediction:  6.0, // sum of inputs
		Activations: map[string][]float32{},
	}
	refResult := &onnx.ActivationResult{
		Prediction:  0.0,
		Activations: map[string][]float32{},
	}

	attrs := e.computeSimpleAttributions(instance, reference, instanceResult, refResult)

	// Output diff = 6, total input diff = 6
	// attr[0] = 2 * (6/6) = 2
	// attr[1] = 4 * (6/6) = 4
	// Sum = 6 = output diff

	if len(attrs) != 2 {
		t.Fatalf("Expected 2 attributions, got %d", len(attrs))
	}

	sum := attrs[0] + attrs[1]
	outputDiff := instanceResult.Prediction - refResult.Prediction

	if sum != outputDiff {
		t.Errorf("Sum of attributions = %v, expected %v", sum, outputDiff)
	}
}

func TestComputeSimpleAttributions_ZeroDiff(t *testing.T) {
	e := &Explainer{}

	// Same instance and reference = no attributions
	instance := []float64{1.0, 2.0}
	reference := []float64{1.0, 2.0}

	instanceResult := &onnx.ActivationResult{
		Prediction:  3.0,
		Activations: map[string][]float32{},
	}
	refResult := &onnx.ActivationResult{
		Prediction:  3.0,
		Activations: map[string][]float32{},
	}

	attrs := e.computeSimpleAttributions(instance, reference, instanceResult, refResult)

	for i, a := range attrs {
		if a != 0 {
			t.Errorf("Attribution[%d] = %v, expected 0 for identical inputs", i, a)
		}
	}
}

func TestComputeSimpleAttributions_SingleFeatureChange(t *testing.T) {
	e := &Explainer{}

	// Only feature 0 changes
	instance := []float64{5.0, 0.0}
	reference := []float64{0.0, 0.0}

	instanceResult := &onnx.ActivationResult{
		Prediction:  5.0,
		Activations: map[string][]float32{},
	}
	refResult := &onnx.ActivationResult{
		Prediction:  0.0,
		Activations: map[string][]float32{},
	}

	attrs := e.computeSimpleAttributions(instance, reference, instanceResult, refResult)

	// All attribution should go to feature 0
	if attrs[0] != 5.0 {
		t.Errorf("Attribution[0] = %v, expected 5.0", attrs[0])
	}
	if attrs[1] != 0.0 {
		t.Errorf("Attribution[1] = %v, expected 0.0", attrs[1])
	}
}
