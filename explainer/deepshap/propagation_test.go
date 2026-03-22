package deepshap

import (
	"math"
	"testing"

	"github.com/plexusone/shap-go/model/onnx"
)

func TestNewPropagationEngine(t *testing.T) {
	graphInfo := &onnx.GraphInfo{
		Nodes: []onnx.NodeInfo{
			{Name: "dense1", LayerType: onnx.LayerTypeDense},
			{Name: "relu1", LayerType: onnx.LayerTypeReLU},
		},
		TopologicalOrder: []string{"dense1", "relu1"},
	}

	engine := NewPropagationEngine(graphInfo)

	if engine == nil {
		t.Fatal("NewPropagationEngine returned nil")
	}

	if len(engine.rules) != 2 {
		t.Errorf("Expected 2 rules, got %d", len(engine.rules))
	}

	if _, ok := engine.rules["dense1"]; !ok {
		t.Error("Missing rule for dense1")
	}

	if _, ok := engine.rules["relu1"]; !ok {
		t.Error("Missing rule for relu1")
	}
}

func TestPropagateSimple_Linear(t *testing.T) {
	// Test with a simple linear model (no hidden layers)
	// f(x) = sum(x), so each feature contributes equally

	instanceInput := []float64{1.0, 2.0, 3.0}
	referenceInput := []float64{0.0, 0.0, 0.0}

	// No intermediate activations for linear model
	instanceActivations := [][]float64{}
	referenceActivations := [][]float64{}

	instanceOutput := 6.0 // sum(1, 2, 3)
	referenceOutput := 0.0

	attributions := PropagateSimple(
		instanceInput, referenceInput,
		instanceActivations, referenceActivations,
		instanceOutput, referenceOutput,
	)

	if len(attributions) != 3 {
		t.Fatalf("Expected 3 attributions, got %d", len(attributions))
	}

	// For a linear model, attributions should equal input differences
	// scaled by output difference / total input difference
	// Total diff = 6, output diff = 6
	// So attribution[i] = input[i] * (6/6) = input[i]
	totalSum := 0.0
	for _, a := range attributions {
		totalSum += a
	}

	// Sum of attributions should equal output difference
	expectedSum := instanceOutput - referenceOutput
	if math.Abs(totalSum-expectedSum) > 0.1 {
		t.Errorf("Sum of attributions = %v, expected %v", totalSum, expectedSum)
	}
}

func TestPropagateSimple_WithActivations(t *testing.T) {
	// Test with one hidden layer
	instanceInput := []float64{1.0, 2.0}
	referenceInput := []float64{0.0, 0.0}

	// Hidden layer activations (after some transform)
	instanceActivations := [][]float64{
		{2.0, 4.0}, // e.g., 2x each input
	}
	referenceActivations := [][]float64{
		{0.0, 0.0},
	}

	instanceOutput := 6.0
	referenceOutput := 0.0

	attributions := PropagateSimple(
		instanceInput, referenceInput,
		instanceActivations, referenceActivations,
		instanceOutput, referenceOutput,
	)

	if len(attributions) != 2 {
		t.Fatalf("Expected 2 attributions, got %d", len(attributions))
	}

	// Attributions should be non-zero
	for i, a := range attributions {
		if a == 0 && instanceInput[i] != referenceInput[i] {
			t.Errorf("Attribution[%d] should be non-zero for different inputs", i)
		}
	}
}

func TestPropagationEngine_Propagate(t *testing.T) {
	// Create a simple graph: input -> relu -> output (identity-like)
	// This tests basic propagation without dimension changes
	graphInfo := &onnx.GraphInfo{
		Nodes: []onnx.NodeInfo{
			{
				Name:      "relu1",
				LayerType: onnx.LayerTypeReLU,
				Inputs:    []string{"input"},
				Outputs:   []string{"output"},
			},
		},
		TopologicalOrder: []string{"relu1"},
		InputNames:       []string{"input"},
		OutputNames:      []string{"output"},
		Initializers:     map[string]int{},
	}

	engine := NewPropagationEngine(graphInfo)

	instanceAct := &ActivationData{
		Input: []float64{1.0, 2.0},
		LayerActivations: map[string][]float64{
			"output": {1.0, 2.0}, // ReLU of positive inputs = identity
		},
		Output: 3.0, // sum
	}

	referenceAct := &ActivationData{
		Input: []float64{0.0, 0.0},
		LayerActivations: map[string][]float64{
			"output": {0.0, 0.0},
		},
		Output: 0.0,
	}

	result, err := engine.Propagate(instanceAct, referenceAct, 1.0)
	if err != nil {
		t.Fatalf("Propagate failed: %v", err)
	}

	if result == nil {
		t.Fatal("Propagate returned nil result")
	}

	if len(result.Attributions) != 2 {
		t.Errorf("Expected 2 attributions, got %d", len(result.Attributions))
	}

	// For ReLU with positive inputs, the rescale ratio is 1
	// So attributions should equal input differences
	for i, attr := range result.Attributions {
		expected := instanceAct.Input[i] - referenceAct.Input[i]
		if math.Abs(attr-expected) > 0.1 {
			t.Errorf("Attribution[%d] = %v, expected %v", i, attr, expected)
		}
	}
}

func TestTotalInputDiff(t *testing.T) {
	tests := []struct {
		inst     []float64
		ref      []float64
		expected float64
	}{
		{[]float64{1, 2, 3}, []float64{0, 0, 0}, 6},
		{[]float64{1, 2, 3}, []float64{1, 1, 1}, 3},
		{[]float64{0, 0, 0}, []float64{0, 0, 0}, 1}, // Returns 1 to avoid div by zero
		{[]float64{-1, 1}, []float64{0, 0}, 0},      // Sums to 0, returns 1
	}

	for _, tt := range tests {
		got := totalInputDiff(tt.inst, tt.ref)
		// Handle the special case where result should be 1 for zero sum
		sum := 0.0
		for i := range tt.inst {
			sum += tt.inst[i] - tt.ref[i]
		}
		if sum == 0 && got != 1 {
			t.Errorf("totalInputDiff for zero sum should be 1, got %v", got)
		} else if sum != 0 && math.Abs(got-tt.expected) > 1e-10 {
			t.Errorf("totalInputDiff(%v, %v) = %v, expected %v", tt.inst, tt.ref, got, tt.expected)
		}
	}
}
