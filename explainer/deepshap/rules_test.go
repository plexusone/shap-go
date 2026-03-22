package deepshap

import (
	"math"
	"testing"

	"github.com/plexusone/shap-go/model/onnx"
)

func TestRuleFactory(t *testing.T) {
	tests := []struct {
		layerType onnx.LayerType
		ruleType  string
	}{
		{onnx.LayerTypeDense, "*deepshap.LinearRule"},
		{onnx.LayerTypeReLU, "*deepshap.ReLURescaleRule"},
		{onnx.LayerTypeSigmoid, "*deepshap.SigmoidRule"},
		{onnx.LayerTypeTanh, "*deepshap.TanhRule"},
		{onnx.LayerTypeSoftmax, "*deepshap.SoftmaxRule"},
		{onnx.LayerTypeAdd, "*deepshap.AddRule"},
		{onnx.LayerTypeIdentity, "*deepshap.IdentityRule"},
		{onnx.LayerTypeUnknown, "*deepshap.IdentityRule"}, // Falls back to identity
	}

	for _, tt := range tests {
		t.Run(string(tt.layerType), func(t *testing.T) {
			rule := RuleFactory(tt.layerType)
			if rule == nil {
				t.Errorf("RuleFactory(%v) returned nil", tt.layerType)
			}
		})
	}
}

func TestLinearRule_Simple(t *testing.T) {
	rule := &LinearRule{}

	// Test with a simple 2x2 weight matrix
	// W = [[1, 2], [3, 4]]
	// y = Wx means:
	//   y[0] = w[0][0]*x[0] + w[1][0]*x[1] = 1*x[0] + 3*x[1]
	//   y[1] = w[0][1]*x[0] + w[1][1]*x[1] = 2*x[0] + 4*x[1]
	// For backprop: mult_in[i] = sum_j(w[i][j] * mult_out[j])
	weights := [][]float64{
		{1, 2},
		{3, 4},
	}

	outputMult := []float64{1, 1}
	inputAct := []float64{0, 0}  // not used for linear
	inputRef := []float64{0, 0}  // not used for linear
	outputAct := []float64{0, 0} // not used for linear
	outputRef := []float64{0, 0} // not used for linear

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, weights)

	// mult_in[0] = w[0][0]*mult_out[0] + w[0][1]*mult_out[1] = 1*1 + 2*1 = 3
	// mult_in[1] = w[1][0]*mult_out[0] + w[1][1]*mult_out[1] = 3*1 + 4*1 = 7
	expected := []float64{3, 7}

	if len(inputMult) != len(expected) {
		t.Fatalf("LinearRule returned %d values, expected %d", len(inputMult), len(expected))
	}

	for i, v := range inputMult {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}

func TestReLURescaleRule_Positive(t *testing.T) {
	rule := &ReLURescaleRule{}

	// For ReLU with positive inputs, y = x
	// So (y - y_ref) == (x - x_ref), and the rescale ratio is 1
	inputAct := []float64{2.0, 3.0}
	inputRef := []float64{1.0, 1.0}
	outputAct := []float64{2.0, 3.0} // ReLU(2) = 2, ReLU(3) = 3
	outputRef := []float64{1.0, 1.0} // ReLU(1) = 1
	outputMult := []float64{1.0, 1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	// x_diff = [1, 2], y_diff = [1, 2]
	// mult_in = mult_out * x_diff / y_diff = [1, 1]
	expected := []float64{1.0, 1.0}

	for i, v := range inputMult {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}

func TestReLURescaleRule_NegativeToPositive(t *testing.T) {
	rule := &ReLURescaleRule{}

	// When input goes from negative to positive:
	// x_ref = -1 -> y_ref = ReLU(-1) = 0
	// x = 2 -> y = ReLU(2) = 2
	// y_diff = 2, x_diff = 3
	// mult_in = mult_out * 3/2 = 1.5
	inputAct := []float64{2.0}
	inputRef := []float64{-1.0}
	outputAct := []float64{2.0}
	outputRef := []float64{0.0}
	outputMult := []float64{1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	expected := []float64{1.5}

	for i, v := range inputMult {
		if i < len(expected) && math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}

func TestReLURescaleRule_GradientFallback(t *testing.T) {
	rule := &ReLURescaleRule{}

	// When y == y_ref (no output change), use gradient
	// For positive input, gradient is 1
	inputAct := []float64{2.0}
	inputRef := []float64{2.0} // Same as inputAct
	outputAct := []float64{2.0}
	outputRef := []float64{2.0} // Same as outputAct -> triggers fallback
	outputMult := []float64{1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	// Gradient of ReLU at x=2 is 1
	expected := []float64{1.0}

	for i, v := range inputMult {
		if i < len(expected) && math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}

func TestReLURescaleRule_NegativeGradient(t *testing.T) {
	rule := &ReLURescaleRule{}

	// For negative input, gradient is 0
	inputAct := []float64{-2.0}
	inputRef := []float64{-2.0}
	outputAct := []float64{0.0}
	outputRef := []float64{0.0}
	outputMult := []float64{1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	expected := []float64{0.0}

	for i, v := range inputMult {
		if i < len(expected) && math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}

func TestSigmoidRule_Rescale(t *testing.T) {
	rule := &SigmoidRule{}

	// Test with non-zero output difference
	inputAct := []float64{0.0}
	inputRef := []float64{-1.0}
	outputAct := []float64{0.5}    // sigmoid(0) = 0.5
	outputRef := []float64{0.2689} // sigmoid(-1) ≈ 0.2689
	outputMult := []float64{1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	// x_diff = 1, y_diff ≈ 0.2311
	// mult_in = 1 * 1 / 0.2311 ≈ 4.33
	xDiff := inputAct[0] - inputRef[0]
	yDiff := outputAct[0] - outputRef[0]
	expected := outputMult[0] * xDiff / yDiff

	if math.Abs(inputMult[0]-expected) > 1e-6 {
		t.Errorf("inputMult[0] = %v, expected %v", inputMult[0], expected)
	}
}

func TestTanhRule_Rescale(t *testing.T) {
	rule := &TanhRule{}

	// Test with non-zero output difference
	inputAct := []float64{1.0}
	inputRef := []float64{0.0}
	outputAct := []float64{math.Tanh(1.0)} // ≈ 0.7616
	outputRef := []float64{0.0}            // tanh(0) = 0
	outputMult := []float64{1.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	xDiff := inputAct[0] - inputRef[0]
	yDiff := outputAct[0] - outputRef[0]
	expected := outputMult[0] * xDiff / yDiff

	if math.Abs(inputMult[0]-expected) > 1e-6 {
		t.Errorf("inputMult[0] = %v, expected %v", inputMult[0], expected)
	}
}

func TestIdentityRule(t *testing.T) {
	rule := &IdentityRule{}

	outputMult := []float64{1.0, 2.0, 3.0}
	inputAct := []float64{0, 0, 0}
	inputRef := []float64{0, 0, 0}
	outputAct := []float64{0, 0, 0}
	outputRef := []float64{0, 0, 0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	for i, v := range inputMult {
		if v != outputMult[i] {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, outputMult[i])
		}
	}
}

func TestAddRule(t *testing.T) {
	rule := &AddRule{}

	outputMult := []float64{1.0, 2.0}
	inputAct := []float64{3.0, 4.0}
	inputRef := []float64{1.0, 2.0}
	outputAct := []float64{5.0, 6.0}
	outputRef := []float64{3.0, 4.0}

	inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, nil)

	// Add rule passes through multipliers
	for i, v := range inputMult {
		if v != outputMult[i] {
			t.Errorf("inputMult[%d] = %v, expected %v", i, v, outputMult[i])
		}
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		x        float64
		expected float64
	}{
		{0, 0.5},
		{-10, 0.0000453978687},
		{10, 0.9999546021},
		{-100, 0}, // Very small, effectively 0
	}

	for _, tt := range tests {
		got := sigmoid(tt.x)
		if math.Abs(got-tt.expected) > 1e-6 {
			t.Errorf("sigmoid(%v) = %v, expected %v", tt.x, got, tt.expected)
		}
	}
}
