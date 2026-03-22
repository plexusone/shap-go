package deepshap

import (
	"math"

	"github.com/plexusone/shap-go/model/onnx"
)

// AttributionRule computes attribution multipliers for a layer type.
// Given output multipliers, it computes input multipliers.
type AttributionRule interface {
	// Apply computes input multipliers given output multipliers.
	//
	// Parameters:
	//   - outputMult: multipliers from the layer above (already computed)
	//   - inputAct: input activations for the current instance
	//   - inputRef: input activations for the reference (baseline)
	//   - outputAct: output activations for the current instance
	//   - outputRef: output activations for the reference (baseline)
	//   - weights: layer weights (nil if not applicable)
	//
	// Returns input multipliers that propagate attribution to the layer's inputs.
	Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64
}

// RuleFactory returns the appropriate AttributionRule for a layer type.
func RuleFactory(layerType onnx.LayerType) AttributionRule {
	switch layerType {
	case onnx.LayerTypeDense:
		return &LinearRule{}
	case onnx.LayerTypeReLU:
		return &ReLURescaleRule{}
	case onnx.LayerTypeSigmoid:
		return &SigmoidRule{}
	case onnx.LayerTypeTanh:
		return &TanhRule{}
	case onnx.LayerTypeSoftmax:
		return &SoftmaxRule{}
	case onnx.LayerTypeAdd:
		return &AddRule{}
	case onnx.LayerTypeIdentity:
		return &IdentityRule{}
	default:
		return &IdentityRule{} // Fall back to pass-through for unknown layers
	}
}

// LinearRule implements attribution for dense/linear layers.
// For a linear layer y = Wx + b, the attribution rule is:
//
//	mult_in[i] = sum_j(W[i,j] * mult_out[j])
//
// This distributes the output multipliers back through the weight matrix.
type LinearRule struct{}

// Apply implements AttributionRule for linear layers.
func (r *LinearRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	if weights == nil || len(weights) == 0 {
		// If no weights available, use identity-like behavior
		return outputMult
	}

	numInputs := len(weights)
	inputMult := make([]float64, numInputs)

	for i := 0; i < numInputs; i++ {
		for j := 0; j < len(outputMult) && j < len(weights[i]); j++ {
			inputMult[i] += weights[i][j] * outputMult[j]
		}
	}

	return inputMult
}

// ReLURescaleRule implements the DeepLIFT rescale rule for ReLU activations.
// The rescale rule computes:
//
//	mult_in = mult_out * (x - x_ref) / (y - y_ref)
//
// When y == y_ref (division by zero), we fall back to the gradient:
//
//	mult_in = mult_out * 1(x > 0)
//
// where 1(x > 0) is 1 if x > 0, 0.5 if x == 0, 0 otherwise.
type ReLURescaleRule struct{}

// Apply implements AttributionRule for ReLU layers.
func (r *ReLURescaleRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	inputMult := make([]float64, len(inputAct))

	for i := range inputAct {
		outputDiff := outputAct[i] - outputRef[i]
		inputDiff := inputAct[i] - inputRef[i]

		if math.Abs(outputDiff) > 1e-10 {
			// Normal rescale rule
			inputMult[i] = outputMult[i] * inputDiff / outputDiff
		} else {
			// Fall back to gradient (derivative of ReLU)
			if inputAct[i] > 0 {
				inputMult[i] = outputMult[i]
			} else if inputAct[i] == 0 {
				inputMult[i] = outputMult[i] * 0.5
			} else {
				inputMult[i] = 0
			}
		}
	}

	return inputMult
}

// SigmoidRule implements the DeepLIFT rescale rule for sigmoid activations.
// Similar to ReLU, but using sigmoid's properties:
//
//	mult_in = mult_out * (x - x_ref) / (y - y_ref)
//
// With gradient fallback: mult_in = mult_out * sigmoid(x) * (1 - sigmoid(x))
type SigmoidRule struct{}

// Apply implements AttributionRule for sigmoid layers.
func (r *SigmoidRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	inputMult := make([]float64, len(inputAct))

	for i := range inputAct {
		outputDiff := outputAct[i] - outputRef[i]
		inputDiff := inputAct[i] - inputRef[i]

		if math.Abs(outputDiff) > 1e-10 {
			// Normal rescale rule
			inputMult[i] = outputMult[i] * inputDiff / outputDiff
		} else {
			// Fall back to sigmoid derivative: sig(x) * (1 - sig(x))
			sig := sigmoid(inputAct[i])
			inputMult[i] = outputMult[i] * sig * (1 - sig)
		}
	}

	return inputMult
}

// TanhRule implements the DeepLIFT rescale rule for tanh activations.
// Similar to sigmoid, but with tanh's derivative fallback:
//
//	mult_in = mult_out * (1 - tanh(x)^2)
type TanhRule struct{}

// Apply implements AttributionRule for tanh layers.
func (r *TanhRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	inputMult := make([]float64, len(inputAct))

	for i := range inputAct {
		outputDiff := outputAct[i] - outputRef[i]
		inputDiff := inputAct[i] - inputRef[i]

		if math.Abs(outputDiff) > 1e-10 {
			// Normal rescale rule
			inputMult[i] = outputMult[i] * inputDiff / outputDiff
		} else {
			// Fall back to tanh derivative: 1 - tanh(x)^2
			tanhX := math.Tanh(inputAct[i])
			inputMult[i] = outputMult[i] * (1 - tanhX*tanhX)
		}
	}

	return inputMult
}

// SoftmaxRule implements attribution for softmax layers.
// For DeepSHAP, we attribute to the pre-softmax logits using the rescale rule.
// Since softmax is typically the final layer, we use a simplified approach
// that preserves the total attribution.
type SoftmaxRule struct{}

// Apply implements AttributionRule for softmax layers.
func (r *SoftmaxRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	inputMult := make([]float64, len(inputAct))

	for i := range inputAct {
		outputDiff := outputAct[i] - outputRef[i]
		inputDiff := inputAct[i] - inputRef[i]

		if math.Abs(outputDiff) > 1e-10 {
			// Rescale rule
			inputMult[i] = outputMult[i] * inputDiff / outputDiff
		} else {
			// For softmax, use the difference directly when outputs are similar
			inputMult[i] = outputMult[i]
		}
	}

	return inputMult
}

// AddRule implements attribution for element-wise addition layers.
// For residual connections (y = x1 + x2), we split attribution proportionally
// based on the contribution of each input to the total difference.
//
// If both inputs changed by the same amount, each gets 50% of the attribution.
type AddRule struct{}

// Apply implements AttributionRule for add layers.
// For add layers, we assume the inputs are concatenated or paired.
// This implementation handles the simple case of two equal-sized inputs.
func (r *AddRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	// For simple element-wise add, each input gets the full multiplier
	// (attributions will naturally split based on input differences)
	inputMult := make([]float64, len(inputAct))
	copy(inputMult, outputMult)
	return inputMult
}

// IdentityRule implements attribution for identity/pass-through layers.
// Layers like Dropout (during inference), Flatten, Reshape, etc.
// simply pass through the multipliers unchanged.
type IdentityRule struct{}

// Apply implements AttributionRule for identity layers.
func (r *IdentityRule) Apply(outputMult, inputAct, inputRef, outputAct, outputRef []float64, weights [][]float64) []float64 {
	// Direct pass-through
	inputMult := make([]float64, len(outputMult))
	copy(inputMult, outputMult)
	return inputMult
}

// sigmoid computes the sigmoid function: 1 / (1 + exp(-x))
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// For numerical stability with negative x
	expX := math.Exp(x)
	return expX / (1.0 + expX)
}
