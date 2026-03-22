package deepshap

import (
	"fmt"

	"github.com/plexusone/shap-go/model/onnx"
)

// PropagationEngine handles backward propagation of attribution multipliers.
type PropagationEngine struct {
	graphInfo *onnx.GraphInfo
	rules     map[string]AttributionRule // node name -> rule
}

// NewPropagationEngine creates a new propagation engine for the given graph.
func NewPropagationEngine(graphInfo *onnx.GraphInfo) *PropagationEngine {
	rules := make(map[string]AttributionRule)

	// Create rules for each node
	for _, node := range graphInfo.Nodes {
		rules[node.Name] = RuleFactory(node.LayerType)
	}

	return &PropagationEngine{
		graphInfo: graphInfo,
		rules:     rules,
	}
}

// ActivationData contains activations for a single forward pass.
type ActivationData struct {
	// Input is the model input.
	Input []float64

	// LayerActivations maps layer output names to activation values.
	LayerActivations map[string][]float64

	// Output is the final model output.
	Output float64
}

// PropagationResult contains the result of backward propagation.
type PropagationResult struct {
	// InputMultipliers are the attribution multipliers for each input feature.
	InputMultipliers []float64

	// Attributions are the final SHAP values (multipliers * input differences).
	Attributions []float64
}

// Propagate performs backward propagation from output to input.
//
// Parameters:
//   - instanceAct: activations for the instance being explained
//   - referenceAct: activations for the reference (baseline)
//   - outputMultiplier: initial multiplier for the output (typically 1.0)
//
// Returns attribution multipliers for each input feature.
func (pe *PropagationEngine) Propagate(
	instanceAct *ActivationData,
	referenceAct *ActivationData,
	outputMultiplier float64,
) (*PropagationResult, error) {
	// Maps tensor name to its current multipliers
	tensorMultipliers := make(map[string][]float64)

	// Initialize output multipliers
	// We need to find the output tensor of the last node
	if len(pe.graphInfo.OutputNames) == 0 {
		return nil, fmt.Errorf("graph has no output names")
	}

	// Get the output node
	outputName := pe.graphInfo.OutputNames[0]
	outputNode := pe.graphInfo.GetNodeByOutput(outputName)
	if outputNode != nil {
		// Initialize multipliers for the output
		outAct := instanceAct.LayerActivations[outputName]
		if outAct == nil {
			// Fall back to scalar output
			tensorMultipliers[outputName] = []float64{outputMultiplier}
		} else {
			// Initialize all outputs with the same multiplier
			mults := make([]float64, len(outAct))
			for i := range mults {
				mults[i] = outputMultiplier
			}
			tensorMultipliers[outputName] = mults
		}
	} else {
		// Output is an input tensor (no processing nodes)
		tensorMultipliers[outputName] = []float64{outputMultiplier}
	}

	// Traverse nodes in reverse topological order
	reverseOrder := pe.graphInfo.ReverseTopologicalOrder()

	for _, nodeName := range reverseOrder {
		node := pe.graphInfo.GetNode(nodeName)
		if node == nil {
			continue
		}

		rule := pe.rules[nodeName]
		if rule == nil {
			continue
		}

		// Get output multipliers for this node
		var outputMult []float64
		for _, outName := range node.Outputs {
			if mults, ok := tensorMultipliers[outName]; ok {
				outputMult = mults
				break
			}
		}

		if outputMult == nil {
			// No multipliers to propagate
			continue
		}

		// Get activations for this node
		var outputAct, outputRef []float64
		for _, outName := range node.Outputs {
			if act, ok := instanceAct.LayerActivations[outName]; ok {
				outputAct = act
				outputRef = referenceAct.LayerActivations[outName]
				break
			}
		}

		// Get input activations
		var inputAct, inputRef []float64
		var inputName string
		for _, inName := range node.Inputs {
			// Skip initializers (weights/biases)
			if _, isInit := pe.graphInfo.Initializers[inName]; isInit {
				continue
			}

			inputName = inName
			if act, ok := instanceAct.LayerActivations[inName]; ok {
				inputAct = act
				inputRef = referenceAct.LayerActivations[inName]
				break
			}
		}

		// Check if input is the model input
		if inputAct == nil {
			for _, graphInput := range pe.graphInfo.InputNames {
				for _, nodeInput := range node.Inputs {
					if nodeInput == graphInput {
						inputAct = instanceAct.Input
						inputRef = referenceAct.Input
						inputName = graphInput
						break
					}
				}
				if inputAct != nil {
					break
				}
			}
		}

		if inputAct == nil || outputAct == nil {
			// Can't propagate without activations
			// Pass through the multipliers to inputs
			for _, inName := range node.Inputs {
				if _, isInit := pe.graphInfo.Initializers[inName]; !isInit {
					tensorMultipliers[inName] = outputMult
				}
			}
			continue
		}

		// Get weights if applicable (for linear layers)
		var weights [][]float64
		if node.LayerType == onnx.LayerTypeDense {
			weights = pe.extractWeights(node)
		}

		// Apply the attribution rule
		inputMult := rule.Apply(outputMult, inputAct, inputRef, outputAct, outputRef, weights)

		// Store the input multipliers
		if inputName != "" {
			tensorMultipliers[inputName] = inputMult
		}
	}

	// Extract multipliers for the model inputs
	var inputMultipliers []float64
	for _, inputName := range pe.graphInfo.InputNames {
		if mults, ok := tensorMultipliers[inputName]; ok {
			inputMultipliers = mults
			break
		}
	}

	if inputMultipliers == nil {
		// Fall back to the first tensor that connects to input
		inputMultipliers = make([]float64, len(instanceAct.Input))
		for i := range inputMultipliers {
			inputMultipliers[i] = 1.0
		}
	}

	// Compute final attributions: mult * (x - x_ref)
	attributions := make([]float64, len(instanceAct.Input))
	for i := range attributions {
		diff := instanceAct.Input[i] - referenceAct.Input[i]
		if i < len(inputMultipliers) {
			attributions[i] = inputMultipliers[i] * diff
		}
	}

	return &PropagationResult{
		InputMultipliers: inputMultipliers,
		Attributions:     attributions,
	}, nil
}

// extractWeights extracts weight matrix from a dense layer node.
// This is a placeholder - actual implementation depends on how weights are stored.
func (pe *PropagationEngine) extractWeights(node *onnx.NodeInfo) [][]float64 {
	// In a real implementation, we would:
	// 1. Find the weight initializer by name
	// 2. Parse the tensor data
	// 3. Reshape to 2D matrix

	// For now, return nil to use fallback behavior in LinearRule
	// This will be enhanced when we integrate with weight extraction
	return nil
}

// PropagateSimple performs a simplified backward propagation for networks
// where we have activations but not full graph structure.
//
// This uses a chain of rescale rules from output to input.
func PropagateSimple(
	instanceInput, referenceInput []float64,
	instanceActivations, referenceActivations [][]float64,
	instanceOutput, referenceOutput float64,
) []float64 {
	numLayers := len(instanceActivations)
	numFeatures := len(instanceInput)

	// Start with output multiplier
	outputDiff := instanceOutput - referenceOutput

	// Initialize attributions
	attributions := make([]float64, numFeatures)

	// If no activations or simple direct mapping
	if numLayers == 0 {
		// Direct rescale from input to output
		for i := range attributions {
			inputDiff := instanceInput[i] - referenceInput[i]
			if inputDiff != 0 && outputDiff != 0 {
				attributions[i] = inputDiff * (outputDiff / totalInputDiff(instanceInput, referenceInput))
			}
		}
		return attributions
	}

	// Work backward through layers
	currentMult := make([]float64, len(instanceActivations[numLayers-1]))
	for i := range currentMult {
		currentMult[i] = 1.0
	}

	for layer := numLayers - 1; layer >= 0; layer-- {
		instAct := instanceActivations[layer]
		refAct := referenceActivations[layer]

		var prevInstAct, prevRefAct []float64
		if layer > 0 {
			prevInstAct = instanceActivations[layer-1]
			prevRefAct = referenceActivations[layer-1]
		} else {
			prevInstAct = instanceInput
			prevRefAct = referenceInput
		}

		// Apply rescale rule
		newMult := make([]float64, len(prevInstAct))
		for i := range newMult {
			// Distribute multipliers proportionally
			var contribution float64
			for j := range instAct {
				outDiff := instAct[j] - refAct[j]
				inDiff := prevInstAct[i] - prevRefAct[i]
				if outDiff != 0 {
					contribution += currentMult[j] * inDiff / outDiff
				}
			}
			newMult[i] = contribution / float64(len(instAct))
		}
		currentMult = newMult
	}

	// Final attributions
	for i := range attributions {
		diff := instanceInput[i] - referenceInput[i]
		if i < len(currentMult) {
			attributions[i] = currentMult[i] * diff
		}
	}

	return attributions
}

func totalInputDiff(inst, ref []float64) float64 {
	var total float64
	for i := range inst {
		total += inst[i] - ref[i]
	}
	if total == 0 {
		return 1 // Avoid division by zero
	}
	return total
}
