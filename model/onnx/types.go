package onnx

// LayerType represents the type of a neural network layer.
type LayerType string

// Supported layer types for DeepSHAP attribution.
const (
	LayerTypeDense    LayerType = "dense"
	LayerTypeReLU     LayerType = "relu"
	LayerTypeSigmoid  LayerType = "sigmoid"
	LayerTypeTanh     LayerType = "tanh"
	LayerTypeSoftmax  LayerType = "softmax"
	LayerTypeAdd      LayerType = "add"
	LayerTypeIdentity LayerType = "identity"
	LayerTypeUnknown  LayerType = "unknown"
)

// NodeInfo contains information about a single node in the ONNX graph.
type NodeInfo struct {
	// Name is the unique identifier for this node.
	Name string

	// OpType is the ONNX operation type (e.g., "Gemm", "Relu", "MatMul").
	OpType string

	// LayerType is the normalized layer type for DeepSHAP attribution rules.
	LayerType LayerType

	// Inputs are the names of input tensors to this node.
	Inputs []string

	// Outputs are the names of output tensors from this node.
	Outputs []string

	// Attributes contains ONNX node attributes (e.g., alpha, beta for Gemm).
	Attributes map[string]interface{}
}

// GraphInfo contains parsed information about an ONNX model's computation graph.
type GraphInfo struct {
	// Nodes contains all nodes in the graph.
	Nodes []NodeInfo

	// TopologicalOrder contains node names in topological order.
	TopologicalOrder []string

	// InputNames are the names of the graph's input tensors.
	InputNames []string

	// OutputNames are the names of the graph's output tensors.
	OutputNames []string

	// Initializers maps initializer names to their indices (for weights/biases).
	Initializers map[string]int
}

// GetNode returns the node with the given name, or nil if not found.
func (g *GraphInfo) GetNode(name string) *NodeInfo {
	for i := range g.Nodes {
		if g.Nodes[i].Name == name {
			return &g.Nodes[i]
		}
	}
	return nil
}

// GetNodeByOutput returns the node that produces the given output tensor.
func (g *GraphInfo) GetNodeByOutput(outputName string) *NodeInfo {
	for i := range g.Nodes {
		for _, out := range g.Nodes[i].Outputs {
			if out == outputName {
				return &g.Nodes[i]
			}
		}
	}
	return nil
}

// ReverseTopologicalOrder returns nodes in reverse topological order.
// This is useful for backward propagation in DeepSHAP.
func (g *GraphInfo) ReverseTopologicalOrder() []string {
	result := make([]string, len(g.TopologicalOrder))
	for i, name := range g.TopologicalOrder {
		result[len(g.TopologicalOrder)-1-i] = name
	}
	return result
}

// MapONNXOpToLayerType converts an ONNX operation type to a normalized LayerType.
func MapONNXOpToLayerType(opType string) LayerType {
	switch opType {
	case "Gemm", "MatMul":
		return LayerTypeDense
	case "Relu":
		return LayerTypeReLU
	case "Sigmoid":
		return LayerTypeSigmoid
	case "Tanh":
		return LayerTypeTanh
	case "Softmax":
		return LayerTypeSoftmax
	case "Add":
		return LayerTypeAdd
	case "Identity", "Dropout", "Flatten", "Reshape", "Transpose":
		return LayerTypeIdentity
	default:
		return LayerTypeUnknown
	}
}
