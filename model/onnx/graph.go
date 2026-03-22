package onnx

import (
	"fmt"
	"os"

	pb "github.com/advancedclimatesystems/gonnx/onnx"
	"google.golang.org/protobuf/proto"
)

// ParseGraph parses an ONNX model file and returns graph information.
func ParseGraph(modelPath string) (*GraphInfo, error) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}
	return ParseGraphFromBytes(data)
}

// ParseGraphFromBytes parses an ONNX model from bytes and returns graph information.
func ParseGraphFromBytes(data []byte) (*GraphInfo, error) {
	model := &pb.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}

	graph := model.GetGraph()
	if graph == nil {
		return nil, fmt.Errorf("ONNX model has no graph")
	}

	return parseGraphProto(graph)
}

// parseGraphProto converts an ONNX GraphProto to our GraphInfo structure.
func parseGraphProto(graph *pb.GraphProto) (*GraphInfo, error) {
	nodes := make([]NodeInfo, 0, len(graph.GetNode()))
	nodeByOutput := make(map[string]string) // output name -> node name

	// Parse nodes
	for i, node := range graph.GetNode() {
		nodeName := node.GetName()
		if nodeName == "" {
			// Generate name if not provided
			nodeName = fmt.Sprintf("node_%d", i)
		}

		opType := node.GetOpType()
		layerType := MapONNXOpToLayerType(opType)

		// Parse attributes
		attrs := make(map[string]interface{})
		for _, attr := range node.GetAttribute() {
			attrName := attr.GetName()
			switch attr.GetType() {
			case pb.AttributeProto_FLOAT:
				attrs[attrName] = attr.GetF()
			case pb.AttributeProto_INT:
				attrs[attrName] = attr.GetI()
			case pb.AttributeProto_STRING:
				attrs[attrName] = string(attr.GetS())
			case pb.AttributeProto_FLOATS:
				attrs[attrName] = attr.GetFloats()
			case pb.AttributeProto_INTS:
				attrs[attrName] = attr.GetInts()
			}
		}

		nodeInfo := NodeInfo{
			Name:       nodeName,
			OpType:     opType,
			LayerType:  layerType,
			Inputs:     node.GetInput(),
			Outputs:    node.GetOutput(),
			Attributes: attrs,
		}
		nodes = append(nodes, nodeInfo)

		// Map outputs to node
		for _, output := range node.GetOutput() {
			nodeByOutput[output] = nodeName
		}
	}

	// Build topological order (nodes are already in topological order in ONNX)
	topOrder := make([]string, len(nodes))
	for i, node := range nodes {
		topOrder[i] = node.Name
	}

	// Extract input names
	inputNames := make([]string, 0, len(graph.GetInput()))
	for _, input := range graph.GetInput() {
		inputNames = append(inputNames, input.GetName())
	}

	// Extract output names
	outputNames := make([]string, 0, len(graph.GetOutput()))
	for _, output := range graph.GetOutput() {
		outputNames = append(outputNames, output.GetName())
	}

	// Map initializers
	initializers := make(map[string]int)
	for i, init := range graph.GetInitializer() {
		initializers[init.GetName()] = i
	}

	return &GraphInfo{
		Nodes:            nodes,
		TopologicalOrder: topOrder,
		InputNames:       inputNames,
		OutputNames:      outputNames,
		Initializers:     initializers,
	}, nil
}

// GetLayerOutputs returns the output tensor names for all layers of a given type.
func (g *GraphInfo) GetLayerOutputs(layerType LayerType) []string {
	var outputs []string
	for _, node := range g.Nodes {
		if node.LayerType == layerType {
			outputs = append(outputs, node.Outputs...)
		}
	}
	return outputs
}

// GetAllLayerOutputs returns all intermediate tensor names in the graph.
func (g *GraphInfo) GetAllLayerOutputs() []string {
	var outputs []string
	for _, node := range g.Nodes {
		outputs = append(outputs, node.Outputs...)
	}
	return outputs
}

// GetNodeInputTypes returns the layer types of nodes that produce inputs to the given node.
func (g *GraphInfo) GetNodeInputTypes(nodeName string) []LayerType {
	node := g.GetNode(nodeName)
	if node == nil {
		return nil
	}

	var types []LayerType
	for _, inputName := range node.Inputs {
		// Check if input is from another node
		if producer := g.GetNodeByOutput(inputName); producer != nil {
			types = append(types, producer.LayerType)
		}
	}
	return types
}
