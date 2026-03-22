package onnx

import (
	"testing"

	pb "github.com/advancedclimatesystems/gonnx/onnx"
	"google.golang.org/protobuf/proto"
)

func TestMapONNXOpToLayerType(t *testing.T) {
	tests := []struct {
		opType   string
		expected LayerType
	}{
		{"Gemm", LayerTypeDense},
		{"MatMul", LayerTypeDense},
		{"Relu", LayerTypeReLU},
		{"Sigmoid", LayerTypeSigmoid},
		{"Tanh", LayerTypeTanh},
		{"Softmax", LayerTypeSoftmax},
		{"Add", LayerTypeAdd},
		{"Identity", LayerTypeIdentity},
		{"Dropout", LayerTypeIdentity},
		{"Flatten", LayerTypeIdentity},
		{"Reshape", LayerTypeIdentity},
		{"Transpose", LayerTypeIdentity},
		{"Unknown", LayerTypeUnknown},
		{"Conv2D", LayerTypeUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.opType, func(t *testing.T) {
			got := MapONNXOpToLayerType(tt.opType)
			if got != tt.expected {
				t.Errorf("MapONNXOpToLayerType(%q) = %q, want %q", tt.opType, got, tt.expected)
			}
		})
	}
}

func TestGraphInfo_ReverseTopologicalOrder(t *testing.T) {
	g := &GraphInfo{
		TopologicalOrder: []string{"input", "layer1", "layer2", "output"},
	}

	reversed := g.ReverseTopologicalOrder()
	expected := []string{"output", "layer2", "layer1", "input"}

	if len(reversed) != len(expected) {
		t.Fatalf("ReverseTopologicalOrder() length = %d, want %d", len(reversed), len(expected))
	}

	for i, name := range reversed {
		if name != expected[i] {
			t.Errorf("ReverseTopologicalOrder()[%d] = %q, want %q", i, name, expected[i])
		}
	}
}

func TestGraphInfo_GetNode(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "node1", OpType: "Relu", LayerType: LayerTypeReLU},
			{Name: "node2", OpType: "Gemm", LayerType: LayerTypeDense},
		},
	}

	// Test finding existing node
	node := g.GetNode("node1")
	if node == nil {
		t.Fatal("GetNode(\"node1\") returned nil")
	}
	if node.OpType != "Relu" {
		t.Errorf("GetNode(\"node1\").OpType = %q, want \"Relu\"", node.OpType)
	}

	// Test finding non-existing node
	node = g.GetNode("nonexistent")
	if node != nil {
		t.Error("GetNode(\"nonexistent\") should return nil")
	}
}

func TestGraphInfo_GetNodeByOutput(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "node1", OpType: "Relu", Outputs: []string{"relu_out"}},
			{Name: "node2", OpType: "Gemm", Outputs: []string{"gemm_out"}},
		},
	}

	// Test finding node by output
	node := g.GetNodeByOutput("relu_out")
	if node == nil {
		t.Fatal("GetNodeByOutput(\"relu_out\") returned nil")
	}
	if node.Name != "node1" {
		t.Errorf("GetNodeByOutput(\"relu_out\").Name = %q, want \"node1\"", node.Name)
	}

	// Test finding non-existing output
	node = g.GetNodeByOutput("nonexistent")
	if node != nil {
		t.Error("GetNodeByOutput(\"nonexistent\") should return nil")
	}
}

func TestGraphInfo_GetLayerOutputs(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "node1", LayerType: LayerTypeReLU, Outputs: []string{"relu1_out"}},
			{Name: "node2", LayerType: LayerTypeDense, Outputs: []string{"dense_out"}},
			{Name: "node3", LayerType: LayerTypeReLU, Outputs: []string{"relu2_out"}},
		},
	}

	outputs := g.GetLayerOutputs(LayerTypeReLU)
	if len(outputs) != 2 {
		t.Fatalf("GetLayerOutputs(ReLU) returned %d outputs, want 2", len(outputs))
	}

	if outputs[0] != "relu1_out" || outputs[1] != "relu2_out" {
		t.Errorf("GetLayerOutputs(ReLU) = %v, want [relu1_out, relu2_out]", outputs)
	}
}

func TestGraphInfo_GetAllLayerOutputs(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "node1", Outputs: []string{"out1"}},
			{Name: "node2", Outputs: []string{"out2", "out3"}},
		},
	}

	outputs := g.GetAllLayerOutputs()
	if len(outputs) != 3 {
		t.Fatalf("GetAllLayerOutputs() returned %d outputs, want 3", len(outputs))
	}
}

func TestGraphInfo_GetNodeInputTypes(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "dense1", LayerType: LayerTypeDense, Outputs: []string{"dense1_out"}},
			{Name: "relu1", LayerType: LayerTypeReLU, Inputs: []string{"dense1_out"}, Outputs: []string{"relu1_out"}},
			{Name: "dense2", LayerType: LayerTypeDense, Inputs: []string{"relu1_out"}, Outputs: []string{"dense2_out"}},
			{Name: "output", LayerType: LayerTypeSoftmax, Inputs: []string{"dense2_out"}, Outputs: []string{"output"}},
		},
	}

	// Test node with single input from another node
	types := g.GetNodeInputTypes("relu1")
	if len(types) != 1 {
		t.Fatalf("GetNodeInputTypes(relu1) returned %d types, want 1", len(types))
	}
	if types[0] != LayerTypeDense {
		t.Errorf("GetNodeInputTypes(relu1)[0] = %v, want %v", types[0], LayerTypeDense)
	}

	// Test node with input from ReLU
	types = g.GetNodeInputTypes("dense2")
	if len(types) != 1 {
		t.Fatalf("GetNodeInputTypes(dense2) returned %d types, want 1", len(types))
	}
	if types[0] != LayerTypeReLU {
		t.Errorf("GetNodeInputTypes(dense2)[0] = %v, want %v", types[0], LayerTypeReLU)
	}

	// Test node that doesn't exist
	types = g.GetNodeInputTypes("nonexistent")
	if types != nil {
		t.Errorf("GetNodeInputTypes(nonexistent) should return nil, got %v", types)
	}

	// Test first node (no inputs from other nodes)
	types = g.GetNodeInputTypes("dense1")
	if len(types) != 0 {
		t.Errorf("GetNodeInputTypes(dense1) should return empty, got %v", types)
	}
}

func TestGraphInfo_GetNodeInputTypes_MultipleInputs(t *testing.T) {
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "branch1", LayerType: LayerTypeDense, Outputs: []string{"b1_out"}},
			{Name: "branch2", LayerType: LayerTypeReLU, Outputs: []string{"b2_out"}},
			{Name: "merge", LayerType: LayerTypeAdd, Inputs: []string{"b1_out", "b2_out"}, Outputs: []string{"merged"}},
		},
	}

	types := g.GetNodeInputTypes("merge")
	if len(types) != 2 {
		t.Fatalf("GetNodeInputTypes(merge) returned %d types, want 2", len(types))
	}

	// Check both input types are captured
	haseDense := false
	hasRelu := false
	for _, lt := range types {
		if lt == LayerTypeDense {
			haseDense = true
		}
		if lt == LayerTypeReLU {
			hasRelu = true
		}
	}
	if !haseDense || !hasRelu {
		t.Errorf("GetNodeInputTypes(merge) = %v, expected both Dense and ReLU", types)
	}
}

func TestGraphInfo_GetNodeInputTypes_InputFromInitializer(t *testing.T) {
	// Test case where inputs include initializers (weights/biases), not other nodes
	g := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "dense1", LayerType: LayerTypeDense, Inputs: []string{"input", "weight", "bias"}, Outputs: []string{"dense_out"}},
		},
	}

	// Inputs that don't come from other nodes should not be included
	types := g.GetNodeInputTypes("dense1")
	if len(types) != 0 {
		t.Errorf("GetNodeInputTypes(dense1) should return empty for initializer inputs, got %v", types)
	}
}

func TestParseGraphFromBytes_InvalidData(t *testing.T) {
	// Test with invalid protobuf data
	invalidData := []byte("not a valid protobuf")
	_, err := ParseGraphFromBytes(invalidData)
	if err == nil {
		t.Error("ParseGraphFromBytes with invalid data should return error")
	}
}

func TestParseGraph_NonexistentFile(t *testing.T) {
	_, err := ParseGraph("/nonexistent/path/model.onnx")
	if err == nil {
		t.Error("ParseGraph with nonexistent file should return error")
	}
}

// createTestONNXModel creates a minimal valid ONNX model for testing.
func createTestONNXModel(t *testing.T) []byte {
	t.Helper()

	// Create a simple model: input -> Gemm (dense) -> Relu -> output
	model := &pb.ModelProto{
		IrVersion: 7,
		Graph: &pb.GraphProto{
			Name: "test_graph",
			Input: []*pb.ValueInfoProto{
				{Name: "input"},
			},
			Output: []*pb.ValueInfoProto{
				{Name: "output"},
			},
			Node: []*pb.NodeProto{
				{
					Name:   "dense1",
					OpType: "Gemm",
					Input:  []string{"input", "weight", "bias"},
					Output: []string{"dense1_out"},
					Attribute: []*pb.AttributeProto{
						{Name: "alpha", Type: pb.AttributeProto_FLOAT, F: 1.0},
						{Name: "beta", Type: pb.AttributeProto_FLOAT, F: 1.0},
						{Name: "transB", Type: pb.AttributeProto_INT, I: 1},
					},
				},
				{
					Name:   "relu1",
					OpType: "Relu",
					Input:  []string{"dense1_out"},
					Output: []string{"relu1_out"},
				},
				{
					Name:   "dense2",
					OpType: "Gemm",
					Input:  []string{"relu1_out", "weight2", "bias2"},
					Output: []string{"output"},
				},
			},
			Initializer: []*pb.TensorProto{
				{Name: "weight"},
				{Name: "bias"},
				{Name: "weight2"},
				{Name: "bias2"},
			},
		},
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Failed to marshal test model: %v", err)
	}
	return data
}

func TestParseGraphFromBytes_ValidModel(t *testing.T) {
	data := createTestONNXModel(t)

	graph, err := ParseGraphFromBytes(data)
	if err != nil {
		t.Fatalf("ParseGraphFromBytes failed: %v", err)
	}

	// Check nodes were parsed
	if len(graph.Nodes) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(graph.Nodes))
	}

	// Check node names
	expectedNames := []string{"dense1", "relu1", "dense2"}
	for i, expected := range expectedNames {
		if graph.Nodes[i].Name != expected {
			t.Errorf("Node[%d].Name = %q, want %q", i, graph.Nodes[i].Name, expected)
		}
	}

	// Check layer types
	if graph.Nodes[0].LayerType != LayerTypeDense {
		t.Errorf("Node[0].LayerType = %v, want Dense", graph.Nodes[0].LayerType)
	}
	if graph.Nodes[1].LayerType != LayerTypeReLU {
		t.Errorf("Node[1].LayerType = %v, want ReLU", graph.Nodes[1].LayerType)
	}

	// Check topological order
	if len(graph.TopologicalOrder) != 3 {
		t.Errorf("TopologicalOrder length = %d, want 3", len(graph.TopologicalOrder))
	}

	// Check inputs
	if len(graph.InputNames) != 1 || graph.InputNames[0] != "input" {
		t.Errorf("InputNames = %v, want [input]", graph.InputNames)
	}

	// Check outputs
	if len(graph.OutputNames) != 1 || graph.OutputNames[0] != "output" {
		t.Errorf("OutputNames = %v, want [output]", graph.OutputNames)
	}

	// Check initializers
	if len(graph.Initializers) != 4 {
		t.Errorf("Initializers count = %d, want 4", len(graph.Initializers))
	}

	// Check GetNode works
	node := graph.GetNode("relu1")
	if node == nil {
		t.Fatal("GetNode(relu1) returned nil")
	}
	if node.OpType != "Relu" {
		t.Errorf("GetNode(relu1).OpType = %q, want Relu", node.OpType)
	}

	// Check GetNodeByOutput works
	node = graph.GetNodeByOutput("dense1_out")
	if node == nil {
		t.Fatal("GetNodeByOutput(dense1_out) returned nil")
	}
	if node.Name != "dense1" {
		t.Errorf("GetNodeByOutput(dense1_out).Name = %q, want dense1", node.Name)
	}

	// Check attributes were parsed
	if graph.Nodes[0].Attributes["alpha"] != float32(1.0) {
		t.Errorf("Node[0].Attributes[alpha] = %v, want 1.0", graph.Nodes[0].Attributes["alpha"])
	}
	if graph.Nodes[0].Attributes["transB"] != int64(1) {
		t.Errorf("Node[0].Attributes[transB] = %v, want 1", graph.Nodes[0].Attributes["transB"])
	}
}

func TestParseGraphFromBytes_EmptyGraph(t *testing.T) {
	// Model without a graph
	model := &pb.ModelProto{
		IrVersion: 7,
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}

	_, err = ParseGraphFromBytes(data)
	if err == nil {
		t.Error("ParseGraphFromBytes with no graph should return error")
	}
}

func TestParseGraphFromBytes_NodeWithoutName(t *testing.T) {
	// Model with nodes that have no names (should generate names)
	model := &pb.ModelProto{
		IrVersion: 7,
		Graph: &pb.GraphProto{
			Name: "test",
			Node: []*pb.NodeProto{
				{
					OpType: "Relu",
					Input:  []string{"input"},
					Output: []string{"output"},
					// No name specified
				},
			},
		},
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}

	graph, err := ParseGraphFromBytes(data)
	if err != nil {
		t.Fatalf("ParseGraphFromBytes failed: %v", err)
	}

	// Should generate a name like "node_0"
	if len(graph.Nodes) != 1 {
		t.Fatalf("Expected 1 node, got %d", len(graph.Nodes))
	}
	if graph.Nodes[0].Name != "node_0" {
		t.Errorf("Node without name should get generated name, got %q", graph.Nodes[0].Name)
	}
}

func TestParseGraphFromBytes_AllAttributeTypes(t *testing.T) {
	// Test parsing various attribute types
	model := &pb.ModelProto{
		IrVersion: 7,
		Graph: &pb.GraphProto{
			Name: "test",
			Node: []*pb.NodeProto{
				{
					Name:   "test_node",
					OpType: "TestOp",
					Attribute: []*pb.AttributeProto{
						{Name: "float_attr", Type: pb.AttributeProto_FLOAT, F: 3.14},
						{Name: "int_attr", Type: pb.AttributeProto_INT, I: 42},
						{Name: "string_attr", Type: pb.AttributeProto_STRING, S: []byte("hello")},
						{Name: "floats_attr", Type: pb.AttributeProto_FLOATS, Floats: []float32{1.0, 2.0, 3.0}},
						{Name: "ints_attr", Type: pb.AttributeProto_INTS, Ints: []int64{1, 2, 3}},
					},
				},
			},
		},
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Failed to marshal model: %v", err)
	}

	graph, err := ParseGraphFromBytes(data)
	if err != nil {
		t.Fatalf("ParseGraphFromBytes failed: %v", err)
	}

	attrs := graph.Nodes[0].Attributes

	// Check float attribute
	if v, ok := attrs["float_attr"].(float32); !ok || v != 3.14 {
		t.Errorf("float_attr = %v, want 3.14", attrs["float_attr"])
	}

	// Check int attribute
	if v, ok := attrs["int_attr"].(int64); !ok || v != 42 {
		t.Errorf("int_attr = %v, want 42", attrs["int_attr"])
	}

	// Check string attribute
	if v, ok := attrs["string_attr"].(string); !ok || v != "hello" {
		t.Errorf("string_attr = %v, want hello", attrs["string_attr"])
	}

	// Check floats attribute
	if floats, ok := attrs["floats_attr"].([]float32); !ok || len(floats) != 3 {
		t.Errorf("floats_attr = %v, want [1.0, 2.0, 3.0]", attrs["floats_attr"])
	}

	// Check ints attribute
	if ints, ok := attrs["ints_attr"].([]int64); !ok || len(ints) != 3 {
		t.Errorf("ints_attr = %v, want [1, 2, 3]", attrs["ints_attr"])
	}
}
