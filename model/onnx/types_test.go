package onnx

import (
	"testing"
)

// Additional edge case tests for types.go not covered in graph_test.go

func TestMapONNXOpToLayerType_EdgeCases(t *testing.T) {
	// Test additional edge cases not covered in graph_test.go
	tests := []struct {
		opType   string
		expected LayerType
	}{
		{"", LayerTypeUnknown},
		{"BatchNormalization", LayerTypeUnknown},
		{"MaxPool", LayerTypeUnknown},
		{"AveragePool", LayerTypeUnknown},
		{"Conv", LayerTypeUnknown},
		{"ConvTranspose", LayerTypeUnknown},
		{"GEMM", LayerTypeUnknown}, // Case sensitive
		{"relu", LayerTypeUnknown}, // Case sensitive
	}

	for _, tc := range tests {
		t.Run(tc.opType, func(t *testing.T) {
			got := MapONNXOpToLayerType(tc.opType)
			if got != tc.expected {
				t.Errorf("MapONNXOpToLayerType(%q) = %q, want %q", tc.opType, got, tc.expected)
			}
		})
	}
}

func TestGraphInfo_EmptyGraph(t *testing.T) {
	graph := &GraphInfo{}

	if node := graph.GetNode("any"); node != nil {
		t.Errorf("GetNode on empty graph should return nil")
	}

	if node := graph.GetNodeByOutput("any"); node != nil {
		t.Errorf("GetNodeByOutput on empty graph should return nil")
	}

	reversed := graph.ReverseTopologicalOrder()
	if len(reversed) != 0 {
		t.Errorf("ReverseTopologicalOrder on empty graph should return empty slice")
	}

	outputs := graph.GetLayerOutputs(LayerTypeReLU)
	if len(outputs) != 0 {
		t.Errorf("GetLayerOutputs on empty graph should return empty slice")
	}

	allOutputs := graph.GetAllLayerOutputs()
	if len(allOutputs) != 0 {
		t.Errorf("GetAllLayerOutputs on empty graph should return empty slice")
	}
}

func TestGraphInfo_SingleElementOrder(t *testing.T) {
	graph := &GraphInfo{
		TopologicalOrder: []string{"only_node"},
	}

	reversed := graph.ReverseTopologicalOrder()
	if len(reversed) != 1 || reversed[0] != "only_node" {
		t.Errorf("ReverseTopologicalOrder with single element = %v, want [only_node]", reversed)
	}
}

func TestNodeInfo_Fields(t *testing.T) {
	node := NodeInfo{
		Name:      "test_node",
		OpType:    "Gemm",
		LayerType: LayerTypeDense,
		Inputs:    []string{"input1", "weight", "bias"},
		Outputs:   []string{"output1"},
		Attributes: map[string]interface{}{
			"alpha":  1.0,
			"beta":   1.0,
			"transB": 1,
		},
	}

	if node.Name != "test_node" {
		t.Errorf("Name = %q, want %q", node.Name, "test_node")
	}
	if node.OpType != "Gemm" {
		t.Errorf("OpType = %q, want %q", node.OpType, "Gemm")
	}
	if node.LayerType != LayerTypeDense {
		t.Errorf("LayerType = %q, want %q", node.LayerType, LayerTypeDense)
	}
	if len(node.Inputs) != 3 {
		t.Errorf("len(Inputs) = %d, want 3", len(node.Inputs))
	}
	if len(node.Outputs) != 1 {
		t.Errorf("len(Outputs) = %d, want 1", len(node.Outputs))
	}
	if node.Attributes["alpha"] != 1.0 {
		t.Errorf("Attributes[alpha] = %v, want 1.0", node.Attributes["alpha"])
	}
}

func TestLayerType_String(t *testing.T) {
	tests := []struct {
		lt       LayerType
		expected string
	}{
		{LayerTypeDense, "dense"},
		{LayerTypeReLU, "relu"},
		{LayerTypeSigmoid, "sigmoid"},
		{LayerTypeTanh, "tanh"},
		{LayerTypeSoftmax, "softmax"},
		{LayerTypeAdd, "add"},
		{LayerTypeIdentity, "identity"},
		{LayerTypeUnknown, "unknown"},
	}

	for _, tc := range tests {
		if string(tc.lt) != tc.expected {
			t.Errorf("LayerType %v string = %q, want %q", tc.lt, string(tc.lt), tc.expected)
		}
	}
}

func TestConfig_Default(t *testing.T) {
	config := DefaultConfig()

	if config.InputName != "float_input" {
		t.Errorf("DefaultConfig().InputName = %q, want %q", config.InputName, "float_input")
	}
	if config.OutputName != "probabilities" {
		t.Errorf("DefaultConfig().OutputName = %q, want %q", config.OutputName, "probabilities")
	}
	if config.ModelPath != "" {
		t.Errorf("DefaultConfig().ModelPath = %q, want empty", config.ModelPath)
	}
	if config.NumFeatures != 0 {
		t.Errorf("DefaultConfig().NumFeatures = %d, want 0", config.NumFeatures)
	}
	if config.UseGPU {
		t.Errorf("DefaultConfig().UseGPU = true, want false")
	}
}

func TestGraphInfo_Initializers(t *testing.T) {
	graph := &GraphInfo{
		Initializers: map[string]int{
			"weight1": 0,
			"bias1":   1,
			"weight2": 2,
		},
	}

	if idx, ok := graph.Initializers["weight1"]; !ok || idx != 0 {
		t.Errorf("Initializers[weight1] = %d, %v, want 0, true", idx, ok)
	}
	if idx, ok := graph.Initializers["bias1"]; !ok || idx != 1 {
		t.Errorf("Initializers[bias1] = %d, %v, want 1, true", idx, ok)
	}
	if _, ok := graph.Initializers["nonexistent"]; ok {
		t.Errorf("Initializers[nonexistent] should not exist")
	}
}

func TestGraphInfo_InputOutputNames(t *testing.T) {
	graph := &GraphInfo{
		InputNames:  []string{"input", "hidden_state"},
		OutputNames: []string{"output", "new_hidden_state"},
	}

	if len(graph.InputNames) != 2 {
		t.Errorf("len(InputNames) = %d, want 2", len(graph.InputNames))
	}
	if graph.InputNames[0] != "input" {
		t.Errorf("InputNames[0] = %q, want %q", graph.InputNames[0], "input")
	}
	if len(graph.OutputNames) != 2 {
		t.Errorf("len(OutputNames) = %d, want 2", len(graph.OutputNames))
	}
	if graph.OutputNames[0] != "output" {
		t.Errorf("OutputNames[0] = %q, want %q", graph.OutputNames[0], "output")
	}
}

func TestGraphInfo_MultipleOutputsPerNode(t *testing.T) {
	graph := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "multi_output", Outputs: []string{"out1", "out2", "out3"}},
		},
	}

	// Test GetNodeByOutput finds all outputs
	for _, out := range []string{"out1", "out2", "out3"} {
		node := graph.GetNodeByOutput(out)
		if node == nil || node.Name != "multi_output" {
			t.Errorf("GetNodeByOutput(%q) should find multi_output", out)
		}
	}

	// Test GetAllLayerOutputs returns all
	outputs := graph.GetAllLayerOutputs()
	if len(outputs) != 3 {
		t.Errorf("GetAllLayerOutputs() = %d outputs, want 3", len(outputs))
	}
}

func TestGraphInfo_GetLayerOutputs_MultipleMatches(t *testing.T) {
	graph := &GraphInfo{
		Nodes: []NodeInfo{
			{Name: "relu1", LayerType: LayerTypeReLU, Outputs: []string{"r1"}},
			{Name: "dense1", LayerType: LayerTypeDense, Outputs: []string{"d1"}},
			{Name: "relu2", LayerType: LayerTypeReLU, Outputs: []string{"r2"}},
			{Name: "dense2", LayerType: LayerTypeDense, Outputs: []string{"d2"}},
			{Name: "relu3", LayerType: LayerTypeReLU, Outputs: []string{"r3"}},
		},
	}

	reluOutputs := graph.GetLayerOutputs(LayerTypeReLU)
	if len(reluOutputs) != 3 {
		t.Errorf("GetLayerOutputs(ReLU) = %d, want 3", len(reluOutputs))
	}

	denseOutputs := graph.GetLayerOutputs(LayerTypeDense)
	if len(denseOutputs) != 2 {
		t.Errorf("GetLayerOutputs(Dense) = %d, want 2", len(denseOutputs))
	}

	sigmoidOutputs := graph.GetLayerOutputs(LayerTypeSigmoid)
	if len(sigmoidOutputs) != 0 {
		t.Errorf("GetLayerOutputs(Sigmoid) = %d, want 0", len(sigmoidOutputs))
	}
}

func TestNodeInfo_EmptyFields(t *testing.T) {
	node := NodeInfo{}

	if node.Name != "" {
		t.Error("Empty NodeInfo should have empty Name")
	}
	if node.OpType != "" {
		t.Error("Empty NodeInfo should have empty OpType")
	}
	if node.LayerType != "" {
		t.Error("Empty NodeInfo should have empty LayerType")
	}
	if len(node.Inputs) != 0 {
		t.Error("Empty NodeInfo should have empty Inputs")
	}
	if len(node.Outputs) != 0 {
		t.Error("Empty NodeInfo should have empty Outputs")
	}
	if node.Attributes != nil {
		t.Error("Empty NodeInfo should have nil Attributes")
	}
}

func TestConfig_CustomValues(t *testing.T) {
	config := Config{
		ModelPath:   "/path/to/model.onnx",
		InputName:   "custom_input",
		OutputName:  "custom_output",
		NumFeatures: 10,
		UseGPU:      true,
	}

	if config.ModelPath != "/path/to/model.onnx" {
		t.Errorf("ModelPath = %q, want %q", config.ModelPath, "/path/to/model.onnx")
	}
	if config.InputName != "custom_input" {
		t.Errorf("InputName = %q, want %q", config.InputName, "custom_input")
	}
	if config.OutputName != "custom_output" {
		t.Errorf("OutputName = %q, want %q", config.OutputName, "custom_output")
	}
	if config.NumFeatures != 10 {
		t.Errorf("NumFeatures = %d, want 10", config.NumFeatures)
	}
	if !config.UseGPU {
		t.Error("UseGPU = false, want true")
	}
}
