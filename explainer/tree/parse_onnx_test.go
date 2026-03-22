package tree

import (
	"testing"

	pb "github.com/advancedclimatesystems/gonnx/onnx"
	"google.golang.org/protobuf/proto"
)

// createTestONNXModel creates a simple ONNX model with a TreeEnsembleRegressor.
// Tree structure: single tree with one split
//
//	  0 (feature 0, threshold 0.5)
//	 / \
//	1   2 (leaves with values 1.0 and 2.0)
func createTestONNXModel() []byte {
	// Create TreeEnsembleRegressor node with attributes
	node := &pb.NodeProto{
		Name:   "tree_ensemble",
		OpType: "TreeEnsembleRegressor",
		Input:  []string{"X"},
		Output: []string{"Y"},
		Attribute: []*pb.AttributeProto{
			{
				Name: "nodes_treeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0, 0}, // All nodes in tree 0
			},
			{
				Name: "nodes_nodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 1, 2}, // Node IDs within tree
			},
			{
				Name: "nodes_featureids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0, 0}, // Feature 0 for split, 0 for leaves (ignored)
			},
			{
				Name:   "nodes_values",
				Type:   pb.AttributeProto_FLOATS,
				Floats: []float32{0.5, 0.0, 0.0}, // Threshold for split, 0 for leaves
			},
			{
				Name:    "nodes_modes",
				Type:    pb.AttributeProto_STRINGS,
				Strings: [][]byte{[]byte("BRANCH_LEQ"), []byte("LEAF"), []byte("LEAF")},
			},
			{
				Name: "nodes_truenodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{1, 0, 0}, // True child is node 1
			},
			{
				Name: "nodes_falsenodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{2, 0, 0}, // False child is node 2
			},
			{
				Name: "target_nodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{1, 2}, // Leaf node IDs
			},
			{
				Name: "target_treeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0}, // Both leaves in tree 0
			},
			{
				Name:   "target_weights",
				Type:   pb.AttributeProto_FLOATS,
				Floats: []float32{1.0, 2.0}, // Leaf values
			},
			{
				Name: "n_targets",
				Type: pb.AttributeProto_INT,
				I:    1,
			},
		},
	}

	// Create graph
	graph := &pb.GraphProto{
		Name: "test_tree",
		Node: []*pb.NodeProto{node},
		Input: []*pb.ValueInfoProto{
			{
				Name: "X",
				Type: &pb.TypeProto{
					Value: &pb.TypeProto_TensorType{
						TensorType: &pb.TypeProto_Tensor{
							ElemType: int32(pb.TensorProto_FLOAT),
							Shape: &pb.TensorShapeProto{
								Dim: []*pb.TensorShapeProto_Dimension{
									{Value: &pb.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
									{Value: &pb.TensorShapeProto_Dimension_DimValue{DimValue: 2}},
								},
							},
						},
					},
				},
			},
		},
		Output: []*pb.ValueInfoProto{
			{
				Name: "Y",
			},
		},
	}

	// Create model
	model := &pb.ModelProto{
		IrVersion: 7,
		Graph:     graph,
	}

	data, _ := proto.Marshal(model)
	return data
}

func TestParseONNXTreeEnsembleFromBytes_Simple(t *testing.T) {
	data := createTestONNXModel()

	ensemble, err := ParseONNXTreeEnsembleFromBytes(data)
	if err != nil {
		t.Fatalf("ParseONNXTreeEnsembleFromBytes failed: %v", err)
	}

	// Check ensemble properties
	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
	if ensemble.NumFeatures != 2 {
		t.Errorf("NumFeatures = %d, want 2", ensemble.NumFeatures)
	}

	// Should have 3 nodes (1 split + 2 leaves)
	if len(ensemble.Nodes) != 3 {
		t.Errorf("len(Nodes) = %d, want 3", len(ensemble.Nodes))
	}

	// Check root node
	root := &ensemble.Nodes[0]
	if root.IsLeaf {
		t.Error("Root node should not be a leaf")
	}
	if root.Feature != 0 {
		t.Errorf("Root.Feature = %d, want 0", root.Feature)
	}
	if root.Threshold != 0.5 {
		t.Errorf("Root.Threshold = %v, want 0.5", root.Threshold)
	}
	if root.DecisionType != DecisionLessEqual {
		t.Errorf("Root.DecisionType = %q, want %q", root.DecisionType, DecisionLessEqual)
	}

	// Check leaves
	leaf1 := &ensemble.Nodes[1]
	if !leaf1.IsLeaf {
		t.Error("Node 1 should be a leaf")
	}
	if leaf1.Prediction != 1.0 {
		t.Errorf("Leaf1.Prediction = %v, want 1.0", leaf1.Prediction)
	}

	leaf2 := &ensemble.Nodes[2]
	if !leaf2.IsLeaf {
		t.Error("Node 2 should be a leaf")
	}
	if leaf2.Prediction != 2.0 {
		t.Errorf("Leaf2.Prediction = %v, want 2.0", leaf2.Prediction)
	}
}

func TestParseONNXTreeEnsembleFromBytes_NoTreeEnsemble(t *testing.T) {
	// Create model without tree ensemble
	node := &pb.NodeProto{
		Name:   "identity",
		OpType: "Identity",
		Input:  []string{"X"},
		Output: []string{"Y"},
	}

	graph := &pb.GraphProto{
		Name: "test",
		Node: []*pb.NodeProto{node},
	}

	model := &pb.ModelProto{
		IrVersion: 7,
		Graph:     graph,
	}

	data, _ := proto.Marshal(model)

	_, err := ParseONNXTreeEnsembleFromBytes(data)
	if err == nil {
		t.Error("ParseONNXTreeEnsembleFromBytes should fail without tree ensemble")
	}
}

func TestParseONNXTreeEnsembleFromBytes_InvalidProtobuf(t *testing.T) {
	_, err := ParseONNXTreeEnsembleFromBytes([]byte("not valid protobuf"))
	if err == nil {
		t.Error("ParseONNXTreeEnsembleFromBytes should fail with invalid protobuf")
	}
}

func TestParseONNXTreeEnsembleFromBytes_NoGraph(t *testing.T) {
	model := &pb.ModelProto{
		IrVersion: 7,
		// No graph
	}

	data, _ := proto.Marshal(model)

	_, err := ParseONNXTreeEnsembleFromBytes(data)
	if err == nil {
		t.Error("ParseONNXTreeEnsembleFromBytes should fail without graph")
	}
}

func TestConvertONNXTreeEnsemble_Empty(t *testing.T) {
	onnx := &ONNXTreeEnsemble{
		NodesTreeIDs: []int64{}, // Empty
	}

	_, err := convertONNXTreeEnsemble(onnx, "TreeEnsembleRegressor", 1)
	if err == nil {
		t.Error("convertONNXTreeEnsemble should fail with empty ensemble")
	}
}

func TestBytesToStrings(t *testing.T) {
	input := [][]byte{
		[]byte("hello"),
		[]byte("world"),
	}
	result := bytesToStrings(input)

	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
	if result[0] != "hello" {
		t.Errorf("result[0] = %q, want %q", result[0], "hello")
	}
	if result[1] != "world" {
		t.Errorf("result[1] = %q, want %q", result[1], "world")
	}
}

// createMultiTreeONNXModel creates a model with 2 trees.
func createMultiTreeONNXModel() []byte {
	// Tree 0: single split at feature 0
	// Tree 1: single split at feature 1
	node := &pb.NodeProto{
		Name:   "tree_ensemble",
		OpType: "TreeEnsembleRegressor",
		Input:  []string{"X"},
		Output: []string{"Y"},
		Attribute: []*pb.AttributeProto{
			{
				Name: "nodes_treeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0, 0, 1, 1, 1}, // 3 nodes each in trees 0 and 1
			},
			{
				Name: "nodes_nodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 1, 2, 0, 1, 2},
			},
			{
				Name: "nodes_featureids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0, 0, 1, 0, 0}, // Tree 0 splits on f0, tree 1 on f1
			},
			{
				Name:   "nodes_values",
				Type:   pb.AttributeProto_FLOATS,
				Floats: []float32{0.5, 0.0, 0.0, 0.3, 0.0, 0.0},
			},
			{
				Name:    "nodes_modes",
				Type:    pb.AttributeProto_STRINGS,
				Strings: [][]byte{[]byte("BRANCH_LEQ"), []byte("LEAF"), []byte("LEAF"), []byte("BRANCH_LEQ"), []byte("LEAF"), []byte("LEAF")},
			},
			{
				Name: "nodes_truenodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{1, 0, 0, 1, 0, 0},
			},
			{
				Name: "nodes_falsenodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{2, 0, 0, 2, 0, 0},
			},
			{
				Name: "target_nodeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{1, 2, 1, 2},
			},
			{
				Name: "target_treeids",
				Type: pb.AttributeProto_INTS,
				Ints: []int64{0, 0, 1, 1},
			},
			{
				Name:   "target_weights",
				Type:   pb.AttributeProto_FLOATS,
				Floats: []float32{1.0, 2.0, 0.5, 1.5},
			},
		},
	}

	graph := &pb.GraphProto{
		Name: "multi_tree",
		Node: []*pb.NodeProto{node},
		Input: []*pb.ValueInfoProto{
			{
				Name: "X",
				Type: &pb.TypeProto{
					Value: &pb.TypeProto_TensorType{
						TensorType: &pb.TypeProto_Tensor{
							ElemType: int32(pb.TensorProto_FLOAT),
							Shape: &pb.TensorShapeProto{
								Dim: []*pb.TensorShapeProto_Dimension{
									{Value: &pb.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
									{Value: &pb.TensorShapeProto_Dimension_DimValue{DimValue: 2}},
								},
							},
						},
					},
				},
			},
		},
		Output: []*pb.ValueInfoProto{{Name: "Y"}},
	}

	model := &pb.ModelProto{
		IrVersion: 7,
		Graph:     graph,
	}

	data, _ := proto.Marshal(model)
	return data
}

func TestParseONNXTreeEnsembleFromBytes_MultipleTrees(t *testing.T) {
	data := createMultiTreeONNXModel()

	ensemble, err := ParseONNXTreeEnsembleFromBytes(data)
	if err != nil {
		t.Fatalf("ParseONNXTreeEnsembleFromBytes failed: %v", err)
	}

	if ensemble.NumTrees != 2 {
		t.Errorf("NumTrees = %d, want 2", ensemble.NumTrees)
	}

	// Should have 6 nodes total (3 per tree)
	if len(ensemble.Nodes) != 6 {
		t.Errorf("len(Nodes) = %d, want 6", len(ensemble.Nodes))
	}

	// Check roots
	if ensemble.Roots[0] != 0 {
		t.Errorf("Roots[0] = %d, want 0", ensemble.Roots[0])
	}
	if ensemble.Roots[1] != 3 {
		t.Errorf("Roots[1] = %d, want 3", ensemble.Roots[1])
	}

	// Check tree 1's root splits on feature 1
	tree1Root := &ensemble.Nodes[3]
	if tree1Root.Feature != 1 {
		t.Errorf("Tree1 root feature = %d, want 1", tree1Root.Feature)
	}
}

func TestParseONNXTreeEnsembleFromBytes_EndToEnd(t *testing.T) {
	data := createTestONNXModel()

	ensemble, err := ParseONNXTreeEnsembleFromBytes(data)
	if err != nil {
		t.Fatalf("ParseONNXTreeEnsembleFromBytes failed: %v", err)
	}

	// Create explainer
	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New explainer failed: %v", err)
	}

	baseValue := exp.BaseValue()
	t.Logf("Base value: %v", baseValue)
}

func BenchmarkParseONNXTreeEnsembleFromBytes(b *testing.B) {
	data := createTestONNXModel()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ParseONNXTreeEnsembleFromBytes(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}
