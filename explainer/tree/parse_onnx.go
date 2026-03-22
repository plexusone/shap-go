package tree

import (
	"fmt"
	"os"

	pb "github.com/advancedclimatesystems/gonnx/onnx"
	"google.golang.org/protobuf/proto"
)

// ONNXTreeEnsemble represents tree ensemble attributes from ONNX-ML operators
// (TreeEnsembleClassifier or TreeEnsembleRegressor).
type ONNXTreeEnsemble struct {
	// Node structure (parallel arrays)
	NodesTreeIDs      []int64   // Tree ID for each node
	NodesNodeIDs      []int64   // Node ID within each tree
	NodesFeatureIDs   []int64   // Feature index for split nodes
	NodesValues       []float32 // Threshold values for splits
	NodesModes        []string  // Split modes (BRANCH_LEQ, BRANCH_LT, LEAF)
	NodesTrueNodeIDs  []int64   // Left/true child node IDs
	NodesFalseNodeIDs []int64   // Right/false child node IDs
	NodesMissing      []int64   // Direction for missing values (0=false, 1=true)
	NodesHitrates     []float32 // Hit rates for nodes (optional)

	// Leaf values (for regressor)
	TargetIDs     []int64   // Target indices (usually 0 for single output)
	TargetNodeIDs []int64   // Node IDs that are leaves
	TargetTreeIDs []int64   // Tree IDs for leaves
	TargetWeights []float32 // Leaf prediction values

	// For classifier
	ClassIDs         []int64   // Class indices
	ClassNodeIDs     []int64   // Leaf node IDs for class labels
	ClassTreeIDs     []int64   // Tree IDs for class leaves
	ClassWeights     []float32 // Class weights
	ClassLabelsInt64 []int64   // Class labels as int64
	ClassLabelsStr   []string  // Class labels as strings

	// Metadata
	NumTargets    int64     // Number of targets (usually 1)
	BaseValues    []float32 // Base values
	Aggregate     string    // Aggregation function (SUM, AVERAGE, etc.)
	PostTransform string    // Post-transform (NONE, LOGISTIC, SOFTMAX, etc.)
}

// ParseONNXTreeEnsemble parses an ONNX model file containing a tree ensemble.
func ParseONNXTreeEnsemble(path string) (*TreeEnsemble, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %w", err)
	}
	return ParseONNXTreeEnsembleFromBytes(data)
}

// ParseONNXTreeEnsembleFromBytes parses an ONNX model from bytes.
func ParseONNXTreeEnsembleFromBytes(data []byte) (*TreeEnsemble, error) {
	model := &pb.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %w", err)
	}

	graph := model.GetGraph()
	if graph == nil {
		return nil, fmt.Errorf("ONNX model has no graph")
	}

	// Find tree ensemble operator
	var treeNode *pb.NodeProto
	var opType string
	for _, node := range graph.GetNode() {
		op := node.GetOpType()
		if op == "TreeEnsembleRegressor" || op == "TreeEnsembleClassifier" {
			treeNode = node
			opType = op
			break
		}
	}

	if treeNode == nil {
		return nil, fmt.Errorf("no TreeEnsembleRegressor or TreeEnsembleClassifier found in model")
	}

	// Extract attributes
	onnxTree := &ONNXTreeEnsemble{}
	for _, attr := range treeNode.GetAttribute() {
		name := attr.GetName()
		switch name {
		case "nodes_treeids":
			onnxTree.NodesTreeIDs = attr.GetInts()
		case "nodes_nodeids":
			onnxTree.NodesNodeIDs = attr.GetInts()
		case "nodes_featureids":
			onnxTree.NodesFeatureIDs = attr.GetInts()
		case "nodes_values":
			onnxTree.NodesValues = attr.GetFloats()
		case "nodes_modes":
			onnxTree.NodesModes = bytesToStrings(attr.GetStrings())
		case "nodes_truenodeids":
			onnxTree.NodesTrueNodeIDs = attr.GetInts()
		case "nodes_falsenodeids":
			onnxTree.NodesFalseNodeIDs = attr.GetInts()
		case "nodes_missing_value_tracks_true":
			onnxTree.NodesMissing = attr.GetInts()
		case "nodes_hitrates":
			onnxTree.NodesHitrates = attr.GetFloats()
		case "target_ids":
			onnxTree.TargetIDs = attr.GetInts()
		case "target_nodeids":
			onnxTree.TargetNodeIDs = attr.GetInts()
		case "target_treeids":
			onnxTree.TargetTreeIDs = attr.GetInts()
		case "target_weights":
			onnxTree.TargetWeights = attr.GetFloats()
		case "class_ids":
			onnxTree.ClassIDs = attr.GetInts()
		case "class_nodeids":
			onnxTree.ClassNodeIDs = attr.GetInts()
		case "class_treeids":
			onnxTree.ClassTreeIDs = attr.GetInts()
		case "class_weights":
			onnxTree.ClassWeights = attr.GetFloats()
		case "classlabels_int64s":
			onnxTree.ClassLabelsInt64 = attr.GetInts()
		case "classlabels_strings":
			onnxTree.ClassLabelsStr = bytesToStrings(attr.GetStrings())
		case "n_targets":
			onnxTree.NumTargets = attr.GetI()
		case "base_values":
			onnxTree.BaseValues = attr.GetFloats()
		case "aggregate_function":
			onnxTree.Aggregate = string(attr.GetS())
		case "post_transform":
			onnxTree.PostTransform = string(attr.GetS())
		}
	}

	// Determine number of features from graph inputs
	numFeatures := 0
	for _, input := range graph.GetInput() {
		inputType := input.GetType()
		if inputType != nil {
			tensorType := inputType.GetTensorType()
			if tensorType != nil {
				shape := tensorType.GetShape()
				if shape != nil {
					dims := shape.GetDim()
					if len(dims) >= 2 {
						// Shape is typically [batch, features]
						dimVal := dims[1].GetDimValue()
						if dimVal > 0 {
							numFeatures = int(dimVal)
							break
						}
					}
				}
			}
		}
	}

	// If we couldn't get from shape, infer from max feature ID
	if numFeatures == 0 {
		for _, fid := range onnxTree.NodesFeatureIDs {
			if int(fid) >= numFeatures {
				numFeatures = int(fid) + 1
			}
		}
	}

	if numFeatures == 0 {
		numFeatures = 1 // Minimum
	}

	return convertONNXTreeEnsemble(onnxTree, opType, numFeatures)
}

// bytesToStrings converts [][]byte to []string.
func bytesToStrings(b [][]byte) []string {
	result := make([]string, len(b))
	for i, v := range b {
		result[i] = string(v)
	}
	return result
}

// convertONNXTreeEnsemble converts ONNX tree ensemble attributes to TreeEnsemble.
func convertONNXTreeEnsemble(onnx *ONNXTreeEnsemble, opType string, numFeatures int) (*TreeEnsemble, error) {
	if len(onnx.NodesTreeIDs) == 0 {
		return nil, fmt.Errorf("empty tree ensemble")
	}

	// Count trees and nodes per tree
	numTrees := 0
	nodesPerTree := make(map[int64]int)
	for _, treeID := range onnx.NodesTreeIDs {
		nodesPerTree[treeID]++
		if int(treeID)+1 > numTrees {
			numTrees = int(treeID) + 1
		}
	}

	// Create leaf value map
	leafValues := make(map[int64]map[int64]float64) // tree_id -> node_id -> value
	if opType == "TreeEnsembleRegressor" {
		for i, nodeID := range onnx.TargetNodeIDs {
			treeID := int64(0)
			if i < len(onnx.TargetTreeIDs) {
				treeID = onnx.TargetTreeIDs[i]
			}
			if leafValues[treeID] == nil {
				leafValues[treeID] = make(map[int64]float64)
			}
			if i < len(onnx.TargetWeights) {
				leafValues[treeID][nodeID] = float64(onnx.TargetWeights[i])
			}
		}
	} else { // Classifier
		for i, nodeID := range onnx.ClassNodeIDs {
			treeID := int64(0)
			if i < len(onnx.ClassTreeIDs) {
				treeID = onnx.ClassTreeIDs[i]
			}
			if leafValues[treeID] == nil {
				leafValues[treeID] = make(map[int64]float64)
			}
			// For classifiers, we might have multiple class outputs per leaf
			// For simplicity, we take the weight directly
			if i < len(onnx.ClassWeights) {
				leafValues[treeID][nodeID] = float64(onnx.ClassWeights[i])
			}
		}
	}

	// Base score
	baseScore := 0.0
	if len(onnx.BaseValues) > 0 {
		baseScore = float64(onnx.BaseValues[0])
	}

	// Build tree structure
	ensemble := &TreeEnsemble{
		NumTrees:    numTrees,
		NumFeatures: numFeatures,
		Roots:       make([]int, numTrees),
		BaseScore:   baseScore,
		Objective:   opType,
	}

	// Group nodes by tree
	nodesByTree := make(map[int64][]int) // tree_id -> indices in original arrays
	for i, treeID := range onnx.NodesTreeIDs {
		nodesByTree[treeID] = append(nodesByTree[treeID], i)
	}

	nodeOffset := 0
	for treeIdx := 0; treeIdx < numTrees; treeIdx++ {
		treeID := int64(treeIdx)
		nodeIndices := nodesByTree[treeID]

		if len(nodeIndices) == 0 {
			// Empty tree - add dummy leaf
			ensemble.Roots[treeIdx] = nodeOffset
			ensemble.Nodes = append(ensemble.Nodes, Node{
				Tree:       treeIdx,
				NodeID:     0,
				IsLeaf:     true,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				Prediction: 0,
			})
			nodeOffset++
			continue
		}

		ensemble.Roots[treeIdx] = nodeOffset

		// Build node ID to index mapping for this tree
		nodeIDToIdx := make(map[int64]int)
		for localIdx, origIdx := range nodeIndices {
			nodeID := onnx.NodesNodeIDs[origIdx]
			nodeIDToIdx[nodeID] = localIdx
		}

		// Convert nodes
		for localIdx, origIdx := range nodeIndices {
			nodeID := onnx.NodesNodeIDs[origIdx]
			mode := ""
			if origIdx < len(onnx.NodesModes) {
				mode = onnx.NodesModes[origIdx]
			}

			isLeaf := mode == "LEAF"

			node := Node{
				Tree:   treeIdx,
				NodeID: localIdx,
				IsLeaf: isLeaf,
			}

			if isLeaf {
				node.Feature = -1
				node.Yes = -1
				node.No = -1
				node.Missing = -1
				if leafVal, ok := leafValues[treeID][nodeID]; ok {
					node.Prediction = leafVal
				}
			} else {
				// Internal node
				featureID := int64(0)
				if origIdx < len(onnx.NodesFeatureIDs) {
					featureID = onnx.NodesFeatureIDs[origIdx]
				}
				node.Feature = int(featureID)

				threshold := float64(0)
				if origIdx < len(onnx.NodesValues) {
					threshold = float64(onnx.NodesValues[origIdx])
				}
				node.Threshold = threshold

				// Determine decision type from mode
				switch mode {
				case "BRANCH_LEQ":
					node.DecisionType = DecisionLessEqual
				case "BRANCH_LT":
					node.DecisionType = DecisionLess
				default:
					node.DecisionType = DecisionLessEqual // Default
				}

				// Map children
				trueNodeID := int64(0)
				if origIdx < len(onnx.NodesTrueNodeIDs) {
					trueNodeID = onnx.NodesTrueNodeIDs[origIdx]
				}
				falseNodeID := int64(0)
				if origIdx < len(onnx.NodesFalseNodeIDs) {
					falseNodeID = onnx.NodesFalseNodeIDs[origIdx]
				}

				if localTrueIdx, ok := nodeIDToIdx[trueNodeID]; ok {
					node.Yes = nodeOffset + localTrueIdx
				} else {
					node.Yes = -1
				}

				if localFalseIdx, ok := nodeIDToIdx[falseNodeID]; ok {
					node.No = nodeOffset + localFalseIdx
				} else {
					node.No = -1
				}

				// Missing value handling
				missingGoesTrue := false
				if origIdx < len(onnx.NodesMissing) && onnx.NodesMissing[origIdx] == 1 {
					missingGoesTrue = true
				}
				if missingGoesTrue {
					node.Missing = node.Yes
				} else {
					node.Missing = node.No
				}
			}

			// Hit rate as cover (optional)
			if origIdx < len(onnx.NodesHitrates) {
				node.Cover = float64(onnx.NodesHitrates[origIdx])
			}

			ensemble.Nodes = append(ensemble.Nodes, node)
		}

		nodeOffset += len(nodeIndices)
	}

	if err := ensemble.Validate(); err != nil {
		return nil, fmt.Errorf("invalid ensemble: %w", err)
	}

	return ensemble, nil
}
