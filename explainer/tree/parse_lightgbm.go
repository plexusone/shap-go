package tree

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// LightGBMModel represents the top-level structure of a LightGBM dump_model() JSON output.
type LightGBMModel struct {
	Name                string            `json:"name"`
	Version             string            `json:"version"`
	NumClass            int               `json:"num_class"`
	NumTreePerIteration int               `json:"num_tree_per_iteration"`
	LabelIndex          int               `json:"label_index"`
	MaxFeatureIdx       int               `json:"max_feature_idx"`
	Objective           string            `json:"objective"`
	AverageOutput       bool              `json:"average_output"`
	FeatureNames        []string          `json:"feature_names"`
	MonotoneConstraints []int             `json:"monotone_constraints"`
	FeatureInfos        []string          `json:"feature_infos"`
	TreeInfo            []LightGBMTree    `json:"tree_info"`
	PandasCategorical   []json.RawMessage `json:"pandas_categorical"`
}

// LightGBMTree represents a single tree in the LightGBM model.
type LightGBMTree struct {
	TreeIndex     int           `json:"tree_index"`
	NumLeaves     int           `json:"num_leaves"`
	NumCat        int           `json:"num_cat"`
	Shrinkage     float64       `json:"shrinkage"`
	TreeStructure *LightGBMNode `json:"tree_structure"`
}

// LightGBMNode represents a node in a LightGBM tree.
// It can be either an internal node (with split info) or a leaf node.
type LightGBMNode struct {
	// Internal node fields
	SplitIndex    int           `json:"split_index,omitempty"`
	SplitFeature  int           `json:"split_feature,omitempty"`
	SplitGain     float64       `json:"split_gain,omitempty"`
	Threshold     float64       `json:"threshold,omitempty"`
	DecisionType  string        `json:"decision_type,omitempty"`
	DefaultLeft   bool          `json:"default_left,omitempty"`
	MissingType   string        `json:"missing_type,omitempty"`
	InternalValue float64       `json:"internal_value,omitempty"`
	InternalCount int           `json:"internal_count,omitempty"`
	LeftChild     *LightGBMNode `json:"left_child,omitempty"`
	RightChild    *LightGBMNode `json:"right_child,omitempty"`

	// Leaf node fields
	LeafIndex  int     `json:"leaf_index,omitempty"`
	LeafParent int     `json:"leaf_parent,omitempty"`
	LeafValue  float64 `json:"leaf_value,omitempty"`
	LeafWeight float64 `json:"leaf_weight,omitempty"`
	LeafCount  int     `json:"leaf_count,omitempty"`
}

// IsLeaf returns true if this node is a leaf node.
func (n *LightGBMNode) IsLeaf() bool {
	return n.LeftChild == nil && n.RightChild == nil
}

// ParseLightGBMJSON parses a LightGBM dump_model() JSON output into a TreeEnsemble.
func ParseLightGBMJSON(data []byte) (*TreeEnsemble, error) {
	var model LightGBMModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to parse LightGBM JSON: %w", err)
	}

	return convertLightGBMModel(&model)
}

// LoadLightGBMModel loads a LightGBM model from a JSON file.
func LoadLightGBMModel(path string) (*TreeEnsemble, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return LoadLightGBMModelFromReader(f)
}

// LoadLightGBMModelFromReader loads a LightGBM model from an io.Reader.
func LoadLightGBMModelFromReader(r io.Reader) (*TreeEnsemble, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}
	return ParseLightGBMJSON(data)
}

// convertLightGBMModel converts a LightGBMModel to a TreeEnsemble.
func convertLightGBMModel(model *LightGBMModel) (*TreeEnsemble, error) {
	if len(model.TreeInfo) == 0 {
		return nil, fmt.Errorf("model has no trees")
	}

	// Determine number of features
	numFeatures := model.MaxFeatureIdx + 1
	if numFeatures <= 0 {
		// Try to infer from feature names
		numFeatures = len(model.FeatureNames)
	}
	if numFeatures <= 0 {
		return nil, fmt.Errorf("could not determine number of features")
	}

	// Create ensemble
	ensemble := &TreeEnsemble{
		NumTrees:     len(model.TreeInfo),
		NumFeatures:  numFeatures,
		FeatureNames: model.FeatureNames,
		Roots:        make([]int, len(model.TreeInfo)),
		BaseScore:    0, // LightGBM doesn't export base_score in dump_model
		Objective:    model.Objective,
	}

	// Convert each tree
	nodeOffset := 0
	for treeIdx, tree := range model.TreeInfo {
		if tree.TreeStructure == nil {
			return nil, fmt.Errorf("tree %d has no structure", treeIdx)
		}

		ensemble.Roots[treeIdx] = nodeOffset
		nodes, err := convertLightGBMTree(tree.TreeStructure, treeIdx, nodeOffset)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tree %d: %w", treeIdx, err)
		}
		ensemble.Nodes = append(ensemble.Nodes, nodes...)
		nodeOffset += len(nodes)
	}

	if err := ensemble.Validate(); err != nil {
		return nil, fmt.Errorf("invalid ensemble: %w", err)
	}

	return ensemble, nil
}

// convertLightGBMTree converts a LightGBM tree to a flat array of nodes.
// LightGBM trees are nested (recursive), so we flatten them.
func convertLightGBMTree(root *LightGBMNode, treeIdx, nodeOffset int) ([]Node, error) {
	if root == nil {
		return nil, fmt.Errorf("root node is nil")
	}

	// First pass: count nodes and assign indices
	nodeCount := countNodes(root)
	nodes := make([]Node, nodeCount)

	// Second pass: convert nodes with proper indices
	nextIdx := 0
	err := convertNode(root, treeIdx, nodeOffset, nodes, &nextIdx)
	if err != nil {
		return nil, err
	}

	return nodes, nil
}

// countNodes counts the total number of nodes in a tree.
func countNodes(node *LightGBMNode) int {
	if node == nil {
		return 0
	}
	return 1 + countNodes(node.LeftChild) + countNodes(node.RightChild)
}

// convertNode recursively converts LightGBM nodes to our flat format.
// It returns the index of the converted node.
func convertNode(lgbNode *LightGBMNode, treeIdx, nodeOffset int, nodes []Node, nextIdx *int) error {
	if lgbNode == nil {
		return fmt.Errorf("nil node")
	}

	currentIdx := *nextIdx
	*nextIdx++

	node := Node{
		Tree:   treeIdx,
		NodeID: currentIdx,
	}

	if lgbNode.IsLeaf() {
		// Leaf node
		node.IsLeaf = true
		node.Feature = -1
		node.Yes = -1
		node.No = -1
		node.Missing = -1
		node.Prediction = lgbNode.LeafValue
		node.Cover = float64(lgbNode.LeafCount)
		if node.Cover == 0 {
			node.Cover = lgbNode.LeafWeight
		}
	} else {
		// Internal node
		node.IsLeaf = false
		node.Feature = lgbNode.SplitFeature
		node.Threshold = lgbNode.Threshold
		node.Cover = float64(lgbNode.InternalCount)

		// Convert decision type
		// LightGBM uses "<=", "==", etc.
		switch lgbNode.DecisionType {
		case "<=":
			node.DecisionType = DecisionLessEqual
		case "<":
			node.DecisionType = DecisionLess
		default:
			// Default to <= which is most common in LightGBM
			node.DecisionType = DecisionLessEqual
		}

		// Reserve indices for children
		leftIdx := *nextIdx
		if err := convertNode(lgbNode.LeftChild, treeIdx, nodeOffset, nodes, nextIdx); err != nil {
			return fmt.Errorf("failed to convert left child: %w", err)
		}

		rightIdx := *nextIdx
		if err := convertNode(lgbNode.RightChild, treeIdx, nodeOffset, nodes, nextIdx); err != nil {
			return fmt.Errorf("failed to convert right child: %w", err)
		}

		// In LightGBM, left child is "yes" (condition is true)
		node.Yes = leftIdx + nodeOffset
		node.No = rightIdx + nodeOffset

		// Handle missing values
		if lgbNode.DefaultLeft {
			node.Missing = node.Yes
		} else {
			node.Missing = node.No
		}
	}

	nodes[currentIdx] = node
	return nil
}

// Note: ParseLightGBMText is implemented in parse_lightgbm_text.go
