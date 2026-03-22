package tree

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// CatBoostModel represents the top-level structure of a CatBoost JSON model.
type CatBoostModel struct {
	ModelInfo      CatBoostModelInfo    `json:"model_info"`
	ObliviousTrees []CatBoostTree       `json:"oblivious_trees"`
	FeaturesInfo   CatBoostFeaturesInfo `json:"features_info"`
	ScaleAndBias   []float64            `json:"scale_and_bias,omitempty"`
}

// CatBoostModelInfo contains model metadata.
type CatBoostModelInfo struct {
	CatFeatureCount   int               `json:"cat_feature_count"`
	FloatFeatureCount int               `json:"float_feature_count"`
	ModelType         string            `json:"model_type"`
	Params            map[string]any    `json:"params"`
	TrainingOptions   map[string]string `json:"training_options"`
}

// CatBoostTree represents a single oblivious tree in CatBoost.
// In oblivious trees, all nodes at the same depth split on the same feature.
type CatBoostTree struct {
	// Splits contains the split conditions for each level of the tree.
	// The first split is the root, subsequent splits apply to all nodes at that depth.
	Splits []CatBoostSplit `json:"splits"`

	// LeafValues contains the prediction values for each leaf.
	// For a tree of depth d, there are 2^d leaves.
	LeafValues []float64 `json:"leaf_values"`

	// LeafWeights contains the training sample counts for each leaf.
	LeafWeights []float64 `json:"leaf_weights,omitempty"`
}

// CatBoostSplit represents a split condition in the tree.
type CatBoostSplit struct {
	// FloatFeatureIndex is the index of the float feature to split on.
	// nil if this is a categorical split.
	FloatFeatureIndex *int `json:"float_feature_index,omitempty"`

	// Border is the threshold value for numerical splits.
	Border float64 `json:"border,omitempty"`

	// CatFeatureIndex is the index of the categorical feature.
	// nil if this is a numerical split.
	CatFeatureIndex *int `json:"cat_feature_index,omitempty"`

	// SplitType indicates the type of split ("FloatFeature" or "OneHotFeature").
	SplitType string `json:"split_type,omitempty"`

	// Value is used for categorical splits (one-hot encoding).
	Value int `json:"value,omitempty"`
}

// CatBoostFeaturesInfo contains information about model features.
type CatBoostFeaturesInfo struct {
	FloatFeatures       []CatBoostFloatFeature `json:"float_features"`
	CategoricalFeatures []CatBoostCatFeature   `json:"categorical_features,omitempty"`
}

// CatBoostFloatFeature represents a numerical feature.
type CatBoostFloatFeature struct {
	FeatureIndex int       `json:"feature_index"`
	FlatIndex    int       `json:"flat_feature_index"`
	HasNans      bool      `json:"has_nans"`
	Borders      []float64 `json:"borders,omitempty"`
	FeatureName  string    `json:"feature_name,omitempty"`
}

// CatBoostCatFeature represents a categorical feature.
type CatBoostCatFeature struct {
	FeatureIndex int    `json:"feature_index"`
	FlatIndex    int    `json:"flat_feature_index"`
	FeatureName  string `json:"feature_name,omitempty"`
}

// ParseCatBoostJSON parses a CatBoost JSON model into a TreeEnsemble.
func ParseCatBoostJSON(data []byte) (*TreeEnsemble, error) {
	var model CatBoostModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to parse CatBoost JSON: %w", err)
	}

	return convertCatBoostModel(&model)
}

// LoadCatBoostModel loads a CatBoost model from a JSON file.
func LoadCatBoostModel(path string) (*TreeEnsemble, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return LoadCatBoostModelFromReader(f)
}

// LoadCatBoostModelFromReader loads a CatBoost model from an io.Reader.
func LoadCatBoostModelFromReader(r io.Reader) (*TreeEnsemble, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}
	return ParseCatBoostJSON(data)
}

// convertCatBoostModel converts a CatBoostModel to a TreeEnsemble.
func convertCatBoostModel(model *CatBoostModel) (*TreeEnsemble, error) {
	if len(model.ObliviousTrees) == 0 {
		return nil, fmt.Errorf("model has no trees")
	}

	// Determine number of features
	numFloatFeatures := model.ModelInfo.FloatFeatureCount
	if numFloatFeatures == 0 {
		numFloatFeatures = len(model.FeaturesInfo.FloatFeatures)
	}
	numCatFeatures := model.ModelInfo.CatFeatureCount
	if numCatFeatures == 0 {
		numCatFeatures = len(model.FeaturesInfo.CategoricalFeatures)
	}
	numFeatures := numFloatFeatures + numCatFeatures

	if numFeatures == 0 {
		return nil, fmt.Errorf("could not determine number of features")
	}

	// Extract feature names
	featureNames := make([]string, numFeatures)
	for _, f := range model.FeaturesInfo.FloatFeatures {
		if f.FlatIndex >= 0 && f.FlatIndex < len(featureNames) {
			if f.FeatureName != "" {
				featureNames[f.FlatIndex] = f.FeatureName
			} else {
				featureNames[f.FlatIndex] = fmt.Sprintf("float_%d", f.FeatureIndex)
			}
		}
	}
	for _, f := range model.FeaturesInfo.CategoricalFeatures {
		if f.FlatIndex >= 0 && f.FlatIndex < len(featureNames) {
			if f.FeatureName != "" {
				featureNames[f.FlatIndex] = f.FeatureName
			} else {
				featureNames[f.FlatIndex] = fmt.Sprintf("cat_%d", f.FeatureIndex)
			}
		}
	}

	// Fill in any empty names
	for i := range featureNames {
		if featureNames[i] == "" {
			featureNames[i] = fmt.Sprintf("feature_%d", i)
		}
	}

	// Parse base score from scale_and_bias
	baseScore := 0.0
	if len(model.ScaleAndBias) >= 2 {
		baseScore = model.ScaleAndBias[1] // bias is second element
	}

	// Create ensemble
	ensemble := &TreeEnsemble{
		NumTrees:     len(model.ObliviousTrees),
		NumFeatures:  numFeatures,
		FeatureNames: featureNames,
		Roots:        make([]int, len(model.ObliviousTrees)),
		BaseScore:    baseScore,
		Objective:    model.ModelInfo.ModelType,
	}

	nodeOffset := 0
	for treeIdx, tree := range model.ObliviousTrees {
		ensemble.Roots[treeIdx] = nodeOffset
		nodes, err := convertCatBoostTree(&tree, treeIdx, nodeOffset, numFloatFeatures)
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

// convertCatBoostTree converts a CatBoost oblivious tree to standard nodes.
// CatBoost uses oblivious (symmetric) trees where all nodes at the same depth
// split on the same feature with the same threshold.
func convertCatBoostTree(tree *CatBoostTree, treeIdx, nodeOffset, numFloatFeatures int) ([]Node, error) {
	depth := len(tree.Splits)
	if depth == 0 {
		// Single leaf tree
		if len(tree.LeafValues) == 0 {
			return nil, fmt.Errorf("tree has no leaf values")
		}
		cover := 1.0
		if len(tree.LeafWeights) > 0 {
			cover = tree.LeafWeights[0]
		}
		return []Node{{
			Tree:       treeIdx,
			NodeID:     0,
			IsLeaf:     true,
			Feature:    -1,
			Yes:        -1,
			No:         -1,
			Missing:    -1,
			Prediction: tree.LeafValues[0],
			Cover:      cover,
		}}, nil
	}

	numLeaves := 1 << depth           // 2^depth
	numInternalNodes := numLeaves - 1 // Full binary tree
	numNodes := numLeaves + numInternalNodes

	if len(tree.LeafValues) != numLeaves {
		return nil, fmt.Errorf("expected %d leaf values, got %d", numLeaves, len(tree.LeafValues))
	}

	nodes := make([]Node, numNodes)

	// Build the tree structure
	// For oblivious trees, all nodes at level i use split[i]
	for nodeID := 0; nodeID < numInternalNodes; nodeID++ {
		// Calculate which level this node is at
		level := nodeLevel(nodeID)
		split := tree.Splits[level]

		// Determine feature index
		var featureIdx int
		if split.FloatFeatureIndex != nil {
			// Numerical feature
			featureIdx = *split.FloatFeatureIndex
		} else if split.CatFeatureIndex != nil {
			// Categorical feature - offset by number of float features
			featureIdx = numFloatFeatures + *split.CatFeatureIndex
		} else {
			// Default to 0 if neither is specified (shouldn't happen)
			featureIdx = 0
		}

		// Calculate children indices
		leftChild := 2*nodeID + 1
		rightChild := 2*nodeID + 2

		nodes[nodeID] = Node{
			Tree:         treeIdx,
			NodeID:       nodeID,
			IsLeaf:       false,
			Feature:      featureIdx,
			Threshold:    split.Border,
			DecisionType: DecisionLessEqual, // CatBoost uses <=
			Yes:          leftChild + nodeOffset,
			No:           rightChild + nodeOffset,
			Missing:      leftChild + nodeOffset, // Default to left for missing
			Cover:        computeNodeCover(tree.LeafWeights, nodeID, depth),
		}
	}

	// Add leaf nodes
	for leafIdx := 0; leafIdx < numLeaves; leafIdx++ {
		nodeID := numInternalNodes + leafIdx
		cover := 1.0
		if leafIdx < len(tree.LeafWeights) {
			cover = tree.LeafWeights[leafIdx]
		}

		nodes[nodeID] = Node{
			Tree:       treeIdx,
			NodeID:     nodeID,
			IsLeaf:     true,
			Feature:    -1,
			Yes:        -1,
			No:         -1,
			Missing:    -1,
			Prediction: tree.LeafValues[leafIdx],
			Cover:      cover,
		}
	}

	// Update node indices to be absolute (add offset)
	for i := range nodes {
		if !nodes[i].IsLeaf {
			// Already added offset above
		}
	}

	return nodes, nil
}

// nodeLevel returns the depth level of a node in a complete binary tree.
// Root is level 0, its children are level 1, etc.
func nodeLevel(nodeID int) int {
	if nodeID == 0 {
		return 0
	}
	level := 0
	for (1<<(level+1))-1 <= nodeID {
		level++
	}
	return level
}

// computeNodeCover computes the cover (sample count) for an internal node
// by summing the covers of all leaves in its subtree.
func computeNodeCover(leafWeights []float64, nodeID, depth int) float64 {
	if len(leafWeights) == 0 {
		return 1.0
	}

	numLeaves := 1 << depth
	numInternalNodes := numLeaves - 1

	// Find the range of leaf indices that belong to this node's subtree
	// For a complete binary tree, leaf indices for a subtree rooted at nodeID
	// can be calculated
	level := nodeLevel(nodeID)
	nodesAtLevel := 1 << level
	positionInLevel := nodeID - (nodesAtLevel - 1)

	// Number of leaves per subtree at this level
	leavesPerSubtree := numLeaves / nodesAtLevel
	startLeaf := positionInLevel * leavesPerSubtree
	endLeaf := startLeaf + leavesPerSubtree

	// Ensure we don't go out of bounds
	if startLeaf >= numLeaves {
		startLeaf = numLeaves - 1
	}
	if endLeaf > numLeaves {
		endLeaf = numLeaves
	}

	// Map leaf indices to leaf values indices
	// Leaves in nodes array start at index numInternalNodes
	_ = numInternalNodes

	var cover float64
	for i := startLeaf; i < endLeaf; i++ {
		if i < len(leafWeights) {
			cover += leafWeights[i]
		} else {
			cover += 1.0
		}
	}

	return cover
}
