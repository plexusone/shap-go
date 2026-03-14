// Package tree provides a TreeSHAP explainer for tree-based models.
//
// TreeSHAP computes exact SHAP values for tree ensembles using the algorithm
// described in "Consistent Individualized Feature Attribution for Tree Ensembles"
// by Lundberg et al. (2018).
//
// Unlike sampling-based SHAP methods, TreeSHAP is:
//   - Exact: Computes precise SHAP values, not approximations
//   - Fast: O(TLD²) complexity where T=trees, L=max depth, D=features
//   - Consistent: Guarantees local accuracy property
//
// Supported model formats:
//   - XGBoost JSON models
//   - LightGBM JSON models (future)
//   - ONNX-ML TreeEnsemble (future)
package tree

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// DecisionType represents the comparison operator for splits.
type DecisionType string

const (
	// DecisionLess means feature < threshold goes left.
	DecisionLess DecisionType = "<"
	// DecisionLessEqual means feature <= threshold goes left.
	DecisionLessEqual DecisionType = "<="
)

// Node represents a node in a decision tree.
// This is a unified format that can represent nodes from XGBoost, LightGBM, etc.
type Node struct {
	// Tree is the 0-indexed tree ID this node belongs to.
	Tree int `json:"tree"`

	// NodeID is the 0-indexed node ID within the tree.
	NodeID int `json:"node_id"`

	// Feature is the feature index for splits (-1 for leaf nodes).
	Feature int `json:"feature"`

	// DecisionType is the comparison operator ("<" or "<=").
	DecisionType DecisionType `json:"decision_type"`

	// Threshold is the split threshold value.
	Threshold float64 `json:"threshold"`

	// Yes is the index of the left/yes child node (-1 if none).
	Yes int `json:"yes"`

	// No is the index of the right/no child node (-1 if none).
	No int `json:"no"`

	// Missing is the index of the node when feature value is missing (-1 if no support).
	Missing int `json:"missing"`

	// Prediction is the leaf value (0 for internal nodes).
	Prediction float64 `json:"prediction"`

	// Cover is the number of training samples that reach this node.
	Cover float64 `json:"cover"`

	// IsLeaf indicates whether this is a leaf node.
	IsLeaf bool `json:"is_leaf"`
}

// TreeEnsemble holds a unified representation of a tree ensemble model.
type TreeEnsemble struct {
	// Nodes contains all nodes from all trees in a flat array.
	Nodes []Node `json:"nodes"`

	// NumTrees is the total number of trees.
	NumTrees int `json:"num_trees"`

	// NumFeatures is the number of input features.
	NumFeatures int `json:"num_features"`

	// FeatureNames optionally holds the names of features.
	FeatureNames []string `json:"feature_names,omitempty"`

	// Roots contains the index into Nodes for each tree's root.
	Roots []int `json:"roots"`

	// BaseScore is the initial prediction value (bias).
	BaseScore float64 `json:"base_score"`

	// Objective is the model objective (e.g., "reg:squarederror").
	Objective string `json:"objective,omitempty"`
}

// Validate checks the ensemble for internal consistency.
func (e *TreeEnsemble) Validate() error {
	if len(e.Nodes) == 0 {
		return fmt.Errorf("ensemble has no nodes")
	}
	if e.NumTrees <= 0 {
		return fmt.Errorf("ensemble has invalid num_trees: %d", e.NumTrees)
	}
	if len(e.Roots) != e.NumTrees {
		return fmt.Errorf("roots length %d does not match num_trees %d", len(e.Roots), e.NumTrees)
	}
	if e.NumFeatures <= 0 {
		return fmt.Errorf("ensemble has invalid num_features: %d", e.NumFeatures)
	}

	// Validate each tree's root index
	for i, rootIdx := range e.Roots {
		if rootIdx < 0 || rootIdx >= len(e.Nodes) {
			return fmt.Errorf("tree %d has invalid root index %d", i, rootIdx)
		}
	}

	// Validate each node
	for i, node := range e.Nodes {
		if node.IsLeaf {
			continue
		}

		// Internal node must have valid feature
		if node.Feature < 0 || node.Feature >= e.NumFeatures {
			return fmt.Errorf("node %d has invalid feature index %d", i, node.Feature)
		}

		// Internal node must have valid children
		if node.Yes < 0 || node.Yes >= len(e.Nodes) {
			return fmt.Errorf("node %d has invalid yes child %d", i, node.Yes)
		}
		if node.No < 0 || node.No >= len(e.Nodes) {
			return fmt.Errorf("node %d has invalid no child %d", i, node.No)
		}
	}

	return nil
}

// TreeNodes returns all nodes belonging to a specific tree.
func (e *TreeEnsemble) TreeNodes(treeIdx int) []Node {
	if treeIdx < 0 || treeIdx >= e.NumTrees {
		return nil
	}

	var nodes []Node
	for _, node := range e.Nodes {
		if node.Tree == treeIdx {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// ExpectedValue computes the expected prediction value from the tree structure.
// This is calculated as the cover-weighted average of all leaf predictions.
// For TreeSHAP, this is the base value that SHAP values are computed relative to.
func (e *TreeEnsemble) ExpectedValue() float64 {
	// Sum the expected value from each tree
	totalExpected := e.BaseScore

	for treeIdx := 0; treeIdx < e.NumTrees; treeIdx++ {
		rootIdx := e.Roots[treeIdx]
		treeExpected := e.treeExpectedValue(rootIdx)
		totalExpected += treeExpected
	}

	return totalExpected
}

// treeExpectedValue computes the expected value for a single tree.
func (e *TreeEnsemble) treeExpectedValue(nodeIdx int) float64 {
	if nodeIdx < 0 || nodeIdx >= len(e.Nodes) {
		return 0
	}

	node := &e.Nodes[nodeIdx]
	if node.IsLeaf {
		return node.Prediction
	}

	// Get children covers
	yesCover := e.Nodes[node.Yes].Cover
	noCover := e.Nodes[node.No].Cover
	totalCover := yesCover + noCover

	if totalCover == 0 {
		// If no cover info, use simple average
		yesExpected := e.treeExpectedValue(node.Yes)
		noExpected := e.treeExpectedValue(node.No)
		return (yesExpected + noExpected) / 2
	}

	// Weighted average based on cover
	yesExpected := e.treeExpectedValue(node.Yes)
	noExpected := e.treeExpectedValue(node.No)

	return (yesCover*yesExpected + noCover*noExpected) / totalCover
}

// MaxDepth returns the maximum depth across all trees.
func (e *TreeEnsemble) MaxDepth() int {
	maxDepth := 0
	for treeIdx := 0; treeIdx < e.NumTrees; treeIdx++ {
		depth := e.treeDepth(e.Roots[treeIdx], 0)
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	return maxDepth
}

func (e *TreeEnsemble) treeDepth(nodeIdx, currentDepth int) int {
	if nodeIdx < 0 || nodeIdx >= len(e.Nodes) {
		return currentDepth
	}

	node := &e.Nodes[nodeIdx]
	if node.IsLeaf {
		return currentDepth
	}

	leftDepth := e.treeDepth(node.Yes, currentDepth+1)
	rightDepth := e.treeDepth(node.No, currentDepth+1)

	if leftDepth > rightDepth {
		return leftDepth
	}
	return rightDepth
}

// ToJSON serializes the ensemble to JSON.
func (e *TreeEnsemble) ToJSON() ([]byte, error) {
	return json.Marshal(e)
}

// ToJSONPretty serializes the ensemble to indented JSON.
func (e *TreeEnsemble) ToJSONPretty() ([]byte, error) {
	return json.MarshalIndent(e, "", "  ")
}

// EnsembleFromJSON deserializes an ensemble from JSON.
func EnsembleFromJSON(data []byte) (*TreeEnsemble, error) {
	var e TreeEnsemble
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ensemble: %w", err)
	}
	if err := e.Validate(); err != nil {
		return nil, fmt.Errorf("invalid ensemble: %w", err)
	}
	return &e, nil
}

// LoadEnsemble loads a TreeEnsemble from a JSON file.
func LoadEnsemble(path string) (*TreeEnsemble, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return LoadEnsembleFromReader(f)
}

// LoadEnsembleFromReader loads a TreeEnsemble from an io.Reader.
func LoadEnsembleFromReader(r io.Reader) (*TreeEnsemble, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}
	return EnsembleFromJSON(data)
}
