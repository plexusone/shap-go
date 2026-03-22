package tree

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LightGBMTextModel holds parsed data from LightGBM text format.
type LightGBMTextModel struct {
	Version       string
	NumClass      int
	NumTreePerItr int
	LabelIndex    int
	MaxFeatureIdx int
	Objective     string
	FeatureNames  []string
	Trees         []LightGBMTextTree
}

// LightGBMTextTree represents a tree parsed from text format.
type LightGBMTextTree struct {
	TreeIndex     int
	NumLeaves     int
	NumCat        int
	SplitFeature  []int
	SplitGain     []float64
	Threshold     []float64
	DecisionType  []string
	LeftChild     []int
	RightChild    []int
	LeafValue     []float64
	LeafWeight    []float64
	LeafCount     []int
	InternalValue []float64
	InternalCount []int
	Shrinkage     float64
}

// ParseLightGBMText parses a LightGBM text-format model.
// The text format is produced by LightGBM's save_model() function.
func ParseLightGBMText(data []byte) (*TreeEnsemble, error) {
	model, err := parseLightGBMTextModel(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to parse LightGBM text model: %w", err)
	}
	return convertLightGBMTextModel(model)
}

// LoadLightGBMTextModel loads a LightGBM model from a text file.
func LoadLightGBMTextModel(path string) (*TreeEnsemble, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return LoadLightGBMTextModelFromReader(f)
}

// LoadLightGBMTextModelFromReader loads a LightGBM text model from a reader.
func LoadLightGBMTextModelFromReader(r io.Reader) (*TreeEnsemble, error) {
	model, err := parseLightGBMTextModel(r)
	if err != nil {
		return nil, err
	}
	return convertLightGBMTextModel(model)
}

// parseLightGBMTextModel parses the text format into intermediate structures.
func parseLightGBMTextModel(r io.Reader) (*LightGBMTextModel, error) {
	scanner := bufio.NewScanner(r)
	model := &LightGBMTextModel{
		NumClass:      1,
		NumTreePerItr: 1,
	}

	var currentTree *LightGBMTextTree
	inTree := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Check for tree header
		if strings.HasPrefix(line, "Tree=") {
			if currentTree != nil {
				model.Trees = append(model.Trees, *currentTree)
			}
			treeIdx, err := strconv.Atoi(strings.TrimPrefix(line, "Tree="))
			if err != nil {
				return nil, fmt.Errorf("invalid tree index: %s", line)
			}
			currentTree = &LightGBMTextTree{
				TreeIndex: treeIdx,
				Shrinkage: 1.0,
			}
			inTree = true
			continue
		}

		// Check for end of trees marker
		if line == "end of trees" {
			if currentTree != nil {
				model.Trees = append(model.Trees, *currentTree)
				currentTree = nil
			}
			inTree = false
			continue
		}

		// Parse key=value pairs
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		if inTree && currentTree != nil {
			if err := parseTreeField(currentTree, key, value); err != nil {
				return nil, fmt.Errorf("failed to parse tree field %s: %w", key, err)
			}
		} else {
			if err := parseModelField(model, key, value); err != nil {
				return nil, fmt.Errorf("failed to parse model field %s: %w", key, err)
			}
		}
	}

	// Don't forget last tree if file doesn't end with "end of trees"
	if currentTree != nil {
		model.Trees = append(model.Trees, *currentTree)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading model: %w", err)
	}

	if len(model.Trees) == 0 {
		return nil, fmt.Errorf("no trees found in model")
	}

	return model, nil
}

// parseModelField parses a header-level key=value pair.
func parseModelField(model *LightGBMTextModel, key, value string) error {
	switch key {
	case "version":
		model.Version = value
	case "num_class":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		model.NumClass = v
	case "num_tree_per_iteration":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		model.NumTreePerItr = v
	case "label_index":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		model.LabelIndex = v
	case "max_feature_idx":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		model.MaxFeatureIdx = v
	case "objective":
		model.Objective = value
	case "feature_names":
		model.FeatureNames = strings.Fields(value)
	}
	return nil
}

// parseTreeField parses a tree-level key=value pair.
func parseTreeField(tree *LightGBMTextTree, key, value string) error {
	switch key {
	case "num_leaves":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		tree.NumLeaves = v
	case "num_cat":
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		tree.NumCat = v
	case "split_feature":
		tree.SplitFeature = parseIntArray(value)
	case "split_gain":
		tree.SplitGain = parseFloatArray(value)
	case "threshold":
		tree.Threshold = parseFloatArray(value)
	case "decision_type":
		tree.DecisionType = strings.Fields(value)
	case "left_child":
		tree.LeftChild = parseIntArray(value)
	case "right_child":
		tree.RightChild = parseIntArray(value)
	case "leaf_value":
		tree.LeafValue = parseFloatArray(value)
	case "leaf_weight":
		tree.LeafWeight = parseFloatArray(value)
	case "leaf_count":
		tree.LeafCount = parseIntArray(value)
	case "internal_value":
		tree.InternalValue = parseFloatArray(value)
	case "internal_count":
		tree.InternalCount = parseIntArray(value)
	case "shrinkage":
		v, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return err
		}
		tree.Shrinkage = v
	}
	return nil
}

// parseIntArray parses a space-separated array of integers.
func parseIntArray(s string) []int {
	fields := strings.Fields(s)
	result := make([]int, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		result = append(result, v)
	}
	return result
}

// parseFloatArray parses a space-separated array of floats.
func parseFloatArray(s string) []float64 {
	fields := strings.Fields(s)
	result := make([]float64, 0, len(fields))
	for _, f := range fields {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			continue
		}
		result = append(result, v)
	}
	return result
}

// convertLightGBMTextModel converts parsed text model to TreeEnsemble.
func convertLightGBMTextModel(model *LightGBMTextModel) (*TreeEnsemble, error) {
	numFeatures := model.MaxFeatureIdx + 1
	if numFeatures <= 0 {
		numFeatures = len(model.FeatureNames)
	}
	if numFeatures <= 0 {
		return nil, fmt.Errorf("could not determine number of features")
	}

	ensemble := &TreeEnsemble{
		NumTrees:     len(model.Trees),
		NumFeatures:  numFeatures,
		FeatureNames: model.FeatureNames,
		Roots:        make([]int, len(model.Trees)),
		BaseScore:    0,
		Objective:    model.Objective,
	}

	nodeOffset := 0
	for treeIdx, tree := range model.Trees {
		ensemble.Roots[treeIdx] = nodeOffset
		nodes, err := convertLightGBMTextTree(&tree, treeIdx, nodeOffset)
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

// convertLightGBMTextTree converts a text-parsed tree to nodes.
//
// LightGBM text format stores trees in a compact array format:
//   - Internal nodes: indices 0 to num_internal_nodes-1
//   - Leaves: encoded as negative indices in left_child/right_child
//   - left_child[i] < 0 means it's a leaf with index ~left_child[i]
func convertLightGBMTextTree(tree *LightGBMTextTree, treeIdx, nodeOffset int) ([]Node, error) {
	if tree.NumLeaves <= 0 {
		return nil, fmt.Errorf("tree has no leaves")
	}

	// Number of internal nodes = num_leaves - 1
	numInternal := tree.NumLeaves - 1
	totalNodes := numInternal + tree.NumLeaves
	nodes := make([]Node, totalNodes)

	// Convert internal nodes (indices 0 to numInternal-1)
	for i := 0; i < numInternal; i++ {
		if i >= len(tree.SplitFeature) {
			return nil, fmt.Errorf("missing split_feature for internal node %d", i)
		}

		node := Node{
			Tree:    treeIdx,
			NodeID:  i,
			IsLeaf:  false,
			Feature: tree.SplitFeature[i],
		}

		if i < len(tree.Threshold) {
			node.Threshold = tree.Threshold[i]
		}

		// Decision type
		if i < len(tree.DecisionType) {
			switch tree.DecisionType[i] {
			case "<=":
				node.DecisionType = DecisionLessEqual
			case "<":
				node.DecisionType = DecisionLess
			default:
				node.DecisionType = DecisionLessEqual
			}
		} else {
			node.DecisionType = DecisionLessEqual
		}

		// Child indices: negative means leaf (bitwise complement ~x = -x-1)
		if i < len(tree.LeftChild) {
			leftIdx := tree.LeftChild[i]
			if leftIdx < 0 {
				// It's a leaf: convert to node index
				leafIdx := ^leftIdx // Bitwise NOT to get leaf index
				node.Yes = numInternal + leafIdx + nodeOffset
			} else {
				node.Yes = leftIdx + nodeOffset
			}
		}

		if i < len(tree.RightChild) {
			rightIdx := tree.RightChild[i]
			if rightIdx < 0 {
				leafIdx := ^rightIdx
				node.No = numInternal + leafIdx + nodeOffset
			} else {
				node.No = rightIdx + nodeOffset
			}
		}

		// Missing defaults to left in LightGBM
		node.Missing = node.Yes

		// Cover
		if i < len(tree.InternalCount) {
			node.Cover = float64(tree.InternalCount[i])
		}

		nodes[i] = node
	}

	// Convert leaf nodes (indices numInternal to totalNodes-1)
	for i := 0; i < tree.NumLeaves; i++ {
		nodeIdx := numInternal + i
		node := Node{
			Tree:    treeIdx,
			NodeID:  nodeIdx,
			IsLeaf:  true,
			Feature: -1,
			Yes:     -1,
			No:      -1,
			Missing: -1,
		}

		if i < len(tree.LeafValue) {
			node.Prediction = tree.LeafValue[i]
		}

		if i < len(tree.LeafCount) {
			node.Cover = float64(tree.LeafCount[i])
		} else if i < len(tree.LeafWeight) {
			node.Cover = tree.LeafWeight[i]
		}

		nodes[nodeIdx] = node
	}

	return nodes, nil
}
