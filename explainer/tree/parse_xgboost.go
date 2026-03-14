package tree

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// XGBoostModel represents the top-level structure of an XGBoost JSON model.
type XGBoostModel struct {
	Learner XGBoostLearner `json:"learner"`
	Version []int          `json:"version"`
}

// XGBoostLearner contains the gradient booster and objective.
type XGBoostLearner struct {
	Attributes        map[string]string        `json:"attributes"`
	FeatureNames      []string                 `json:"feature_names"`
	FeatureTypes      []string                 `json:"feature_types"`
	GradientBooster   XGBoostGradientBooster   `json:"gradient_booster"`
	Objective         XGBoostObjective         `json:"objective"`
	LearnerModelParam XGBoostLearnerModelParam `json:"learner_model_param"`
}

// XGBoostGradientBooster contains the tree model.
type XGBoostGradientBooster struct {
	Name  string        `json:"name"`
	Model XGBoostGBTree `json:"model"`
}

// XGBoostGBTree contains the individual trees.
type XGBoostGBTree struct {
	GBTreeModelParam XGBoostGBTreeModelParam `json:"gbtree_model_param"`
	Trees            []XGBoostTree           `json:"trees"`
	TreeInfo         []int                   `json:"tree_info"`
}

// XGBoostGBTreeModelParam contains model parameters.
type XGBoostGBTreeModelParam struct {
	NumTrees string `json:"num_trees"`
}

// XGBoostTree represents a single decision tree.
type XGBoostTree struct {
	TreeParam          XGBoostTreeParam `json:"tree_param"`
	ID                 int              `json:"id"`
	SplitIndices       []int            `json:"split_indices"`
	SplitConditions    []float64        `json:"split_conditions"`
	SplitType          []int            `json:"split_type"`
	LeftChildren       []int            `json:"left_children"`
	RightChildren      []int            `json:"right_children"`
	Parents            []int            `json:"parents"`
	DefaultLeft        []int            `json:"default_left"`
	BaseWeights        []float64        `json:"base_weights"`
	Categories         []int            `json:"categories"`
	CategoriesNodes    []int            `json:"categories_nodes"`
	CategoriesSegments []int            `json:"categories_segments"`
	CategoriesSizes    []int            `json:"categories_sizes"`
	SumHessian         []float64        `json:"sum_hessian"`
	LossChanges        []float64        `json:"loss_changes"`
}

// XGBoostTreeParam contains tree-specific parameters.
type XGBoostTreeParam struct {
	NumDeleted     string `json:"num_deleted"`
	NumFeature     string `json:"num_feature"`
	NumNodes       string `json:"num_nodes"`
	SizeLeafVector string `json:"size_leaf_vector"`
}

// XGBoostObjective contains objective function details.
type XGBoostObjective struct {
	Name string `json:"name"`
	Reg  struct {
		LossParam map[string]string `json:"loss_param"`
	} `json:"reg_loss_param"`
}

// XGBoostLearnerModelParam contains learner parameters.
type XGBoostLearnerModelParam struct {
	BaseScore  string `json:"base_score"`
	NumClass   string `json:"num_class"`
	NumFeature string `json:"num_feature"`
}

// ParseXGBoostJSON parses an XGBoost JSON model into a TreeEnsemble.
func ParseXGBoostJSON(data []byte) (*TreeEnsemble, error) {
	var model XGBoostModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("failed to parse XGBoost JSON: %w", err)
	}

	return convertXGBoostModel(&model)
}

// LoadXGBoostModel loads an XGBoost model from a JSON file.
func LoadXGBoostModel(path string) (*TreeEnsemble, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	return LoadXGBoostModelFromReader(f)
}

// LoadXGBoostModelFromReader loads an XGBoost model from an io.Reader.
func LoadXGBoostModelFromReader(r io.Reader) (*TreeEnsemble, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read data: %w", err)
	}
	return ParseXGBoostJSON(data)
}

// convertXGBoostModel converts an XGBoostModel to a TreeEnsemble.
func convertXGBoostModel(model *XGBoostModel) (*TreeEnsemble, error) {
	learner := &model.Learner
	gbtree := &learner.GradientBooster.Model
	trees := gbtree.Trees

	if len(trees) == 0 {
		return nil, fmt.Errorf("model has no trees")
	}

	// Parse number of features
	numFeatures := 0
	if len(trees) > 0 && trees[0].TreeParam.NumFeature != "" {
		if _, err := fmt.Sscanf(trees[0].TreeParam.NumFeature, "%d", &numFeatures); err != nil {
			return nil, fmt.Errorf("failed to parse num_feature: %w", err)
		}
	}
	if numFeatures == 0 && learner.LearnerModelParam.NumFeature != "" {
		if _, err := fmt.Sscanf(learner.LearnerModelParam.NumFeature, "%d", &numFeatures); err != nil {
			return nil, fmt.Errorf("failed to parse num_feature from learner: %w", err)
		}
	}
	if numFeatures == 0 {
		return nil, fmt.Errorf("could not determine number of features")
	}

	// Parse base score
	baseScore := 0.5 // default for binary classification
	if learner.LearnerModelParam.BaseScore != "" {
		if _, err := fmt.Sscanf(learner.LearnerModelParam.BaseScore, "%f", &baseScore); err != nil {
			// Use default if parsing fails
			baseScore = 0.5
		}
	}

	// Convert trees
	ensemble := &TreeEnsemble{
		NumTrees:     len(trees),
		NumFeatures:  numFeatures,
		FeatureNames: learner.FeatureNames,
		Roots:        make([]int, len(trees)),
		BaseScore:    baseScore,
		Objective:    learner.Objective.Name,
	}

	nodeOffset := 0
	for treeIdx, tree := range trees {
		ensemble.Roots[treeIdx] = nodeOffset
		nodes, err := convertXGBoostTree(&tree, treeIdx, nodeOffset)
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

// convertXGBoostTree converts a single XGBoost tree to nodes.
func convertXGBoostTree(tree *XGBoostTree, treeIdx, nodeOffset int) ([]Node, error) {
	numNodes := len(tree.BaseWeights)
	if numNodes == 0 {
		return nil, fmt.Errorf("tree has no nodes")
	}

	nodes := make([]Node, numNodes)

	for i := 0; i < numNodes; i++ {
		leftChild := tree.LeftChildren[i]
		rightChild := tree.RightChildren[i]
		isLeaf := leftChild == -1 && rightChild == -1

		node := Node{
			Tree:       treeIdx,
			NodeID:     i,
			IsLeaf:     isLeaf,
			Prediction: 0,
		}

		if isLeaf {
			// Leaf node
			node.Feature = -1
			node.Yes = -1
			node.No = -1
			node.Missing = -1
			node.Prediction = tree.BaseWeights[i]
		} else {
			// Internal node
			node.Feature = tree.SplitIndices[i]
			node.Threshold = tree.SplitConditions[i]
			node.DecisionType = DecisionLess // XGBoost uses <

			// Convert children to absolute indices
			node.Yes = leftChild + nodeOffset
			node.No = rightChild + nodeOffset

			// Handle missing values
			if i < len(tree.DefaultLeft) && tree.DefaultLeft[i] == 1 {
				node.Missing = node.Yes
			} else {
				node.Missing = node.No
			}
		}

		// Set cover (sum of hessians)
		if i < len(tree.SumHessian) {
			node.Cover = tree.SumHessian[i]
		}

		nodes[i] = node
	}

	return nodes, nil
}
