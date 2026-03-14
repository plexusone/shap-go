package tree

import (
	"context"
	"encoding/json"
	"testing"
)

// Sample LightGBM model JSON for testing.
// This represents a simple tree: feature 0 <= 0.5 -> leaf -0.5, else -> leaf 0.5
var simpleLightGBMModel = `{
  "name": "tree",
  "version": "v3",
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "label_index": 0,
  "max_feature_idx": 0,
  "objective": "regression",
  "average_output": false,
  "feature_names": ["x0"],
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 2,
      "num_cat": 0,
      "shrinkage": 1.0,
      "tree_structure": {
        "split_index": 0,
        "split_feature": 0,
        "split_gain": 100.0,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": true,
        "internal_value": 0.0,
        "internal_count": 100,
        "left_child": {
          "leaf_index": 0,
          "leaf_value": -0.5,
          "leaf_count": 50
        },
        "right_child": {
          "leaf_index": 1,
          "leaf_value": 0.5,
          "leaf_count": 50
        }
      }
    }
  ]
}`

// Two-feature model: x0 <= 0.5 -> (x1 <= 0.5 -> 1.0, else -> 2.0), else -> 4.0
var twoFeatureLightGBMModel = `{
  "name": "tree",
  "version": "v3",
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "label_index": 0,
  "max_feature_idx": 1,
  "objective": "regression",
  "average_output": false,
  "feature_names": ["x0", "x1"],
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 3,
      "num_cat": 0,
      "shrinkage": 1.0,
      "tree_structure": {
        "split_index": 0,
        "split_feature": 0,
        "split_gain": 100.0,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": true,
        "internal_value": 2.75,
        "internal_count": 100,
        "left_child": {
          "split_index": 1,
          "split_feature": 1,
          "split_gain": 50.0,
          "threshold": 0.5,
          "decision_type": "<=",
          "default_left": true,
          "internal_value": 1.5,
          "internal_count": 50,
          "left_child": {
            "leaf_index": 0,
            "leaf_value": 1.0,
            "leaf_count": 25
          },
          "right_child": {
            "leaf_index": 1,
            "leaf_value": 2.0,
            "leaf_count": 25
          }
        },
        "right_child": {
          "leaf_index": 2,
          "leaf_value": 4.0,
          "leaf_count": 50
        }
      }
    }
  ]
}`

// Two-tree ensemble
var ensembleLightGBMModel = `{
  "name": "tree",
  "version": "v3",
  "num_class": 1,
  "num_tree_per_iteration": 1,
  "label_index": 0,
  "max_feature_idx": 0,
  "objective": "regression",
  "average_output": false,
  "feature_names": ["x0"],
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 2,
      "num_cat": 0,
      "shrinkage": 1.0,
      "tree_structure": {
        "split_index": 0,
        "split_feature": 0,
        "split_gain": 100.0,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": true,
        "internal_value": 0.0,
        "internal_count": 100,
        "left_child": {
          "leaf_index": 0,
          "leaf_value": 1.0,
          "leaf_count": 50
        },
        "right_child": {
          "leaf_index": 1,
          "leaf_value": 2.0,
          "leaf_count": 50
        }
      }
    },
    {
      "tree_index": 1,
      "num_leaves": 2,
      "num_cat": 0,
      "shrinkage": 1.0,
      "tree_structure": {
        "split_index": 0,
        "split_feature": 0,
        "split_gain": 50.0,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": true,
        "internal_value": 0.0,
        "internal_count": 100,
        "left_child": {
          "leaf_index": 0,
          "leaf_value": 0.5,
          "leaf_count": 50
        },
        "right_child": {
          "leaf_index": 1,
          "leaf_value": 1.5,
          "leaf_count": 50
        }
      }
    }
  ]
}`

func TestParseLightGBMJSON_SimpleTree(t *testing.T) {
	ensemble, err := ParseLightGBMJSON([]byte(simpleLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	// Verify basic structure
	if ensemble.NumTrees != 1 {
		t.Errorf("Expected 1 tree, got %d", ensemble.NumTrees)
	}
	if ensemble.NumFeatures != 1 {
		t.Errorf("Expected 1 feature, got %d", ensemble.NumFeatures)
	}
	if len(ensemble.Nodes) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(ensemble.Nodes))
	}

	// Verify root node
	root := ensemble.Nodes[0]
	if root.IsLeaf {
		t.Error("Root should not be a leaf")
	}
	if root.Feature != 0 {
		t.Errorf("Root feature should be 0, got %d", root.Feature)
	}
	if root.Threshold != 0.5 {
		t.Errorf("Root threshold should be 0.5, got %f", root.Threshold)
	}

	// Verify leaves
	leftLeaf := ensemble.Nodes[1]
	if !leftLeaf.IsLeaf {
		t.Error("Left child should be a leaf")
	}
	if leftLeaf.Prediction != -0.5 {
		t.Errorf("Left leaf value should be -0.5, got %f", leftLeaf.Prediction)
	}

	rightLeaf := ensemble.Nodes[2]
	if !rightLeaf.IsLeaf {
		t.Error("Right child should be a leaf")
	}
	if rightLeaf.Prediction != 0.5 {
		t.Errorf("Right leaf value should be 0.5, got %f", rightLeaf.Prediction)
	}
}

func TestParseLightGBMJSON_TwoFeatureTree(t *testing.T) {
	ensemble, err := ParseLightGBMJSON([]byte(twoFeatureLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("Expected 1 tree, got %d", ensemble.NumTrees)
	}
	if ensemble.NumFeatures != 2 {
		t.Errorf("Expected 2 features, got %d", ensemble.NumFeatures)
	}
	if len(ensemble.Nodes) != 5 {
		t.Errorf("Expected 5 nodes, got %d", len(ensemble.Nodes))
	}

	// Verify tree structure
	// Node 0: root (feature 0)
	// Node 1: left child of root (feature 1)
	// Node 2: left-left leaf (value 1.0)
	// Node 3: left-right leaf (value 2.0)
	// Node 4: right leaf (value 4.0)

	root := ensemble.Nodes[0]
	if root.Feature != 0 {
		t.Errorf("Root feature should be 0, got %d", root.Feature)
	}

	leftChild := ensemble.Nodes[1]
	if leftChild.Feature != 1 {
		t.Errorf("Left child feature should be 1, got %d", leftChild.Feature)
	}

	// Check leaf values
	leafValues := make(map[float64]bool)
	for _, node := range ensemble.Nodes {
		if node.IsLeaf {
			leafValues[node.Prediction] = true
		}
	}
	expectedLeaves := []float64{1.0, 2.0, 4.0}
	for _, v := range expectedLeaves {
		if !leafValues[v] {
			t.Errorf("Expected leaf value %f not found", v)
		}
	}
}

func TestParseLightGBMJSON_Ensemble(t *testing.T) {
	ensemble, err := ParseLightGBMJSON([]byte(ensembleLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	if ensemble.NumTrees != 2 {
		t.Errorf("Expected 2 trees, got %d", ensemble.NumTrees)
	}
	if len(ensemble.Roots) != 2 {
		t.Errorf("Expected 2 roots, got %d", len(ensemble.Roots))
	}

	// Tree 0 starts at node 0, Tree 1 starts at node 3
	if ensemble.Roots[0] != 0 {
		t.Errorf("Tree 0 root should be 0, got %d", ensemble.Roots[0])
	}
	if ensemble.Roots[1] != 3 {
		t.Errorf("Tree 1 root should be 3, got %d", ensemble.Roots[1])
	}

	// Total nodes: 3 (tree 0) + 3 (tree 1) = 6
	if len(ensemble.Nodes) != 6 {
		t.Errorf("Expected 6 nodes, got %d", len(ensemble.Nodes))
	}
}

func TestLightGBMNodeIsLeaf(t *testing.T) {
	// Internal node
	internal := &LightGBMNode{
		SplitIndex:   0,
		SplitFeature: 0,
		LeftChild:    &LightGBMNode{LeafValue: 1.0},
		RightChild:   &LightGBMNode{LeafValue: 2.0},
	}
	if internal.IsLeaf() {
		t.Error("Internal node should not be a leaf")
	}

	// Leaf node
	leaf := &LightGBMNode{
		LeafIndex: 0,
		LeafValue: 1.0,
	}
	if !leaf.IsLeaf() {
		t.Error("Leaf node should be a leaf")
	}
}

func TestParseLightGBMJSON_DecisionTypes(t *testing.T) {
	// Test with <= decision type (default LightGBM)
	ensemble, err := ParseLightGBMJSON([]byte(simpleLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	root := ensemble.Nodes[0]
	if root.DecisionType != DecisionLessEqual {
		t.Errorf("Expected DecisionLessEqual, got %s", root.DecisionType)
	}
}

func TestParseLightGBMJSON_FeatureNames(t *testing.T) {
	ensemble, err := ParseLightGBMJSON([]byte(twoFeatureLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	if len(ensemble.FeatureNames) != 2 {
		t.Errorf("Expected 2 feature names, got %d", len(ensemble.FeatureNames))
	}
	if ensemble.FeatureNames[0] != "x0" {
		t.Errorf("Expected feature name 'x0', got '%s'", ensemble.FeatureNames[0])
	}
	if ensemble.FeatureNames[1] != "x1" {
		t.Errorf("Expected feature name 'x1', got '%s'", ensemble.FeatureNames[1])
	}
}

func TestParseLightGBMJSON_InvalidJSON(t *testing.T) {
	_, err := ParseLightGBMJSON([]byte("not valid json"))
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}

func TestParseLightGBMJSON_EmptyModel(t *testing.T) {
	emptyModel := `{"tree_info": []}`
	_, err := ParseLightGBMJSON([]byte(emptyModel))
	if err == nil {
		t.Error("Expected error for empty model")
	}
}

func TestLightGBMWithTreeSHAP(t *testing.T) {
	// Integration test: parse LightGBM model and compute SHAP values
	ensemble, err := ParseLightGBMJSON([]byte(twoFeatureLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	// Create explainer
	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Test instances
	tests := []struct {
		name     string
		instance []float64
		wantPred float64
	}{
		{"x0<=0.5,x1<=0.5", []float64{0.3, 0.3}, 1.0},
		{"x0<=0.5,x1>0.5", []float64{0.3, 0.7}, 2.0},
		{"x0>0.5", []float64{0.7, 0.3}, 4.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := exp.Explain(context.Background(), tt.instance)
			if err != nil {
				t.Fatalf("Explain failed: %v", err)
			}

			// Check prediction
			if result.Prediction != tt.wantPred {
				t.Errorf("Prediction = %f, want %f", result.Prediction, tt.wantPred)
			}

			// Verify local accuracy: sum(SHAP) = prediction - base_value
			verifyResult := result.Verify(1e-6)
			if !verifyResult.Valid {
				t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
					verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
			}
		})
	}
}

func TestLightGBMEnsembleWithTreeSHAP(t *testing.T) {
	// Integration test: parse LightGBM ensemble and compute SHAP values
	ensemble, err := ParseLightGBMJSON([]byte(ensembleLightGBMModel))
	if err != nil {
		t.Fatalf("ParseLightGBMJSON failed: %v", err)
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	tests := []struct {
		name     string
		instance []float64
		wantPred float64
	}{
		{"x0<=0.5", []float64{0.3}, 1.5}, // 1.0 + 0.5
		{"x0>0.5", []float64{0.7}, 3.5},  // 2.0 + 1.5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := exp.Explain(context.Background(), tt.instance)
			if err != nil {
				t.Fatalf("Explain failed: %v", err)
			}

			if result.Prediction != tt.wantPred {
				t.Errorf("Prediction = %f, want %f", result.Prediction, tt.wantPred)
			}

			verifyResult := result.Verify(1e-6)
			if !verifyResult.Valid {
				t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
					verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
			}
		})
	}
}

func TestLightGBMModelStructure(t *testing.T) {
	// Test that our struct correctly parses all fields
	var model LightGBMModel
	if err := json.Unmarshal([]byte(twoFeatureLightGBMModel), &model); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if model.NumClass != 1 {
		t.Errorf("Expected num_class=1, got %d", model.NumClass)
	}
	if model.MaxFeatureIdx != 1 {
		t.Errorf("Expected max_feature_idx=1, got %d", model.MaxFeatureIdx)
	}
	if model.Objective != "regression" {
		t.Errorf("Expected objective='regression', got '%s'", model.Objective)
	}
	if len(model.TreeInfo) != 1 {
		t.Errorf("Expected 1 tree, got %d", len(model.TreeInfo))
	}

	tree := model.TreeInfo[0]
	if tree.NumLeaves != 3 {
		t.Errorf("Expected 3 leaves, got %d", tree.NumLeaves)
	}
	if tree.TreeStructure == nil {
		t.Fatal("Tree structure is nil")
	}
	if tree.TreeStructure.SplitFeature != 0 {
		t.Errorf("Expected root split_feature=0, got %d", tree.TreeStructure.SplitFeature)
	}
}
