package tree

import (
	"context"
	"math"
	"strings"
	"testing"
)

// TestRepeatedFeatureSplits tests TreeSHAP on a tree where the same feature
// appears at multiple levels. This exercises the unwindPath code path.
//
// Tree structure:
//
//	    [0] x0 < 0.5
//	   /          \
//	[1] x0 < 0.25  [2] 4.0
//	   /      \
//	[3] 1.0  [4] 2.0
//
// This tree has x0 splitting at the root AND at node 1, which triggers
// the unwindPath logic in the TreeSHAP algorithm.
func TestRepeatedFeatureSplits(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
			// Root: split on x0 < 0.5
			{
				Tree:         0,
				NodeID:       0,
				Feature:      0,
				DecisionType: DecisionLess,
				Threshold:    0.5,
				Yes:          1,
				No:           2,
				Missing:      1,
				IsLeaf:       false,
				Cover:        100,
			},
			// Node 1: split on x0 < 0.25 (same feature!)
			{
				Tree:         0,
				NodeID:       1,
				Feature:      0, // Same feature as root
				DecisionType: DecisionLess,
				Threshold:    0.25,
				Yes:          3,
				No:           4,
				Missing:      3,
				IsLeaf:       false,
				Cover:        50,
			},
			// Node 2: leaf, prediction = 4.0
			{
				Tree:       0,
				NodeID:     2,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				IsLeaf:     true,
				Prediction: 4.0,
				Cover:      50,
			},
			// Node 3: leaf, prediction = 1.0
			{
				Tree:       0,
				NodeID:     3,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				IsLeaf:     true,
				Prediction: 1.0,
				Cover:      25,
			},
			// Node 4: leaf, prediction = 2.0
			{
				Tree:       0,
				NodeID:     4,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				IsLeaf:     true,
				Prediction: 2.0,
				Cover:      25,
			},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	testCases := []struct {
		name     string
		instance []float64
		wantPred float64
	}{
		{"x0=0.1 (far left)", []float64{0.1}, 1.0},   // < 0.5, < 0.25 -> 1.0
		{"x0=0.35 (mid left)", []float64{0.35}, 2.0}, // < 0.5, >= 0.25 -> 2.0
		{"x0=0.7 (right)", []float64{0.7}, 4.0},      // >= 0.5 -> 4.0
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := exp.Explain(ctx, tc.instance)
			if err != nil {
				t.Fatalf("Explain failed: %v", err)
			}

			// Check prediction
			if math.Abs(result.Prediction-tc.wantPred) > 1e-9 {
				t.Errorf("expected prediction %f, got %f", tc.wantPred, result.Prediction)
			}

			// Verify local accuracy
			verifyResult := result.Verify(1e-6)
			if !verifyResult.Valid {
				t.Errorf("local accuracy failed: sum=%f, expected=%f, diff=%f",
					verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
			}
		})
	}
}

// TestEnsembleUtilityMethods tests TreeEnsemble utility methods.
func TestEnsembleUtilityMethods(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    2,
		NumFeatures: 2,
		Roots:       []int{0, 4},
		BaseScore:   0.0,
		Nodes: []Node{
			// Tree 0 (3 levels deep)
			{Tree: 0, NodeID: 0, Feature: 0, DecisionType: DecisionLess, Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100},
			{Tree: 0, NodeID: 1, Feature: 1, DecisionType: DecisionLess, Threshold: 0.5, Yes: 3, No: 3, Missing: 3, Cover: 50},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 3.0, Cover: 50},
			{Tree: 0, NodeID: 3, IsLeaf: true, Prediction: 1.0, Cover: 50},
			// Tree 1 (single level)
			{Tree: 1, NodeID: 0, Feature: 0, DecisionType: DecisionLess, Threshold: 0.5, Yes: 5, No: 6, Missing: 5, Cover: 100},
			{Tree: 1, NodeID: 1, IsLeaf: true, Prediction: 0.5, Cover: 50},
			{Tree: 1, NodeID: 2, IsLeaf: true, Prediction: 1.5, Cover: 50},
		},
	}

	t.Run("TreeNodes", func(t *testing.T) {
		// Valid tree index
		nodes := ensemble.TreeNodes(0)
		if len(nodes) != 4 {
			t.Errorf("expected 4 nodes for tree 0, got %d", len(nodes))
		}

		nodes = ensemble.TreeNodes(1)
		if len(nodes) != 3 {
			t.Errorf("expected 3 nodes for tree 1, got %d", len(nodes))
		}

		// Invalid tree index (negative)
		nodes = ensemble.TreeNodes(-1)
		if nodes != nil {
			t.Error("expected nil for negative tree index")
		}

		// Invalid tree index (too large)
		nodes = ensemble.TreeNodes(10)
		if nodes != nil {
			t.Error("expected nil for out-of-range tree index")
		}
	})

	t.Run("MaxDepth", func(t *testing.T) {
		// Tree 0 has depth 2 (root -> node1 -> node3)
		// Tree 1 has depth 1 (root -> leaf)
		maxDepth := ensemble.MaxDepth()
		if maxDepth != 2 {
			t.Errorf("expected max depth 2, got %d", maxDepth)
		}
	})

	t.Run("ToJSON_and_FromJSON", func(t *testing.T) {
		// Serialize
		jsonData, err := ensemble.ToJSON()
		if err != nil {
			t.Fatalf("ToJSON failed: %v", err)
		}
		if len(jsonData) == 0 {
			t.Error("ToJSON returned empty data")
		}

		// Deserialize
		loaded, err := EnsembleFromJSON(jsonData)
		if err != nil {
			t.Fatalf("EnsembleFromJSON failed: %v", err)
		}

		// Verify
		if loaded.NumTrees != ensemble.NumTrees {
			t.Errorf("expected %d trees, got %d", ensemble.NumTrees, loaded.NumTrees)
		}
		if len(loaded.Nodes) != len(ensemble.Nodes) {
			t.Errorf("expected %d nodes, got %d", len(ensemble.Nodes), len(loaded.Nodes))
		}
	})

	t.Run("ToJSONPretty", func(t *testing.T) {
		prettyJSON, err := ensemble.ToJSONPretty()
		if err != nil {
			t.Fatalf("ToJSONPretty failed: %v", err)
		}
		if !strings.Contains(string(prettyJSON), "\n") {
			t.Error("ToJSONPretty should produce formatted JSON with newlines")
		}
	})

	t.Run("LoadEnsembleFromReader", func(t *testing.T) {
		jsonData, _ := ensemble.ToJSON()
		reader := strings.NewReader(string(jsonData))

		loaded, err := LoadEnsembleFromReader(reader)
		if err != nil {
			t.Fatalf("LoadEnsembleFromReader failed: %v", err)
		}
		if loaded.NumTrees != ensemble.NumTrees {
			t.Errorf("expected %d trees, got %d", ensemble.NumTrees, loaded.NumTrees)
		}
	})
}

// TestEnsembleFromJSONInvalid tests error handling in EnsembleFromJSON.
func TestEnsembleFromJSONInvalid(t *testing.T) {
	t.Run("invalid_json", func(t *testing.T) {
		_, err := EnsembleFromJSON([]byte("not valid json"))
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})

	t.Run("invalid_ensemble", func(t *testing.T) {
		// Valid JSON but invalid ensemble (no nodes)
		_, err := EnsembleFromJSON([]byte(`{"num_trees":1,"num_features":1,"roots":[0],"nodes":[]}`))
		if err == nil {
			t.Error("expected error for invalid ensemble")
		}
	})
}

// TestExplainerAccessors tests Explainer accessor methods.
func TestExplainerAccessors(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:     1,
		NumFeatures:  2,
		FeatureNames: []string{"feature_a", "feature_b"},
		Roots:        []int{0},
		Nodes: []Node{
			{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	t.Run("FeatureNames", func(t *testing.T) {
		names := exp.FeatureNames()
		if len(names) != 2 {
			t.Errorf("expected 2 feature names, got %d", len(names))
		}
	})

	t.Run("Ensemble", func(t *testing.T) {
		ens := exp.Ensemble()
		if ens == nil {
			t.Error("Ensemble() returned nil")
		}
		if ens != ensemble {
			t.Error("Ensemble() should return the same ensemble")
		}
	})
}

// TestXGBoostJSONParsing tests XGBoost JSON parsing.
func TestXGBoostJSONParsing(t *testing.T) {
	// Minimal XGBoost JSON model structure
	xgboostJSON := `{
		"learner": {
			"attributes": {},
			"feature_names": ["x0", "x1"],
			"feature_types": ["float", "float"],
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {"num_trees": "1"},
					"trees": [{
						"tree_param": {
							"num_deleted": "0",
							"num_feature": "2",
							"num_nodes": "3",
							"size_leaf_vector": "0"
						},
						"id": 0,
						"split_indices": [0, 0, 0],
						"split_conditions": [0.5, 0.0, 0.0],
						"split_type": [0, 0, 0],
						"left_children": [1, -1, -1],
						"right_children": [2, -1, -1],
						"parents": [-1, 0, 0],
						"default_left": [1, 0, 0],
						"base_weights": [0.0, 1.0, 2.0],
						"sum_hessian": [100.0, 50.0, 50.0]
					}],
					"tree_info": [0]
				}
			},
			"objective": {"name": "reg:squarederror"},
			"learner_model_param": {
				"base_score": "0.5",
				"num_class": "0",
				"num_feature": "2"
			}
		},
		"version": [2, 1, 0]
	}`

	t.Run("ParseXGBoostJSON", func(t *testing.T) {
		ensemble, err := ParseXGBoostJSON([]byte(xgboostJSON))
		if err != nil {
			t.Fatalf("ParseXGBoostJSON failed: %v", err)
		}

		if ensemble.NumTrees != 1 {
			t.Errorf("expected 1 tree, got %d", ensemble.NumTrees)
		}
		if ensemble.NumFeatures != 2 {
			t.Errorf("expected 2 features, got %d", ensemble.NumFeatures)
		}
		if len(ensemble.Nodes) != 3 {
			t.Errorf("expected 3 nodes, got %d", len(ensemble.Nodes))
		}
		if math.Abs(ensemble.BaseScore-0.5) > 1e-9 {
			t.Errorf("expected base_score 0.5, got %f", ensemble.BaseScore)
		}
	})

	t.Run("LoadXGBoostModelFromReader", func(t *testing.T) {
		reader := strings.NewReader(xgboostJSON)
		ensemble, err := LoadXGBoostModelFromReader(reader)
		if err != nil {
			t.Fatalf("LoadXGBoostModelFromReader failed: %v", err)
		}
		if ensemble.NumTrees != 1 {
			t.Errorf("expected 1 tree, got %d", ensemble.NumTrees)
		}
	})

	t.Run("invalid_json", func(t *testing.T) {
		_, err := ParseXGBoostJSON([]byte("not valid json"))
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})

	t.Run("no_trees", func(t *testing.T) {
		emptyJSON := `{
			"learner": {
				"gradient_booster": {
					"name": "gbtree",
					"model": {"trees": []}
				},
				"learner_model_param": {}
			}
		}`
		_, err := ParseXGBoostJSON([]byte(emptyJSON))
		if err == nil {
			t.Error("expected error for model with no trees")
		}
	})
}

// TestXGBoostExplainerIntegration tests creating an explainer from XGBoost JSON.
func TestXGBoostExplainerIntegration(t *testing.T) {
	// Simple XGBoost model: x0 < 0.5 -> 1.0, else -> 2.0
	xgboostJSON := `{
		"learner": {
			"feature_names": ["x0"],
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {"num_trees": "1"},
					"trees": [{
						"tree_param": {"num_feature": "1", "num_nodes": "3"},
						"id": 0,
						"split_indices": [0, 0, 0],
						"split_conditions": [0.5, 0.0, 0.0],
						"split_type": [0, 0, 0],
						"left_children": [1, -1, -1],
						"right_children": [2, -1, -1],
						"parents": [-1, 0, 0],
						"default_left": [1, 0, 0],
						"base_weights": [0.0, 1.0, 2.0],
						"sum_hessian": [100.0, 50.0, 50.0]
					}]
				}
			},
			"objective": {"name": "reg:squarederror"},
			"learner_model_param": {"base_score": "0.0", "num_feature": "1"}
		},
		"version": [2, 1, 0]
	}`

	ensemble, err := ParseXGBoostJSON([]byte(xgboostJSON))
	if err != nil {
		t.Fatalf("ParseXGBoostJSON failed: %v", err)
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx := context.Background()

	t.Run("left_branch", func(t *testing.T) {
		result, err := exp.Explain(ctx, []float64{0.3})
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		if math.Abs(result.Prediction-1.0) > 1e-9 {
			t.Errorf("expected prediction 1.0, got %f", result.Prediction)
		}

		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: diff=%f", verifyResult.Difference)
		}
	})

	t.Run("right_branch", func(t *testing.T) {
		result, err := exp.Explain(ctx, []float64{0.7})
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		if math.Abs(result.Prediction-2.0) > 1e-9 {
			t.Errorf("expected prediction 2.0, got %f", result.Prediction)
		}

		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: diff=%f", verifyResult.Difference)
		}
	})
}

// TestDecisionTypeLessEqual tests DecisionLessEqual comparison.
func TestDecisionTypeLessEqual(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		Nodes: []Node{
			{
				Tree:         0,
				NodeID:       0,
				Feature:      0,
				DecisionType: DecisionLessEqual, // Use <= instead of <
				Threshold:    0.5,
				Yes:          1,
				No:           2,
				Missing:      1,
				Cover:        100,
			},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0, Cover: 50},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0, Cover: 50},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	// With <=, exactly 0.5 should go LEFT
	t.Run("exactly_at_threshold", func(t *testing.T) {
		result, err := exp.Explain(ctx, []float64{0.5})
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		// 0.5 <= 0.5 is true, so should go left to leaf 1.0
		if math.Abs(result.Prediction-1.0) > 1e-9 {
			t.Errorf("expected prediction 1.0 (left branch), got %f", result.Prediction)
		}
	})

	t.Run("just_above_threshold", func(t *testing.T) {
		result, err := exp.Explain(ctx, []float64{0.50001})
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		// 0.50001 <= 0.5 is false, so should go right to leaf 2.0
		if math.Abs(result.Prediction-2.0) > 1e-9 {
			t.Errorf("expected prediction 2.0 (right branch), got %f", result.Prediction)
		}
	})
}

// TestMissingValueHandling tests NaN (missing value) handling.
func TestMissingValueHandling(t *testing.T) {
	// Tree with missing values going to right (default_left = false)
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		Nodes: []Node{
			{
				Tree:         0,
				NodeID:       0,
				Feature:      0,
				DecisionType: DecisionLess,
				Threshold:    0.5,
				Yes:          1,
				No:           2,
				Missing:      2, // Missing goes to No (right)
				Cover:        100,
			},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0, Cover: 50},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0, Cover: 50},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	result, err := exp.Explain(ctx, []float64{math.NaN()})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// NaN should go to Missing branch (right, prediction 2.0)
	if math.Abs(result.Prediction-2.0) > 1e-9 {
		t.Errorf("expected prediction 2.0 for NaN, got %f", result.Prediction)
	}
}

// TestValidationErrors tests ensemble validation error paths.
func TestValidationErrors(t *testing.T) {
	t.Run("invalid_feature_index", func(t *testing.T) {
		ensemble := &TreeEnsemble{
			NumTrees:    1,
			NumFeatures: 2,
			Roots:       []int{0},
			Nodes: []Node{
				{Tree: 0, NodeID: 0, Feature: 5, Yes: 1, No: 2, Cover: 100}, // Feature 5 > NumFeatures-1
				{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0},
				{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0},
			},
		}

		err := ensemble.Validate()
		if err == nil {
			t.Error("expected validation error for invalid feature index")
		}
	})

	t.Run("invalid_child_index", func(t *testing.T) {
		ensemble := &TreeEnsemble{
			NumTrees:    1,
			NumFeatures: 2,
			Roots:       []int{0},
			Nodes: []Node{
				{Tree: 0, NodeID: 0, Feature: 0, Yes: 10, No: 2, Cover: 100}, // Yes=10 out of range
				{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0},
				{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0},
			},
		}

		err := ensemble.Validate()
		if err == nil {
			t.Error("expected validation error for invalid child index")
		}
	})

	t.Run("invalid_root_index", func(t *testing.T) {
		ensemble := &TreeEnsemble{
			NumTrees:    1,
			NumFeatures: 2,
			Roots:       []int{100}, // Out of range
			Nodes: []Node{
				{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
			},
		}

		err := ensemble.Validate()
		if err == nil {
			t.Error("expected validation error for invalid root index")
		}
	})

	t.Run("roots_mismatch", func(t *testing.T) {
		ensemble := &TreeEnsemble{
			NumTrees:    2, // Says 2 trees
			NumFeatures: 2,
			Roots:       []int{0}, // But only 1 root
			Nodes: []Node{
				{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
			},
		}

		err := ensemble.Validate()
		if err == nil {
			t.Error("expected validation error for roots mismatch")
		}
	})
}

// TestTreeDepthEdgeCases tests treeDepth with edge cases.
func TestTreeDepthEdgeCases(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		Nodes: []Node{
			{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
		},
	}

	// Single-leaf tree has depth 0
	maxDepth := ensemble.MaxDepth()
	if maxDepth != 0 {
		t.Errorf("expected max depth 0 for single-leaf tree, got %d", maxDepth)
	}
}

// TestExpectedValueNoCover tests expected value calculation when cover is 0.
func TestExpectedValueNoCover(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.5,
		Nodes: []Node{
			{Tree: 0, NodeID: 0, Feature: 0, DecisionType: DecisionLess, Threshold: 0.5, Yes: 1, No: 2, Cover: 0},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0, Cover: 0},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 3.0, Cover: 0},
		},
	}

	// When cover is 0, should use simple average
	expected := ensemble.ExpectedValue()
	// BaseScore (0.5) + average of leaves ((1.0 + 3.0) / 2 = 2.0) = 2.5
	if math.Abs(expected-2.5) > 1e-9 {
		t.Errorf("expected 2.5, got %f", expected)
	}
}
