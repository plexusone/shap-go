package tree

import (
	"strings"
	"testing"
)

// Sample CatBoost model JSON for testing
// This is a simple model with one tree of depth 2
const sampleCatBoostJSON = `{
  "model_info": {
    "cat_feature_count": 0,
    "float_feature_count": 2,
    "model_type": "Regression"
  },
  "oblivious_trees": [
    {
      "splits": [
        {"float_feature_index": 0, "border": 0.5, "split_type": "FloatFeature"},
        {"float_feature_index": 1, "border": 0.3, "split_type": "FloatFeature"}
      ],
      "leaf_values": [1.0, 2.0, 3.0, 4.0],
      "leaf_weights": [10.0, 20.0, 30.0, 40.0]
    }
  ],
  "features_info": {
    "float_features": [
      {"feature_index": 0, "flat_feature_index": 0, "has_nans": false, "feature_name": "x0"},
      {"feature_index": 1, "flat_feature_index": 1, "has_nans": false, "feature_name": "x1"}
    ],
    "categorical_features": []
  },
  "scale_and_bias": [1.0, 0.0]
}`

func TestParseCatBoostJSON_Simple(t *testing.T) {
	ensemble, err := ParseCatBoostJSON([]byte(sampleCatBoostJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	// Check ensemble properties
	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
	if ensemble.NumFeatures != 2 {
		t.Errorf("NumFeatures = %d, want 2", ensemble.NumFeatures)
	}
	if ensemble.Objective != "Regression" {
		t.Errorf("Objective = %q, want %q", ensemble.Objective, "Regression")
	}

	// Check feature names
	if len(ensemble.FeatureNames) != 2 {
		t.Errorf("len(FeatureNames) = %d, want 2", len(ensemble.FeatureNames))
	} else {
		if ensemble.FeatureNames[0] != "x0" {
			t.Errorf("FeatureNames[0] = %q, want %q", ensemble.FeatureNames[0], "x0")
		}
		if ensemble.FeatureNames[1] != "x1" {
			t.Errorf("FeatureNames[1] = %q, want %q", ensemble.FeatureNames[1], "x1")
		}
	}

	// For depth 2, we have 4 leaves + 3 internal nodes = 7 nodes
	if len(ensemble.Nodes) != 7 {
		t.Errorf("len(Nodes) = %d, want 7", len(ensemble.Nodes))
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
}

func TestParseCatBoostJSON_SingleLeaf(t *testing.T) {
	modelJSON := `{
		"model_info": {
			"cat_feature_count": 0,
			"float_feature_count": 1,
			"model_type": "Regression"
		},
		"oblivious_trees": [
			{
				"splits": [],
				"leaf_values": [5.0],
				"leaf_weights": [100.0]
			}
		],
		"features_info": {
			"float_features": [
				{"feature_index": 0, "flat_feature_index": 0, "has_nans": false}
			]
		}
	}`

	ensemble, err := ParseCatBoostJSON([]byte(modelJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}

	// Single leaf means 1 node
	if len(ensemble.Nodes) != 1 {
		t.Errorf("len(Nodes) = %d, want 1", len(ensemble.Nodes))
	}

	leaf := &ensemble.Nodes[0]
	if !leaf.IsLeaf {
		t.Error("Single node should be a leaf")
	}
	if leaf.Prediction != 5.0 {
		t.Errorf("Leaf.Prediction = %v, want 5.0", leaf.Prediction)
	}
	if leaf.Cover != 100.0 {
		t.Errorf("Leaf.Cover = %v, want 100.0", leaf.Cover)
	}
}

func TestParseCatBoostJSON_MultipleTrees(t *testing.T) {
	modelJSON := `{
		"model_info": {
			"cat_feature_count": 0,
			"float_feature_count": 2,
			"model_type": "Regression"
		},
		"oblivious_trees": [
			{
				"splits": [
					{"float_feature_index": 0, "border": 0.5}
				],
				"leaf_values": [1.0, 2.0],
				"leaf_weights": [50.0, 50.0]
			},
			{
				"splits": [
					{"float_feature_index": 1, "border": 0.3}
				],
				"leaf_values": [0.5, 1.5],
				"leaf_weights": [60.0, 40.0]
			}
		],
		"features_info": {
			"float_features": [
				{"feature_index": 0, "flat_feature_index": 0, "has_nans": false},
				{"feature_index": 1, "flat_feature_index": 1, "has_nans": false}
			]
		}
	}`

	ensemble, err := ParseCatBoostJSON([]byte(modelJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	if ensemble.NumTrees != 2 {
		t.Errorf("NumTrees = %d, want 2", ensemble.NumTrees)
	}

	// Each tree has depth 1: 1 internal + 2 leaves = 3 nodes per tree
	if len(ensemble.Nodes) != 6 {
		t.Errorf("len(Nodes) = %d, want 6", len(ensemble.Nodes))
	}

	// Check roots
	if ensemble.Roots[0] != 0 {
		t.Errorf("Roots[0] = %d, want 0", ensemble.Roots[0])
	}
	if ensemble.Roots[1] != 3 { // Second tree starts at index 3
		t.Errorf("Roots[1] = %d, want 3", ensemble.Roots[1])
	}
}

func TestParseCatBoostJSON_WithBias(t *testing.T) {
	modelJSON := `{
		"model_info": {
			"cat_feature_count": 0,
			"float_feature_count": 1,
			"model_type": "Regression"
		},
		"oblivious_trees": [
			{
				"splits": [],
				"leaf_values": [1.0]
			}
		],
		"features_info": {
			"float_features": [
				{"feature_index": 0, "flat_feature_index": 0, "has_nans": false}
			]
		},
		"scale_and_bias": [1.0, 2.5]
	}`

	ensemble, err := ParseCatBoostJSON([]byte(modelJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	if ensemble.BaseScore != 2.5 {
		t.Errorf("BaseScore = %v, want 2.5", ensemble.BaseScore)
	}
}

func TestParseCatBoostJSON_NoTrees(t *testing.T) {
	modelJSON := `{
		"model_info": {
			"float_feature_count": 1
		},
		"oblivious_trees": [],
		"features_info": {
			"float_features": []
		}
	}`

	_, err := ParseCatBoostJSON([]byte(modelJSON))
	if err == nil {
		t.Error("ParseCatBoostJSON should fail with no trees")
	}
}

func TestParseCatBoostJSON_InvalidJSON(t *testing.T) {
	_, err := ParseCatBoostJSON([]byte("not valid json"))
	if err == nil {
		t.Error("ParseCatBoostJSON should fail with invalid JSON")
	}
}

func TestLoadCatBoostModelFromReader(t *testing.T) {
	reader := strings.NewReader(sampleCatBoostJSON)
	ensemble, err := LoadCatBoostModelFromReader(reader)
	if err != nil {
		t.Fatalf("LoadCatBoostModelFromReader failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
}

func TestNodeLevel(t *testing.T) {
	tests := []struct {
		nodeID int
		want   int
	}{
		{0, 0}, // root
		{1, 1}, // left child of root
		{2, 1}, // right child of root
		{3, 2}, // grandchildren
		{4, 2},
		{5, 2},
		{6, 2},
		{7, 3}, // great-grandchildren
		{14, 3},
	}

	for _, tc := range tests {
		got := nodeLevel(tc.nodeID)
		if got != tc.want {
			t.Errorf("nodeLevel(%d) = %d, want %d", tc.nodeID, got, tc.want)
		}
	}
}

func TestParseCatBoostJSON_EndToEnd(t *testing.T) {
	// Test that parsed model can be used with TreeSHAP
	ensemble, err := ParseCatBoostJSON([]byte(sampleCatBoostJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	// Create explainer
	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New explainer failed: %v", err)
	}

	// Verify base value can be computed
	baseValue := exp.BaseValue()
	t.Logf("Base value: %v", baseValue)
}

func TestParseCatBoostJSON_CategoricalFeature(t *testing.T) {
	modelJSON := `{
		"model_info": {
			"cat_feature_count": 1,
			"float_feature_count": 1,
			"model_type": "Classification"
		},
		"oblivious_trees": [
			{
				"splits": [
					{"float_feature_index": 0, "border": 0.5, "split_type": "FloatFeature"}
				],
				"leaf_values": [0.1, 0.9],
				"leaf_weights": [50.0, 50.0]
			}
		],
		"features_info": {
			"float_features": [
				{"feature_index": 0, "flat_feature_index": 0, "has_nans": false, "feature_name": "num_feat"}
			],
			"categorical_features": [
				{"feature_index": 0, "flat_feature_index": 1, "feature_name": "cat_feat"}
			]
		}
	}`

	ensemble, err := ParseCatBoostJSON([]byte(modelJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	// Should have 2 features total
	if ensemble.NumFeatures != 2 {
		t.Errorf("NumFeatures = %d, want 2", ensemble.NumFeatures)
	}

	// Check feature names
	if len(ensemble.FeatureNames) != 2 {
		t.Fatalf("len(FeatureNames) = %d, want 2", len(ensemble.FeatureNames))
	}
	if ensemble.FeatureNames[0] != "num_feat" {
		t.Errorf("FeatureNames[0] = %q, want %q", ensemble.FeatureNames[0], "num_feat")
	}
	if ensemble.FeatureNames[1] != "cat_feat" {
		t.Errorf("FeatureNames[1] = %q, want %q", ensemble.FeatureNames[1], "cat_feat")
	}
}

func TestParseCatBoostJSON_Depth3(t *testing.T) {
	// Tree with depth 3: 8 leaves, 7 internal nodes = 15 total
	modelJSON := `{
		"model_info": {
			"cat_feature_count": 0,
			"float_feature_count": 3,
			"model_type": "Regression"
		},
		"oblivious_trees": [
			{
				"splits": [
					{"float_feature_index": 0, "border": 0.5},
					{"float_feature_index": 1, "border": 0.3},
					{"float_feature_index": 2, "border": 0.7}
				],
				"leaf_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
				"leaf_weights": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
			}
		],
		"features_info": {
			"float_features": [
				{"feature_index": 0, "flat_feature_index": 0, "has_nans": false},
				{"feature_index": 1, "flat_feature_index": 1, "has_nans": false},
				{"feature_index": 2, "flat_feature_index": 2, "has_nans": false}
			]
		}
	}`

	ensemble, err := ParseCatBoostJSON([]byte(modelJSON))
	if err != nil {
		t.Fatalf("ParseCatBoostJSON failed: %v", err)
	}

	// 8 leaves + 7 internal = 15 nodes
	if len(ensemble.Nodes) != 15 {
		t.Errorf("len(Nodes) = %d, want 15", len(ensemble.Nodes))
	}

	// Validate tree structure
	if err := ensemble.Validate(); err != nil {
		t.Errorf("Validation failed: %v", err)
	}
}

func BenchmarkParseCatBoostJSON(b *testing.B) {
	data := []byte(sampleCatBoostJSON)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ParseCatBoostJSON(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}
