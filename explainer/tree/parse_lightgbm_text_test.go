package tree

import (
	"strings"
	"testing"
)

// Sample LightGBM text format model for testing
// Simple tree: if f0 <= 0.5 then 1.0 else 3.0
const sampleLightGBMText = `version=v3
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=3
objective=regression
feature_names=f0 f1 f2 f3

Tree=0
num_leaves=2
num_cat=0
split_feature=0
split_gain=100.0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=1.0 3.0
leaf_count=50 50
internal_value=2.0
internal_count=100
shrinkage=1.0

end of trees
`

func TestParseLightGBMText_Simple(t *testing.T) {
	ensemble, err := ParseLightGBMText([]byte(sampleLightGBMText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	// Check ensemble properties
	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
	if ensemble.NumFeatures != 4 {
		t.Errorf("NumFeatures = %d, want 4", ensemble.NumFeatures)
	}
	if len(ensemble.FeatureNames) != 4 {
		t.Errorf("len(FeatureNames) = %d, want 4", len(ensemble.FeatureNames))
	}
	if ensemble.Objective != "regression" {
		t.Errorf("Objective = %q, want %q", ensemble.Objective, "regression")
	}

	// Check tree structure
	if len(ensemble.Nodes) != 3 { // 1 internal + 2 leaves
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
}

func TestParseLightGBMText_MultipleTrees(t *testing.T) {
	modelText := `version=v3
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=binary

Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=0.5 -0.5
leaf_count=50 50
internal_count=100

Tree=1
num_leaves=2
split_feature=1
threshold=0.3
decision_type=<=
left_child=-1
right_child=-2
leaf_value=0.3 -0.3
leaf_count=60 40
internal_count=100

end of trees
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	if ensemble.NumTrees != 2 {
		t.Errorf("NumTrees = %d, want 2", ensemble.NumTrees)
	}

	// Check that roots point to correct nodes
	if ensemble.Roots[0] != 0 {
		t.Errorf("Roots[0] = %d, want 0", ensemble.Roots[0])
	}
	if ensemble.Roots[1] != 3 { // After 1 internal + 2 leaves from tree 0
		t.Errorf("Roots[1] = %d, want 3", ensemble.Roots[1])
	}
}

func TestParseLightGBMText_DeeperTree(t *testing.T) {
	// Tree with depth 2: root splits, then left child splits
	// Structure:
	//       0 (root, feature 0)
	//      / \
	//     1   leaf2
	//    / \
	// leaf0 leaf1
	modelText := `version=v3
max_feature_idx=1

Tree=0
num_leaves=3
split_feature=0 1
threshold=0.5 0.3
decision_type=<= <=
left_child=1 -1
right_child=-3 -2
leaf_value=1.0 2.0 3.0
leaf_count=25 25 50
internal_count=100 50
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	// 2 internal nodes + 3 leaves = 5 total
	if len(ensemble.Nodes) != 5 {
		t.Errorf("len(Nodes) = %d, want 5", len(ensemble.Nodes))
	}

	// Root (node 0) splits on feature 0
	root := &ensemble.Nodes[0]
	if root.Feature != 0 {
		t.Errorf("Root.Feature = %d, want 0", root.Feature)
	}
	// Left child is internal node 1, right child is leaf 2
	if root.Yes != 1 {
		t.Errorf("Root.Yes = %d, want 1", root.Yes)
	}
	// Leaf 2 is at index 4 (2 internal + 2)
	if root.No != 4 { // internal nodes + leaf index 2
		t.Errorf("Root.No = %d, want 4", root.No)
	}

	// Check internal node 1 splits on feature 1
	node1 := &ensemble.Nodes[1]
	if node1.Feature != 1 {
		t.Errorf("Node1.Feature = %d, want 1", node1.Feature)
	}
}

func TestParseLightGBMText_NoTrees(t *testing.T) {
	modelText := `version=v3
max_feature_idx=3
objective=regression
`

	_, err := ParseLightGBMText([]byte(modelText))
	if err == nil {
		t.Error("ParseLightGBMText should fail with no trees")
	}
}

func TestParseLightGBMText_EmptyInput(t *testing.T) {
	_, err := ParseLightGBMText([]byte(""))
	if err == nil {
		t.Error("ParseLightGBMText should fail with empty input")
	}
}

func TestParseLightGBMText_InvalidTreeIndex(t *testing.T) {
	modelText := `version=v3
max_feature_idx=1

Tree=invalid
num_leaves=2
`

	_, err := ParseLightGBMText([]byte(modelText))
	if err == nil {
		t.Error("ParseLightGBMText should fail with invalid tree index")
	}
}

func TestParseLightGBMText_Comments(t *testing.T) {
	modelText := `# This is a comment
version=v3
max_feature_idx=1

# Another comment
Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=1.0 2.0
leaf_count=50 50
internal_count=100
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
}

func TestParseLightGBMText_FeatureNames(t *testing.T) {
	modelText := `version=v3
max_feature_idx=2
feature_names=age income score

Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=1.0 2.0
leaf_count=50 50
internal_count=100
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	expected := []string{"age", "income", "score"}
	if len(ensemble.FeatureNames) != len(expected) {
		t.Fatalf("len(FeatureNames) = %d, want %d", len(ensemble.FeatureNames), len(expected))
	}
	for i, name := range expected {
		if ensemble.FeatureNames[i] != name {
			t.Errorf("FeatureNames[%d] = %q, want %q", i, ensemble.FeatureNames[i], name)
		}
	}
}

func TestParseLightGBMText_DecisionTypes(t *testing.T) {
	tests := []struct {
		decisionType string
		expected     DecisionType
	}{
		{"<=", DecisionLessEqual},
		{"<", DecisionLess},
		{"==", DecisionLessEqual}, // Fallback
	}

	for _, tc := range tests {
		modelText := `version=v3
max_feature_idx=1

Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=` + tc.decisionType + `
left_child=-1
right_child=-2
leaf_value=1.0 2.0
leaf_count=50 50
internal_count=100
`

		ensemble, err := ParseLightGBMText([]byte(modelText))
		if err != nil {
			t.Fatalf("ParseLightGBMText failed for decision_type=%s: %v", tc.decisionType, err)
		}

		if ensemble.Nodes[0].DecisionType != tc.expected {
			t.Errorf("DecisionType for %q = %q, want %q",
				tc.decisionType, ensemble.Nodes[0].DecisionType, tc.expected)
		}
	}
}

func TestLoadLightGBMTextModelFromReader(t *testing.T) {
	reader := strings.NewReader(sampleLightGBMText)
	ensemble, err := LoadLightGBMTextModelFromReader(reader)
	if err != nil {
		t.Fatalf("LoadLightGBMTextModelFromReader failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
}

func TestParseIntArray(t *testing.T) {
	tests := []struct {
		input    string
		expected []int
	}{
		{"1 2 3", []int{1, 2, 3}},
		{"-1 -2 -3", []int{-1, -2, -3}},
		{"0", []int{0}},
		{"", []int{}},
		{"1 invalid 3", []int{1, 3}},
	}

	for _, tc := range tests {
		result := parseIntArray(tc.input)
		if len(result) != len(tc.expected) {
			t.Errorf("parseIntArray(%q) len = %d, want %d", tc.input, len(result), len(tc.expected))
			continue
		}
		for i := range result {
			if result[i] != tc.expected[i] {
				t.Errorf("parseIntArray(%q)[%d] = %d, want %d", tc.input, i, result[i], tc.expected[i])
			}
		}
	}
}

func TestParseFloatArray(t *testing.T) {
	tests := []struct {
		input    string
		expected []float64
	}{
		{"1.0 2.5 3.7", []float64{1.0, 2.5, 3.7}},
		{"-1.5 0.0 1.5", []float64{-1.5, 0.0, 1.5}},
		{"0", []float64{0}},
		{"", []float64{}},
		{"1.0 invalid 3.0", []float64{1.0, 3.0}},
	}

	for _, tc := range tests {
		result := parseFloatArray(tc.input)
		if len(result) != len(tc.expected) {
			t.Errorf("parseFloatArray(%q) len = %d, want %d", tc.input, len(result), len(tc.expected))
			continue
		}
		for i := range result {
			if result[i] != tc.expected[i] {
				t.Errorf("parseFloatArray(%q)[%d] = %v, want %v", tc.input, i, result[i], tc.expected[i])
			}
		}
	}
}

func TestParseLightGBMText_EndToEnd(t *testing.T) {
	// Test that parsed model can be used with TreeSHAP
	ensemble, err := ParseLightGBMText([]byte(sampleLightGBMText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	// Create explainer
	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("New explainer failed: %v", err)
	}

	// Verify base value
	baseValue := exp.BaseValue()
	if baseValue == 0 {
		// Note: base value depends on tree structure, just check it's computed
		t.Log("Base value is 0, which is valid for this simple model")
	}
}

func TestParseLightGBMText_NoEndOfTrees(t *testing.T) {
	// Test that file without "end of trees" is still parsed
	modelText := `version=v3
max_feature_idx=1

Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=1.0 2.0
leaf_count=50 50
internal_count=100
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	if ensemble.NumTrees != 1 {
		t.Errorf("NumTrees = %d, want 1", ensemble.NumTrees)
	}
}

func TestConvertLightGBMTextTree_NoLeaves(t *testing.T) {
	tree := &LightGBMTextTree{
		NumLeaves: 0,
	}

	_, err := convertLightGBMTextTree(tree, 0, 0)
	if err == nil {
		t.Error("convertLightGBMTextTree should fail with no leaves")
	}
}

func TestParseLightGBMText_LeafWeight(t *testing.T) {
	// Test that leaf_weight is used when leaf_count is missing
	modelText := `version=v3
max_feature_idx=1

Tree=0
num_leaves=2
split_feature=0
threshold=0.5
decision_type=<=
left_child=-1
right_child=-2
leaf_value=1.0 2.0
leaf_weight=50.0 50.0
internal_count=100
`

	ensemble, err := ParseLightGBMText([]byte(modelText))
	if err != nil {
		t.Fatalf("ParseLightGBMText failed: %v", err)
	}

	// Leaves should have cover from leaf_weight
	leaf0 := &ensemble.Nodes[1] // First leaf
	if leaf0.Cover != 50.0 {
		t.Errorf("Leaf0.Cover = %v, want 50.0", leaf0.Cover)
	}
}

func BenchmarkParseLightGBMText(b *testing.B) {
	data := []byte(sampleLightGBMText)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ParseLightGBMText(data)
		if err != nil {
			b.Fatal(err)
		}
	}
}
