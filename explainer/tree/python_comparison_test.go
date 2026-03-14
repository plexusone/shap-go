package tree

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/plexusone/shap-go/explainer"
)

// TestCase represents a single test instance from the JSON file.
type TestCase struct {
	Name       string    `json:"name"`
	Instance   []float64 `json:"instance"`
	Prediction float64   `json:"prediction"`
	SHAPValues []float64 `json:"shap_values"`
	Sum        float64   `json:"sum,omitempty"`
	Comment    string    `json:"comment,omitempty"`
}

// TreeNode represents a node in the test case tree.
type TreeNode struct {
	Feature   int     `json:"feature,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
	Yes       int     `json:"yes,omitempty"`
	No        int     `json:"no,omitempty"`
	Cover     float64 `json:"cover,omitempty"`
	IsLeaf    bool    `json:"is_leaf,omitempty"`
	Value     float64 `json:"value,omitempty"`
}

// TreeDef represents a tree definition in the test case.
type TreeDef struct {
	NFeatures int        `json:"n_features"`
	Nodes     []TreeNode `json:"nodes,omitempty"`
	NTrees    int        `json:"n_trees,omitempty"`
	Trees     []struct {
		Nodes []TreeNode `json:"nodes"`
	} `json:"trees,omitempty"`
}

// TestSuite represents a test suite from the JSON file.
type TestSuite struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Tree        TreeDef    `json:"tree"`
	BaseValue   float64    `json:"base_value"`
	Cases       []TestCase `json:"cases"`
}

// TestCases represents the full JSON test file.
type TestCases struct {
	Version    string      `json:"version"`
	TestSuites []TestSuite `json:"test_suites"`
}

// buildEnsembleFromSuite builds a TreeEnsemble from a test suite definition.
func buildEnsembleFromSuite(suite TestSuite) *TreeEnsemble {
	if suite.Tree.NTrees > 0 && len(suite.Tree.Trees) > 0 {
		// Multi-tree ensemble
		return buildMultiTreeEnsemble(suite)
	}
	// Single tree
	return buildSingleTreeEnsemble(suite)
}

func buildSingleTreeEnsemble(suite TestSuite) *TreeEnsemble {
	nodes := make([]Node, len(suite.Tree.Nodes))
	for i, n := range suite.Tree.Nodes {
		nodes[i] = Node{
			Tree:         0,
			NodeID:       i,
			Feature:      n.Feature,
			DecisionType: DecisionLess,
			Threshold:    n.Threshold,
			Yes:          n.Yes,
			No:           n.No,
			Missing:      n.Yes,
			Prediction:   n.Value,
			Cover:        n.Cover,
			IsLeaf:       n.IsLeaf,
		}
		if n.IsLeaf {
			nodes[i].Feature = -1
			nodes[i].Yes = -1
			nodes[i].No = -1
			nodes[i].Missing = -1
		}
	}

	return &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: suite.Tree.NFeatures,
		Roots:       []int{0},
		BaseScore:   0,
		Nodes:       nodes,
	}
}

func buildMultiTreeEnsemble(suite TestSuite) *TreeEnsemble {
	var allNodes []Node
	roots := make([]int, len(suite.Tree.Trees))

	nodeOffset := 0
	for treeIdx, tree := range suite.Tree.Trees {
		roots[treeIdx] = nodeOffset
		for i, n := range tree.Nodes {
			node := Node{
				Tree:         treeIdx,
				NodeID:       i,
				Feature:      n.Feature,
				DecisionType: DecisionLess,
				Threshold:    n.Threshold,
				Yes:          n.Yes + nodeOffset,
				No:           n.No + nodeOffset,
				Missing:      n.Yes + nodeOffset,
				Prediction:   n.Value,
				Cover:        n.Cover,
				IsLeaf:       n.IsLeaf,
			}
			if n.IsLeaf {
				node.Feature = -1
				node.Yes = -1
				node.No = -1
				node.Missing = -1
			}
			allNodes = append(allNodes, node)
		}
		nodeOffset += len(tree.Nodes)
	}

	return &TreeEnsemble{
		NumTrees:    len(suite.Tree.Trees),
		NumFeatures: suite.Tree.NFeatures,
		Roots:       roots,
		BaseScore:   0,
		Nodes:       allNodes,
	}
}

// TestAgainstPythonSHAP tests the Go TreeSHAP implementation against
// known-correct values from the Python SHAP library.
func TestAgainstPythonSHAP(t *testing.T) {
	// Find the test cases file
	testDataPath := filepath.Join("..", "..", "testdata", "treeshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v (run generate_test_cases.py first)", err)
	}

	var testCases TestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()
	tolerance := 1e-6

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			ensemble := buildEnsembleFromSuite(suite)

			exp, err := New(ensemble, explainer.WithFeatureNames(makeFeatureNames(suite.Tree.NFeatures)))
			if err != nil {
				t.Fatalf("Failed to create explainer: %v", err)
			}

			// Verify base value
			gotBase := exp.BaseValue()
			if math.Abs(gotBase-suite.BaseValue) > tolerance {
				t.Errorf("Base value mismatch: got %f, expected %f", gotBase, suite.BaseValue)
			}

			for _, tc := range suite.Cases {
				t.Run(tc.Name, func(t *testing.T) {
					result, err := exp.Explain(ctx, tc.Instance)
					if err != nil {
						t.Fatalf("Explain failed: %v", err)
					}

					// Check prediction
					if math.Abs(result.Prediction-tc.Prediction) > tolerance {
						t.Errorf("Prediction mismatch: got %f, expected %f",
							result.Prediction, tc.Prediction)
					}

					// Check local accuracy
					verifyResult := result.Verify(tolerance)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}

					// Check individual SHAP values against Python SHAP
					for i, expected := range tc.SHAPValues {
						featureName := makeFeatureName(i)
						got, ok := result.Values[featureName]
						if !ok {
							t.Errorf("Missing SHAP value for feature %s", featureName)
							continue
						}
						if math.Abs(got-expected) > tolerance {
							t.Errorf("SHAP[%s] mismatch: got %f, expected %f (diff=%f)",
								featureName, got, expected, math.Abs(got-expected))
						}
					}
				})
			}
		})
	}
}

func makeFeatureNames(n int) []string {
	names := make([]string, n)
	for i := 0; i < n; i++ {
		names[i] = makeFeatureName(i)
	}
	return names
}

func makeFeatureName(i int) string {
	return fmt.Sprintf("x%d", i)
}

// TestLocalAccuracyProperty verifies the fundamental SHAP property:
// sum(SHAP values) = prediction - base_value
func TestLocalAccuracyProperty(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "treeshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v", err)
	}

	var testCases TestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			ensemble := buildEnsembleFromSuite(suite)
			exp, err := New(ensemble)
			if err != nil {
				t.Fatalf("Failed to create explainer: %v", err)
			}

			for _, tc := range suite.Cases {
				t.Run(tc.Name, func(t *testing.T) {
					result, err := exp.Explain(ctx, tc.Instance)
					if err != nil {
						t.Fatalf("Explain failed: %v", err)
					}

					// Verify local accuracy with 1e-9 tolerance
					verifyResult := result.Verify(1e-9)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy violated: sum=%f, expected=%f, diff=%e",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}
				})
			}
		})
	}
}
