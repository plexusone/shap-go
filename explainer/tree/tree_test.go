package tree

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
)

// TestSimpleTree tests TreeSHAP on a simple 2-node tree.
//
// Tree structure:
//
//	    [0] x0 < 0.5
//	   /          \
//	[1] 1.0    [2] 3.0
//
// For input x0=0.3: goes left, prediction = 1.0
// For input x0=0.7: goes right, prediction = 3.0
func TestSimpleTree(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
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
			{
				Tree:       0,
				NodeID:     1,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				IsLeaf:     true,
				Prediction: 1.0,
				Cover:      50,
			},
			{
				Tree:       0,
				NodeID:     2,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				IsLeaf:     true,
				Prediction: 3.0,
				Cover:      50,
			},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	// Test instance going left
	t.Run("goes_left", func(t *testing.T) {
		instance := []float64{0.3}
		result, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		// Prediction should be 1.0
		if math.Abs(result.Prediction-1.0) > 1e-9 {
			t.Errorf("expected prediction 1.0, got %f", result.Prediction)
		}

		// Verify local accuracy: sum(SHAP) = prediction - base
		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: sum=%f, expected=%f, diff=%f",
				verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
		}
	})

	// Test instance going right
	t.Run("goes_right", func(t *testing.T) {
		instance := []float64{0.7}
		result, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		// Prediction should be 3.0
		if math.Abs(result.Prediction-3.0) > 1e-9 {
			t.Errorf("expected prediction 3.0, got %f", result.Prediction)
		}

		// Verify local accuracy
		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: sum=%f, expected=%f, diff=%f",
				verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
		}
	})
}

// TestTwoFeatureTree tests TreeSHAP on a tree with 2 features.
//
// Tree structure:
//
//	        [0] x0 < 0.5
//	       /          \
//	   [1] x1<0.5   [2] 4.0
//	   /      \
//	[3] 1.0  [4] 2.0
func TestTwoFeatureTree(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
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
			{
				Tree:         0,
				NodeID:       1,
				Feature:      1,
				DecisionType: DecisionLess,
				Threshold:    0.5,
				Yes:          3,
				No:           4,
				Missing:      3,
				IsLeaf:       false,
				Cover:        50,
			},
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

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"x0", "x1"}))
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	testCases := []struct {
		name     string
		instance []float64
		wantPred float64
	}{
		{"x0<0.5,x1<0.5", []float64{0.3, 0.3}, 1.0},
		{"x0<0.5,x1>=0.5", []float64{0.3, 0.7}, 2.0},
		{"x0>=0.5", []float64{0.7, 0.3}, 4.0},
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

			// Ensure feature names are correct
			if _, ok := result.Values["x0"]; !ok {
				t.Error("missing SHAP value for x0")
			}
			if _, ok := result.Values["x1"]; !ok {
				t.Error("missing SHAP value for x1")
			}
		})
	}
}

// TestEnsembleOfTrees tests TreeSHAP on an ensemble with multiple trees.
func TestEnsembleOfTrees(t *testing.T) {
	// Two simple trees, each with a single split
	// Tree 0: x0 < 0.5 -> 1.0, else -> 2.0
	// Tree 1: x0 < 0.5 -> 0.5, else -> 1.5
	ensemble := &TreeEnsemble{
		NumTrees:    2,
		NumFeatures: 1,
		Roots:       []int{0, 3},
		BaseScore:   0.0,
		Nodes: []Node{
			// Tree 0
			{
				Tree: 0, NodeID: 0, Feature: 0, DecisionType: DecisionLess,
				Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100,
			},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0, Cover: 50},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0, Cover: 50},
			// Tree 1
			{
				Tree: 1, NodeID: 0, Feature: 0, DecisionType: DecisionLess,
				Threshold: 0.5, Yes: 4, No: 5, Missing: 4, Cover: 100,
			},
			{Tree: 1, NodeID: 1, IsLeaf: true, Prediction: 0.5, Cover: 50},
			{Tree: 1, NodeID: 2, IsLeaf: true, Prediction: 1.5, Cover: 50},
		},
	}

	exp, err := New(ensemble)
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	// Test x0 < 0.5: prediction = 1.0 + 0.5 = 1.5
	t.Run("x0<0.5", func(t *testing.T) {
		instance := []float64{0.3}
		result, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		expectedPred := 1.0 + 0.5 // sum of leaf values
		if math.Abs(result.Prediction-expectedPred) > 1e-9 {
			t.Errorf("expected prediction %f, got %f", expectedPred, result.Prediction)
		}

		// Verify local accuracy
		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: sum=%f, expected=%f, diff=%f",
				verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
		}
	})

	// Test x0 >= 0.5: prediction = 2.0 + 1.5 = 3.5
	t.Run("x0>=0.5", func(t *testing.T) {
		instance := []float64{0.7}
		result, err := exp.Explain(ctx, instance)
		if err != nil {
			t.Fatalf("Explain failed: %v", err)
		}

		expectedPred := 2.0 + 1.5
		if math.Abs(result.Prediction-expectedPred) > 1e-9 {
			t.Errorf("expected prediction %f, got %f", expectedPred, result.Prediction)
		}

		// Verify local accuracy
		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("local accuracy failed: sum=%f, expected=%f, diff=%f",
				verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
		}
	})
}

// TestExplainBatch tests batch explanation.
func TestExplainBatch(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
			{
				Tree: 0, NodeID: 0, Feature: 0, DecisionType: DecisionLess,
				Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100,
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
	instances := [][]float64{
		{0.3}, // goes left
		{0.7}, // goes right
		{0.1}, // goes left
	}

	results, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch failed: %v", err)
	}

	if len(results) != len(instances) {
		t.Fatalf("expected %d results, got %d", len(instances), len(results))
	}

	// Check each result
	expectedPreds := []float64{1.0, 2.0, 1.0}
	for i, result := range results {
		if math.Abs(result.Prediction-expectedPreds[i]) > 1e-9 {
			t.Errorf("instance %d: expected prediction %f, got %f",
				i, expectedPreds[i], result.Prediction)
		}

		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("instance %d: local accuracy failed", i)
		}
	}
}

// TestExplainBatchParallel tests parallel batch explanation.
func TestExplainBatchParallel(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
			{
				Tree: 0, NodeID: 0, Feature: 0, DecisionType: DecisionLess,
				Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100,
			},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: 1.0, Cover: 50},
			{Tree: 0, NodeID: 2, IsLeaf: true, Prediction: 2.0, Cover: 50},
		},
	}

	exp, err := New(ensemble, explainer.WithNumWorkers(4))
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	ctx := context.Background()

	// Create many instances
	instances := make([][]float64, 100)
	for i := range instances {
		instances[i] = []float64{float64(i) / 100.0}
	}

	results, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		t.Fatalf("ExplainBatch failed: %v", err)
	}

	if len(results) != len(instances) {
		t.Fatalf("expected %d results, got %d", len(instances), len(results))
	}

	// Verify all results
	for i, result := range results {
		verifyResult := result.Verify(1e-6)
		if !verifyResult.Valid {
			t.Errorf("instance %d: local accuracy failed: diff=%f",
				i, verifyResult.Difference)
		}
	}
}

// TestValidation tests ensemble validation.
func TestValidation(t *testing.T) {
	t.Run("nil_ensemble", func(t *testing.T) {
		_, err := New(nil)
		if err == nil {
			t.Error("expected error for nil ensemble")
		}
	})

	t.Run("empty_ensemble", func(t *testing.T) {
		ensemble := &TreeEnsemble{}
		_, err := New(ensemble)
		if err == nil {
			t.Error("expected error for empty ensemble")
		}
	})

	t.Run("wrong_instance_size", func(t *testing.T) {
		ensemble := &TreeEnsemble{
			NumTrees:    1,
			NumFeatures: 2,
			Roots:       []int{0},
			Nodes: []Node{
				{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
			},
		}

		exp, err := New(ensemble)
		if err != nil {
			t.Fatalf("failed to create explainer: %v", err)
		}

		_, err = exp.Explain(context.Background(), []float64{0.5}) // wrong size
		if err == nil {
			t.Error("expected error for wrong instance size")
		}
	})
}

// TestMetadata tests that metadata is correctly populated.
func TestMetadata(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 1,
		Roots:       []int{0},
		BaseScore:   0.0,
		Nodes: []Node{
			{Tree: 0, NodeID: 0, IsLeaf: true, Prediction: 1.0},
		},
	}

	exp, err := New(ensemble, explainer.WithModelID("test-model"))
	if err != nil {
		t.Fatalf("failed to create explainer: %v", err)
	}

	result, err := exp.Explain(context.Background(), []float64{0.5})
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	if result.ModelID != "test-model" {
		t.Errorf("expected model ID 'test-model', got '%s'", result.ModelID)
	}

	if result.Metadata.Algorithm != "treeshap" {
		t.Errorf("expected algorithm 'treeshap', got '%s'", result.Metadata.Algorithm)
	}

	if result.Timestamp.IsZero() {
		t.Error("timestamp should not be zero")
	}
}

// BenchmarkTreeSHAP benchmarks TreeSHAP computation.
func BenchmarkTreeSHAP(b *testing.B) {
	// Create a more realistic ensemble with 10 trees, depth 4, 10 features
	ensemble := createBenchmarkEnsemble(10, 4, 10)

	exp, err := New(ensemble)
	if err != nil {
		b.Fatalf("failed to create explainer: %v", err)
	}

	// Create test instance
	instance := make([]float64, 10)
	for i := range instance {
		instance[i] = 0.5
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := exp.Explain(ctx, instance)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// createBenchmarkEnsemble creates a synthetic ensemble for benchmarking.
func createBenchmarkEnsemble(numTrees, depth, numFeatures int) *TreeEnsemble {
	ensemble := &TreeEnsemble{
		NumTrees:    numTrees,
		NumFeatures: numFeatures,
		Roots:       make([]int, numTrees),
		BaseScore:   0.0,
	}

	nodeOffset := 0
	for treeIdx := 0; treeIdx < numTrees; treeIdx++ {
		ensemble.Roots[treeIdx] = nodeOffset
		nodes := createCompleteTree(treeIdx, depth, numFeatures, nodeOffset)
		ensemble.Nodes = append(ensemble.Nodes, nodes...)
		nodeOffset += len(nodes)
	}

	return ensemble
}

// createCompleteTree creates a complete binary tree.
func createCompleteTree(treeIdx, depth, numFeatures, nodeOffset int) []Node {
	nodesPerTree := (1 << (depth + 1)) - 1
	nodes := make([]Node, nodesPerTree)

	for i := 0; i < nodesPerTree; i++ {
		leftChild := 2*i + 1
		rightChild := 2*i + 2
		isLeaf := leftChild >= nodesPerTree

		node := Node{
			Tree:   treeIdx,
			NodeID: i,
			IsLeaf: isLeaf,
			Cover:  100.0 / float64(int(1)<<nodeDepth(i)),
		}

		if isLeaf {
			node.Feature = -1
			node.Yes = -1
			node.No = -1
			node.Missing = -1
			node.Prediction = float64(i) * 0.01
		} else {
			node.Feature = i % numFeatures
			node.DecisionType = DecisionLess
			node.Threshold = 0.5
			node.Yes = leftChild + nodeOffset
			node.No = rightChild + nodeOffset
			node.Missing = node.Yes
		}

		nodes[i] = node
	}

	return nodes
}

// nodeDepth returns the depth of a node in a complete binary tree (0-indexed).
func nodeDepth(nodeIdx int) int {
	depth := 0
	for (1 << depth) <= nodeIdx {
		depth++
	}
	return depth
}
