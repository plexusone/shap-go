// Example: Computing SHAP values for tree-based models using TreeSHAP
//
// TreeSHAP computes exact SHAP values in O(TLD²) time, making it
// 40-100x faster than permutation-based methods for tree ensembles.
//
// This example demonstrates:
// 1. Loading XGBoost and LightGBM models
// 2. Computing SHAP values with TreeSHAP
// 3. Verifying local accuracy
// 4. Batch processing multiple instances
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/tree"
)

func main() {
	fmt.Println("TreeSHAP Example")
	fmt.Println("================")
	fmt.Println()

	// Example 1: Using a manually constructed tree ensemble
	// This creates a simple tree: x0 < 0.5 -> 1.0, else -> 2.0
	fmt.Println("Example 1: Simple Decision Tree")
	fmt.Println("--------------------------------")
	simpleTreeExample()

	fmt.Println()

	// Example 2: Two-feature tree with depth 2
	fmt.Println("Example 2: Two-Feature Tree")
	fmt.Println("----------------------------")
	twoFeatureTreeExample()

	fmt.Println()

	// Example 3: Batch processing
	fmt.Println("Example 3: Batch Processing")
	fmt.Println("----------------------------")
	batchExample()

	fmt.Println()

	// Example 4: Loading from JSON (commented out - requires actual model files)
	fmt.Println("Example 4: Loading Models from JSON")
	fmt.Println("------------------------------------")
	fmt.Println("// XGBoost: tree.LoadXGBoostModel(\"model.json\")")
	fmt.Println("// LightGBM: tree.LoadLightGBMModel(\"model.json\")")
	fmt.Println("// See README.md for Python export instructions")
}

// simpleTreeExample demonstrates TreeSHAP with a simple single-split tree.
func simpleTreeExample() {
	// Create a simple tree ensemble manually
	// Tree structure: x0 < 0.5 -> leaf(1.0), else -> leaf(2.0)
	ensemble := &tree.TreeEnsemble{
		NumTrees:     1,
		NumFeatures:  1,
		FeatureNames: []string{"x0"},
		Roots:        []int{0},
		BaseScore:    0,
		Nodes: []tree.Node{
			// Root node: split on feature 0 at threshold 0.5
			{
				Tree:         0,
				NodeID:       0,
				Feature:      0,
				DecisionType: tree.DecisionLess,
				Threshold:    0.5,
				Yes:          1, // left child (< 0.5)
				No:           2, // right child (>= 0.5)
				Missing:      1,
				Cover:        100,
				IsLeaf:       false,
			},
			// Left leaf: prediction = 1.0
			{
				Tree:       0,
				NodeID:     1,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				Prediction: 1.0,
				Cover:      50,
				IsLeaf:     true,
			},
			// Right leaf: prediction = 2.0
			{
				Tree:       0,
				NodeID:     2,
				Feature:    -1,
				Yes:        -1,
				No:         -1,
				Missing:    -1,
				Prediction: 2.0,
				Cover:      50,
				IsLeaf:     true,
			},
		},
	}

	// Create TreeSHAP explainer
	exp, err := tree.New(ensemble,
		explainer.WithFeatureNames([]string{"x0"}),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	fmt.Printf("Tree: x0 < 0.5 -> 1.0, else -> 2.0\n")
	fmt.Printf("Base Value (expected value): %.4f\n\n", exp.BaseValue())

	// Explain instances
	ctx := context.Background()
	instances := [][]float64{
		{0.3}, // goes left -> prediction 1.0
		{0.7}, // goes right -> prediction 2.0
	}

	for _, instance := range instances {
		explanation, err := exp.Explain(ctx, instance)
		if err != nil {
			log.Fatalf("Failed to explain: %v", err)
		}

		fmt.Printf("Instance: x0=%.1f\n", instance[0])
		fmt.Printf("  Prediction: %.4f\n", explanation.Prediction)
		fmt.Printf("  SHAP[x0]: %.4f\n", explanation.Values["x0"])

		// Verify local accuracy
		result := explanation.Verify(1e-9)
		fmt.Printf("  Local accuracy: %v (diff=%.2e)\n", result.Valid, result.Difference)
	}
}

// twoFeatureTreeExample demonstrates TreeSHAP with a deeper tree.
func twoFeatureTreeExample() {
	// Tree structure:
	//   x0 < 0.5 (root)
	//     |-- x1 < 0.5 -> leaf(1.0)
	//     |-- x1 >= 0.5 -> leaf(2.0)
	//   x0 >= 0.5 -> leaf(4.0)
	ensemble := &tree.TreeEnsemble{
		NumTrees:     1,
		NumFeatures:  2,
		FeatureNames: []string{"x0", "x1"},
		Roots:        []int{0},
		BaseScore:    0,
		Nodes: []tree.Node{
			// Node 0: root, split on x0 < 0.5
			{Tree: 0, NodeID: 0, Feature: 0, DecisionType: tree.DecisionLess, Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100, IsLeaf: false},
			// Node 1: left child, split on x1 < 0.5
			{Tree: 0, NodeID: 1, Feature: 1, DecisionType: tree.DecisionLess, Threshold: 0.5, Yes: 3, No: 4, Missing: 3, Cover: 50, IsLeaf: false},
			// Node 2: right leaf, prediction = 4.0
			{Tree: 0, NodeID: 2, Feature: -1, Yes: -1, No: -1, Missing: -1, Prediction: 4.0, Cover: 50, IsLeaf: true},
			// Node 3: left-left leaf, prediction = 1.0
			{Tree: 0, NodeID: 3, Feature: -1, Yes: -1, No: -1, Missing: -1, Prediction: 1.0, Cover: 25, IsLeaf: true},
			// Node 4: left-right leaf, prediction = 2.0
			{Tree: 0, NodeID: 4, Feature: -1, Yes: -1, No: -1, Missing: -1, Prediction: 2.0, Cover: 25, IsLeaf: true},
		},
	}

	exp, err := tree.New(ensemble)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	fmt.Printf("Tree: x0<0.5 -> (x1<0.5 -> 1.0, else -> 2.0), else -> 4.0\n")
	fmt.Printf("Base Value: %.4f\n\n", exp.BaseValue())

	ctx := context.Background()
	instances := []struct {
		name     string
		instance []float64
	}{
		{"x0<0.5, x1<0.5", []float64{0.3, 0.3}},
		{"x0<0.5, x1>=0.5", []float64{0.3, 0.7}},
		{"x0>=0.5", []float64{0.7, 0.3}},
	}

	for _, tc := range instances {
		explanation, err := exp.Explain(ctx, tc.instance)
		if err != nil {
			log.Fatalf("Failed to explain: %v", err)
		}

		fmt.Printf("Instance: %s (x0=%.1f, x1=%.1f)\n", tc.name, tc.instance[0], tc.instance[1])
		fmt.Printf("  Prediction: %.4f\n", explanation.Prediction)
		fmt.Printf("  SHAP[x0]: %.4f\n", explanation.Values["x0"])
		fmt.Printf("  SHAP[x1]: %.4f\n", explanation.Values["x1"])

		result := explanation.Verify(1e-9)
		fmt.Printf("  Local accuracy: %v\n", result.Valid)
	}
}

// batchExample demonstrates batch processing of multiple instances.
func batchExample() {
	// Simple tree for batch processing demo
	ensemble := &tree.TreeEnsemble{
		NumTrees:     1,
		NumFeatures:  2,
		FeatureNames: []string{"x0", "x1"},
		Roots:        []int{0},
		Nodes: []tree.Node{
			{Tree: 0, NodeID: 0, Feature: 0, DecisionType: tree.DecisionLess, Threshold: 0.5, Yes: 1, No: 2, Missing: 1, Cover: 100, IsLeaf: false},
			{Tree: 0, NodeID: 1, Feature: -1, Yes: -1, No: -1, Missing: -1, Prediction: 1.0, Cover: 50, IsLeaf: true},
			{Tree: 0, NodeID: 2, Feature: -1, Yes: -1, No: -1, Missing: -1, Prediction: 2.0, Cover: 50, IsLeaf: true},
		},
	}

	exp, err := tree.New(ensemble)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	// Batch of instances to explain
	instances := [][]float64{
		{0.1, 0.5},
		{0.2, 0.5},
		{0.3, 0.5},
		{0.6, 0.5},
		{0.7, 0.5},
		{0.8, 0.5},
	}

	ctx := context.Background()
	explanations, err := exp.ExplainBatch(ctx, instances)
	if err != nil {
		log.Fatalf("Failed to explain batch: %v", err)
	}

	fmt.Printf("Processed %d instances\n\n", len(explanations))

	for i, explanation := range explanations {
		fmt.Printf("Instance %d: x0=%.1f -> pred=%.1f, SHAP[x0]=%.4f\n",
			i+1, instances[i][0], explanation.Prediction, explanation.Values["x0"])
	}
}
