// Example: Batch processing with TreeSHAP
//
// This example demonstrates how to efficiently explain multiple instances
// using TreeSHAP with parallel processing. Batch processing is useful for:
// - Explaining entire test datasets
// - Computing global feature importance
// - Real-time batch inference with explanations
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/tree"
	"github.com/plexusone/shap-go/render"
)

func main() {
	fmt.Println("Batch Processing Example with TreeSHAP")
	fmt.Println("======================================")
	fmt.Println()

	// Create a tree ensemble for demonstration
	// This represents a simple model: x0 and x1 both contribute to prediction
	ensemble := createDemoEnsemble()

	// Create explainer with parallel workers
	exp, err := tree.New(ensemble,
		explainer.WithNumWorkers(4), // Use 4 parallel workers
		explainer.WithFeatureNames([]string{"income", "age", "credit_score"}),
		explainer.WithModelID("loan-approval-model"),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	fmt.Printf("Model: %d trees, %d features\n", ensemble.NumTrees, ensemble.NumFeatures)
	fmt.Printf("Base Value: %.4f\n\n", exp.BaseValue())

	// Generate batch of instances to explain
	// In production, this would be loaded from a database or file
	instances := generateTestInstances(100)
	fmt.Printf("Explaining %d instances...\n\n", len(instances))

	// Benchmark: Sequential vs Parallel
	ctx := context.Background()

	// Sequential processing
	seqExp, _ := tree.New(ensemble, explainer.WithNumWorkers(1))
	start := time.Now()
	_, _ = seqExp.ExplainBatch(ctx, instances)
	seqDuration := time.Since(start)

	// Parallel processing (4 workers)
	parExp, _ := tree.New(ensemble, explainer.WithNumWorkers(4))
	start = time.Now()
	explanations, err := parExp.ExplainBatch(ctx, instances)
	parDuration := time.Since(start)

	if err != nil {
		log.Fatalf("Batch explanation failed: %v", err)
	}

	fmt.Println("Performance Comparison")
	fmt.Println("----------------------")
	fmt.Printf("Sequential (1 worker):  %v\n", seqDuration)
	fmt.Printf("Parallel (4 workers):   %v\n", parDuration)
	fmt.Printf("Speedup:                %.2fx\n\n", float64(seqDuration)/float64(parDuration))

	// Show sample explanations
	fmt.Println("Sample Explanations (first 3)")
	fmt.Println("-----------------------------")
	for i := 0; i < 3 && i < len(explanations); i++ {
		exp := explanations[i]
		fmt.Printf("Instance %d: prediction=%.4f\n", i+1, exp.Prediction)
		for _, name := range exp.FeatureNames {
			fmt.Printf("  %s: value=%.2f, SHAP=%.4f\n",
				name, exp.FeatureValues[name], exp.Values[name])
		}
		fmt.Println()
	}

	// Compute global feature importance from batch
	fmt.Println("Global Feature Importance")
	fmt.Println("-------------------------")

	expSet := render.NewExplanationSet(explanations)
	meanAbs := expSet.MeanAbsoluteSHAP()

	// Sort and display
	for _, name := range []string{"income", "age", "credit_score"} {
		importance := meanAbs[name]
		fmt.Printf("  %s: %.4f\n", name, importance)
	}
	fmt.Println()

	// Verify local accuracy for all explanations
	fmt.Println("Local Accuracy Verification")
	fmt.Println("---------------------------")
	validCount := 0
	maxDiff := 0.0
	for _, exp := range explanations {
		result := exp.Verify(1e-6)
		if result.Valid {
			validCount++
		}
		if result.Difference > maxDiff {
			maxDiff = result.Difference
		}
	}
	fmt.Printf("Valid explanations: %d/%d (%.1f%%)\n",
		validCount, len(explanations), 100*float64(validCount)/float64(len(explanations)))
	fmt.Printf("Max difference: %.2e\n", maxDiff)
}

// createDemoEnsemble creates a simple tree ensemble for demonstration.
// In production, you would load this from XGBoost/LightGBM JSON files.
func createDemoEnsemble() *tree.TreeEnsemble {
	return &tree.TreeEnsemble{
		NumTrees:    3,
		NumFeatures: 3,
		Roots:       []int{0, 4, 8},
		BaseScore:   0.5,
		Nodes: []tree.Node{
			// Tree 0: Split on income
			{Tree: 0, NodeID: 0, Feature: 0, DecisionType: tree.DecisionLess, Threshold: 50000, Yes: 1, No: 2, Missing: 1, Cover: 100},
			{Tree: 0, NodeID: 1, IsLeaf: true, Prediction: -0.2, Cover: 40},
			{Tree: 0, NodeID: 2, Feature: 2, DecisionType: tree.DecisionLess, Threshold: 700, Yes: 3, No: 3, Missing: 3, Cover: 60},
			{Tree: 0, NodeID: 3, IsLeaf: true, Prediction: 0.3, Cover: 60},

			// Tree 1: Split on age
			{Tree: 1, NodeID: 0, Feature: 1, DecisionType: tree.DecisionLess, Threshold: 30, Yes: 5, No: 6, Missing: 5, Cover: 100},
			{Tree: 1, NodeID: 1, IsLeaf: true, Prediction: -0.1, Cover: 30},
			{Tree: 1, NodeID: 2, Feature: 0, DecisionType: tree.DecisionLess, Threshold: 75000, Yes: 7, No: 7, Missing: 7, Cover: 70},
			{Tree: 1, NodeID: 3, IsLeaf: true, Prediction: 0.2, Cover: 70},

			// Tree 2: Split on credit score
			{Tree: 2, NodeID: 0, Feature: 2, DecisionType: tree.DecisionLess, Threshold: 650, Yes: 9, No: 10, Missing: 9, Cover: 100},
			{Tree: 2, NodeID: 1, IsLeaf: true, Prediction: -0.15, Cover: 20},
			{Tree: 2, NodeID: 2, Feature: 1, DecisionType: tree.DecisionLess, Threshold: 45, Yes: 11, No: 11, Missing: 11, Cover: 80},
			{Tree: 2, NodeID: 3, IsLeaf: true, Prediction: 0.25, Cover: 80},
		},
	}
}

// generateTestInstances creates synthetic test data.
func generateTestInstances(n int) [][]float64 {
	instances := make([][]float64, n)
	for i := 0; i < n; i++ {
		// Generate realistic-looking feature values
		income := 30000 + float64((i*1000)%100000) // 30k-130k
		age := 20 + float64((i*7)%50)              // 20-70
		creditScore := 550 + float64((i*11)%300)   // 550-850
		instances[i] = []float64{income, age, creditScore}
	}
	return instances
}
