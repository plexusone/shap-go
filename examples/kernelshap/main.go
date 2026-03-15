// Example: Computing SHAP values using KernelSHAP
//
// KernelSHAP uses weighted linear regression to approximate SHAP values.
// It's model-agnostic and guarantees local accuracy (SHAP values sum to
// prediction - baseline). It has lower variance than SamplingSHAP due to
// the principled Shapley kernel weighting.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/kernel"
	"github.com/plexusone/shap-go/model"
)

func main() {
	fmt.Println("KernelSHAP Example")
	fmt.Println("==================")
	fmt.Println()

	// Create a non-linear model to demonstrate
	// f(x) = x0^2 + 2*x1 + x0*x2
	predict := func(ctx context.Context, input []float64) (float64, error) {
		x0, x1, x2 := input[0], input[1], input[2]
		return x0*x0 + 2*x1 + x0*x2, nil
	}

	m := model.NewFuncModel(predict, 3)

	// Background data: representative samples from the feature distribution
	// KernelSHAP averages over background samples for masked features
	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.5, 0.5, 0.5},
		{1.0, 1.0, 1.0},
	}

	// Create KernelSHAP explainer
	// KernelSHAP uses Shapley kernel weights to emphasize informative coalitions
	exp, err := kernel.New(m, background,
		explainer.WithNumSamples(200),
		explainer.WithSeed(42),
		explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	ctx := context.Background()

	// Explain different instances
	instances := []struct {
		name   string
		values []float64
	}{
		{"Origin", []float64{0.0, 0.0, 0.0}},
		{"Unit x0", []float64{1.0, 0.0, 0.0}},
		{"Unit x1", []float64{0.0, 1.0, 0.0}},
		{"Mixed", []float64{1.0, 1.0, 1.0}},
		{"Large x0", []float64{2.0, 0.5, 0.5}},
	}

	fmt.Printf("Base Value: %.4f\n\n", exp.BaseValue())

	for _, inst := range instances {
		explanation, err := exp.Explain(ctx, inst.values)
		if err != nil {
			log.Fatalf("Failed to explain %s: %v", inst.name, err)
		}

		fmt.Printf("Instance: %s = %v\n", inst.name, inst.values)
		fmt.Printf("  Prediction: %.4f\n", explanation.Prediction)
		fmt.Printf("  SHAP Values:\n")
		for _, name := range explanation.FeatureNames {
			fmt.Printf("    %s: %+.4f\n", name, explanation.Values[name])
		}

		// KernelSHAP guarantees local accuracy via constrained regression
		result := explanation.Verify(1e-6)
		fmt.Printf("  Local Accuracy: %v (diff=%.2e)\n", result.Valid, result.Difference)
		fmt.Println()
	}

	// Compare KernelSHAP with different sample counts
	fmt.Println("Effect of Sample Count on SHAP Estimates")
	fmt.Println("-----------------------------------------")
	fmt.Println()

	instance := []float64{2.0, 1.0, 0.5}
	sampleCounts := []int{20, 50, 100, 200, 500}

	fmt.Println("Instance: [2.0, 1.0, 0.5]")
	fmt.Println()
	fmt.Printf("%-10s %-12s %-12s %-12s %-12s\n", "Samples", "SHAP(x0)", "SHAP(x1)", "SHAP(x2)", "Sum")
	fmt.Println("--------------------------------------------------------------")

	for _, samples := range sampleCounts {
		exp, _ := kernel.New(m, background,
			explainer.WithNumSamples(samples),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
		)

		explanation, _ := exp.Explain(ctx, instance)
		sum := explanation.Values["x0"] + explanation.Values["x1"] + explanation.Values["x2"]

		fmt.Printf("%-10d %+.4f      %+.4f      %+.4f      %.4f\n",
			samples,
			explanation.Values["x0"],
			explanation.Values["x1"],
			explanation.Values["x2"],
			sum)
	}

	fmt.Println()
	fmt.Println("Note: Local accuracy is always guaranteed regardless of sample count.")
	fmt.Println("More samples improve the accuracy of individual SHAP values.")
}
