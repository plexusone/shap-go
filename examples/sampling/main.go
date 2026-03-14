// Example: Computing SHAP values using Sampling SHAP
//
// SamplingSHAP uses simple Monte Carlo sampling to approximate SHAP values.
// It's faster than PermutationSHAP but has higher variance. Best for quick
// estimates when exact values aren't critical.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/sampling"
	"github.com/plexusone/shap-go/model"
)

func main() {
	fmt.Println("Sampling SHAP Example")
	fmt.Println("=====================")
	fmt.Println()

	// Create a non-linear model to demonstrate
	// f(x) = x0^2 + 2*x1 + x0*x2
	predict := func(ctx context.Context, input []float64) (float64, error) {
		x0, x1, x2 := input[0], input[1], input[2]
		return x0*x0 + 2*x1 + x0*x2, nil
	}

	m := model.NewFuncModel(predict, 3)

	// Background data: representative samples from the feature distribution
	// More background samples generally improves accuracy
	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.5, 0.5, 0.5},
	}

	// Create sampling explainer
	// Lower NumSamples = faster but less accurate
	// Higher NumSamples = slower but more accurate
	exp, err := sampling.New(m, background,
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
	}

	for _, inst := range instances {
		explanation, err := exp.Explain(ctx, inst.values)
		if err != nil {
			log.Fatalf("Failed to explain %s: %v", inst.name, err)
		}

		fmt.Printf("Instance: %s = %v\n", inst.name, inst.values)
		fmt.Printf("  Prediction: %.4f\n", explanation.Prediction)
		fmt.Printf("  Base Value: %.4f\n", explanation.BaseValue)
		fmt.Printf("  SHAP Values:\n")
		for _, name := range explanation.FeatureNames {
			fmt.Printf("    %s: %.4f\n", name, explanation.Values[name])
		}

		// Note: SamplingSHAP doesn't guarantee local accuracy
		// Check how close we are
		result := explanation.Verify(0.5) // Use larger tolerance
		fmt.Printf("  Local Accuracy Check:\n")
		fmt.Printf("    Sum of SHAP: %.4f\n", result.SumSHAP)
		fmt.Printf("    Expected:    %.4f\n", result.Expected)
		fmt.Printf("    Difference:  %.4f\n", result.Difference)
		fmt.Println()
	}

	// Compare with different sample counts
	fmt.Println("Effect of Sample Count on Accuracy")
	fmt.Println("-----------------------------------")

	instance := []float64{1.0, 1.0, 1.0}
	sampleCounts := []int{10, 50, 100, 500}

	for _, samples := range sampleCounts {
		exp, _ := sampling.New(m, background,
			explainer.WithNumSamples(samples),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
		)

		explanation, _ := exp.Explain(ctx, instance)
		result := explanation.Verify(1.0)

		fmt.Printf("Samples=%3d: diff=%.4f (valid at 0.5 tolerance: %v)\n",
			samples, result.Difference, result.Difference < 0.5)
	}
}
