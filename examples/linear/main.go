// Example: Computing SHAP values for a simple linear model
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/permutation"
	"github.com/plexusone/shap-go/model"
)

func main() {
	// Create a simple linear model: y = 2*x0 + 3*x1 + 1*x2
	// This makes it easy to verify SHAP values are correct
	weights := []float64{2.0, 3.0, 1.0}
	predict := func(ctx context.Context, input []float64) (float64, error) {
		var sum float64
		for i, v := range input {
			sum += weights[i] * v
		}
		return sum, nil
	}

	m := model.NewFuncModel(predict, 3)

	// Background data: using zeros so SHAP values equal weighted feature values
	background := [][]float64{
		{0.0, 0.0, 0.0},
	}

	// Create a permutation explainer
	exp, err := permutation.New(m, background,
		explainer.WithNumSamples(100),
		explainer.WithSeed(42),
		explainer.WithFeatureNames([]string{"feature_a", "feature_b", "feature_c"}),
		explainer.WithModelID("linear-model"),
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	// Instance to explain
	instance := []float64{1.0, 2.0, 3.0}
	// Expected prediction: 2*1 + 3*2 + 1*3 = 2 + 6 + 3 = 11

	ctx := context.Background()
	explanation, err := exp.Explain(ctx, instance)
	if err != nil {
		log.Fatalf("Failed to explain: %v", err)
	}

	// Print results
	fmt.Println("SHAP Explanation for Linear Model")
	fmt.Println("==================================")
	fmt.Printf("Model: y = 2*x0 + 3*x1 + 1*x2\n")
	fmt.Printf("Instance: %v\n\n", instance)
	fmt.Printf("Prediction: %.4f\n", explanation.Prediction)
	fmt.Printf("Base Value: %.4f\n\n", explanation.BaseValue)

	fmt.Println("SHAP Values:")
	for _, name := range explanation.FeatureNames {
		value := explanation.FeatureValues[name]
		shap := explanation.Values[name]
		fmt.Printf("  %s: value=%.2f, SHAP=%.4f\n", name, value, shap)
	}

	// Verify local accuracy
	fmt.Println("\nVerification:")
	result := explanation.Verify(1e-10)
	fmt.Printf("  Sum of SHAP values: %.4f\n", result.SumSHAP)
	fmt.Printf("  Expected (pred - base): %.4f\n", result.Expected)
	fmt.Printf("  Difference: %.10f\n", result.Difference)
	fmt.Printf("  Valid: %v\n", result.Valid)

	// Show top features
	fmt.Println("\nTop Features by |SHAP|:")
	topFeatures := explanation.TopFeatures(3)
	for i, f := range topFeatures {
		fmt.Printf("  %d. %s: %.4f\n", i+1, f.Name, f.SHAPValue)
	}

	// Print JSON
	fmt.Println("\nJSON Output:")
	jsonData, err := explanation.ToJSONPretty()
	if err != nil {
		log.Fatalf("Failed to serialize: %v", err)
	}
	fmt.Println(string(jsonData))
}
