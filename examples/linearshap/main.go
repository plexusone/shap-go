// Example linearshap demonstrates LinearSHAP for exact SHAP values on linear models.
//
// LinearSHAP provides exact (not approximate) SHAP values using a closed-form
// solution. For a linear model f(x) = bias + Σ wᵢxᵢ, the SHAP value for
// feature i is: SHAP[i] = wᵢ × (xᵢ - E[xᵢ])
package main

import (
	"context"
	"fmt"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/linear"
)

func main() {
	// Example: Linear regression model for house price prediction
	// f(price) = 50000 + 100*sqft + 5000*bedrooms + 10000*bathrooms - 500*age

	weights := []float64{100.0, 5000.0, 10000.0, -500.0}
	bias := 50000.0
	featureNames := []string{"sqft", "bedrooms", "bathrooms", "age"}

	// Background data: representative samples from the housing market
	background := [][]float64{
		{1500, 3, 2, 20}, // 1500 sqft, 3 bed, 2 bath, 20 years old
		{2000, 4, 2, 10},
		{1200, 2, 1, 30},
		{2500, 4, 3, 5},
		{1800, 3, 2, 15},
	}

	// Create LinearSHAP explainer
	exp, err := linear.New(weights, bias, background,
		explainer.WithFeatureNames(featureNames),
		explainer.WithModelID("house-price-linear-v1"),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println("LinearSHAP Example: House Price Prediction")
	fmt.Println("==========================================")
	fmt.Println()

	// Show model info
	fmt.Printf("Model: f(x) = %.0f", bias)
	for i, w := range weights {
		if w >= 0 {
			fmt.Printf(" + %.0f×%s", w, featureNames[i])
		} else {
			fmt.Printf(" - %.0f×%s", -w, featureNames[i])
		}
	}
	fmt.Println()
	fmt.Println()

	// Show feature means from background
	fmt.Println("Background feature means:")
	means := exp.FeatureMeans()
	for i, name := range featureNames {
		fmt.Printf("  E[%s] = %.1f\n", name, means[i])
	}
	fmt.Printf("\nBase value (expected price): $%.2f\n", exp.BaseValue())
	fmt.Println()

	// Explain a specific house
	ctx := context.Background()
	house := []float64{2200, 4, 3, 8} // 2200 sqft, 4 bed, 3 bath, 8 years old

	fmt.Println("House to explain:")
	for i, name := range featureNames {
		fmt.Printf("  %s = %.0f\n", name, house[i])
	}
	fmt.Println()

	explanation, err := exp.Explain(ctx, house)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Predicted price: $%.2f\n", explanation.Prediction)
	fmt.Printf("Base value:      $%.2f\n", explanation.BaseValue)
	fmt.Printf("Difference:      $%.2f\n", explanation.Prediction-explanation.BaseValue)
	fmt.Println()

	fmt.Println("Feature contributions (SHAP values):")
	for _, f := range explanation.TopFeatures(len(featureNames)) {
		sign := "+"
		if f.SHAPValue < 0 {
			sign = "-"
		}
		fmt.Printf("  %s %s: $%.2f\n", sign, f.Name, f.SHAPValue)
	}
	fmt.Println()

	// Verify local accuracy
	result := explanation.Verify(1e-10)
	fmt.Printf("Local accuracy check: %v\n", result.Valid)
	fmt.Printf("  Sum of SHAP values: $%.2f\n", result.SumSHAP)
	fmt.Printf("  Expected (pred - base): $%.2f\n", result.Expected)
	fmt.Println()

	// Explain another house (below average)
	fmt.Println("---")
	fmt.Println()
	cheapHouse := []float64{1000, 2, 1, 40}
	fmt.Println("Explaining a smaller, older house:")
	for i, name := range featureNames {
		fmt.Printf("  %s = %.0f\n", name, cheapHouse[i])
	}
	fmt.Println()

	explanation2, _ := exp.Explain(ctx, cheapHouse)
	fmt.Printf("Predicted price: $%.2f\n", explanation2.Prediction)
	fmt.Println()
	fmt.Println("Feature contributions:")
	for _, f := range explanation2.TopFeatures(len(featureNames)) {
		sign := "+"
		if f.SHAPValue < 0 {
			sign = "-"
		}
		fmt.Printf("  %s %s: $%.2f\n", sign, f.Name, f.SHAPValue)
	}
}
