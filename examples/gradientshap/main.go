// Example: Computing SHAP values using GradientSHAP (Expected Gradients)
//
// GradientSHAP uses the Expected Gradients method to compute SHAP values.
// It computes gradients at interpolated points between the input and
// background samples, combining ideas from Integrated Gradients and SHAP.
//
// Key properties:
// - Model-agnostic: works with any differentiable model
// - Uses numerical gradients (finite differences)
// - Lower variance than pure sampling methods
// - Satisfies local accuracy: sum(SHAP) = prediction - baseline
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/gradient"
	"github.com/plexusone/shap-go/model"
)

func main() {
	fmt.Println("GradientSHAP (Expected Gradients) Example")
	fmt.Println("==========================================")
	fmt.Println()

	// Create a nonlinear model to demonstrate GradientSHAP
	// f(x) = x0^2 + 2*x0*x1 + sin(x2)
	predict := func(ctx context.Context, input []float64) (float64, error) {
		x0, x1, x2 := input[0], input[1], input[2]
		return x0*x0 + 2*x0*x1 + math.Sin(x2), nil
	}

	m := model.NewFuncModel(predict, 3)

	// Background data: representative samples from the feature distribution
	background := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.5, 0.5, 0.5},
		{1.0, 1.0, 1.0},
		{-0.5, 0.5, 0.0},
		{0.5, -0.5, 0.0},
	}

	// Create GradientSHAP explainer
	exp, err := gradient.New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(300),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
		},
	)
	if err != nil {
		log.Fatalf("Failed to create explainer: %v", err)
	}

	ctx := context.Background()

	fmt.Printf("Base Value: %.4f\n\n", exp.BaseValue())

	// Explain different instances
	instances := []struct {
		name   string
		values []float64
	}{
		{"Origin", []float64{0.0, 0.0, 0.0}},
		{"Quadratic dominant", []float64{2.0, 0.0, 0.0}},
		{"Interaction dominant", []float64{1.0, 2.0, 0.0}},
		{"Sine dominant", []float64{0.0, 0.0, 1.57}}, // pi/2
		{"Mixed", []float64{1.0, 1.0, 1.0}},
	}

	for _, inst := range instances {
		explanation, err := exp.Explain(ctx, inst.values)
		if err != nil {
			log.Fatalf("Failed to explain %s: %v", inst.name, err)
		}

		pred := explanation.Prediction
		fmt.Printf("Instance: %s = %v\n", inst.name, inst.values)
		fmt.Printf("  Prediction: %.4f\n", pred)
		fmt.Printf("  SHAP Values:\n")

		// Show SHAP values with visual bars
		for _, name := range explanation.FeatureNames {
			val := explanation.Values[name]
			bar := visualBar(val)
			fmt.Printf("    %s: %+.4f  %s\n", name, val, bar)
		}

		// Verify local accuracy
		result := explanation.Verify(0.5)
		fmt.Printf("  Local Accuracy: %v (diff=%.2e)\n", result.Valid, result.Difference)
		fmt.Println()
	}

	// Demonstrate confidence intervals
	fmt.Println("GradientSHAP with Confidence Intervals")
	fmt.Println("---------------------------------------")
	fmt.Println()

	expCI, err := gradient.New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(500),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
			explainer.WithConfidenceLevel(0.95),
		},
	)
	if err != nil {
		log.Fatalf("Failed to create explainer with CI: %v", err)
	}

	instance := []float64{1.5, 1.0, 0.5}
	explanation, err := expCI.Explain(ctx, instance)
	if err != nil {
		log.Fatalf("Failed to explain: %v", err)
	}

	fmt.Printf("Instance: %v\n", instance)
	fmt.Printf("Prediction: %.4f (Base: %.4f)\n", explanation.Prediction, exp.BaseValue())
	fmt.Println()
	fmt.Printf("%-10s %-12s %-12s %-12s %-12s\n", "Feature", "SHAP Value", "Std Error", "95% CI Low", "95% CI High")
	fmt.Println("---------------------------------------------------------------")

	for _, name := range explanation.FeatureNames {
		shapVal := explanation.Values[name]
		low, high, _ := explanation.GetConfidenceInterval(name)
		se := explanation.Metadata.ConfidenceIntervals.StandardErrors[name]
		fmt.Printf("%-10s %+.4f      %.4f       %+.4f      %+.4f\n",
			name, shapVal, se, low, high)
	}

	fmt.Println()

	// Compare sequential vs parallel
	fmt.Println("Sequential vs Parallel Performance")
	fmt.Println("-----------------------------------")
	fmt.Println()

	// Sequential
	expSeq, _ := gradient.New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(200),
			explainer.WithSeed(42),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
		},
	)

	seqExp, _ := expSeq.Explain(ctx, instance)

	// Parallel
	expPar, _ := gradient.New(m, background,
		[]explainer.Option{
			explainer.WithNumSamples(200),
			explainer.WithSeed(42),
			explainer.WithNumWorkers(4),
			explainer.WithFeatureNames([]string{"x0", "x1", "x2"}),
		},
	)

	parExp, _ := expPar.Explain(ctx, instance)

	fmt.Println("Both methods produce similar results:")
	fmt.Printf("%-10s %-15s %-15s\n", "Feature", "Sequential", "Parallel")
	fmt.Println("----------------------------------------")
	for _, name := range seqExp.FeatureNames {
		fmt.Printf("%-10s %+.4f         %+.4f\n", name, seqExp.Values[name], parExp.Values[name])
	}
	fmt.Println()

	// Demonstrate with custom epsilon
	fmt.Println("Effect of Epsilon on Gradient Accuracy")
	fmt.Println("---------------------------------------")
	fmt.Println()

	// Simple quadratic for demonstration
	quadratic := func(ctx context.Context, input []float64) (float64, error) {
		return input[0] * input[0], nil
	}
	mQuad := model.NewFuncModel(quadratic, 1)
	bgSimple := [][]float64{{0.0}, {1.0}}

	epsilons := []float64{1e-4, 1e-6, 1e-8, 1e-10}
	point := []float64{2.0}

	fmt.Printf("f(x) = x^2, x = 2.0\n")
	fmt.Printf("True gradient: 2*x = 4.0\n\n")
	fmt.Printf("%-12s %-15s\n", "Epsilon", "SHAP Value")
	fmt.Println("---------------------------")

	for _, eps := range epsilons {
		expEps, _ := gradient.New(mQuad, bgSimple,
			[]explainer.Option{
				explainer.WithNumSamples(100),
				explainer.WithSeed(42),
			},
			gradient.WithEpsilon(eps),
		)

		result, _ := expEps.Explain(ctx, point)
		shapVal := result.Values["feature_0"]
		fmt.Printf("%-12.0e %+.6f\n", eps, shapVal)
	}

	fmt.Println()
	fmt.Println("Note: Default epsilon (1e-7) provides a good balance between")
	fmt.Println("accuracy and numerical stability for most models.")
}

func visualBar(val float64) string {
	maxBars := 20
	absBars := int(math.Min(math.Abs(val)*10, float64(maxBars)))

	if val > 0 {
		return repeat("█", absBars) + "+"
	} else if val < 0 {
		return "-" + repeat("█", absBars)
	}
	return ""
}

func repeat(s string, n int) string {
	if n <= 0 {
		return ""
	}
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
