// Example: Multi-class Classification with SHAP
//
// This example demonstrates how to explain multi-class classification models
// using KernelSHAP. For multi-class models, we explain each class separately
// by wrapping the model to return the probability for a specific class.
//
// The example uses a simple 3-class classifier with 4 features, showing:
// - How to wrap a multi-class model for SHAP explanation
// - Computing SHAP values for each class
// - Interpreting which features contribute to each class prediction
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explainer/kernel"
	"github.com/plexusone/shap-go/model"
)

// softmax converts logits to probabilities
func softmax(logits []float64) []float64 {
	// Find max for numerical stability
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}

	// Compute exp(logit - max)
	expSum := 0.0
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(v - max)
		expSum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= expSum
	}
	return probs
}

// MultiClassModel represents a 3-class classifier.
// Features: sepal_length, sepal_width, petal_length, petal_width (Iris-like)
// Classes: setosa (0), versicolor (1), virginica (2)
type MultiClassModel struct {
	// Weights for each class (4 features x 3 classes)
	weights [][]float64
	// Biases for each class
	biases []float64
}

// NewMultiClassModel creates a simple linear multi-class classifier.
// The weights are chosen to roughly mimic Iris classification patterns:
// - Setosa: small petal length/width
// - Versicolor: medium measurements
// - Virginica: large petal length/width
func NewMultiClassModel() *MultiClassModel {
	return &MultiClassModel{
		// Weights: [sepal_length, sepal_width, petal_length, petal_width]
		weights: [][]float64{
			{-0.5, 1.0, -2.0, -2.0}, // Setosa: negative petal weights
			{0.0, 0.0, 0.5, 0.5},    // Versicolor: moderate petal weights
			{0.3, -0.5, 1.5, 1.5},   // Virginica: positive petal weights
		},
		biases: []float64{3.0, 0.0, -1.0},
	}
}

// PredictProba returns class probabilities for the input
func (m *MultiClassModel) PredictProba(input []float64) []float64 {
	logits := make([]float64, len(m.weights))
	for class := range m.weights {
		logit := m.biases[class]
		for i, w := range m.weights[class] {
			logit += w * input[i]
		}
		logits[class] = logit
	}
	return softmax(logits)
}

// PredictClass returns the predicted class (argmax of probabilities)
func (m *MultiClassModel) PredictClass(input []float64) int {
	probs := m.PredictProba(input)
	maxIdx := 0
	maxProb := probs[0]
	for i, p := range probs[1:] {
		if p > maxProb {
			maxProb = p
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// ClassWrapper wraps a multi-class model to return probability for a specific class.
// This allows us to use standard SHAP explainers which expect a single output.
type ClassWrapper struct {
	model    *MultiClassModel
	classIdx int
}

// NewClassWrapper creates a wrapper for a specific class
func NewClassWrapper(m *MultiClassModel, classIdx int) *ClassWrapper {
	return &ClassWrapper{
		model:    m,
		classIdx: classIdx,
	}
}

// Predict returns the probability for this wrapper's class
func (w *ClassWrapper) Predict(_ context.Context, input []float64) (float64, error) {
	probs := w.model.PredictProba(input)
	return probs[w.classIdx], nil
}

func main() {
	fmt.Println("Multi-class Classification SHAP Example")
	fmt.Println("========================================")
	fmt.Println()

	// Create multi-class model
	classifier := NewMultiClassModel()
	classNames := []string{"Setosa", "Versicolor", "Virginica"}
	featureNames := []string{"sepal_length", "sepal_width", "petal_length", "petal_width"}

	// Background data: representative samples spanning feature space
	// Values normalized to roughly [0, 3] range
	background := [][]float64{
		{1.0, 1.0, 0.5, 0.2}, // Small petals (setosa-like)
		{1.5, 1.0, 1.5, 0.8}, // Medium petals (versicolor-like)
		{2.0, 1.0, 2.5, 1.5}, // Large petals (virginica-like)
		{1.2, 0.8, 1.0, 0.5},
		{1.8, 1.2, 2.0, 1.0},
		{1.0, 1.5, 0.3, 0.1},
		{2.2, 0.9, 2.8, 1.8},
		{1.5, 1.1, 1.2, 0.6},
	}

	// Create explainers for each class
	fmt.Println("Creating KernelSHAP explainers for each class...")
	fmt.Println()

	ctx := context.Background()

	// Test instances representing different classes
	instances := []struct {
		name   string
		values []float64
	}{
		{"Typical Setosa", []float64{1.0, 1.2, 0.3, 0.1}},
		{"Typical Versicolor", []float64{1.5, 1.0, 1.5, 0.8}},
		{"Typical Virginica", []float64{2.2, 1.0, 2.5, 1.5}},
		{"Ambiguous Sample", []float64{1.3, 1.1, 1.0, 0.5}},
	}

	for _, inst := range instances {
		fmt.Printf("Instance: %s\n", inst.name)
		fmt.Printf("  Features: %v\n", inst.values)

		// Get class probabilities
		probs := classifier.PredictProba(inst.values)
		predictedClass := classifier.PredictClass(inst.values)

		fmt.Printf("  Predicted Class: %s (%.1f%% confidence)\n",
			classNames[predictedClass], probs[predictedClass]*100)
		fmt.Printf("  Class Probabilities: ")
		for i, p := range probs {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%s=%.1f%%", classNames[i], p*100)
		}
		fmt.Println()
		fmt.Println()

		// Explain each class
		fmt.Println("  SHAP Values by Class:")
		fmt.Println("  " + "─────────────────────────────────────────────────────────────────")

		for classIdx := range classNames {
			// Create model wrapper for this class
			wrapper := NewClassWrapper(classifier, classIdx)
			shapModel := model.NewFuncModel(wrapper.Predict, 4)

			// Create KernelSHAP explainer
			exp, err := kernel.New(shapModel, background,
				explainer.WithNumSamples(200),
				explainer.WithSeed(42),
				explainer.WithFeatureNames(featureNames),
			)
			if err != nil {
				log.Fatalf("Failed to create explainer for class %d: %v", classIdx, err)
			}

			// Explain
			explanation, err := exp.Explain(ctx, inst.values)
			if err != nil {
				log.Fatalf("Failed to explain: %v", err)
			}

			fmt.Printf("  %s (base=%.3f, pred=%.3f):\n",
				classNames[classIdx], exp.BaseValue(), explanation.Prediction)

			// Show SHAP values with visual bars
			for _, name := range featureNames {
				val := explanation.Values[name]
				bar := ""
				if val > 0 {
					bar = fmt.Sprintf("%s+", repeatChar("█", int(math.Abs(val)*20)))
				} else if val < 0 {
					bar = fmt.Sprintf("-%s", repeatChar("█", int(math.Abs(val)*20)))
				}
				fmt.Printf("    %-14s: %+.4f  %s\n", name, val, bar)
			}

			// Verify local accuracy
			result := explanation.Verify(1e-4)
			if !result.Valid {
				fmt.Printf("    [WARNING] Local accuracy check failed: diff=%.2e\n", result.Difference)
			}
		}
		fmt.Println()
		fmt.Println("  " + "═════════════════════════════════════════════════════════════════")
		fmt.Println()
	}

	// Summary section
	fmt.Println("Key Observations:")
	fmt.Println("-----------------")
	fmt.Println()
	fmt.Println("1. SHAP values are computed separately for each class output")
	fmt.Println("2. Petal features (length/width) have the strongest influence")
	fmt.Println("3. For Setosa: Small petal values → positive SHAP contribution")
	fmt.Println("4. For Virginica: Large petal values → positive SHAP contribution")
	fmt.Println("5. SHAP values for each class sum to (prediction - base value)")
	fmt.Println()
	fmt.Println("This pattern allows you to understand not just which class was")
	fmt.Println("predicted, but WHY each class was or wasn't predicted based on")
	fmt.Println("the input features.")
}

// repeatChar repeats a character n times
func repeatChar(char string, n int) string {
	if n <= 0 {
		return ""
	}
	result := ""
	for i := 0; i < n; i++ {
		result += char
	}
	return result
}
