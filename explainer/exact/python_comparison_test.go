package exact

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/model"
)

// TestCase represents a single test instance from the JSON file.
type TestCase struct {
	Name       string    `json:"name"`
	Instance   []float64 `json:"instance"`
	Prediction float64   `json:"prediction"`
	SHAPValues []float64 `json:"shap_values"`
	Comment    string    `json:"comment,omitempty"`
}

// TestSuite represents a test suite from the JSON file.
type TestSuite struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Model       string      `json:"model"`
	NFeatures   int         `json:"n_features"`
	Background  [][]float64 `json:"background"`
	BaseValue   float64     `json:"base_value"`
	Cases       []TestCase  `json:"cases"`
}

// TestCases represents the full JSON test file.
type TestCases struct {
	Version    string      `json:"version"`
	TestSuites []TestSuite `json:"test_suites"`
}

// modelFromSuite creates a model function based on the test suite definition.
func modelFromSuite(suite TestSuite) func(ctx context.Context, input []float64) (float64, error) {
	switch suite.Model {
	case "linear":
		// Linear model with weights 1, 2, 3, ... for each feature
		return func(ctx context.Context, input []float64) (float64, error) {
			sum := 0.0
			for i, v := range input {
				sum += float64(i+1) * v
			}
			return sum, nil
		}
	case "weighted_linear":
		// Weighted linear: y = 5 + 2*x0 + 3*x1
		return func(ctx context.Context, input []float64) (float64, error) {
			return 5.0 + 2.0*input[0] + 3.0*input[1], nil
		}
	case "quadratic":
		// Quadratic: y = x0 * x1
		return func(ctx context.Context, input []float64) (float64, error) {
			return input[0] * input[1], nil
		}
	default:
		return nil
	}
}

// TestAgainstKnownValues tests the Go ExactSHAP implementation against
// mathematically derived correct Shapley values.
func TestAgainstKnownValues(t *testing.T) {
	// Find the test cases file
	testDataPath := filepath.Join("..", "..", "testdata", "exactshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v", err)
	}

	var testCases TestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()
	tolerance := 1e-10

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			modelFunc := modelFromSuite(suite)
			if modelFunc == nil {
				t.Skipf("Unknown model type: %s", suite.Model)
			}

			fm := model.NewFuncModel(modelFunc, suite.NFeatures)

			exp, err := New(fm, suite.Background,
				explainer.WithFeatureNames(makeFeatureNames(suite.NFeatures)),
			)
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

					// Check local accuracy (ExactSHAP should always satisfy this)
					verifyResult := result.Verify(tolerance)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}

					// Check individual SHAP values against expected
					for i, expected := range tc.SHAPValues {
						featureName := makeFeatureName(i)
						got, ok := result.Values[featureName]
						if !ok {
							t.Errorf("Missing SHAP value for feature %s", featureName)
							continue
						}
						if math.Abs(got-expected) > tolerance {
							t.Errorf("SHAP[%s] mismatch: got %f, expected %f (diff=%e)",
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
	testDataPath := filepath.Join("..", "..", "testdata", "exactshap_test_cases.json")
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
			modelFunc := modelFromSuite(suite)
			if modelFunc == nil {
				t.Skipf("Unknown model type: %s", suite.Model)
			}

			fm := model.NewFuncModel(modelFunc, suite.NFeatures)

			exp, err := New(fm, suite.Background)
			if err != nil {
				t.Fatalf("Failed to create explainer: %v", err)
			}

			for _, tc := range suite.Cases {
				t.Run(tc.Name, func(t *testing.T) {
					result, err := exp.Explain(ctx, tc.Instance)
					if err != nil {
						t.Fatalf("Explain failed: %v", err)
					}

					// Verify local accuracy with very tight tolerance
					// ExactSHAP should satisfy this within floating point precision
					verifyResult := result.Verify(1e-12)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy violated: sum=%f, expected=%f, diff=%e",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}
				})
			}
		})
	}
}
