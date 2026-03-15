package kernel

import (
	"context"
	"encoding/json"
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
}

// ModelDef represents the model definition in the test case.
type ModelDef struct {
	Type    string    `json:"type"`
	Weights []float64 `json:"weights,omitempty"`
	Bias    float64   `json:"bias,omitempty"`
}

// TestSuite represents a test suite from the JSON file.
type TestSuite struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Model       ModelDef    `json:"model"`
	Background  [][]float64 `json:"background"`
	BaseValue   float64     `json:"base_value"`
	Cases       []TestCase  `json:"cases"`
}

// TestCases represents the full JSON test file.
type TestCases struct {
	Version    string      `json:"version"`
	TestSuites []TestSuite `json:"test_suites"`
}

// createModelFromDef creates a model.Model from the test suite definition.
func createModelFromDef(def ModelDef) model.Model {
	switch def.Type {
	case "linear":
		fn := func(ctx context.Context, input []float64) (float64, error) {
			sum := def.Bias
			for i, w := range def.Weights {
				if i < len(input) {
					sum += w * input[i]
				}
			}
			return sum, nil
		}
		return model.NewFuncModel(fn, len(def.Weights))

	case "quadratic":
		// f(x) = x0^2 + 2*x1 + x0*x2
		fn := func(ctx context.Context, input []float64) (float64, error) {
			if len(input) < 3 {
				return 0, nil
			}
			return input[0]*input[0] + 2*input[1] + input[0]*input[2], nil
		}
		return model.NewFuncModel(fn, 3)

	default:
		return nil
	}
}

// TestAgainstPythonSHAP tests the Go KernelSHAP implementation against
// known-correct values from the Python SHAP library.
func TestAgainstPythonSHAP(t *testing.T) {
	// Find the test cases file
	testDataPath := filepath.Join("..", "..", "testdata", "kernelshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v (run generate_kernelshap_test_cases.py first)", err)
	}

	var testCases TestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			m := createModelFromDef(suite.Model)
			if m == nil {
				t.Skipf("Unknown model type: %s", suite.Model.Type)
			}

			// Create KernelSHAP explainer with high sample count for accuracy
			exp, err := New(m, suite.Background,
				explainer.WithNumSamples(500),
				explainer.WithSeed(42),
				explainer.WithFeatureNames(makeFeatureNames(len(suite.Background[0]))),
			)
			if err != nil {
				t.Fatalf("Failed to create explainer: %v", err)
			}

			// Verify base value (with tolerance for sampling variance)
			gotBase := exp.BaseValue()
			baseTolerance := 0.1 // Base value computed from background, should be close
			if math.Abs(gotBase-suite.BaseValue) > baseTolerance {
				t.Errorf("Base value mismatch: got %f, expected %f (diff=%f)",
					gotBase, suite.BaseValue, math.Abs(gotBase-suite.BaseValue))
			}

			for _, tc := range suite.Cases {
				t.Run(tc.Name, func(t *testing.T) {
					result, err := exp.Explain(ctx, tc.Instance)
					if err != nil {
						t.Fatalf("Explain failed: %v", err)
					}

					// Check prediction (should be exact)
					predTolerance := 1e-6
					if math.Abs(result.Prediction-tc.Prediction) > predTolerance {
						t.Errorf("Prediction mismatch: got %f, expected %f",
							result.Prediction, tc.Prediction)
					}

					// Check local accuracy (should always be satisfied)
					verifyResult := result.Verify(1e-6)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy failed: sum=%f, expected=%f, diff=%f",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}

					// Check individual SHAP values against Python SHAP
					// Use larger tolerance since KernelSHAP is approximate
					shapTolerance := 0.3 // KernelSHAP has variance
					for i, expected := range tc.SHAPValues {
						featureName := makeFeatureName(i)
						got, ok := result.Values[featureName]
						if !ok {
							t.Errorf("Missing SHAP value for feature %s", featureName)
							continue
						}
						if math.Abs(got-expected) > shapTolerance {
							t.Errorf("SHAP[%s] mismatch: got %f, expected %f (diff=%f)",
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
	return "x" + itoa(i)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	if i < 0 {
		return "-" + itoa(-i)
	}
	digits := ""
	for i > 0 {
		digits = string(rune('0'+i%10)) + digits
		i /= 10
	}
	return digits
}

// TestLocalAccuracyProperty verifies the fundamental SHAP property:
// sum(SHAP values) = prediction - base_value
// This should ALWAYS be satisfied for KernelSHAP due to constrained regression.
func TestLocalAccuracyProperty(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "kernelshap_test_cases.json")
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
			m := createModelFromDef(suite.Model)
			if m == nil {
				t.Skipf("Unknown model type: %s", suite.Model.Type)
			}

			exp, err := New(m, suite.Background,
				explainer.WithNumSamples(200),
				explainer.WithSeed(42),
			)
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
					// KernelSHAP uses constrained regression, so this should be exact
					verifyResult := result.Verify(1e-9)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy violated: sum=%f, expected=%f, diff=%e",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}
				})
			}
		})
	}
}

// TestCorrelationWithPythonSHAP tests that our SHAP values correlate well
// with Python SHAP, even if they're not exactly equal (due to sampling variance).
func TestCorrelationWithPythonSHAP(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "kernelshap_test_cases.json")
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
			m := createModelFromDef(suite.Model)
			if m == nil {
				t.Skipf("Unknown model type: %s", suite.Model.Type)
			}

			exp, err := New(m, suite.Background,
				explainer.WithNumSamples(500),
				explainer.WithSeed(42),
			)
			if err != nil {
				t.Fatalf("Failed to create explainer: %v", err)
			}

			// Collect all SHAP values for correlation analysis
			var pythonValues, goValues []float64

			for _, tc := range suite.Cases {
				result, err := exp.Explain(ctx, tc.Instance)
				if err != nil {
					t.Fatalf("Explain failed: %v", err)
				}

				for i, expected := range tc.SHAPValues {
					featureName := makeFeatureName(i)
					got, ok := result.Values[featureName]
					if ok {
						pythonValues = append(pythonValues, expected)
						goValues = append(goValues, got)
					}
				}
			}

			// Compute correlation coefficient
			if len(pythonValues) > 0 {
				corr := pearsonCorrelation(pythonValues, goValues)
				t.Logf("Correlation with Python SHAP: %.4f", corr)

				// Correlation should be high (> 0.9)
				if corr < 0.9 {
					t.Errorf("Low correlation with Python SHAP: %.4f (expected > 0.9)", corr)
				}
			}
		})
	}
}

// pearsonCorrelation computes the Pearson correlation coefficient.
func pearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	n := float64(len(x))

	// Compute means
	var sumX, sumY float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	// Compute correlation
	var sumXY, sumX2, sumY2 float64
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		sumXY += dx * dy
		sumX2 += dx * dx
		sumY2 += dy * dy
	}

	if sumX2 == 0 || sumY2 == 0 {
		return 0
	}

	return sumXY / math.Sqrt(sumX2*sumY2)
}
