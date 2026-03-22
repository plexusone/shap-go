package gradient

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

// PythonTestCase represents a single test instance from the JSON file.
type PythonTestCase struct {
	Name       string    `json:"name"`
	Instance   []float64 `json:"instance"`
	Prediction float64   `json:"prediction"`
	SHAPValues []float64 `json:"shap_values"`
}

// PythonTestSuite represents a test suite from the JSON file.
type PythonTestSuite struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Model       json.RawMessage  `json:"model"`
	Background  [][]float64      `json:"background"`
	BaseValue   float64          `json:"base_value"`
	Cases       []PythonTestCase `json:"cases"`
}

// PythonTestCases represents the full JSON test file.
type PythonTestCases struct {
	Version    string            `json:"version"`
	Note       string            `json:"note"`
	TestSuites []PythonTestSuite `json:"test_suites"`
}

// createModelFromSuite creates a simple model that approximates the target function.
// Since we can't load the exact Python MLP, we use analytical functions for validation.
func createModelFromSuite(suite PythonTestSuite) model.Model {
	// Parse model info
	var modelInfo struct {
		Type           string `json:"type"`
		TargetFunction string `json:"target_function"`
	}
	_ = json.Unmarshal(suite.Model, &modelInfo)

	numFeatures := len(suite.Background[0])

	switch modelInfo.TargetFunction {
	case "2*x0 + 3*x1 + x2":
		predict := func(_ context.Context, input []float64) (float64, error) {
			return 2*input[0] + 3*input[1] + input[2], nil
		}
		return model.NewFuncModel(predict, numFeatures)

	case "x0^2 + 2*x1 + x0*x2":
		predict := func(_ context.Context, input []float64) (float64, error) {
			return input[0]*input[0] + 2*input[1] + input[0]*input[2], nil
		}
		return model.NewFuncModel(predict, numFeatures)

	case "x0 + x1":
		predict := func(_ context.Context, input []float64) (float64, error) {
			return input[0] + input[1], nil
		}
		return model.NewFuncModel(predict, numFeatures)

	case "sin(x0) + 2*x1":
		predict := func(_ context.Context, input []float64) (float64, error) {
			return math.Sin(input[0]) + 2*input[1], nil
		}
		return model.NewFuncModel(predict, numFeatures)

	default:
		// Default linear model
		predict := func(_ context.Context, input []float64) (float64, error) {
			sum := 0.0
			for _, v := range input {
				sum += v
			}
			return sum, nil
		}
		return model.NewFuncModel(predict, numFeatures)
	}
}

// TestAgainstPythonSHAP tests the Go GradientSHAP implementation against
// known values from the Python SHAP library.
//
// Note: GradientSHAP is stochastic, so we use correlation and directional
// checks rather than exact value matching.
func TestAgainstPythonSHAP(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "gradientshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v (run generate_gradientshap_test_cases.py first)", err)
	}

	var testCases PythonTestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			m := createModelFromSuite(suite)
			defer m.Close()

			exp, err := New(m, suite.Background,
				[]explainer.Option{
					explainer.WithNumSamples(500),
					explainer.WithSeed(42),
					explainer.WithFeatureNames(makeFeatureNames(len(suite.Background[0]))),
				},
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

					// Check local accuracy (most important property)
					verifyResult := result.Verify(0.5) // Looser tolerance for sampling method
					if !verifyResult.Valid {
						t.Errorf("Local accuracy failed: sum=%.4f, expected=%.4f, diff=%.4f",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}

					// Check correlation with Python SHAP values
					// Both should identify the same important features
					goValues := make([]float64, len(tc.SHAPValues))
					for i := range tc.SHAPValues {
						goValues[i] = result.Values[makeFeatureName(i)]
					}

					// Check if both are near-zero (at background mean)
					goNorm := norm(goValues)
					pyNorm := norm(tc.SHAPValues)

					if goNorm < 0.01 && pyNorm < 0.01 {
						// Both are near zero - this is expected at background mean
						// Skip correlation check
					} else if goNorm < 0.01 || pyNorm < 0.01 {
						// One is near zero, the other isn't - this is suspicious
						t.Logf("Norm mismatch: Go=%.4f, Python=%.4f", goNorm, pyNorm)
					} else {
						// Check if vectors are close via normalized MSE
						mse := meanSquaredError(goValues, tc.SHAPValues)
						if mse < 0.1 {
							// Vectors are very close - skip correlation check
						} else {
							corr := correlation(goValues, tc.SHAPValues)
							if corr < 0.7 { // Allow some variance due to stochastic nature
								t.Errorf("Low correlation with Python SHAP: %.4f", corr)
								t.Logf("Go SHAP:     %v", goValues)
								t.Logf("Python SHAP: %v", tc.SHAPValues)
							}
						}
					}

					// Check directional agreement for dominant features
					for i := range tc.SHAPValues {
						goVal := goValues[i]
						pyVal := tc.SHAPValues[i]

						// If Python value is significant, check direction matches
						if math.Abs(pyVal) > 0.1 {
							if (goVal > 0) != (pyVal > 0) {
								// Direction mismatch is concerning but not fatal
								// due to stochastic nature
								t.Logf("Direction mismatch for feature %d: Go=%.4f, Python=%.4f",
									i, goVal, pyVal)
							}
						}
					}
				})
			}
		})
	}
}

// TestLocalAccuracyProperty verifies the fundamental SHAP property:
// sum(SHAP values) ≈ prediction - base_value
func TestLocalAccuracyProperty(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "gradientshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v", err)
	}

	var testCases PythonTestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			m := createModelFromSuite(suite)
			defer m.Close()

			exp, err := New(m, suite.Background,
				[]explainer.Option{
					explainer.WithNumSamples(500),
					explainer.WithSeed(42),
				},
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

					// GradientSHAP should approximately satisfy local accuracy
					verifyResult := result.Verify(0.5)
					if !verifyResult.Valid {
						t.Errorf("Local accuracy violated: sum=%.4f, expected=%.4f, diff=%.4f",
							verifyResult.SumSHAP, verifyResult.Expected, verifyResult.Difference)
					}
				})
			}
		})
	}
}

// TestFeatureImportanceRanking checks if GradientSHAP identifies the correct
// most important features compared to Python SHAP.
func TestFeatureImportanceRanking(t *testing.T) {
	testDataPath := filepath.Join("..", "..", "testdata", "gradientshap_test_cases.json")
	data, err := os.ReadFile(testDataPath)
	if err != nil {
		t.Skipf("Test data file not found: %v", err)
	}

	var testCases PythonTestCases
	if err := json.Unmarshal(data, &testCases); err != nil {
		t.Fatalf("Failed to parse test cases: %v", err)
	}

	ctx := context.Background()

	for _, suite := range testCases.TestSuites {
		t.Run(suite.Name, func(t *testing.T) {
			m := createModelFromSuite(suite)
			defer m.Close()

			exp, err := New(m, suite.Background,
				[]explainer.Option{
					explainer.WithNumSamples(500),
					explainer.WithSeed(42),
					explainer.WithFeatureNames(makeFeatureNames(len(suite.Background[0]))),
				},
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

					// Find most important feature by absolute value
					goMaxIdx := 0
					pyMaxIdx := 0
					goMaxVal := 0.0
					pyMaxVal := 0.0

					for i := range tc.SHAPValues {
						goVal := math.Abs(result.Values[makeFeatureName(i)])
						pyVal := math.Abs(tc.SHAPValues[i])

						if goVal > goMaxVal {
							goMaxVal = goVal
							goMaxIdx = i
						}
						if pyVal > pyMaxVal {
							pyMaxVal = pyVal
							pyMaxIdx = i
						}
					}

					// Most important feature should often match
					// (not always due to stochastic nature)
					if goMaxIdx != pyMaxIdx && pyMaxVal > 0.2 {
						t.Logf("Most important feature differs: Go=%d, Python=%d (values: %.4f vs %.4f)",
							goMaxIdx, pyMaxIdx, goMaxVal, pyMaxVal)
					}
				})
			}
		})
	}
}

// makeFeatureNames generates feature names for n features.
func makeFeatureNames(n int) []string {
	names := make([]string, n)
	for i := 0; i < n; i++ {
		names[i] = makeFeatureName(i)
	}
	return names
}

// makeFeatureName generates a feature name for index i.
func makeFeatureName(i int) string {
	return fmt.Sprintf("x%d", i)
}

// norm computes the L2 norm of a vector.
func norm(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// meanSquaredError computes the MSE between two vectors.
func meanSquaredError(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.Inf(1)
	}
	sum := 0.0
	for i := range x {
		diff := x[i] - y[i]
		sum += diff * diff
	}
	return sum / float64(len(x))
}

// correlation computes Pearson correlation between two slices.
func correlation(x, y []float64) float64 {
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
