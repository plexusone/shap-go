// Package explanation provides the core types for SHAP explanations.
package explanation

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"
)

// Explanation represents a SHAP explanation for a single prediction.
// It includes the base value, SHAP values for each feature, and metadata.
type Explanation struct {
	// ID is an optional unique identifier for this explanation.
	ID string `json:"id,omitempty"`

	// ModelID identifies the model that produced this explanation.
	ModelID string `json:"model_id,omitempty"`

	// Prediction is the model's output for the explained instance.
	Prediction float64 `json:"prediction"`

	// BaseValue is the expected model output (typically mean prediction on background).
	BaseValue float64 `json:"base_value"`

	// Values contains the SHAP value for each feature.
	// Keys are feature names, values are SHAP contributions.
	Values map[string]float64 `json:"shap_values"`

	// FeatureNames lists features in their original order.
	FeatureNames []string `json:"feature_names,omitempty"`

	// FeatureValues contains the actual feature values for the explained instance.
	FeatureValues map[string]float64 `json:"feature_values,omitempty"`

	// Timestamp records when this explanation was generated.
	Timestamp time.Time `json:"timestamp,omitempty"`

	// Metadata contains additional information about the explanation.
	Metadata ExplanationMetadata `json:"metadata,omitempty"`
}

// ExplanationMetadata contains additional information about how an explanation was computed.
type ExplanationMetadata struct {
	// Algorithm is the name of the algorithm used (e.g., "permutation", "sampling").
	Algorithm string `json:"algorithm,omitempty"`

	// NumSamples is the number of samples used in Monte Carlo estimation.
	NumSamples int `json:"num_samples,omitempty"`

	// BackgroundSize is the number of background samples used.
	BackgroundSize int `json:"background_size,omitempty"`

	// ComputeTimeMS is the computation time in milliseconds.
	ComputeTimeMS int64 `json:"compute_time_ms,omitempty"`

	// Version is the library version used to generate this explanation.
	Version string `json:"version,omitempty"`

	// ConfidenceIntervals contains confidence intervals for SHAP values.
	// Only populated for sampling-based methods when requested.
	ConfidenceIntervals *ConfidenceIntervals `json:"confidence_intervals,omitempty"`
}

// ConfidenceIntervals contains uncertainty bounds for SHAP value estimates.
// These are computed from the variance of Monte Carlo samples.
type ConfidenceIntervals struct {
	// Level is the confidence level (e.g., 0.95 for 95% confidence).
	Level float64 `json:"level"`

	// Lower contains the lower bounds for each feature.
	Lower map[string]float64 `json:"lower"`

	// Upper contains the upper bounds for each feature.
	Upper map[string]float64 `json:"upper"`

	// StandardErrors contains the standard error for each feature.
	StandardErrors map[string]float64 `json:"standard_errors"`
}

// GetConfidenceInterval returns the confidence interval for a specific feature.
// Returns (lower, upper, ok) where ok is false if the feature is not found.
func (e *Explanation) GetConfidenceInterval(featureName string) (lower, upper float64, ok bool) {
	if e.Metadata.ConfidenceIntervals == nil {
		return 0, 0, false
	}
	lower, ok1 := e.Metadata.ConfidenceIntervals.Lower[featureName]
	upper, ok2 := e.Metadata.ConfidenceIntervals.Upper[featureName]
	return lower, upper, ok1 && ok2
}

// HasConfidenceIntervals returns true if confidence intervals are available.
func (e *Explanation) HasConfidenceIntervals() bool {
	return e.Metadata.ConfidenceIntervals != nil
}

// VerifyResult contains the results of verifying a SHAP explanation.
type VerifyResult struct {
	// Valid indicates whether the explanation satisfies local accuracy.
	Valid bool `json:"valid"`

	// SumSHAP is the sum of all SHAP values.
	SumSHAP float64 `json:"sum_shap"`

	// Expected is the expected sum (Prediction - BaseValue).
	Expected float64 `json:"expected"`

	// Difference is the absolute difference between SumSHAP and Expected.
	Difference float64 `json:"difference"`

	// Tolerance is the tolerance used for the check.
	Tolerance float64 `json:"tolerance"`
}

// Verify checks that the SHAP values satisfy local accuracy:
// sum(SHAP values) ≈ Prediction - BaseValue
// Returns a VerifyResult with details about the check.
func (e *Explanation) Verify(tolerance float64) VerifyResult {
	var sumSHAP float64
	for _, v := range e.Values {
		sumSHAP += v
	}

	expected := e.Prediction - e.BaseValue
	diff := math.Abs(sumSHAP - expected)

	return VerifyResult{
		Valid:      diff <= tolerance,
		SumSHAP:    sumSHAP,
		Expected:   expected,
		Difference: diff,
		Tolerance:  tolerance,
	}
}

// SortedFeatures returns feature names sorted by absolute SHAP value (descending).
func (e *Explanation) SortedFeatures() []string {
	features := make([]string, 0, len(e.Values))
	for f := range e.Values {
		features = append(features, f)
	}

	sort.Slice(features, func(i, j int) bool {
		return math.Abs(e.Values[features[i]]) > math.Abs(e.Values[features[j]])
	})

	return features
}

// TopFeatures returns the top n features by absolute SHAP value.
func (e *Explanation) TopFeatures(n int) []FeatureContribution {
	sorted := e.SortedFeatures()
	if n > len(sorted) {
		n = len(sorted)
	}

	result := make([]FeatureContribution, n)
	for i := 0; i < n; i++ {
		name := sorted[i]
		result[i] = FeatureContribution{
			Name:      name,
			SHAPValue: e.Values[name],
		}
		if e.FeatureValues != nil {
			if v, ok := e.FeatureValues[name]; ok {
				result[i].Value = &v
			}
		}
	}

	return result
}

// FeatureContribution represents a single feature's contribution to the prediction.
type FeatureContribution struct {
	// Name is the feature name.
	Name string `json:"name"`

	// Value is the feature's value for this instance (optional).
	Value *float64 `json:"value,omitempty"`

	// SHAPValue is the SHAP contribution of this feature.
	SHAPValue float64 `json:"shap_value"`
}

// ToJSON serializes the explanation to JSON.
func (e *Explanation) ToJSON() ([]byte, error) {
	return json.Marshal(e)
}

// ToJSONPretty serializes the explanation to indented JSON.
func (e *Explanation) ToJSONPretty() ([]byte, error) {
	return json.MarshalIndent(e, "", "  ")
}

// FromJSON deserializes an explanation from JSON.
func FromJSON(data []byte) (*Explanation, error) {
	var e Explanation
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, fmt.Errorf("failed to unmarshal explanation: %w", err)
	}
	return &e, nil
}

// NewExplanation creates a new Explanation with the given parameters.
func NewExplanation(prediction, baseValue float64, values map[string]float64, featureNames []string) *Explanation {
	return &Explanation{
		Prediction:   prediction,
		BaseValue:    baseValue,
		Values:       values,
		FeatureNames: featureNames,
		Timestamp:    time.Now(),
	}
}
