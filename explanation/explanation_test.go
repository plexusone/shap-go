package explanation

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

func TestExplanation_Verify(t *testing.T) {
	tests := []struct {
		name       string
		prediction float64
		baseValue  float64
		values     map[string]float64
		tolerance  float64
		wantValid  bool
	}{
		{
			name:       "valid local accuracy",
			prediction: 0.8,
			baseValue:  0.5,
			values: map[string]float64{
				"feature_0": 0.2,
				"feature_1": 0.1,
			},
			tolerance: 1e-10,
			wantValid: true,
		},
		{
			name:       "invalid local accuracy",
			prediction: 0.8,
			baseValue:  0.5,
			values: map[string]float64{
				"feature_0": 0.1,
				"feature_1": 0.1,
			},
			tolerance: 0.01,
			wantValid: false,
		},
		{
			name:       "within tolerance",
			prediction: 0.8,
			baseValue:  0.5,
			values: map[string]float64{
				"feature_0": 0.15,
				"feature_1": 0.14,
			},
			tolerance: 0.02,
			wantValid: true,
		},
		{
			name:       "negative values",
			prediction: 0.3,
			baseValue:  0.5,
			values: map[string]float64{
				"feature_0": -0.1,
				"feature_1": -0.1,
			},
			tolerance: 1e-10,
			wantValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Explanation{
				Prediction: tt.prediction,
				BaseValue:  tt.baseValue,
				Values:     tt.values,
			}

			result := e.Verify(tt.tolerance)
			if result.Valid != tt.wantValid {
				t.Errorf("Verify() valid = %v, want %v", result.Valid, tt.wantValid)
				t.Errorf("SumSHAP = %v, Expected = %v, Difference = %v",
					result.SumSHAP, result.Expected, result.Difference)
			}
		})
	}
}

func TestExplanation_SortedFeatures(t *testing.T) {
	e := &Explanation{
		Values: map[string]float64{
			"a": 0.1,
			"b": -0.5,
			"c": 0.3,
			"d": -0.05,
		},
	}

	sorted := e.SortedFeatures()

	// Should be sorted by absolute value descending
	expected := []string{"b", "c", "a", "d"}
	if len(sorted) != len(expected) {
		t.Fatalf("SortedFeatures() length = %d, want %d", len(sorted), len(expected))
	}

	for i, name := range expected {
		if sorted[i] != name {
			t.Errorf("SortedFeatures()[%d] = %s, want %s", i, sorted[i], name)
		}
	}
}

func TestExplanation_TopFeatures(t *testing.T) {
	e := &Explanation{
		Values: map[string]float64{
			"a": 0.1,
			"b": -0.5,
			"c": 0.3,
		},
		FeatureValues: map[string]float64{
			"a": 1.0,
			"b": 2.0,
			"c": 3.0,
		},
	}

	top := e.TopFeatures(2)

	if len(top) != 2 {
		t.Fatalf("TopFeatures(2) returned %d features, want 2", len(top))
	}

	if top[0].Name != "b" || top[0].SHAPValue != -0.5 {
		t.Errorf("TopFeatures(2)[0] = %+v, want {Name:b, SHAPValue:-0.5}", top[0])
	}

	if top[1].Name != "c" || top[1].SHAPValue != 0.3 {
		t.Errorf("TopFeatures(2)[1] = %+v, want {Name:c, SHAPValue:0.3}", top[1])
	}

	// Check feature values are included
	if top[0].Value == nil || *top[0].Value != 2.0 {
		t.Errorf("TopFeatures(2)[0].Value = %v, want 2.0", top[0].Value)
	}
}

func TestExplanation_JSON(t *testing.T) {
	e := &Explanation{
		ID:         "test-id",
		ModelID:    "model-1",
		Prediction: 0.8,
		BaseValue:  0.5,
		Values: map[string]float64{
			"feature_0": 0.2,
			"feature_1": 0.1,
		},
		FeatureNames: []string{"feature_0", "feature_1"},
		Timestamp:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		Metadata: ExplanationMetadata{
			Algorithm:  "permutation",
			NumSamples: 100,
		},
	}

	// Test ToJSON
	data, err := e.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON() error = %v", err)
	}

	// Test FromJSON
	e2, err := FromJSON(data)
	if err != nil {
		t.Fatalf("FromJSON() error = %v", err)
	}

	if e2.ID != e.ID {
		t.Errorf("FromJSON().ID = %s, want %s", e2.ID, e.ID)
	}
	if e2.Prediction != e.Prediction {
		t.Errorf("FromJSON().Prediction = %f, want %f", e2.Prediction, e.Prediction)
	}
	if e2.BaseValue != e.BaseValue {
		t.Errorf("FromJSON().BaseValue = %f, want %f", e2.BaseValue, e.BaseValue)
	}
	if len(e2.Values) != len(e.Values) {
		t.Errorf("FromJSON().Values length = %d, want %d", len(e2.Values), len(e.Values))
	}
}

func TestExplanation_JSONPretty(t *testing.T) {
	e := &Explanation{
		Prediction: 0.8,
		BaseValue:  0.5,
		Values: map[string]float64{
			"feature_0": 0.3,
		},
	}

	data, err := e.ToJSONPretty()
	if err != nil {
		t.Fatalf("ToJSONPretty() error = %v", err)
	}

	// Verify it's valid JSON
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("ToJSONPretty() produced invalid JSON: %v", err)
	}

	// Verify it contains newlines (is indented)
	if len(data) <= 50 {
		t.Error("ToJSONPretty() should produce indented output")
	}
}

func TestNewExplanation(t *testing.T) {
	values := map[string]float64{
		"a": 0.1,
		"b": 0.2,
	}
	names := []string{"a", "b"}

	e := NewExplanation(0.8, 0.5, values, names)

	if e.Prediction != 0.8 {
		t.Errorf("NewExplanation().Prediction = %f, want 0.8", e.Prediction)
	}
	if e.BaseValue != 0.5 {
		t.Errorf("NewExplanation().BaseValue = %f, want 0.5", e.BaseValue)
	}
	if len(e.Values) != 2 {
		t.Errorf("NewExplanation().Values length = %d, want 2", len(e.Values))
	}
	if e.Timestamp.IsZero() {
		t.Error("NewExplanation().Timestamp should be set")
	}
}

func TestVerifyResult(t *testing.T) {
	e := &Explanation{
		Prediction: 1.0,
		BaseValue:  0.5,
		Values: map[string]float64{
			"a": 0.3,
			"b": 0.2,
		},
	}

	result := e.Verify(1e-10)

	if !result.Valid {
		t.Error("Verify() should be valid")
	}
	if math.Abs(result.SumSHAP-0.5) > 1e-10 {
		t.Errorf("Verify().SumSHAP = %f, want 0.5", result.SumSHAP)
	}
	if math.Abs(result.Expected-0.5) > 1e-10 {
		t.Errorf("Verify().Expected = %f, want 0.5", result.Expected)
	}
	if math.Abs(result.Difference) > 1e-10 {
		t.Errorf("Verify().Difference = %f, want 0", result.Difference)
	}
	if result.Tolerance != 1e-10 {
		t.Errorf("Verify().Tolerance = %f, want 1e-10", result.Tolerance)
	}
}
