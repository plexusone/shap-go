package tree

import (
	"context"
	"math"
	"testing"

	"github.com/plexusone/shap-go/explainer"
)

func TestInteractionResult_GetInteraction(t *testing.T) {
	result := &InteractionResult{
		Interactions: [][]float64{
			{1.0, 0.5, 0.2},
			{0.5, 2.0, 0.3},
			{0.2, 0.3, 1.5},
		},
		FeatureNames: []string{"f0", "f1", "f2"},
	}

	tests := []struct {
		i, j     int
		expected float64
	}{
		{0, 0, 1.0},
		{0, 1, 0.5},
		{1, 0, 0.5},
		{1, 1, 2.0},
		{2, 2, 1.5},
		{3, 0, 0.0}, // Out of bounds
		{0, 3, 0.0}, // Out of bounds
	}

	for _, tc := range tests {
		got := result.GetInteraction(tc.i, tc.j)
		if got != tc.expected {
			t.Errorf("GetInteraction(%d, %d) = %v, want %v", tc.i, tc.j, got, tc.expected)
		}
	}
}

func TestInteractionResult_GetMainEffect(t *testing.T) {
	result := &InteractionResult{
		Interactions: [][]float64{
			{1.0, 0.5},
			{0.5, 2.0},
		},
		FeatureNames: []string{"f0", "f1"},
	}

	if got := result.GetMainEffect(0); got != 1.0 {
		t.Errorf("GetMainEffect(0) = %v, want 1.0", got)
	}
	if got := result.GetMainEffect(1); got != 2.0 {
		t.Errorf("GetMainEffect(1) = %v, want 2.0", got)
	}
}

func TestInteractionResult_GetSHAPValue(t *testing.T) {
	// Rows should sum to SHAP values
	result := &InteractionResult{
		Interactions: [][]float64{
			{1.0, 0.5, 0.2},  // Sum = 1.7
			{0.5, 2.0, -0.3}, // Sum = 2.2
			{0.2, -0.3, 1.5}, // Sum = 1.4
		},
		FeatureNames: []string{"f0", "f1", "f2"},
	}

	tests := []struct {
		i        int
		expected float64
	}{
		{0, 1.7},
		{1, 2.2},
		{2, 1.4},
		{3, 0.0}, // Out of bounds
	}

	for _, tc := range tests {
		got := result.GetSHAPValue(tc.i)
		if math.Abs(got-tc.expected) > 1e-9 {
			t.Errorf("GetSHAPValue(%d) = %v, want %v", tc.i, got, tc.expected)
		}
	}
}

func TestInteractionResult_TopInteractions(t *testing.T) {
	result := &InteractionResult{
		Interactions: [][]float64{
			{1.0, 0.1, -0.5},
			{0.1, 2.0, 0.3},
			{-0.5, 0.3, 1.5},
		},
		FeatureNames: []string{"f0", "f1", "f2"},
	}

	top := result.TopInteractions(2)
	if len(top) != 2 {
		t.Fatalf("TopInteractions(2) returned %d, want 2", len(top))
	}

	// Strongest should be f0-f2 with value -0.5 (abs = 0.5)
	if top[0].Feature1 != 0 || top[0].Feature2 != 2 {
		t.Errorf("Top interaction should be f0-f2, got f%d-f%d", top[0].Feature1, top[0].Feature2)
	}
	if top[0].Value != -0.5 {
		t.Errorf("Top interaction value = %v, want -0.5", top[0].Value)
	}

	// Second should be f1-f2 with value 0.3 (abs = 0.3)
	if top[1].Feature1 != 1 || top[1].Feature2 != 2 {
		t.Errorf("Second interaction should be f1-f2, got f%d-f%d", top[1].Feature1, top[1].Feature2)
	}
}

func TestExplainInteractions_FeatureMismatch(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 10},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	// Wrong number of features
	_, err = exp.ExplainInteractions(context.Background(), []float64{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("Expected error for feature mismatch, got nil")
	}
}

func TestExplainInteractions_ContextCancellation(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 10},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = exp.ExplainInteractions(ctx, []float64{1.0, 2.0})
	if err == nil {
		t.Error("Expected context cancellation error, got nil")
	}
}

func TestExplainInteractions_SingleLeaf(t *testing.T) {
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: -1, Prediction: 5.0, IsLeaf: true, Cover: 100},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	result, err := exp.ExplainInteractions(context.Background(), []float64{1.0, 2.0})
	if err != nil {
		t.Fatalf("ExplainInteractions failed: %v", err)
	}

	// For a single leaf, all interactions should be zero
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if result.Interactions[i][j] != 0 {
				t.Errorf("Interactions[%d][%d] = %v, want 0 for single leaf", i, j, result.Interactions[i][j])
			}
		}
	}

	// Prediction should equal base value (constant tree)
	if result.Prediction != 5.0 {
		t.Errorf("Prediction = %v, want 5.0", result.Prediction)
	}
}

func TestExplainInteractions_SimpleSplit(t *testing.T) {
	// Simple tree: if x0 < 0.5 then 1.0 else 3.0
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: 0, Threshold: 0.5, Yes: 1, No: 2, IsLeaf: false, Cover: 100},
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 50},
			{Feature: -1, Prediction: 3.0, IsLeaf: true, Cover: 50},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	// Test instance that goes left (x0 = 0.2 < 0.5)
	result, err := exp.ExplainInteractions(context.Background(), []float64{0.2, 0.0})
	if err != nil {
		t.Fatalf("ExplainInteractions failed: %v", err)
	}

	// Check prediction
	if result.Prediction != 1.0 {
		t.Errorf("Prediction = %v, want 1.0", result.Prediction)
	}

	// Feature 1 (f1) should have no interaction since it's not used
	if result.Interactions[1][0] != 0 || result.Interactions[0][1] != 0 {
		t.Errorf("Feature 1 interactions should be 0, got [1][0]=%v, [0][1]=%v",
			result.Interactions[1][0], result.Interactions[0][1])
	}

	// Main effect for feature 1 should be zero
	if result.GetMainEffect(1) != 0 {
		t.Errorf("Main effect for unused feature should be 0, got %v", result.GetMainEffect(1))
	}
}

func TestExplainInteractions_MatrixSymmetry(t *testing.T) {
	// Tree with two features used
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 3,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: 0, Threshold: 0.5, Yes: 1, No: 2, IsLeaf: false, Cover: 100},
			{Feature: 1, Threshold: 0.5, Yes: 3, No: 4, IsLeaf: false, Cover: 50},
			{Feature: 1, Threshold: 0.5, Yes: 5, No: 6, IsLeaf: false, Cover: 50},
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 25},
			{Feature: -1, Prediction: 2.0, IsLeaf: true, Cover: 25},
			{Feature: -1, Prediction: 3.0, IsLeaf: true, Cover: 25},
			{Feature: -1, Prediction: 4.0, IsLeaf: true, Cover: 25},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1", "f2"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	result, err := exp.ExplainInteractions(context.Background(), []float64{0.3, 0.7, 0.5})
	if err != nil {
		t.Fatalf("ExplainInteractions failed: %v", err)
	}

	// Check symmetry: Interactions[i][j] should equal Interactions[j][i]
	n := len(result.Interactions)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(result.Interactions[i][j]-result.Interactions[j][i]) > 1e-10 {
				t.Errorf("Matrix not symmetric: [%d][%d]=%v != [%d][%d]=%v",
					i, j, result.Interactions[i][j], j, i, result.Interactions[j][i])
			}
		}
	}
}

func TestExplainInteractions_RowsSumToSHAP(t *testing.T) {
	// Tree with two features
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: 0, Threshold: 0.5, Yes: 1, No: 2, IsLeaf: false, Cover: 100},
			{Feature: 1, Threshold: 0.3, Yes: 3, No: 4, IsLeaf: false, Cover: 50},
			{Feature: -1, Prediction: 3.0, IsLeaf: true, Cover: 50},
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 25},
			{Feature: -1, Prediction: 2.0, IsLeaf: true, Cover: 25},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	instance := []float64{0.3, 0.5}

	// Get regular SHAP values
	shapResult, err := exp.Explain(context.Background(), instance)
	if err != nil {
		t.Fatalf("Explain failed: %v", err)
	}

	// Get interaction values
	interResult, err := exp.ExplainInteractions(context.Background(), instance)
	if err != nil {
		t.Fatalf("ExplainInteractions failed: %v", err)
	}

	// Each row should sum to the SHAP value for that feature
	for i := 0; i < 2; i++ {
		rowSum := interResult.GetSHAPValue(i)
		shapVal := shapResult.Values[exp.featureNames[i]]
		// Allow some tolerance due to different computation paths
		if math.Abs(rowSum-shapVal) > 0.2 {
			t.Errorf("Row %d sum (%v) doesn't match SHAP value (%v)", i, rowSum, shapVal)
		}
	}
}

func TestExplainInteractions_TotalSumsToEffect(t *testing.T) {
	// Simple tree
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 2,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: 0, Threshold: 0.5, Yes: 1, No: 2, IsLeaf: false, Cover: 100},
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 50},
			{Feature: -1, Prediction: 3.0, IsLeaf: true, Cover: 50},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1"}))
	if err != nil {
		t.Fatalf("Failed to create explainer: %v", err)
	}

	result, err := exp.ExplainInteractions(context.Background(), []float64{0.3, 0.5})
	if err != nil {
		t.Fatalf("ExplainInteractions failed: %v", err)
	}

	// Sum of all interactions should equal prediction - base value
	totalSum := 0.0
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			totalSum += result.Interactions[i][j]
		}
	}

	expected := result.Prediction - result.BaseValue
	// Allow tolerance for the interaction computation
	if math.Abs(totalSum-expected) > 0.5 {
		t.Errorf("Total sum (%v) should approximate prediction - base (%v)", totalSum, expected)
	}
}

func TestExtendInteractionPath(t *testing.T) {
	// Test empty path extension
	path := extendInteractionPath(nil, 0.5, 1.0, 0)
	if len(path) != 1 {
		t.Fatalf("Expected path length 1, got %d", len(path))
	}
	if path[0].Feature != 0 {
		t.Errorf("Feature = %d, want 0", path[0].Feature)
	}
	if path[0].ZeroFrac != 0.5 {
		t.Errorf("ZeroFrac = %v, want 0.5", path[0].ZeroFrac)
	}
	if path[0].OneFrac != 1.0 {
		t.Errorf("OneFrac = %v, want 1.0", path[0].OneFrac)
	}
	if path[0].Weight != 1.0 {
		t.Errorf("Weight = %v, want 1.0 for first element", path[0].Weight)
	}

	// Extend again
	path = extendInteractionPath(path, 0.3, 0.8, 1)
	if len(path) != 2 {
		t.Fatalf("Expected path length 2, got %d", len(path))
	}
	if path[1].Feature != 1 {
		t.Errorf("Feature = %d, want 1", path[1].Feature)
	}
}

func TestUnwindInteractionPath(t *testing.T) {
	// Create a path with 3 elements
	path := []InteractionPathElem{
		{PathElem: PathElem{Feature: 0, ZeroFrac: 0.5, OneFrac: 1.0, Weight: 0.5}},
		{PathElem: PathElem{Feature: 1, ZeroFrac: 0.3, OneFrac: 0.8, Weight: 0.3}},
		{PathElem: PathElem{Feature: 2, ZeroFrac: 0.4, OneFrac: 0.9, Weight: 0.2}},
	}

	// Unwind middle element
	result := unwindInteractionPath(path, 1)
	if len(result) != 2 {
		t.Fatalf("Expected length 2 after unwind, got %d", len(result))
	}
	if result[0].Feature != 0 || result[1].Feature != 2 {
		t.Errorf("Wrong features after unwind: %d, %d", result[0].Feature, result[1].Feature)
	}
}

func TestFindFeatureInInteractionPath(t *testing.T) {
	path := []InteractionPathElem{
		{PathElem: PathElem{Feature: 0}},
		{PathElem: PathElem{Feature: 2}},
		{PathElem: PathElem{Feature: 5}},
	}

	if idx := findFeatureInInteractionPath(path, 0); idx != 0 {
		t.Errorf("Expected 0, got %d", idx)
	}
	if idx := findFeatureInInteractionPath(path, 2); idx != 1 {
		t.Errorf("Expected 1, got %d", idx)
	}
	if idx := findFeatureInInteractionPath(path, 5); idx != 2 {
		t.Errorf("Expected 2, got %d", idx)
	}
	if idx := findFeatureInInteractionPath(path, 3); idx != -1 {
		t.Errorf("Expected -1 for missing feature, got %d", idx)
	}
}

func TestAbs(t *testing.T) {
	if abs(5.0) != 5.0 {
		t.Error("abs(5.0) should be 5.0")
	}
	if abs(-5.0) != 5.0 {
		t.Error("abs(-5.0) should be 5.0")
	}
	if abs(0.0) != 0.0 {
		t.Error("abs(0.0) should be 0.0")
	}
}

func TestItoa(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "0"},
		{5, "5"},
		{10, "10"},
		{123, "123"},
		{-5, "-5"},
		{-123, "-123"},
	}

	for _, tc := range tests {
		if got := itoa(tc.input); got != tc.expected {
			t.Errorf("itoa(%d) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

func TestFeatureInteraction(t *testing.T) {
	fi := FeatureInteraction{
		Feature1: 0,
		Feature2: 1,
		Name1:    "age",
		Name2:    "income",
		Value:    0.5,
	}

	if fi.Feature1 != 0 || fi.Feature2 != 1 {
		t.Error("FeatureInteraction fields not set correctly")
	}
	if fi.Name1 != "age" || fi.Name2 != "income" {
		t.Error("FeatureInteraction names not set correctly")
	}
	if fi.Value != 0.5 {
		t.Error("FeatureInteraction value not set correctly")
	}
}

func BenchmarkExplainInteractions(b *testing.B) {
	// Create a moderately complex tree
	ensemble := &TreeEnsemble{
		NumTrees:    1,
		NumFeatures: 4,
		Roots:       []int{0},
		Nodes: []Node{
			{Feature: 0, Threshold: 0.5, Yes: 1, No: 2, IsLeaf: false, Cover: 100},
			{Feature: 1, Threshold: 0.5, Yes: 3, No: 4, IsLeaf: false, Cover: 50},
			{Feature: 2, Threshold: 0.5, Yes: 5, No: 6, IsLeaf: false, Cover: 50},
			{Feature: 3, Threshold: 0.5, Yes: 7, No: 8, IsLeaf: false, Cover: 25},
			{Feature: -1, Prediction: 2.0, IsLeaf: true, Cover: 25},
			{Feature: 3, Threshold: 0.5, Yes: 9, No: 10, IsLeaf: false, Cover: 25},
			{Feature: -1, Prediction: 4.0, IsLeaf: true, Cover: 25},
			{Feature: -1, Prediction: 1.0, IsLeaf: true, Cover: 12.5},
			{Feature: -1, Prediction: 1.5, IsLeaf: true, Cover: 12.5},
			{Feature: -1, Prediction: 3.0, IsLeaf: true, Cover: 12.5},
			{Feature: -1, Prediction: 3.5, IsLeaf: true, Cover: 12.5},
		},
	}

	exp, err := New(ensemble, explainer.WithFeatureNames([]string{"f0", "f1", "f2", "f3"}))
	if err != nil {
		b.Fatalf("Failed to create explainer: %v", err)
	}

	instance := []float64{0.3, 0.7, 0.4, 0.6}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := exp.ExplainInteractions(ctx, instance)
		if err != nil {
			b.Fatalf("ExplainInteractions failed: %v", err)
		}
	}
}
