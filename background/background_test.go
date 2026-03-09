package background

import (
	"math"
	"math/rand"
	"testing"
)

func TestNewDataset(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	names := []string{"a", "b", "c"}

	ds, err := NewDataset(data, names)
	if err != nil {
		t.Fatalf("NewDataset() error = %v", err)
	}

	if ds.NumSamples() != 3 {
		t.Errorf("NumSamples() = %d, want 3", ds.NumSamples())
	}

	if ds.NumFeatures() != 3 {
		t.Errorf("NumFeatures() = %d, want 3", ds.NumFeatures())
	}

	// Check mean
	expectedMean := []float64{4.0, 5.0, 6.0}
	for i, m := range ds.Mean {
		if m != expectedMean[i] {
			t.Errorf("Mean[%d] = %f, want %f", i, m, expectedMean[i])
		}
	}
}

func TestNewDataset_Empty(t *testing.T) {
	_, err := NewDataset([][]float64{}, []string{})
	if err == nil {
		t.Error("NewDataset() should error with empty data")
	}
}

func TestNewDataset_MismatchedNames(t *testing.T) {
	data := [][]float64{{1.0, 2.0}}
	names := []string{"a", "b", "c"} // Wrong length

	_, err := NewDataset(data, names)
	if err == nil {
		t.Error("NewDataset() should error with mismatched names")
	}
}

func TestNewDatasetFromSlice(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	ds, err := NewDatasetFromSlice(data)
	if err != nil {
		t.Fatalf("NewDatasetFromSlice() error = %v", err)
	}

	if ds.FeatureNames[0] != "feature_0" {
		t.Errorf("FeatureNames[0] = %s, want feature_0", ds.FeatureNames[0])
	}
}

func TestDataset_Sample(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	rng := rand.New(rand.NewSource(42))
	sample := ds.Sample(rng)

	if len(sample) != 2 {
		t.Errorf("Sample() returned %d features, want 2", len(sample))
	}

	// Check it's one of the original samples
	isValid := (sample[0] == 1.0 && sample[1] == 2.0) ||
		(sample[0] == 3.0 && sample[1] == 4.0)
	if !isValid {
		t.Errorf("Sample() = %v, expected one of the original samples", sample)
	}
}

func TestDataset_SampleN(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	rng := rand.New(rand.NewSource(42))
	samples := ds.SampleN(rng, 5)

	if len(samples) != 5 {
		t.Errorf("SampleN(5) returned %d samples, want 5", len(samples))
	}
}

func TestDataset_Subset(t *testing.T) {
	data := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
		{5.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	subset := ds.Subset(3)

	if subset.NumSamples() != 3 {
		t.Errorf("Subset(3).NumSamples() = %d, want 3", subset.NumSamples())
	}

	// Check values
	for i := 0; i < 3; i++ {
		if subset.Data[i][0] != float64(i+1) {
			t.Errorf("Subset(3).Data[%d][0] = %f, want %f", i, subset.Data[i][0], float64(i+1))
		}
	}
}

func TestDataset_RandomSubset(t *testing.T) {
	data := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
		{5.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	rng := rand.New(rand.NewSource(42))
	subset := ds.RandomSubset(rng, 3)

	if subset.NumSamples() != 3 {
		t.Errorf("RandomSubset(3).NumSamples() = %d, want 3", subset.NumSamples())
	}
}

func TestDataset_FeatureIndex(t *testing.T) {
	data := [][]float64{{1.0, 2.0, 3.0}}
	names := []string{"a", "b", "c"}
	ds, _ := NewDataset(data, names)

	tests := []struct {
		name     string
		expected int
	}{
		{"a", 0},
		{"b", 1},
		{"c", 2},
		{"d", -1}, // Not found
	}

	for _, tt := range tests {
		idx := ds.FeatureIndex(tt.name)
		if idx != tt.expected {
			t.Errorf("FeatureIndex(%s) = %d, want %d", tt.name, idx, tt.expected)
		}
	}
}

func TestDataset_FeatureValues(t *testing.T) {
	data := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	values := ds.FeatureValues(0)
	expected := []float64{1.0, 2.0, 3.0}

	if len(values) != len(expected) {
		t.Fatalf("FeatureValues(0) length = %d, want %d", len(values), len(expected))
	}

	for i, v := range values {
		if v != expected[i] {
			t.Errorf("FeatureValues(0)[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestDataset_Stats(t *testing.T) {
	data := [][]float64{
		{1.0},
		{2.0},
		{3.0},
		{4.0},
		{5.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	stats := ds.Stats()

	if len(stats) != 1 {
		t.Fatalf("Stats() length = %d, want 1", len(stats))
	}

	s := stats[0]
	if s.Mean != 3.0 {
		t.Errorf("Stats()[0].Mean = %f, want 3.0", s.Mean)
	}
	if s.Min != 1.0 {
		t.Errorf("Stats()[0].Min = %f, want 1.0", s.Min)
	}
	if s.Max != 5.0 {
		t.Errorf("Stats()[0].Max = %f, want 5.0", s.Max)
	}
	if s.Median != 3.0 {
		t.Errorf("Stats()[0].Median = %f, want 3.0", s.Median)
	}

	// Std should be sqrt(2) ≈ 1.414
	expectedStd := math.Sqrt(2.0)
	if math.Abs(s.Std-expectedStd) > 0.01 {
		t.Errorf("Stats()[0].Std = %f, want ~%f", s.Std, expectedStd)
	}
}

func TestDataset_KMeansSummary(t *testing.T) {
	// Create a dataset with two clear clusters
	data := [][]float64{
		{0.0, 0.0},
		{0.1, 0.1},
		{0.2, 0.0},
		{10.0, 10.0},
		{10.1, 10.1},
		{10.2, 10.0},
	}
	ds, _ := NewDatasetFromSlice(data)

	rng := rand.New(rand.NewSource(42))
	summary := ds.KMeansSummary(2, 100, rng)

	if summary.NumSamples() != 2 {
		t.Errorf("KMeansSummary(2).NumSamples() = %d, want 2", summary.NumSamples())
	}

	// Check that centroids are roughly at (0.1, 0.03) and (10.1, 10.03)
	centroids := summary.Data
	var lowCluster, highCluster []float64

	for _, c := range centroids {
		if c[0] < 5 {
			lowCluster = c
		} else {
			highCluster = c
		}
	}

	if lowCluster == nil || highCluster == nil {
		t.Fatal("KMeansSummary() should produce two distinct clusters")
	}

	if lowCluster[0] > 1.0 || lowCluster[1] > 1.0 {
		t.Errorf("Low cluster centroid = %v, expected near (0.1, 0.03)", lowCluster)
	}

	if highCluster[0] < 9.0 || highCluster[1] < 9.0 {
		t.Errorf("High cluster centroid = %v, expected near (10.1, 10.03)", highCluster)
	}
}
