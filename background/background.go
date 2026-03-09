// Package background provides utilities for managing background datasets used in SHAP computation.
package background

import (
	"fmt"
	"math/rand"
	"sort"
)

// Dataset represents a background dataset for SHAP computation.
type Dataset struct {
	// Data contains the feature vectors (rows x features).
	Data [][]float64

	// FeatureNames contains the names of each feature.
	FeatureNames []string

	// Mean contains the mean value of each feature.
	Mean []float64
}

// NewDataset creates a new Dataset from the given data and feature names.
func NewDataset(data [][]float64, featureNames []string) (*Dataset, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	numFeatures := len(data[0])
	if len(featureNames) != numFeatures {
		return nil, fmt.Errorf("feature names length (%d) does not match data width (%d)",
			len(featureNames), numFeatures)
	}

	for i, row := range data {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("row %d has %d features, expected %d", i, len(row), numFeatures)
		}
	}

	// Compute mean
	mean := computeMean(data)

	return &Dataset{
		Data:         data,
		FeatureNames: featureNames,
		Mean:         mean,
	}, nil
}

// NewDatasetFromSlice creates a new Dataset from a slice of float64 slices.
// Feature names are automatically generated as "feature_0", "feature_1", etc.
func NewDatasetFromSlice(data [][]float64) (*Dataset, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	numFeatures := len(data[0])
	featureNames := make([]string, numFeatures)
	for i := range featureNames {
		featureNames[i] = fmt.Sprintf("feature_%d", i)
	}

	return NewDataset(data, featureNames)
}

// NumSamples returns the number of samples in the dataset.
func (d *Dataset) NumSamples() int {
	return len(d.Data)
}

// NumFeatures returns the number of features in the dataset.
func (d *Dataset) NumFeatures() int {
	if len(d.Data) == 0 {
		return 0
	}
	return len(d.Data[0])
}

// Sample returns a random sample from the dataset.
func (d *Dataset) Sample(rng *rand.Rand) []float64 {
	idx := rng.Intn(len(d.Data))
	result := make([]float64, len(d.Data[idx]))
	copy(result, d.Data[idx])
	return result
}

// SampleN returns n random samples from the dataset (with replacement).
func (d *Dataset) SampleN(rng *rand.Rand, n int) [][]float64 {
	result := make([][]float64, n)
	for i := 0; i < n; i++ {
		result[i] = d.Sample(rng)
	}
	return result
}

// Subset returns a new Dataset containing only the first n samples.
// If n > len(data), all samples are returned.
func (d *Dataset) Subset(n int) *Dataset {
	if n >= len(d.Data) {
		return d
	}

	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		data[i] = make([]float64, len(d.Data[i]))
		copy(data[i], d.Data[i])
	}

	featureNames := make([]string, len(d.FeatureNames))
	copy(featureNames, d.FeatureNames)

	return &Dataset{
		Data:         data,
		FeatureNames: featureNames,
		Mean:         computeMean(data),
	}
}

// RandomSubset returns a new Dataset containing n randomly sampled rows (without replacement).
func (d *Dataset) RandomSubset(rng *rand.Rand, n int) *Dataset {
	if n >= len(d.Data) {
		return d
	}

	// Create shuffled indices
	indices := make([]int, len(d.Data))
	for i := range indices {
		indices[i] = i
	}
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Take first n
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		idx := indices[i]
		data[i] = make([]float64, len(d.Data[idx]))
		copy(data[i], d.Data[idx])
	}

	featureNames := make([]string, len(d.FeatureNames))
	copy(featureNames, d.FeatureNames)

	return &Dataset{
		Data:         data,
		FeatureNames: featureNames,
		Mean:         computeMean(data),
	}
}

// KMeansSummary returns a summarized dataset using k-means clustering.
// This is useful for reducing large datasets while preserving distribution.
// The returned dataset contains the cluster centroids.
func (d *Dataset) KMeansSummary(k int, maxIterations int, rng *rand.Rand) *Dataset {
	if k >= len(d.Data) {
		return d
	}

	centroids := kMeans(d.Data, k, maxIterations, rng)

	featureNames := make([]string, len(d.FeatureNames))
	copy(featureNames, d.FeatureNames)

	return &Dataset{
		Data:         centroids,
		FeatureNames: featureNames,
		Mean:         computeMean(centroids),
	}
}

// FeatureIndex returns the index of the feature with the given name, or -1 if not found.
func (d *Dataset) FeatureIndex(name string) int {
	for i, fn := range d.FeatureNames {
		if fn == name {
			return i
		}
	}
	return -1
}

// FeatureValues returns all values for the feature at the given index.
func (d *Dataset) FeatureValues(featureIndex int) []float64 {
	if featureIndex < 0 || featureIndex >= d.NumFeatures() {
		return nil
	}

	values := make([]float64, len(d.Data))
	for i, row := range d.Data {
		values[i] = row[featureIndex]
	}
	return values
}

// computeMean computes the mean of each feature.
func computeMean(data [][]float64) []float64 {
	if len(data) == 0 {
		return nil
	}

	numFeatures := len(data[0])
	mean := make([]float64, numFeatures)

	for _, row := range data {
		for i, v := range row {
			mean[i] += v
		}
	}

	n := float64(len(data))
	for i := range mean {
		mean[i] /= n
	}

	return mean
}

// kMeans performs k-means clustering and returns the centroids.
func kMeans(data [][]float64, k int, maxIterations int, rng *rand.Rand) [][]float64 {
	if len(data) == 0 || k <= 0 {
		return nil
	}

	numFeatures := len(data[0])

	// Initialize centroids randomly (k-means++)
	centroids := make([][]float64, k)
	centroids[0] = make([]float64, numFeatures)
	copy(centroids[0], data[rng.Intn(len(data))])

	// k-means++ initialization
	for i := 1; i < k; i++ {
		// Compute distances to nearest centroid
		distances := make([]float64, len(data))
		totalDist := 0.0
		for j, point := range data {
			minDist := euclideanDistanceSquared(point, centroids[0])
			for c := 1; c < i; c++ {
				dist := euclideanDistanceSquared(point, centroids[c])
				if dist < minDist {
					minDist = dist
				}
			}
			distances[j] = minDist
			totalDist += minDist
		}

		// Sample proportional to distance squared
		r := rng.Float64() * totalDist
		cumulative := 0.0
		for j, d := range distances {
			cumulative += d
			if cumulative >= r {
				centroids[i] = make([]float64, numFeatures)
				copy(centroids[i], data[j])
				break
			}
		}
		if centroids[i] == nil {
			centroids[i] = make([]float64, numFeatures)
			copy(centroids[i], data[len(data)-1])
		}
	}

	// Iterate
	assignments := make([]int, len(data))
	for iter := 0; iter < maxIterations; iter++ {
		changed := false

		// Assign points to nearest centroid
		for i, point := range data {
			nearest := 0
			nearestDist := euclideanDistanceSquared(point, centroids[0])
			for c := 1; c < k; c++ {
				dist := euclideanDistanceSquared(point, centroids[c])
				if dist < nearestDist {
					nearest = c
					nearestDist = dist
				}
			}
			if assignments[i] != nearest {
				assignments[i] = nearest
				changed = true
			}
		}

		if !changed {
			break
		}

		// Update centroids
		counts := make([]int, k)
		for c := range centroids {
			for f := range centroids[c] {
				centroids[c][f] = 0
			}
		}

		for i, point := range data {
			c := assignments[i]
			counts[c]++
			for f, v := range point {
				centroids[c][f] += v
			}
		}

		for c := range centroids {
			if counts[c] > 0 {
				for f := range centroids[c] {
					centroids[c][f] /= float64(counts[c])
				}
			}
		}
	}

	return centroids
}

// euclideanDistanceSquared computes the squared Euclidean distance between two points.
func euclideanDistanceSquared(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// FeatureStats contains statistics for a single feature.
type FeatureStats struct {
	Name   string  `json:"name"`
	Mean   float64 `json:"mean"`
	Std    float64 `json:"std"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Median float64 `json:"median"`
}

// Stats returns statistics for all features in the dataset.
func (d *Dataset) Stats() []FeatureStats {
	stats := make([]FeatureStats, d.NumFeatures())

	for i := 0; i < d.NumFeatures(); i++ {
		values := d.FeatureValues(i)
		stats[i] = computeFeatureStats(d.FeatureNames[i], values)
	}

	return stats
}

// computeFeatureStats computes statistics for a single feature.
func computeFeatureStats(name string, values []float64) FeatureStats {
	if len(values) == 0 {
		return FeatureStats{Name: name}
	}

	// Sort for median
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	// Compute statistics
	var sum, sumSq float64
	min, max := sorted[0], sorted[len(sorted)-1]

	for _, v := range values {
		sum += v
		sumSq += v * v
	}

	n := float64(len(values))
	mean := sum / n
	variance := (sumSq / n) - (mean * mean)
	if variance < 0 {
		variance = 0
	}
	std := 0.0
	if variance > 0 {
		std = variance
		for i := 0; i < 10; i++ { // Newton's method for sqrt
			std = (std + variance/std) / 2
		}
	}

	// Median
	var median float64
	if len(sorted)%2 == 0 {
		median = (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	} else {
		median = sorted[len(sorted)/2]
	}

	return FeatureStats{
		Name:   name,
		Mean:   mean,
		Std:    std,
		Min:    min,
		Max:    max,
		Median: median,
	}
}
