// Package masker provides feature masking strategies for SHAP computation.
package masker

// Masker defines the interface for feature masking strategies.
// A masker determines how to replace feature values when computing SHAP values.
type Masker interface {
	// Mask creates a masked version of the instance.
	// The mask parameter indicates which features to mask (true = use background/masked value).
	// Returns the masked instance.
	Mask(instance []float64, mask []bool) []float64

	// NumFeatures returns the number of features this masker handles.
	NumFeatures() int
}
