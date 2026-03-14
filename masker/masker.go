// Package masker provides feature masking strategies for SHAP computation.
package masker

import "errors"

// Common errors returned by masker operations.
var (
	ErrInstanceFeatureMismatch   = errors.New("instance feature count mismatch")
	ErrMaskFeatureMismatch       = errors.New("mask length mismatch")
	ErrBackgroundFeatureMismatch = errors.New("background sample feature count mismatch")
)

// Masker defines the interface for feature masking strategies.
// A masker determines how to replace feature values when computing SHAP values.
type Masker interface {
	// Mask creates a masked version of the instance.
	// The mask parameter indicates which features to mask (true = use background/masked value).
	// Returns the masked instance or an error if dimensions don't match.
	Mask(instance []float64, mask []bool) ([]float64, error)

	// NumFeatures returns the number of features this masker handles.
	NumFeatures() int
}
