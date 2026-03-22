package explainer

import (
	"math/rand"
	"time"
)

// Config contains configuration options for SHAP explainers.
type Config struct {
	// NumSamples is the number of Monte Carlo samples to use.
	// Higher values give more accurate SHAP estimates but take longer.
	// Default: 100
	NumSamples int

	// Seed is the random seed for reproducibility.
	// If nil, uses current time.
	Seed *int64

	// NumWorkers is the number of parallel workers for computation.
	// If 0, uses sequential computation.
	// Default: 0 (sequential)
	NumWorkers int

	// ModelID is an optional identifier for the model being explained.
	ModelID string

	// FeatureNames are the names of the input features.
	// If not provided, features are named "feature_0", "feature_1", etc.
	FeatureNames []string

	// ConfidenceLevel specifies the confidence level for computing
	// confidence intervals on SHAP values (e.g., 0.95 for 95% CI).
	// If 0, confidence intervals are not computed.
	// Only applies to sampling-based methods (sampling, permutation).
	// Default: 0 (disabled)
	ConfidenceLevel float64

	// UseBatchedPredictions enables batched model predictions for efficiency.
	// When true, explainers will use PredictBatch instead of individual Predict
	// calls where possible. This can significantly improve performance when
	// the model has optimized batch inference (e.g., neural networks).
	// Default: false
	UseBatchedPredictions bool
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		NumSamples: 100,
		NumWorkers: 0,
	}
}

// Validate validates the configuration and sets defaults.
func (c *Config) Validate(numFeatures int) {
	if c.NumSamples <= 0 {
		c.NumSamples = 100
	}
	if c.NumWorkers < 0 {
		c.NumWorkers = 0
	}
	if len(c.FeatureNames) == 0 {
		c.FeatureNames = generateFeatureNames(numFeatures)
	}
}

// GetRNG returns a random number generator based on the config seed.
func (c *Config) GetRNG() *rand.Rand {
	var seed int64
	if c.Seed != nil {
		seed = *c.Seed
	} else {
		seed = time.Now().UnixNano()
	}
	return rand.New(rand.NewSource(seed)) //nolint:gosec // seeded for reproducibility
}

// generateFeatureNames generates default feature names.
func generateFeatureNames(n int) []string {
	names := make([]string, n)
	for i := range names {
		names[i] = featureName(i)
	}
	return names
}

func featureName(i int) string {
	return "feature_" + itoa(i)
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

// Option is a function that modifies a Config.
type Option func(*Config)

// WithNumSamples sets the number of Monte Carlo samples.
func WithNumSamples(n int) Option {
	return func(c *Config) {
		c.NumSamples = n
	}
}

// WithSeed sets the random seed for reproducibility.
func WithSeed(seed int64) Option {
	return func(c *Config) {
		c.Seed = &seed
	}
}

// WithNumWorkers sets the number of parallel workers.
func WithNumWorkers(n int) Option {
	return func(c *Config) {
		c.NumWorkers = n
	}
}

// WithModelID sets the model identifier.
func WithModelID(id string) Option {
	return func(c *Config) {
		c.ModelID = id
	}
}

// WithFeatureNames sets the feature names.
func WithFeatureNames(names []string) Option {
	return func(c *Config) {
		c.FeatureNames = names
	}
}

// WithConfidenceLevel sets the confidence level for computing confidence intervals.
// Common values are 0.90 (90%), 0.95 (95%), and 0.99 (99%).
// Set to 0 to disable confidence interval computation (default).
func WithConfidenceLevel(level float64) Option {
	return func(c *Config) {
		c.ConfidenceLevel = level
	}
}

// WithBatchedPredictions enables batched model predictions for efficiency.
// When enabled, explainers use PredictBatch instead of individual Predict calls
// where possible. This can significantly improve performance for models with
// optimized batch inference (e.g., neural networks, ONNX models).
func WithBatchedPredictions(enabled bool) Option {
	return func(c *Config) {
		c.UseBatchedPredictions = enabled
	}
}

// ApplyOptions applies the given options to a config.
func ApplyOptions(config *Config, opts ...Option) {
	for _, opt := range opts {
		opt(config)
	}
}

// ZScoreForConfidenceLevel returns the z-score for a given confidence level.
// Uses common values for standard confidence levels and approximates others.
func ZScoreForConfidenceLevel(level float64) float64 {
	// Common z-scores for standard confidence levels
	switch {
	case level >= 0.999:
		return 3.291 // 99.9%
	case level >= 0.99:
		return 2.576 // 99%
	case level >= 0.975:
		return 2.241 // 97.5%
	case level >= 0.95:
		return 1.960 // 95%
	case level >= 0.90:
		return 1.645 // 90%
	case level >= 0.85:
		return 1.440 // 85%
	case level >= 0.80:
		return 1.282 // 80%
	default:
		// For other values, use a simple approximation
		// This is not perfectly accurate but handles edge cases
		return 1.960 // Default to 95%
	}
}
