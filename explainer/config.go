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
	return rand.New(rand.NewSource(seed))
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

// ApplyOptions applies the given options to a config.
func ApplyOptions(config *Config, opts ...Option) {
	for _, opt := range opts {
		opt(config)
	}
}
