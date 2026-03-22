package gradient

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Common errors returned by GradientSHAP.
var (
	ErrNilModel        = errors.New("model cannot be nil")
	ErrNoBackground    = errors.New("background data cannot be empty")
	ErrFeatureMismatch = errors.New("feature count mismatch")
	ErrInvalidEpsilon  = errors.New("epsilon must be positive")
)

// GradientConfig contains GradientSHAP-specific configuration.
type GradientConfig struct {
	// Epsilon is the step size for numerical gradient computation.
	// Default: 1e-7
	Epsilon float64

	// NoiseStdev is the standard deviation of Gaussian noise to add
	// to interpolated points. Set to 0 to disable noise.
	// Default: 0
	NoiseStdev float64

	// LocalSmoothingSamples is the number of samples for local smoothing
	// when noise is enabled. Default: 1 (no local smoothing)
	LocalSmoothingSamples int
}

// DefaultGradientConfig returns the default GradientSHAP configuration.
func DefaultGradientConfig() GradientConfig {
	return GradientConfig{
		Epsilon:               1e-7,
		NoiseStdev:            0,
		LocalSmoothingSamples: 1,
	}
}

// GradientOption modifies GradientConfig.
type GradientOption func(*GradientConfig)

// WithEpsilon sets the step size for numerical gradients.
func WithEpsilon(eps float64) GradientOption {
	return func(c *GradientConfig) {
		c.Epsilon = eps
	}
}

// WithNoiseStdev sets the Gaussian noise standard deviation.
func WithNoiseStdev(stdev float64) GradientOption {
	return func(c *GradientConfig) {
		c.NoiseStdev = stdev
	}
}

// WithLocalSmoothing sets the number of local smoothing samples.
func WithLocalSmoothing(n int) GradientOption {
	return func(c *GradientConfig) {
		c.LocalSmoothingSamples = n
	}
}

// Explainer implements GradientSHAP (Expected Gradients).
type Explainer struct {
	model        model.Model
	background   [][]float64
	baseValue    float64
	featureNames []string
	config       explainer.Config
	gradConfig   GradientConfig
}

// New creates a new GradientSHAP explainer.
//
// Parameters:
//   - m: Model to explain (must implement model.Model)
//   - background: Representative samples for computing expected values
//   - opts: Standard explainer options (WithNumSamples, WithSeed, etc.)
//   - gradOpts: GradientSHAP-specific options (WithEpsilon, WithNoiseStdev, etc.)
//
// The background dataset is used for sampling reference points. Using
// 100-1000 representative samples from training data is recommended.
func New(
	m model.Model,
	background [][]float64,
	opts []explainer.Option,
	gradOpts ...GradientOption,
) (*Explainer, error) {
	if m == nil {
		return nil, ErrNilModel
	}
	if len(background) == 0 {
		return nil, ErrNoBackground
	}

	numFeatures := m.NumFeatures()

	// Validate background dimensions
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("%w: background row %d has %d features, expected %d",
				ErrFeatureMismatch, i, len(row), numFeatures)
		}
	}

	// Apply standard configuration
	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

	// Apply gradient configuration
	gradConfig := DefaultGradientConfig()
	for _, opt := range gradOpts {
		opt(&gradConfig)
	}

	if gradConfig.Epsilon <= 0 {
		return nil, ErrInvalidEpsilon
	}

	// Compute base value (expected prediction on background)
	ctx := context.Background()
	predictions, err := m.PredictBatch(ctx, background)
	if err != nil {
		return nil, fmt.Errorf("failed to compute base value: %w", err)
	}

	var baseValue float64
	for _, p := range predictions {
		baseValue += p
	}
	baseValue /= float64(len(predictions))

	return &Explainer{
		model:        m,
		background:   background,
		baseValue:    baseValue,
		featureNames: config.FeatureNames,
		config:       config,
		gradConfig:   gradConfig,
	}, nil
}

// Explain computes SHAP values for a single instance using Expected Gradients.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.model.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("%w: instance has %d features, expected %d",
			ErrFeatureMismatch, len(instance), numFeatures)
	}

	// Get prediction for the instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute SHAP values using Expected Gradients
	var shapValues []float64
	var shapSE []float64

	if e.config.NumWorkers > 1 {
		shapValues, shapSE, err = e.computeParallel(ctx, instance)
	} else {
		shapValues, shapSE, err = e.computeSequential(ctx, instance)
	}
	if err != nil {
		return nil, err
	}

	// Build the explanation
	values := make(map[string]float64)
	featureValues := make(map[string]float64)
	for i, name := range e.featureNames {
		values[name] = shapValues[i]
		featureValues[name] = instance[i]
	}

	exp := &explanation.Explanation{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Values:        values,
		FeatureNames:  e.featureNames,
		FeatureValues: featureValues,
		Timestamp:     time.Now(),
		ModelID:       e.config.ModelID,
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "gradient",
			NumSamples:     e.config.NumSamples,
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}

	// Add confidence intervals if enabled
	if e.config.ConfidenceLevel > 0 && len(shapSE) > 0 {
		zScore := explainer.ZScoreForConfidenceLevel(e.config.ConfidenceLevel)
		lower := make(map[string]float64)
		upper := make(map[string]float64)
		stdErrs := make(map[string]float64)

		for i, name := range e.featureNames {
			stdErrs[name] = shapSE[i]
			halfWidth := zScore * shapSE[i]
			lower[name] = shapValues[i] - halfWidth
			upper[name] = shapValues[i] + halfWidth
		}

		exp.Metadata.ConfidenceIntervals = &explanation.ConfidenceIntervals{
			Level:          e.config.ConfidenceLevel,
			Lower:          lower,
			Upper:          upper,
			StandardErrors: stdErrs,
		}
	}

	return exp, nil
}

// computeSequential computes SHAP values sequentially.
func (e *Explainer) computeSequential(ctx context.Context, instance []float64) ([]float64, []float64, error) {
	numFeatures := len(instance)
	rng := e.config.GetRNG()

	// Accumulate SHAP contributions
	shapSum := make([]float64, numFeatures)
	shapSumSq := make([]float64, numFeatures) // For variance estimation

	for i := 0; i < e.config.NumSamples; i++ {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		default:
		}

		// Sample random background point
		bgIdx := rng.Intn(len(e.background))
		reference := e.background[bgIdx]

		// Sample alpha uniformly from [0, 1]
		alpha := rng.Float64()

		// Compute contribution for this sample
		contrib, err := e.computeSampleContribution(ctx, instance, reference, alpha, rng)
		if err != nil {
			return nil, nil, err
		}

		// Accumulate
		for j, c := range contrib {
			shapSum[j] += c
			shapSumSq[j] += c * c
		}
	}

	// Compute means
	n := float64(e.config.NumSamples)
	shapValues := make([]float64, numFeatures)
	for i := range shapValues {
		shapValues[i] = shapSum[i] / n
	}

	// Compute standard errors if confidence intervals requested
	var shapSE []float64
	if e.config.ConfidenceLevel > 0 {
		shapSE = make([]float64, numFeatures)
		for i := range shapSE {
			mean := shapValues[i]
			variance := (shapSumSq[i]/n - mean*mean) * n / (n - 1)
			if variance < 0 {
				variance = 0
			}
			shapSE[i] = math.Sqrt(variance / n)
		}
	}

	return shapValues, shapSE, nil
}

// computeParallel computes SHAP values using parallel workers.
func (e *Explainer) computeParallel(ctx context.Context, instance []float64) ([]float64, []float64, error) {
	numFeatures := len(instance)
	numWorkers := e.config.NumWorkers
	if numWorkers > e.config.NumSamples {
		numWorkers = e.config.NumSamples
	}

	// Create worker context
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Channels for work distribution
	jobs := make(chan int, e.config.NumSamples)
	results := make(chan sampleResult, e.config.NumSamples)

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		workerSeed := e.config.GetRNG().Int63()
		go func(seed int64) {
			defer wg.Done()
			e.worker(workerCtx, instance, seed, jobs, results)
		}(workerSeed)
	}

	// Send jobs
	go func() {
		for i := 0; i < e.config.NumSamples; i++ {
			select {
			case <-workerCtx.Done():
				break
			case jobs <- i:
			}
		}
		close(jobs)
	}()

	// Close results when all workers done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	shapSum := make([]float64, numFeatures)
	shapSumSq := make([]float64, numFeatures)
	count := 0
	var firstErr error

	for result := range results {
		if result.err != nil && firstErr == nil {
			firstErr = result.err
			cancel()
			continue
		}
		if result.err == nil {
			for i, c := range result.contrib {
				shapSum[i] += c
				shapSumSq[i] += c * c
			}
			count++
		}
	}

	if firstErr != nil {
		return nil, nil, firstErr
	}

	// Compute means
	n := float64(count)
	shapValues := make([]float64, numFeatures)
	for i := range shapValues {
		shapValues[i] = shapSum[i] / n
	}

	// Compute standard errors if confidence intervals requested
	var shapSE []float64
	if e.config.ConfidenceLevel > 0 && count > 1 {
		shapSE = make([]float64, numFeatures)
		for i := range shapSE {
			mean := shapValues[i]
			variance := (shapSumSq[i]/n - mean*mean) * n / (n - 1)
			if variance < 0 {
				variance = 0
			}
			shapSE[i] = math.Sqrt(variance / n)
		}
	}

	return shapValues, shapSE, nil
}

type sampleResult struct {
	contrib []float64
	err     error
}

// worker processes samples for parallel computation.
func (e *Explainer) worker(ctx context.Context, instance []float64, seed int64, jobs <-chan int, results chan<- sampleResult) {
	rng := newRNG(seed)

	for range jobs {
		select {
		case <-ctx.Done():
			results <- sampleResult{err: ctx.Err()}
			continue
		default:
		}

		// Sample random background and alpha
		bgIdx := rng.Intn(len(e.background))
		reference := e.background[bgIdx]
		alpha := rng.Float64()

		contrib, err := e.computeSampleContribution(ctx, instance, reference, alpha, rng)
		results <- sampleResult{contrib: contrib, err: err}
	}
}

// computeSampleContribution computes the SHAP contribution for a single sample.
func (e *Explainer) computeSampleContribution(
	ctx context.Context,
	instance, reference []float64,
	alpha float64,
	rng RNG,
) ([]float64, error) {
	numFeatures := len(instance)

	// Compute interpolated point: z = reference + alpha * (instance - reference)
	interpolated := make([]float64, numFeatures)
	for i := range interpolated {
		interpolated[i] = reference[i] + alpha*(instance[i]-reference[i])
	}

	// Add noise if configured
	if e.gradConfig.NoiseStdev > 0 {
		for i := range interpolated {
			interpolated[i] += rng.NormFloat64() * e.gradConfig.NoiseStdev
		}
	}

	// Compute gradient at interpolated point
	gradient, err := e.computeNumericalGradient(ctx, interpolated)
	if err != nil {
		return nil, err
	}

	// SHAP contribution: (instance[i] - reference[i]) * gradient[i]
	contrib := make([]float64, numFeatures)
	for i := range contrib {
		contrib[i] = (instance[i] - reference[i]) * gradient[i]
	}

	return contrib, nil
}

// computeNumericalGradient computes gradient using central finite differences.
func (e *Explainer) computeNumericalGradient(ctx context.Context, point []float64) ([]float64, error) {
	numFeatures := len(point)
	eps := e.gradConfig.Epsilon

	// Create perturbed points for all features
	// We'll compute f(x + eps*e_i) and f(x - eps*e_i) for each feature
	forwardPoints := make([][]float64, numFeatures)
	backwardPoints := make([][]float64, numFeatures)

	for i := 0; i < numFeatures; i++ {
		forward := make([]float64, numFeatures)
		backward := make([]float64, numFeatures)
		copy(forward, point)
		copy(backward, point)
		forward[i] += eps
		backward[i] -= eps
		forwardPoints[i] = forward
		backwardPoints[i] = backward
	}

	// Batch predict for efficiency
	allPoints := make([][]float64, 0, 2*numFeatures)
	allPoints = append(allPoints, forwardPoints...)
	allPoints = append(allPoints, backwardPoints...)

	predictions, err := e.model.PredictBatch(ctx, allPoints)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient: %w", err)
	}

	// Compute gradients: (f(x+eps) - f(x-eps)) / (2*eps)
	gradient := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		gradient[i] = (predictions[i] - predictions[numFeatures+i]) / (2 * eps)
	}

	return gradient, nil
}

// ExplainBatch computes SHAP explanations for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, inst := range instances {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		exp, err := e.Explain(ctx, inst)
		if err != nil {
			return nil, fmt.Errorf("failed to explain instance %d: %w", i, err)
		}
		results[i] = exp
	}
	return results, nil
}

// BaseValue returns the expected model output on the background dataset.
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// FeatureNames returns the names of the features.
func (e *Explainer) FeatureNames() []string {
	return e.featureNames
}

// Ensure Explainer implements explainer.Explainer interface.
var _ explainer.Explainer = (*Explainer)(nil)
