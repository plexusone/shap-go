package explainer

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/plexusone/shap-go/explanation"
)

// BatchConfig contains configuration for parallel batch explanation.
type BatchConfig struct {
	// Workers is the number of parallel workers.
	// If 0, defaults to runtime.NumCPU().
	Workers int

	// StopOnError determines whether to stop all workers on first error.
	// If false, continues processing and returns all errors at the end.
	// Default: true
	StopOnError bool
}

// DefaultBatchConfig returns the default batch configuration.
func DefaultBatchConfig() BatchConfig {
	return BatchConfig{
		Workers:     runtime.NumCPU(),
		StopOnError: true,
	}
}

// BatchOption is a function that modifies BatchConfig.
type BatchOption func(*BatchConfig)

// WithWorkers sets the number of parallel workers.
func WithWorkers(n int) BatchOption {
	return func(c *BatchConfig) {
		c.Workers = n
	}
}

// WithStopOnError sets whether to stop on first error.
func WithStopOnError(stop bool) BatchOption {
	return func(c *BatchConfig) {
		c.StopOnError = stop
	}
}

// ExplainBatchParallel computes SHAP explanations for multiple instances in parallel.
// This is a generic function that works with any Explainer implementation.
//
// It distributes instances across worker goroutines, each calling exp.Explain().
// Results are returned in the same order as the input instances.
//
// Example:
//
//	explanations, err := explainer.ExplainBatchParallel(ctx, exp, instances,
//	    explainer.WithWorkers(4))
func ExplainBatchParallel(
	ctx context.Context,
	exp Explainer,
	instances [][]float64,
	opts ...BatchOption,
) ([]*explanation.Explanation, error) {
	if len(instances) == 0 {
		return []*explanation.Explanation{}, nil
	}

	config := DefaultBatchConfig()
	for _, opt := range opts {
		opt(&config)
	}

	if config.Workers <= 0 {
		config.Workers = runtime.NumCPU()
	}

	// For small batches, use fewer workers
	if config.Workers > len(instances) {
		config.Workers = len(instances)
	}

	// Single worker case - just use sequential
	if config.Workers == 1 {
		return exp.ExplainBatch(ctx, instances)
	}

	return explainParallel(ctx, exp, instances, config)
}

// workItem represents a single instance to explain.
type workItem struct {
	index    int
	instance []float64
}

// workResult represents the result of explaining an instance.
type workResult struct {
	index       int
	explanation *explanation.Explanation
	err         error
}

// explainParallel implements the parallel explanation logic.
func explainParallel(
	ctx context.Context,
	exp Explainer,
	instances [][]float64,
	config BatchConfig,
) ([]*explanation.Explanation, error) {
	// Create channels
	jobs := make(chan workItem, len(instances))
	results := make(chan workResult, len(instances))

	// Create cancellable context for stopping workers on error
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < config.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				// Check if context is cancelled
				select {
				case <-workerCtx.Done():
					results <- workResult{
						index: job.index,
						err:   workerCtx.Err(),
					}
					continue
				default:
				}

				// Explain the instance
				exp, err := exp.Explain(workerCtx, job.instance)
				results <- workResult{
					index:       job.index,
					explanation: exp,
					err:         err,
				}

				// If stop on error is enabled and we got an error, signal cancellation
				if err != nil && config.StopOnError {
					cancel()
				}
			}
		}()
	}

	// Send jobs
	go func() {
		for i, inst := range instances {
			select {
			case <-workerCtx.Done():
				// Context cancelled, stop sending jobs
				break
			case jobs <- workItem{index: i, instance: inst}:
			}
		}
		close(jobs)
	}()

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	explanations := make([]*explanation.Explanation, len(instances))
	var firstErr error
	var errCount int

	for result := range results {
		if result.err != nil {
			errCount++
			if firstErr == nil {
				firstErr = fmt.Errorf("instance %d: %w", result.index, result.err)
			}
			continue
		}
		explanations[result.index] = result.explanation
	}

	if firstErr != nil {
		if config.StopOnError {
			return nil, firstErr
		}
		return explanations, fmt.Errorf("%d instances failed, first error: %w", errCount, firstErr)
	}

	return explanations, nil
}

// ExplainBatchWithProgress computes SHAP explanations with progress callback.
// The progress function is called after each instance is explained with the
// current count and total count.
//
// Example:
//
//	explanations, err := explainer.ExplainBatchWithProgress(ctx, exp, instances,
//	    func(done, total int) {
//	        fmt.Printf("Progress: %d/%d\n", done, total)
//	    },
//	    explainer.WithWorkers(4))
func ExplainBatchWithProgress(
	ctx context.Context,
	exp Explainer,
	instances [][]float64,
	progress func(done, total int),
	opts ...BatchOption,
) ([]*explanation.Explanation, error) {
	if len(instances) == 0 {
		return []*explanation.Explanation{}, nil
	}

	config := DefaultBatchConfig()
	for _, opt := range opts {
		opt(&config)
	}

	if config.Workers <= 0 {
		config.Workers = runtime.NumCPU()
	}

	if config.Workers > len(instances) {
		config.Workers = len(instances)
	}

	// Create channels
	jobs := make(chan workItem, len(instances))
	results := make(chan workResult, len(instances))

	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < config.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				select {
				case <-workerCtx.Done():
					results <- workResult{index: job.index, err: workerCtx.Err()}
					continue
				default:
				}

				exp, err := exp.Explain(workerCtx, job.instance)
				results <- workResult{index: job.index, explanation: exp, err: err}

				if err != nil && config.StopOnError {
					cancel()
				}
			}
		}()
	}

	// Send jobs
	go func() {
		for i, inst := range instances {
			select {
			case <-workerCtx.Done():
				break
			case jobs <- workItem{index: i, instance: inst}:
			}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results with progress
	explanations := make([]*explanation.Explanation, len(instances))
	var firstErr error
	var errCount int
	done := 0
	total := len(instances)

	for result := range results {
		done++
		if progress != nil {
			progress(done, total)
		}

		if result.err != nil {
			errCount++
			if firstErr == nil {
				firstErr = fmt.Errorf("instance %d: %w", result.index, result.err)
			}
			continue
		}
		explanations[result.index] = result.explanation
	}

	if firstErr != nil {
		if config.StopOnError {
			return nil, firstErr
		}
		return explanations, fmt.Errorf("%d instances failed, first error: %w", errCount, firstErr)
	}

	return explanations, nil
}
