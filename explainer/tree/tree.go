package tree

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
)

// Explainer implements TreeSHAP for tree ensemble models.
//
// TreeSHAP computes exact SHAP values using a polynomial-time algorithm
// that exploits the tree structure. Unlike sampling-based methods (permutation,
// sampling), TreeSHAP provides:
//
//   - Exact values (not approximations)
//   - Guaranteed local accuracy (SHAP values sum to prediction - baseline)
//   - O(TLD²) complexity where T=trees, L=max depth, D=features
//
// Usage:
//
//	ensemble, err := tree.LoadXGBoostModel("model.json")
//	exp, err := tree.New(ensemble, explainer.WithFeatureNames(names))
//	result, err := exp.Explain(ctx, instance)
type Explainer struct {
	ensemble     *TreeEnsemble
	featureNames []string
	baseValue    float64
	config       explainer.Config
	computer     *treeSHAPComputer
}

// New creates a new TreeSHAP explainer from a tree ensemble.
//
// The ensemble must be a valid TreeEnsemble (e.g., loaded from XGBoost JSON).
// Options can be used to set feature names, model ID, etc.
//
// Note: NumSamples and NumWorkers options are ignored for TreeSHAP since
// it computes exact values without sampling.
func New(ensemble *TreeEnsemble, opts ...explainer.Option) (*Explainer, error) {
	if ensemble == nil {
		return nil, fmt.Errorf("ensemble cannot be nil")
	}
	if err := ensemble.Validate(); err != nil {
		return nil, fmt.Errorf("invalid ensemble: %w", err)
	}

	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(ensemble.NumFeatures)

	// Use ensemble's feature names if not provided via options
	featureNames := config.FeatureNames
	if len(ensemble.FeatureNames) > 0 && len(config.FeatureNames) == ensemble.NumFeatures {
		// Check if config has default names (feature_0, feature_1, etc.)
		isDefault := true
		for i, name := range config.FeatureNames {
			if name != fmt.Sprintf("feature_%d", i) {
				isDefault = false
				break
			}
		}
		if isDefault {
			featureNames = ensemble.FeatureNames
		}
	}

	// Compute expected value from tree structure
	// This is the base value that SHAP values are computed relative to
	expectedValue := ensemble.ExpectedValue()

	return &Explainer{
		ensemble:     ensemble,
		featureNames: featureNames,
		baseValue:    expectedValue,
		config:       config,
		computer:     newTreeSHAPComputer(ensemble),
	}, nil
}

// NewFromXGBoost creates a TreeSHAP explainer from an XGBoost JSON model file.
func NewFromXGBoost(modelPath string, opts ...explainer.Option) (*Explainer, error) {
	ensemble, err := LoadXGBoostModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load XGBoost model: %w", err)
	}
	return New(ensemble, opts...)
}

// Explain computes SHAP values for a single instance.
//
// The instance must have the same number of features as the model.
// Returns an Explanation containing SHAP values for each feature.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	if len(instance) != e.ensemble.NumFeatures {
		return nil, fmt.Errorf("instance has %d features, expected %d",
			len(instance), e.ensemble.NumFeatures)
	}

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Compute prediction by summing leaf values
	prediction := e.predict(instance)

	// Compute SHAP values using TreeSHAP
	shapValues := e.computer.computeSHAP(instance)

	// Build the explanation
	values := make(map[string]float64)
	featureValues := make(map[string]float64)
	for i, name := range e.featureNames {
		values[name] = shapValues[i]
		if i < len(instance) {
			featureValues[name] = instance[i]
		}
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
			Algorithm:      "treeshap",
			NumSamples:     0, // Not applicable for exact computation
			BackgroundSize: 0, // Uses tree structure, not background data
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}

	return exp, nil
}

// ExplainBatch computes SHAP explanations for multiple instances.
//
// If NumWorkers > 1, instances are processed in parallel.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	if e.config.NumWorkers > 1 {
		return e.explainBatchParallel(ctx, instances)
	}

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

// explainBatchParallel processes instances in parallel.
func (e *Explainer) explainBatchParallel(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	numInstances := len(instances)
	numWorkers := e.config.NumWorkers
	if numWorkers > numInstances {
		numWorkers = numInstances
	}

	type result struct {
		idx int
		exp *explanation.Explanation
		err error
	}

	results := make([]*explanation.Explanation, numInstances)
	resultCh := make(chan result, numInstances)

	// Create work queue
	workCh := make(chan int, numInstances)
	for i := 0; i < numInstances; i++ {
		workCh <- i
	}
	close(workCh)

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range workCh {
				select {
				case <-ctx.Done():
					resultCh <- result{idx: idx, err: ctx.Err()}
					return
				default:
				}

				exp, err := e.Explain(ctx, instances[idx])
				resultCh <- result{idx: idx, exp: exp, err: err}
			}
		}()
	}

	// Close result channel when workers are done
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Collect results
	var firstErr error
	for res := range resultCh {
		if res.err != nil && firstErr == nil {
			firstErr = fmt.Errorf("failed to explain instance %d: %w", res.idx, res.err)
		}
		if res.exp != nil {
			results[res.idx] = res.exp
		}
	}

	if firstErr != nil {
		return nil, firstErr
	}

	return results, nil
}

// BaseValue returns the expected model output (base score).
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// FeatureNames returns the names of the features.
func (e *Explainer) FeatureNames() []string {
	return e.featureNames
}

// Ensemble returns the underlying tree ensemble.
func (e *Explainer) Ensemble() *TreeEnsemble {
	return e.ensemble
}

// predict computes the model prediction for an instance by traversing trees.
// Uses ensemble.BaseScore (not the expected value) as the initial prediction.
func (e *Explainer) predict(instance []float64) float64 {
	prediction := e.ensemble.BaseScore

	for treeIdx := 0; treeIdx < e.ensemble.NumTrees; treeIdx++ {
		rootIdx := e.ensemble.Roots[treeIdx]
		leafValue := e.traverseTree(rootIdx, instance)
		prediction += leafValue
	}

	return prediction
}

// traverseTree traverses a tree and returns the leaf value.
func (e *Explainer) traverseTree(nodeIdx int, instance []float64) float64 {
	node := &e.ensemble.Nodes[nodeIdx]

	if node.IsLeaf {
		return node.Prediction
	}

	// Evaluate split
	goesLeft := e.computer.evaluateSplit(node, instance[node.Feature])

	if goesLeft {
		return e.traverseTree(node.Yes, instance)
	}
	return e.traverseTree(node.No, instance)
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
