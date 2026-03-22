package partition

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Node represents a node in the feature hierarchy tree.
// Leaf nodes represent individual features, internal nodes represent groups.
type Node struct {
	// Name is the name of this node (feature or group name).
	Name string

	// FeatureIdx is the index of the feature for leaf nodes.
	// Should be -1 for internal nodes.
	FeatureIdx int

	// Children are the child nodes. Empty for leaf nodes.
	Children []*Node
}

// IsLeaf returns true if this node is a leaf (represents a single feature).
func (n *Node) IsLeaf() bool {
	return len(n.Children) == 0
}

// GetFeatureIndices returns all feature indices under this node.
func (n *Node) GetFeatureIndices() []int {
	if n.IsLeaf() {
		return []int{n.FeatureIdx}
	}

	var indices []int
	for _, child := range n.Children {
		indices = append(indices, child.GetFeatureIndices()...)
	}
	return indices
}

// Explainer implements PartitionSHAP using hierarchical Owen values.
type Explainer struct {
	model        model.Model
	background   [][]float64
	hierarchy    *Node
	featureNames []string
	baseValue    float64
	config       explainer.Config
	rng          *rand.Rand
	mu           sync.Mutex // protects rng
}

// New creates a new PartitionSHAP explainer.
//
// Parameters:
//   - m: The model to explain (implements model.Model interface)
//   - background: Representative samples for baseline/masking
//   - hierarchy: Feature hierarchy tree. If nil, creates a flat hierarchy.
//   - opts: Configuration options (WithNumSamples, WithSeed, etc.)
func New(m model.Model, background [][]float64, hierarchy *Node, opts ...explainer.Option) (*Explainer, error) {
	if m == nil {
		return nil, fmt.Errorf("model cannot be nil")
	}
	if len(background) == 0 {
		return nil, fmt.Errorf("background data cannot be empty")
	}

	numFeatures := m.NumFeatures()
	for i, row := range background {
		if len(row) != numFeatures {
			return nil, fmt.Errorf("background row %d has %d features, expected %d",
				i, len(row), numFeatures)
		}
	}

	// Create default flat hierarchy if none provided
	if hierarchy == nil {
		hierarchy = createFlatHierarchy(numFeatures)
	}

	// Validate hierarchy covers all features
	if err := validateHierarchy(hierarchy, numFeatures); err != nil {
		return nil, err
	}

	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

	// Compute base value
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
		hierarchy:    hierarchy,
		featureNames: config.FeatureNames,
		baseValue:    baseValue,
		config:       config,
		rng:          config.GetRNG(),
	}, nil
}

// createFlatHierarchy creates a hierarchy where each feature is its own group.
func createFlatHierarchy(numFeatures int) *Node {
	children := make([]*Node, numFeatures)
	for i := 0; i < numFeatures; i++ {
		children[i] = &Node{
			Name:       fmt.Sprintf("feature_%d", i),
			FeatureIdx: i,
		}
	}
	return &Node{
		Name:     "root",
		Children: children,
	}
}

// validateHierarchy checks that the hierarchy covers all features exactly once.
func validateHierarchy(root *Node, numFeatures int) error {
	indices := root.GetFeatureIndices()

	// Check for correct number of features
	if len(indices) != numFeatures {
		return fmt.Errorf("hierarchy has %d features, expected %d", len(indices), numFeatures)
	}

	// Check for duplicates and valid indices
	seen := make(map[int]bool)
	for _, idx := range indices {
		if idx < 0 || idx >= numFeatures {
			return fmt.Errorf("invalid feature index %d (must be 0-%d)", idx, numFeatures-1)
		}
		if seen[idx] {
			return fmt.Errorf("duplicate feature index %d in hierarchy", idx)
		}
		seen[idx] = true
	}

	return nil
}

// Explain computes SHAP values for a single instance using PartitionSHAP.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.model.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("instance has %d features, expected %d",
			len(instance), numFeatures)
	}

	// Get prediction for this instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute SHAP values using hierarchical Owen values
	shapValues, err := e.computeOwenValues(ctx, instance, prediction)
	if err != nil {
		return nil, fmt.Errorf("failed to compute Owen values: %w", err)
	}

	// Build result
	values := make(map[string]float64, numFeatures)
	featureValues := make(map[string]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		name := e.featureNames[i]
		values[name] = shapValues[i]
		featureValues[name] = instance[i]
	}

	return &explanation.Explanation{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Values:        values,
		FeatureNames:  e.featureNames,
		FeatureValues: featureValues,
		Timestamp:     time.Now(),
		ModelID:       e.config.ModelID,
		Metadata: explanation.ExplanationMetadata{
			Algorithm:      "partition",
			NumSamples:     e.config.NumSamples,
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}, nil
}

// computeOwenValues computes SHAP values using hierarchical Owen values.
func (e *Explainer) computeOwenValues(ctx context.Context, instance []float64, prediction float64) ([]float64, error) {
	numFeatures := e.model.NumFeatures()
	shapValues := make([]float64, numFeatures)

	// Recursively compute Owen values from root
	totalContribution := prediction - e.baseValue
	err := e.distributeContribution(ctx, e.hierarchy, instance, shapValues, totalContribution)
	if err != nil {
		return nil, err
	}

	return shapValues, nil
}

// distributeContribution recursively distributes the contribution to features.
func (e *Explainer) distributeContribution(ctx context.Context, node *Node, instance []float64, shapValues []float64, contribution float64) error {
	if node.IsLeaf() {
		// Leaf node: assign contribution directly to this feature
		shapValues[node.FeatureIdx] = contribution
		return nil
	}

	// Internal node: compute marginal contributions of each child group
	// and then recursively distribute to features within each group
	numChildren := len(node.Children)
	if numChildren == 1 {
		// Single child: pass through contribution
		return e.distributeContribution(ctx, node.Children[0], instance, shapValues, contribution)
	}

	// Compute contribution for each child using sampling
	childContributions, err := e.computeChildContributions(ctx, node, instance, contribution)
	if err != nil {
		return err
	}

	// Recursively distribute to each child
	for i, child := range node.Children {
		if err := e.distributeContribution(ctx, child, instance, shapValues, childContributions[i]); err != nil {
			return err
		}
	}

	return nil
}

// computeChildContributions computes the contribution for each child of a node.
func (e *Explainer) computeChildContributions(ctx context.Context, node *Node, instance []float64, totalContribution float64) ([]float64, error) {
	numChildren := len(node.Children)
	if numChildren == 0 {
		return nil, nil
	}

	// Get feature indices for each child
	childFeatures := make([][]int, numChildren)
	for i, child := range node.Children {
		childFeatures[i] = child.GetFeatureIndices()
	}

	// Sample permutations to estimate marginal contributions
	numSamples := e.config.NumSamples
	if numSamples < 10 {
		numSamples = 100
	}

	// Accumulate marginal contributions for each child
	marginalSums := make([]float64, numChildren)
	counts := make([]float64, numChildren)

	e.mu.Lock()
	defer e.mu.Unlock()

	for s := 0; s < numSamples; s++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Generate random permutation of children
		perm := e.randomPermutation(numChildren)

		// For each child, compute marginal contribution
		for pos, childIdx := range perm {
			// Create coalition: children before this one in permutation
			coalitionChildren := perm[:pos]

			// Compute f(coalition ∪ {child}) - f(coalition)
			withChild, err := e.evaluateCoalition(ctx, childFeatures, append(coalitionChildren, childIdx), instance)
			if err != nil {
				return nil, err
			}

			withoutChild, err := e.evaluateCoalition(ctx, childFeatures, coalitionChildren, instance)
			if err != nil {
				return nil, err
			}

			marginalSums[childIdx] += withChild - withoutChild
			counts[childIdx]++
		}
	}

	// Average marginal contributions
	contributions := make([]float64, numChildren)
	totalEstimated := 0.0
	for i := 0; i < numChildren; i++ {
		if counts[i] > 0 {
			contributions[i] = marginalSums[i] / counts[i]
			totalEstimated += contributions[i]
		}
	}

	// Scale to ensure sum equals totalContribution (efficiency)
	if totalEstimated != 0 {
		scale := totalContribution / totalEstimated
		for i := range contributions {
			contributions[i] *= scale
		}
	} else {
		// Distribute evenly if all contributions are zero
		perChild := totalContribution / float64(numChildren)
		for i := range contributions {
			contributions[i] = perChild
		}
	}

	return contributions, nil
}

// evaluateCoalition computes the model prediction when features in the coalition
// are from the instance and others are from background.
func (e *Explainer) evaluateCoalition(ctx context.Context, childFeatures [][]int, coalitionIndices []int, instance []float64) (float64, error) {
	// Use batched predictions if enabled
	if e.config.UseBatchedPredictions {
		return e.evaluateCoalitionBatched(ctx, childFeatures, coalitionIndices, instance)
	}

	// Determine which features are in the coalition
	inCoalition := make(map[int]bool)
	for _, childIdx := range coalitionIndices {
		for _, featIdx := range childFeatures[childIdx] {
			inCoalition[featIdx] = true
		}
	}

	// Average over background samples
	var sum float64
	for _, bg := range e.background {
		// Create masked instance
		masked := make([]float64, len(instance))
		for i := range instance {
			if inCoalition[i] {
				masked[i] = instance[i]
			} else {
				masked[i] = bg[i]
			}
		}

		pred, err := e.model.Predict(ctx, masked)
		if err != nil {
			return 0, err
		}
		sum += pred
	}

	return sum / float64(len(e.background)), nil
}

// evaluateCoalitionBatched computes coalition prediction using batched model inference.
// This is more efficient when the model has optimized batch prediction.
func (e *Explainer) evaluateCoalitionBatched(ctx context.Context, childFeatures [][]int, coalitionIndices []int, instance []float64) (float64, error) {
	// Determine which features are in the coalition
	inCoalition := make(map[int]bool)
	for _, childIdx := range coalitionIndices {
		for _, featIdx := range childFeatures[childIdx] {
			inCoalition[featIdx] = true
		}
	}

	numBackground := len(e.background)

	// Build all masked inputs at once
	inputs := make([][]float64, numBackground)
	for b, bg := range e.background {
		masked := make([]float64, len(instance))
		for i := range instance {
			if inCoalition[i] {
				masked[i] = instance[i]
			} else {
				masked[i] = bg[i]
			}
		}
		inputs[b] = masked
	}

	// Batch prediction
	predictions, err := e.model.PredictBatch(ctx, inputs)
	if err != nil {
		return 0, err
	}

	// Average predictions
	var sum float64
	for _, pred := range predictions {
		sum += pred
	}

	return sum / float64(numBackground), nil
}

// randomPermutation generates a random permutation of 0..n-1.
func (e *Explainer) randomPermutation(n int) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	e.rng.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})
	return perm
}

// ExplainBatch computes SHAP values for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, instance := range instances {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		result, err := e.Explain(ctx, instance)
		if err != nil {
			return nil, fmt.Errorf("failed to explain instance %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// BaseValue returns the expected model output on the background data.
func (e *Explainer) BaseValue() float64 {
	return e.baseValue
}

// FeatureNames returns the feature names.
func (e *Explainer) FeatureNames() []string {
	return e.featureNames
}

// Hierarchy returns the feature hierarchy.
func (e *Explainer) Hierarchy() *Node {
	return e.hierarchy
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
