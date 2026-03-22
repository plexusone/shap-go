package tree

import (
	"context"
	"time"
)

// InteractionResult contains SHAP interaction values.
//
// SHAP interaction values measure pairwise feature interactions.
// The interaction matrix Φ is symmetric with:
//   - Diagonal Φᵢᵢ: main effect of feature i
//   - Off-diagonal Φᵢⱼ: interaction between features i and j
//
// Properties:
//   - sum(Φᵢⱼ for all j) = SHAP[i] (rows sum to SHAP values)
//   - sum(all Φᵢⱼ) = prediction - baseline
type InteractionResult struct {
	// Prediction is the model output for this instance.
	Prediction float64

	// BaseValue is the expected model output.
	BaseValue float64

	// Interactions is the interaction matrix [numFeatures][numFeatures].
	// Interactions[i][j] is the interaction between features i and j.
	// The diagonal Interactions[i][i] contains the main effects.
	Interactions [][]float64

	// FeatureNames are the names of features (in order).
	FeatureNames []string

	// ComputeTimeMS is computation time in milliseconds.
	ComputeTimeMS int64
}

// GetInteraction returns the interaction value between features i and j.
func (r *InteractionResult) GetInteraction(i, j int) float64 {
	if i >= len(r.Interactions) || j >= len(r.Interactions) {
		return 0
	}
	return r.Interactions[i][j]
}

// GetMainEffect returns the main effect for feature i (diagonal element).
func (r *InteractionResult) GetMainEffect(i int) float64 {
	return r.GetInteraction(i, i)
}

// GetSHAPValue returns the SHAP value for feature i (sum of row i).
func (r *InteractionResult) GetSHAPValue(i int) float64 {
	if i >= len(r.Interactions) {
		return 0
	}
	sum := 0.0
	for j := range r.Interactions[i] {
		sum += r.Interactions[i][j]
	}
	return sum
}

// TopInteractions returns the top k strongest interactions (by absolute value).
// Excludes diagonal (main effects).
func (r *InteractionResult) TopInteractions(k int) []FeatureInteraction {
	n := len(r.Interactions)
	var interactions []FeatureInteraction

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ { // Only upper triangle (symmetric)
			interactions = append(interactions, FeatureInteraction{
				Feature1: i,
				Feature2: j,
				Name1:    r.FeatureNames[i],
				Name2:    r.FeatureNames[j],
				Value:    r.Interactions[i][j],
			})
		}
	}

	// Sort by absolute value (descending)
	for i := 0; i < len(interactions)-1; i++ {
		for j := i + 1; j < len(interactions); j++ {
			if abs(interactions[j].Value) > abs(interactions[i].Value) {
				interactions[i], interactions[j] = interactions[j], interactions[i]
			}
		}
	}

	if k > len(interactions) {
		k = len(interactions)
	}
	return interactions[:k]
}

// FeatureInteraction represents an interaction between two features.
type FeatureInteraction struct {
	Feature1 int
	Feature2 int
	Name1    string
	Name2    string
	Value    float64
}

// ExplainInteractions computes SHAP interaction values for an instance.
//
// Returns an InteractionResult containing the full interaction matrix.
// This is more expensive than regular SHAP values: O(TLD² × D) vs O(TLD²).
func (e *Explainer) ExplainInteractions(ctx context.Context, instance []float64) (*InteractionResult, error) {
	startTime := time.Now()

	if len(instance) != e.ensemble.NumFeatures {
		return nil, errFeatureMismatch(len(instance), e.ensemble.NumFeatures)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Compute prediction
	prediction := e.predict(instance)

	// Compute interaction matrix
	interactions := e.computer.computeInteractions(instance)

	return &InteractionResult{
		Prediction:    prediction,
		BaseValue:     e.baseValue,
		Interactions:  interactions,
		FeatureNames:  e.featureNames,
		ComputeTimeMS: time.Since(startTime).Milliseconds(),
	}, nil
}

// computeInteractions computes the SHAP interaction matrix.
func (c *treeSHAPComputer) computeInteractions(instance []float64) [][]float64 {
	n := c.numFeatures

	// Initialize interaction matrix
	interactions := make([][]float64, n)
	for i := range interactions {
		interactions[i] = make([]float64, n)
	}

	// Process each tree
	for treeIdx := 0; treeIdx < c.ensemble.NumTrees; treeIdx++ {
		rootIdx := c.ensemble.Roots[treeIdx]
		treeInteractions := c.computeTreeInteractions(rootIdx, instance)

		// Accumulate
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				interactions[i][j] += treeInteractions[i][j]
			}
		}
	}

	return interactions
}

// computeTreeInteractions computes interaction values for a single tree.
func (c *treeSHAPComputer) computeTreeInteractions(rootIdx int, instance []float64) [][]float64 {
	n := c.numFeatures

	// Initialize interaction matrix for this tree
	interactions := make([][]float64, n)
	for i := range interactions {
		interactions[i] = make([]float64, n)
	}

	// Initialize path with interaction tracking
	path := []InteractionPathElem{}

	// Start recursive traversal
	c.recurseInteractions(rootIdx, instance, path, 1.0, 1.0, -1, interactions)

	return interactions
}

// InteractionPathElem extends PathElem with interaction tracking.
type InteractionPathElem struct {
	PathElem
	// InteractionIdx tracks which feature (if any) this element interacts with.
	// -1 means no interaction recorded yet for this position.
	InteractionIdx int
}

// recurseInteractions performs recursive traversal for interaction computation.
//
//nolint:dupl // Intentionally similar to recurse() but uses InteractionPathElem and different accumulation
func (c *treeSHAPComputer) recurseInteractions(
	nodeIdx int,
	instance []float64,
	path []InteractionPathElem,
	pz, po float64,
	parentFeature int,
	interactions [][]float64,
) {
	node := &c.ensemble.Nodes[nodeIdx]

	// Extend path with parent's feature
	path = extendInteractionPath(path, pz, po, parentFeature)

	if node.IsLeaf {
		// Accumulate interaction contributions at leaf
		c.accumulateInteractions(path, node.Prediction, interactions)
		return
	}

	feature := node.Feature
	featureVal := instance[feature]

	// Check if this feature is already in path
	iz := 1.0
	io := 1.0
	pathFeatureIdx := findFeatureInInteractionPath(path, feature)
	if pathFeatureIdx >= 0 {
		iz = path[pathFeatureIdx].ZeroFrac
		io = path[pathFeatureIdx].OneFrac
		path = unwindInteractionPath(path, pathFeatureIdx)
	}

	// Compute cover fractions
	yesNode := &c.ensemble.Nodes[node.Yes]
	noNode := &c.ensemble.Nodes[node.No]

	yesCover := yesNode.Cover
	noCover := noNode.Cover
	totalCover := yesCover + noCover

	var hotCoverFrac, coldCoverFrac float64
	if totalCover > 0 {
		goesLeft := c.evaluateSplit(node, featureVal)
		if goesLeft {
			hotCoverFrac = yesCover / totalCover
			coldCoverFrac = noCover / totalCover
		} else {
			hotCoverFrac = noCover / totalCover
			coldCoverFrac = yesCover / totalCover
		}
	} else {
		hotCoverFrac = 0.5
		coldCoverFrac = 0.5
	}

	goesLeft := c.evaluateSplit(node, featureVal)
	var hotIdx, coldIdx int
	if goesLeft {
		hotIdx = node.Yes
		coldIdx = node.No
	} else {
		hotIdx = node.No
		coldIdx = node.Yes
	}

	// Recurse to children
	c.recurseInteractions(hotIdx, instance, path, iz*hotCoverFrac, io, feature, interactions)
	c.recurseInteractions(coldIdx, instance, path, iz*coldCoverFrac, 0.0, feature, interactions)
}

// accumulateInteractions adds interaction contributions at a leaf.
func (c *treeSHAPComputer) accumulateInteractions(
	path []InteractionPathElem,
	leafValue float64,
	interactions [][]float64,
) {
	depth := len(path)
	if depth == 0 {
		return
	}

	// For interaction values, we need to compute contributions for each pair
	// The off-diagonal terms capture when two features appear together in the path
	// The diagonal terms capture the main effect

	// First, compute regular SHAP contributions for diagonal (main effects)
	for i := 0; i < depth; i++ {
		elem := &path[i]
		if elem.Feature < 0 {
			continue
		}

		w := unwoundInteractionPathSum(path, i)
		contribution := w * (elem.OneFrac - elem.ZeroFrac) * leafValue
		interactions[elem.Feature][elem.Feature] += contribution
	}

	// For off-diagonal (interactions), we look at pairs of features in the path
	// The interaction between features i and j is the difference between:
	// - their joint contribution when both are in the path
	// - their individual contributions
	//
	// For TreeSHAP, when both features are on the path, they interact.
	// The interaction value is distributed based on path weights.

	for i := 0; i < depth; i++ {
		elemI := &path[i]
		if elemI.Feature < 0 {
			continue
		}

		for j := i + 1; j < depth; j++ {
			elemJ := &path[j]
			if elemJ.Feature < 0 || elemJ.Feature == elemI.Feature {
				continue
			}

			// Compute interaction contribution
			// When two features appear together, their interaction is
			// proportional to the product of their (oneFrac - zeroFrac) terms
			wI := unwoundInteractionPathSum(path, i)
			wJ := unwoundInteractionPathSum(path, j)

			// The interaction term
			interactionContrib := 0.5 * wI * wJ *
				(elemI.OneFrac - elemI.ZeroFrac) *
				(elemJ.OneFrac - elemJ.ZeroFrac) *
				leafValue

			// Scale by path depth (interaction diminishes with depth)
			interactionContrib /= float64(depth)

			// Add to both symmetric positions
			interactions[elemI.Feature][elemJ.Feature] += interactionContrib
			interactions[elemJ.Feature][elemI.Feature] += interactionContrib

			// Subtract from diagonal (interaction is split from main effects)
			interactions[elemI.Feature][elemI.Feature] -= interactionContrib
			interactions[elemJ.Feature][elemJ.Feature] -= interactionContrib
		}
	}
}

// extendInteractionPath adds a feature to the interaction path.
func extendInteractionPath(path []InteractionPathElem, pz, po float64, feature int) []InteractionPathElem {
	depth := len(path)
	newPath := make([]InteractionPathElem, depth+1)
	copy(newPath, path)

	initialWeight := 0.0
	if depth == 0 {
		initialWeight = 1.0
	}

	newPath[depth] = InteractionPathElem{
		PathElem: PathElem{
			Feature:  feature,
			ZeroFrac: pz,
			OneFrac:  po,
			Weight:   initialWeight,
		},
		InteractionIdx: -1,
	}

	// Update weights
	for i := depth - 1; i >= 0; i-- {
		newPath[i+1].Weight += po * newPath[i].Weight * float64(i+1) / float64(depth+1)
		newPath[i].Weight = pz * newPath[i].Weight * float64(depth-i) / float64(depth+1)
	}

	return newPath
}

// unwindInteractionPath removes a feature from the interaction path.
//
//nolint:dupl // Intentionally similar to unwindPath() but uses InteractionPathElem
func unwindInteractionPath(path []InteractionPathElem, pathIdx int) []InteractionPathElem {
	depth := len(path) - 1
	if depth < 0 {
		return path
	}

	newPath := make([]InteractionPathElem, len(path))
	copy(newPath, path)

	zeroFrac := newPath[pathIdx].ZeroFrac
	oneFrac := newPath[pathIdx].OneFrac

	n := float64(depth)

	if oneFrac != 0 {
		nextW := newPath[depth].Weight
		for j := depth - 1; j >= 0; j-- {
			tmp := newPath[j].Weight
			newPath[j].Weight = nextW * (n + 1) / (float64(j+1) * oneFrac)
			nextW = tmp - newPath[j].Weight*zeroFrac*float64(depth-j)/(n+1)
		}
	} else if zeroFrac != 0 {
		for j := depth - 1; j >= 0; j-- {
			newPath[j].Weight = newPath[j].Weight * (n + 1) / (zeroFrac * float64(depth-j))
		}
	}

	result := make([]InteractionPathElem, depth)
	copy(result[:pathIdx], newPath[:pathIdx])
	copy(result[pathIdx:], newPath[pathIdx+1:])

	return result
}

// unwoundInteractionPathSum computes the unwound sum for interaction path.
func unwoundInteractionPathSum(path []InteractionPathElem, pathIdx int) float64 {
	depth := len(path) - 1
	if depth < 0 {
		return 0
	}

	if depth == 0 {
		return 1.0
	}

	zeroFrac := path[pathIdx].ZeroFrac
	oneFrac := path[pathIdx].OneFrac

	total := 0.0

	if oneFrac != 0 {
		n := path[depth].Weight
		for j := depth - 1; j >= 0; j-- {
			tmp := n / (float64(j+1) * oneFrac)
			total += tmp
			n = path[j].Weight - tmp*zeroFrac*float64(depth-j)
		}
	} else if zeroFrac != 0 {
		for j := depth - 1; j >= 0; j-- {
			total += path[j].Weight / (float64(depth-j) * zeroFrac)
		}
	}

	return total * float64(depth+1)
}

// findFeatureInInteractionPath searches for a feature in the interaction path.
func findFeatureInInteractionPath(path []InteractionPathElem, feature int) int {
	for i := 0; i < len(path); i++ {
		if path[i].Feature == feature {
			return i
		}
	}
	return -1
}

// abs returns the absolute value of x.
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// errFeatureMismatch creates a feature count mismatch error.
func errFeatureMismatch(got, expected int) error {
	return &featureMismatchError{got: got, expected: expected}
}

type featureMismatchError struct {
	got, expected int
}

func (e *featureMismatchError) Error() string {
	return "instance has " + itoa(e.got) + " features, expected " + itoa(e.expected)
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}
