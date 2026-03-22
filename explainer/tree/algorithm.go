package tree

import (
	"math"
)

// PathElem tracks a feature's contribution along a decision path.
// This is used internally by the TreeSHAP algorithm.
type PathElem struct {
	// Feature is the feature index (-1 for unused positions).
	Feature int

	// ZeroFrac is the fraction of paths that exclude this feature (z).
	ZeroFrac float64

	// OneFrac is the fraction of paths that include this feature (o).
	OneFrac float64

	// Weight is the combinatorial weight used in SHAP computation (w).
	Weight float64
}

// treeSHAPComputer performs TreeSHAP computation for an ensemble.
type treeSHAPComputer struct {
	ensemble    *TreeEnsemble
	numFeatures int
}

// newTreeSHAPComputer creates a new TreeSHAP computer.
func newTreeSHAPComputer(ensemble *TreeEnsemble) *treeSHAPComputer {
	return &treeSHAPComputer{
		ensemble:    ensemble,
		numFeatures: ensemble.NumFeatures,
	}
}

// computeSHAP computes SHAP values for a single instance.
// Returns a slice of SHAP values, one per feature.
func (c *treeSHAPComputer) computeSHAP(instance []float64) []float64 {
	shapValues := make([]float64, c.numFeatures)

	// Process each tree in the ensemble
	for treeIdx := 0; treeIdx < c.ensemble.NumTrees; treeIdx++ {
		rootIdx := c.ensemble.Roots[treeIdx]
		treeShap := c.computeTreeSHAP(rootIdx, instance)

		// Accumulate contributions from this tree
		for i := range shapValues {
			shapValues[i] += treeShap[i]
		}
	}

	return shapValues
}

// computeTreeSHAP computes SHAP values for a single tree.
func (c *treeSHAPComputer) computeTreeSHAP(rootIdx int, instance []float64) []float64 {
	shapValues := make([]float64, c.numFeatures)

	// Initialize empty path
	path := []PathElem{}

	// Start recursive traversal from root with pz=1, po=1, parentFeature=-1
	c.recurse(rootIdx, instance, path, 1.0, 1.0, -1, shapValues)

	return shapValues
}

// recurse performs the recursive tree traversal for TreeSHAP.
// Following the R treeshap algorithm exactly:
// 1. ALWAYS extend path with parent's feature using (pz, po) - even pi=-1 for dummy
// 2. If this node's feature already in path, unwind it and save (i_z, i_o)
// 3. Compute local cover fractions and recurse with i_z * local_fraction
//
// Parameters:
//   - pz: zero fraction for extending parent's feature (LOCAL, not cumulative)
//   - po: one fraction (1.0 for hot path, 0.0 for cold path)
//   - parentFeature: feature index from parent node (-1 for root/dummy)
//
//nolint:dupl // Similar to recurseInteractions() but uses PathElem and different accumulation
func (c *treeSHAPComputer) recurse(
	nodeIdx int,
	instance []float64,
	path []PathElem,
	pz, po float64,
	parentFeature int,
	shapValues []float64,
) {
	node := &c.ensemble.Nodes[nodeIdx]

	// ALWAYS extend path with parent's feature (as in R treeshap)
	// This includes the dummy element (parentFeature=-1) at the start
	// The dummy affects weight calculations but is skipped in contributions
	path = extendPath(path, pz, po, parentFeature)

	if node.IsLeaf {
		// At a leaf, accumulate SHAP contributions for all features in path
		// Skip dummy elements (feature < 0)
		c.accumulateContributions(path, node.Prediction, shapValues)
		return
	}

	// Get the feature value for this split
	feature := node.Feature
	featureVal := instance[feature]

	// Check if this node's feature is already in the path
	// If so, unwind it and save (i_z, i_o) for use in recursive calls
	// This matches R treeshap behavior exactly
	// Note: findFeatureInPath only matches features >= 0, so dummy won't match
	iz := 1.0 // Default zero fraction multiplier
	io := 1.0 // Default one fraction indicator
	pathFeatureIdx := findFeatureInPath(path, feature)
	if pathFeatureIdx >= 0 {
		iz = path[pathFeatureIdx].ZeroFrac
		io = path[pathFeatureIdx].OneFrac
		path = unwindPath(path, pathFeatureIdx)
	}

	// Compute LOCAL cover fractions at this node
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

	// Determine hot/cold child indices
	goesLeft := c.evaluateSplit(node, featureVal)
	var hotIdx, coldIdx int
	if goesLeft {
		hotIdx = node.Yes
		coldIdx = node.No
	} else {
		hotIdx = node.No
		coldIdx = node.Yes
	}

	// Recurse to children with LOCAL fractions (not cumulative)
	// R treeshap passes: i_z * cover[branch] / cover[node]
	// Hot path: p_o = i_o (from unwound feature, or 1 if not found)
	// Cold path: p_o = 0 (instance doesn't go this way)
	c.recurse(hotIdx, instance, path, iz*hotCoverFrac, io, feature, shapValues)
	c.recurse(coldIdx, instance, path, iz*coldCoverFrac, 0.0, feature, shapValues)
}

// evaluateSplit determines if an instance goes left (yes) at a node.
func (c *treeSHAPComputer) evaluateSplit(node *Node, featureVal float64) bool {
	if math.IsNaN(featureVal) {
		// Handle missing value
		return node.Missing == node.Yes
	}

	switch node.DecisionType {
	case DecisionLess:
		return featureVal < node.Threshold
	case DecisionLessEqual:
		return featureVal <= node.Threshold
	default:
		// Default to less-than
		return featureVal < node.Threshold
	}
}

// accumulateContributions adds SHAP contributions at a leaf node.
func (c *treeSHAPComputer) accumulateContributions(
	path []PathElem,
	leafValue float64,
	shapValues []float64,
) {
	depth := len(path)
	if depth == 0 {
		return
	}

	// For each feature in the path
	for i := 0; i < depth; i++ {
		elem := &path[i]
		if elem.Feature < 0 {
			continue
		}

		// Compute unwound sum for this feature
		w := unwoundPathSum(path, i)

		// SHAP contribution = weight * (oneFrac - zeroFrac) * leafValue
		contribution := w * (elem.OneFrac - elem.ZeroFrac) * leafValue
		shapValues[elem.Feature] += contribution
	}
}

// extendPath adds a feature to the path and updates weights.
// Returns a new path with the feature added.
func extendPath(path []PathElem, pz, po float64, feature int) []PathElem {
	depth := len(path)

	// Create a new path with one more element
	newPath := make([]PathElem, depth+1)
	copy(newPath, path)

	// Initialize weight: 1.0 if this is the first element, else 0.0
	initialWeight := 0.0
	if depth == 0 {
		initialWeight = 1.0
	}

	// Add new element
	newPath[depth] = PathElem{
		Feature:  feature,
		ZeroFrac: pz,
		OneFrac:  po,
		Weight:   initialWeight,
	}

	// Update weights using the TreeSHAP formula
	for i := depth - 1; i >= 0; i-- {
		newPath[i+1].Weight += po * newPath[i].Weight * float64(i+1) / float64(depth+1)
		newPath[i].Weight = pz * newPath[i].Weight * float64(depth-i) / float64(depth+1)
	}

	return newPath
}

// unwindPath removes a feature from the path at the given position.
// Returns a new path with the feature removed.
//
//nolint:dupl // Similar to unwindInteractionPath() but uses PathElem
func unwindPath(path []PathElem, pathIdx int) []PathElem {
	depth := len(path) - 1
	if depth < 0 {
		return path
	}

	// Create a copy to modify
	newPath := make([]PathElem, len(path))
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

	// Remove the element at pathIdx
	result := make([]PathElem, depth)
	copy(result[:pathIdx], newPath[:pathIdx])
	copy(result[pathIdx:], newPath[pathIdx+1:])

	return result
}

// unwoundPathSum computes the sum of weights with a feature unwound.
// This is used to compute SHAP contributions without modifying the path.
// Matches the R treeshap unwound_sum() implementation exactly.
//
// For a path with a single element (depth=0), this returns 1.0 since
// removing the only element leaves an empty path with unit weight.
func unwoundPathSum(path []PathElem, pathIdx int) float64 {
	depth := len(path) - 1
	if depth < 0 {
		return 0
	}

	// Special case: single-element path returns 1.0
	if depth == 0 {
		return 1.0
	}

	zeroFrac := path[pathIdx].ZeroFrac
	oneFrac := path[pathIdx].OneFrac

	total := 0.0

	if oneFrac != 0 {
		// R treeshap: n = m[depth].w, then loop j from depth-1 to 0
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

	// R treeshap multiplies by (depth+1) at the end
	return total * float64(depth+1)
}

// findFeatureInPath searches for a feature in the path.
// Returns the path index if found, -1 otherwise.
func findFeatureInPath(path []PathElem, feature int) int {
	for i := 0; i < len(path); i++ {
		if path[i].Feature == feature {
			return i
		}
	}
	return -1
}
