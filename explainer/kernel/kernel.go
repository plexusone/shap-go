// Package kernel provides a KernelSHAP explainer using weighted linear regression.
//
// KernelSHAP is a model-agnostic method that approximates SHAP values by:
// 1. Sampling random feature coalitions (subsets)
// 2. Weighting each coalition using the Shapley kernel
// 3. Fitting a weighted linear regression to estimate feature contributions
//
// This implementation is based on the Python SHAP library's KernelExplainer:
// https://github.com/shap/shap
//
// Reference: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
package kernel

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/plexusone/shap-go/explainer"
	"github.com/plexusone/shap-go/explanation"
	"github.com/plexusone/shap-go/model"
)

// Explainer implements KernelSHAP using weighted linear regression.
//
// The algorithm works by:
//  1. Sampling binary coalition vectors z ∈ {0,1}^M where M is the number of features
//  2. For each coalition, computing the prediction when features in z are from the
//     instance and others are from the background
//  3. Weighting each sample using the Shapley kernel weight
//  4. Solving weighted least squares with the constraint that SHAP values sum to
//     (prediction - baseline)
type Explainer struct {
	model        model.Model
	background   [][]float64
	featureNames []string
	baseValue    float64
	config       explainer.Config
	rng          *rand.Rand
	mu           sync.Mutex // protects rng
}

// New creates a new KernelSHAP explainer.
//
// Parameters:
//   - m: The model to explain (implements model.Model interface)
//   - background: Representative samples for baseline/masking
//   - opts: Configuration options (WithNumSamples, WithSeed, etc.)
//
// The background data defines the "baseline" prediction and is used for
// masking features when computing marginal contributions.
func New(m model.Model, background [][]float64, opts ...explainer.Option) (*Explainer, error) {
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

	config := explainer.DefaultConfig()
	explainer.ApplyOptions(&config, opts...)
	config.Validate(numFeatures)

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
		featureNames: config.FeatureNames,
		baseValue:    baseValue,
		config:       config,
		rng:          config.GetRNG(),
	}, nil
}

// Explain computes SHAP values for a single instance using KernelSHAP.
func (e *Explainer) Explain(ctx context.Context, instance []float64) (*explanation.Explanation, error) {
	startTime := time.Now()

	numFeatures := e.model.NumFeatures()
	if len(instance) != numFeatures {
		return nil, fmt.Errorf("instance has %d features, expected %d",
			len(instance), numFeatures)
	}

	// Get the prediction for the instance
	prediction, err := e.model.Predict(ctx, instance)
	if err != nil {
		return nil, fmt.Errorf("failed to predict instance: %w", err)
	}

	// Compute SHAP values using KernelSHAP algorithm
	var shapValues []float64
	if e.config.NumWorkers > 1 {
		shapValues, err = e.computeSHAPValuesParallel(ctx, instance, prediction)
	} else {
		shapValues, err = e.computeSHAPValues(ctx, instance, prediction)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to compute SHAP values: %w", err)
	}

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
			Algorithm:      "kernel",
			NumSamples:     e.config.NumSamples,
			BackgroundSize: len(e.background),
			ComputeTimeMS:  time.Since(startTime).Milliseconds(),
		},
	}

	return exp, nil
}

// ExplainBatch computes SHAP explanations for multiple instances.
func (e *Explainer) ExplainBatch(ctx context.Context, instances [][]float64) ([]*explanation.Explanation, error) {
	results := make([]*explanation.Explanation, len(instances))
	for i, inst := range instances {
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

// coalitionData holds the data for a single coalition sample.
type coalitionData struct {
	mask   []float64 // binary mask indicating which features are from instance
	weight float64   // Shapley kernel weight
	pred   float64   // model prediction for this coalition
}

// computeSHAPValues computes SHAP values using the KernelSHAP algorithm.
//
// Following Python SHAP's approach:
// 1. Use two-phase sampling: enumerate small coalitions exactly, then sample randomly
// 2. Add both coalition and its complement (variance reduction)
// 3. Solve constrained weighted least squares
func (e *Explainer) computeSHAPValues(ctx context.Context, instance []float64, prediction float64) ([]float64, error) {
	numFeatures := len(instance)

	// Special case: single feature
	// For single feature, SHAP value equals prediction - baseline by definition
	if numFeatures == 1 {
		return []float64{prediction - e.baseValue}, nil
	}

	e.mu.Lock()
	rng := e.rng
	e.mu.Unlock()

	// Collect coalition samples
	samples, err := e.sampleCoalitions(ctx, instance, rng)
	if err != nil {
		return nil, err
	}

	// Build matrices for weighted least squares
	// We solve: minimize Σ w_i * (y_i - φ₀ - Σⱼ x_ij * φⱼ)²
	// Subject to: Σⱼ φⱼ = prediction - baseValue
	return e.solveConstrainedWLS(samples, prediction, numFeatures)
}

// computeSHAPValuesParallel computes SHAP values using parallel workers.
func (e *Explainer) computeSHAPValuesParallel(ctx context.Context, instance []float64, prediction float64) ([]float64, error) {
	numFeatures := len(instance)

	// Special case: single feature
	if numFeatures == 1 {
		return []float64{prediction - e.baseValue}, nil
	}

	numWorkers := e.config.NumWorkers
	numSamples := e.config.NumSamples

	// For parallel, we skip enumeration and do random sampling only
	// Distribute samples across workers
	samplesPerWorker := numSamples / numWorkers
	extraSamples := numSamples % numWorkers

	type workerResult struct {
		samples []coalitionData
		err     error
	}

	results := make(chan workerResult, numWorkers)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		workerSamples := samplesPerWorker
		if w < extraSamples {
			workerSamples++
		}

		e.mu.Lock()
		workerSeed := e.rng.Int63()
		e.mu.Unlock()

		go func(nSamples int, seed int64) {
			defer wg.Done()

			workerRNG := rand.New(rand.NewSource(seed)) //nolint:gosec // seeded for reproducibility
			samples := make([]coalitionData, 0, nSamples*2)

			for i := 0; i < nSamples; i++ {
				select {
				case <-ctx.Done():
					results <- workerResult{nil, ctx.Err()}
					return
				default:
				}

				// Sample coalition and its complement
				coalitionSamples, err := e.sampleCoalitionPair(ctx, instance, numFeatures, workerRNG)
				if err != nil {
					results <- workerResult{nil, err}
					return
				}
				samples = append(samples, coalitionSamples...)
			}

			results <- workerResult{samples, nil}
		}(workerSamples, workerSeed)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Aggregate results
	allSamples := make([]coalitionData, 0, numSamples*2)
	for res := range results {
		if res.err != nil {
			return nil, res.err
		}
		allSamples = append(allSamples, res.samples...)
	}

	return e.solveConstrainedWLS(allSamples, prediction, numFeatures)
}

// sampleCoalitions generates coalition samples using two-phase approach.
// Phase 1: Enumerate all coalitions of sizes that fit in budget
// Phase 2: Random sampling for remaining budget
func (e *Explainer) sampleCoalitions(ctx context.Context, instance []float64, rng *rand.Rand) ([]coalitionData, error) {
	numFeatures := len(instance)
	numSamples := e.config.NumSamples

	samples := make([]coalitionData, 0, numSamples*2)

	// Phase 1: Complete enumeration for small coalition sizes
	// We enumerate pairs (coalition, complement) starting from size 1 and M-1
	budget := numSamples
	enumeratedSizes := make(map[int]bool)

	for size := 1; size <= numFeatures/2 && budget > 0; size++ {
		numCombinations := int(binomialCoefficient(numFeatures, size))

		// Each combination gives us 2 samples (coalition + complement)
		// except when size == numFeatures/2 and numFeatures is even
		samplesNeeded := numCombinations
		if size != numFeatures-size {
			samplesNeeded *= 2
		}

		if samplesNeeded <= budget {
			// Enumerate all coalitions of this size
			enumSamples, err := e.enumerateCoalitions(ctx, instance, size, numFeatures)
			if err != nil {
				return nil, err
			}
			samples = append(samples, enumSamples...)
			budget -= len(enumSamples)
			enumeratedSizes[size] = true
			enumeratedSizes[numFeatures-size] = true
		} else {
			break
		}
	}

	// Phase 2: Random sampling for remaining budget
	for budget > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Sample a coalition size that wasn't fully enumerated
		size := e.sampleCoalitionSize(numFeatures, enumeratedSizes, rng)
		if size == 0 || size == numFeatures {
			continue
		}

		pair, err := e.sampleCoalitionPairOfSize(ctx, instance, numFeatures, size, rng)
		if err != nil {
			return nil, err
		}
		samples = append(samples, pair...)
		budget -= len(pair)
	}

	return samples, nil
}

// enumerateCoalitions generates all coalitions of a given size and their complements.
func (e *Explainer) enumerateCoalitions(ctx context.Context, instance []float64, size, numFeatures int) ([]coalitionData, error) {
	// Use batched evaluation if enabled
	if e.config.UseBatchedPredictions {
		return e.enumerateCoalitionsBatched(ctx, instance, size, numFeatures)
	}

	samples := make([]coalitionData, 0)
	weight := shapleyKernelWeight(numFeatures, size)
	complementWeight := shapleyKernelWeight(numFeatures, numFeatures-size)

	// Generate all combinations of 'size' features
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Create coalition mask
		mask := make([]float64, numFeatures)
		for _, idx := range indices {
			mask[idx] = 1.0
		}

		// Evaluate coalition
		pred, err := e.evaluateCoalition(ctx, instance, mask)
		if err != nil {
			return nil, err
		}
		samples = append(samples, coalitionData{mask: mask, weight: weight, pred: pred})

		// Add complement (if different from coalition)
		if size != numFeatures-size {
			complementMask := make([]float64, numFeatures)
			for i := range complementMask {
				complementMask[i] = 1.0 - mask[i]
			}
			compPred, err := e.evaluateCoalition(ctx, instance, complementMask)
			if err != nil {
				return nil, err
			}
			samples = append(samples, coalitionData{mask: complementMask, weight: complementWeight, pred: compPred})
		}

		// Generate next combination
		if !nextCombination(indices, numFeatures) {
			break
		}
	}

	return samples, nil
}

// enumerateCoalitionsBatched generates all coalitions using batched predictions.
// This collects all masks first, then evaluates them in a single batch call.
func (e *Explainer) enumerateCoalitionsBatched(ctx context.Context, instance []float64, size, numFeatures int) ([]coalitionData, error) {
	weight := shapleyKernelWeight(numFeatures, size)
	complementWeight := shapleyKernelWeight(numFeatures, numFeatures-size)
	includeComplement := size != numFeatures-size

	// First pass: collect all masks
	var masks [][]float64
	var weights []float64
	var isComplement []bool

	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Create coalition mask
		mask := make([]float64, numFeatures)
		for _, idx := range indices {
			mask[idx] = 1.0
		}
		masks = append(masks, mask)
		weights = append(weights, weight)
		isComplement = append(isComplement, false)

		// Add complement (if different from coalition)
		if includeComplement {
			complementMask := make([]float64, numFeatures)
			for i := range complementMask {
				complementMask[i] = 1.0 - mask[i]
			}
			masks = append(masks, complementMask)
			weights = append(weights, complementWeight)
			isComplement = append(isComplement, true)
		}

		// Generate next combination
		if !nextCombination(indices, numFeatures) {
			break
		}
	}

	// Second pass: evaluate all coalitions in batch
	predictions, err := e.evaluateCoalitionsBatched(ctx, instance, masks)
	if err != nil {
		return nil, err
	}

	// Build result
	samples := make([]coalitionData, len(masks))
	for i := range masks {
		samples[i] = coalitionData{
			mask:   masks[i],
			weight: weights[i],
			pred:   predictions[i],
		}
	}

	return samples, nil
}

// nextCombination generates the next combination in lexicographic order.
// Returns false if there are no more combinations.
func nextCombination(indices []int, n int) bool {
	k := len(indices)
	for i := k - 1; i >= 0; i-- {
		if indices[i] < n-k+i {
			indices[i]++
			for j := i + 1; j < k; j++ {
				indices[j] = indices[j-1] + 1
			}
			return true
		}
	}
	return false
}

// sampleCoalitionPair samples a random coalition and its complement.
func (e *Explainer) sampleCoalitionPair(ctx context.Context, instance []float64, numFeatures int, rng *rand.Rand) ([]coalitionData, error) {
	// Sample a coalition size (excluding 0 and M)
	size := 1 + rng.Intn(numFeatures-1)
	return e.sampleCoalitionPairOfSize(ctx, instance, numFeatures, size, rng)
}

// sampleCoalitionPairOfSize samples a coalition of specific size and its complement.
func (e *Explainer) sampleCoalitionPairOfSize(ctx context.Context, instance []float64, numFeatures, size int, rng *rand.Rand) ([]coalitionData, error) {
	// Randomly select 'size' features
	perm := make([]int, numFeatures)
	for i := range perm {
		perm[i] = i
	}
	rng.Shuffle(numFeatures, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})

	// Create coalition mask
	mask := make([]float64, numFeatures)
	for i := 0; i < size; i++ {
		mask[perm[i]] = 1.0
	}

	weight := shapleyKernelWeight(numFeatures, size)

	// Evaluate coalition
	pred, err := e.evaluateCoalition(ctx, instance, mask)
	if err != nil {
		return nil, err
	}

	samples := []coalitionData{{mask: mask, weight: weight, pred: pred}}

	// Add complement
	complementMask := make([]float64, numFeatures)
	for i := range complementMask {
		complementMask[i] = 1.0 - mask[i]
	}
	complementWeight := shapleyKernelWeight(numFeatures, numFeatures-size)

	compPred, err := e.evaluateCoalition(ctx, instance, complementMask)
	if err != nil {
		return nil, err
	}
	samples = append(samples, coalitionData{mask: complementMask, weight: complementWeight, pred: compPred})

	return samples, nil
}

// sampleCoalitionSize samples a coalition size, avoiding fully enumerated sizes.
func (e *Explainer) sampleCoalitionSize(numFeatures int, enumerated map[int]bool, rng *rand.Rand) int {
	// Compute weights for non-enumerated sizes
	weights := make([]float64, numFeatures+1)
	totalWeight := 0.0

	for size := 1; size < numFeatures; size++ {
		if !enumerated[size] {
			w := shapleyKernelWeight(numFeatures, size)
			weights[size] = w
			totalWeight += w
		}
	}

	if totalWeight == 0 {
		// All sizes enumerated, just pick randomly
		return 1 + rng.Intn(numFeatures-1)
	}

	// Sample proportionally to weights
	r := rng.Float64() * totalWeight
	cumulative := 0.0
	for size := 1; size < numFeatures; size++ {
		cumulative += weights[size]
		if r <= cumulative {
			return size
		}
	}

	return numFeatures - 1
}

// evaluateCoalition computes the model prediction for a coalition.
// Uses the mean prediction over background samples for masked features.
func (e *Explainer) evaluateCoalition(ctx context.Context, instance []float64, mask []float64) (float64, error) {
	// Use batched predictions if enabled
	if e.config.UseBatchedPredictions {
		return e.evaluateCoalitionBatched(ctx, instance, mask)
	}

	numFeatures := len(instance)

	// Average prediction over all background samples
	totalPred := 0.0
	for _, bgSample := range e.background {
		input := make([]float64, numFeatures)
		for i := 0; i < numFeatures; i++ {
			if mask[i] > 0.5 {
				input[i] = instance[i]
			} else {
				input[i] = bgSample[i]
			}
		}

		pred, err := e.model.Predict(ctx, input)
		if err != nil {
			return 0, err
		}
		totalPred += pred
	}

	return totalPred / float64(len(e.background)), nil
}

// evaluateCoalitionBatched computes coalition prediction using batched model inference.
// This is more efficient when the model has optimized batch prediction.
func (e *Explainer) evaluateCoalitionBatched(ctx context.Context, instance []float64, mask []float64) (float64, error) {
	numFeatures := len(instance)
	numBackground := len(e.background)

	// Build all masked inputs at once
	inputs := make([][]float64, numBackground)
	for b, bgSample := range e.background {
		input := make([]float64, numFeatures)
		for i := 0; i < numFeatures; i++ {
			if mask[i] > 0.5 {
				input[i] = instance[i]
			} else {
				input[i] = bgSample[i]
			}
		}
		inputs[b] = input
	}

	// Batch prediction
	predictions, err := e.model.PredictBatch(ctx, inputs)
	if err != nil {
		return 0, err
	}

	// Average predictions
	totalPred := 0.0
	for _, pred := range predictions {
		totalPred += pred
	}

	return totalPred / float64(numBackground), nil
}

// evaluateCoalitionsBatched evaluates multiple coalitions in a single batch.
// Returns the average prediction for each coalition (one per mask).
// This is the most efficient method when evaluating many coalitions.
func (e *Explainer) evaluateCoalitionsBatched(ctx context.Context, instance []float64, masks [][]float64) ([]float64, error) {
	numFeatures := len(instance)
	numBackground := len(e.background)
	numCoalitions := len(masks)

	// Total inputs: numCoalitions * numBackground
	totalInputs := numCoalitions * numBackground
	inputs := make([][]float64, totalInputs)

	// Build all inputs for all coalitions and all background samples
	idx := 0
	for _, mask := range masks {
		for _, bgSample := range e.background {
			input := make([]float64, numFeatures)
			for i := 0; i < numFeatures; i++ {
				if mask[i] > 0.5 {
					input[i] = instance[i]
				} else {
					input[i] = bgSample[i]
				}
			}
			inputs[idx] = input
			idx++
		}
	}

	// Single batch prediction for all
	predictions, err := e.model.PredictBatch(ctx, inputs)
	if err != nil {
		return nil, err
	}

	// Aggregate predictions by coalition
	results := make([]float64, numCoalitions)
	for c := 0; c < numCoalitions; c++ {
		start := c * numBackground
		totalPred := 0.0
		for b := 0; b < numBackground; b++ {
			totalPred += predictions[start+b]
		}
		results[c] = totalPred / float64(numBackground)
	}

	return results, nil
}

// shapleyKernelWeight computes the Shapley kernel weight for a coalition of given size.
//
// w(|S|) = (M - 1) / (|S| * (M - |S|))
//
// where M is the total number of features and |S| is the coalition size.
func shapleyKernelWeight(numFeatures, coalitionSize int) float64 {
	if coalitionSize == 0 || coalitionSize == numFeatures {
		return 0 // These coalitions provide no information
	}

	M := float64(numFeatures)
	s := float64(coalitionSize)
	return (M - 1) / (s * (M - s))
}

// binomialCoefficient computes C(n, k) = n! / (k! * (n-k)!)
func binomialCoefficient(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	result := 1.0
	for i := 0; i < k; i++ {
		result *= float64(n-i) / float64(i+1)
	}
	return result
}

// solveConstrainedWLS solves weighted least squares with the constraint
// that SHAP values sum to (prediction - baseValue).
//
// We solve:
//
//	minimize Σᵢ wᵢ * (yᵢ - φ₀ - Σⱼ zᵢⱼ * φⱼ)²
//	subject to: Σⱼ φⱼ = prediction - baseValue
//
// Following Python SHAP, we eliminate one variable using the constraint.
// Let φₘ = (prediction - baseValue) - Σⱼ₌₁^(M-1) φⱼ
// Then substitute and solve the unconstrained problem for φ₁...φₘ₋₁
func (e *Explainer) solveConstrainedWLS(samples []coalitionData, prediction float64, numFeatures int) ([]float64, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("no samples provided")
	}

	numSamples := len(samples)
	targetSum := prediction - e.baseValue

	// Special case: 1 feature
	if numFeatures == 1 {
		return []float64{targetSum}, nil
	}

	// Build the modified design matrix and response vector
	// We eliminate the last feature using the constraint:
	// φₘ = targetSum - Σⱼ₌₁^(M-1) φⱼ
	//
	// Original model: y = φ₀ + Σⱼ zⱼφⱼ
	// After substitution: y = φ₀ + Σⱼ₌₁^(M-1) zⱼφⱼ + zₘ(targetSum - Σⱼ₌₁^(M-1) φⱼ)
	//                     y - φ₀ - zₘ*targetSum = Σⱼ₌₁^(M-1) (zⱼ - zₘ)φⱼ
	//
	// Let y' = y - φ₀ - zₘ*targetSum
	// Let x'ⱼ = zⱼ - zₘ
	// Then: y' = Σⱼ x'ⱼ φⱼ

	lastFeature := numFeatures - 1

	// Build X' and y' for the reduced problem
	Xprime := make([][]float64, numSamples)
	yprime := make([]float64, numSamples)
	weights := make([]float64, numSamples)

	for i, sample := range samples {
		Xprime[i] = make([]float64, numFeatures-1)
		zLast := sample.mask[lastFeature]

		for j := 0; j < numFeatures-1; j++ {
			Xprime[i][j] = sample.mask[j] - zLast
		}

		yprime[i] = sample.pred - e.baseValue - zLast*targetSum
		weights[i] = sample.weight
	}

	// Solve weighted least squares: (X'WX)φ = X'Wy
	reducedSHAP, err := solveWLS(Xprime, yprime, weights)
	if err != nil {
		return nil, err
	}

	// Reconstruct φₘ from constraint
	shapValues := make([]float64, numFeatures)
	sum := 0.0
	for j := 0; j < numFeatures-1; j++ {
		shapValues[j] = reducedSHAP[j]
		sum += reducedSHAP[j]
	}
	shapValues[lastFeature] = targetSum - sum

	return shapValues, nil
}

// solveWLS solves weighted least squares: minimize Σᵢ wᵢ(yᵢ - Xᵢβ)²
// Returns β = (X'WX)⁻¹ X'Wy
func solveWLS(X [][]float64, y []float64, w []float64) ([]float64, error) {
	numSamples := len(X)
	if numSamples == 0 {
		return nil, fmt.Errorf("no samples")
	}
	numFeatures := len(X[0])
	if numFeatures == 0 {
		return []float64{}, nil
	}

	// Compute X'WX
	XtWX := make([][]float64, numFeatures)
	for i := range XtWX {
		XtWX[i] = make([]float64, numFeatures)
	}

	for i := 0; i < numFeatures; i++ {
		for j := 0; j < numFeatures; j++ {
			sum := 0.0
			for k := 0; k < numSamples; k++ {
				sum += X[k][i] * w[k] * X[k][j]
			}
			XtWX[i][j] = sum
		}
	}

	// Compute X'Wy
	XtWy := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		sum := 0.0
		for k := 0; k < numSamples; k++ {
			sum += X[k][i] * w[k] * y[k]
		}
		XtWy[i] = sum
	}

	// Add small regularization for numerical stability
	lambda := 1e-10
	for i := 0; i < numFeatures; i++ {
		XtWX[i][i] += lambda
	}

	// Solve using Gaussian elimination with partial pivoting
	return solveLinearSystem(XtWX, XtWy)
}

// solveLinearSystem solves Ax = b using Gaussian elimination with partial pivoting.
func solveLinearSystem(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	if n == 0 {
		return []float64{}, nil
	}

	// Create augmented matrix [A|b]
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, n+1)
		copy(aug[i], A[i])
		aug[i][n] = b[i]
	}

	// Forward elimination with partial pivoting
	for col := 0; col < n; col++ {
		// Find pivot
		maxRow := col
		maxVal := math.Abs(aug[col][col])
		for row := col + 1; row < n; row++ {
			if absVal := math.Abs(aug[row][col]); absVal > maxVal {
				maxVal = absVal
				maxRow = row
			}
		}

		// Swap rows
		if maxRow != col {
			aug[col], aug[maxRow] = aug[maxRow], aug[col]
		}

		// Check for singular matrix
		if math.Abs(aug[col][col]) < 1e-12 {
			continue
		}

		// Eliminate column
		for row := col + 1; row < n; row++ {
			factor := aug[row][col] / aug[col][col]
			for j := col; j <= n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Back substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		if math.Abs(aug[i][i]) < 1e-12 {
			x[i] = 0
			continue
		}
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
		x[i] /= aug[i][i]
	}

	return x, nil
}

// Ensure Explainer implements explainer.Explainer.
var _ explainer.Explainer = (*Explainer)(nil)
