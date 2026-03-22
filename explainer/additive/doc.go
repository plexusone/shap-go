// Package additive provides an AdditiveExplainer for Generalized Additive Models (GAMs).
//
// AdditiveExplainer computes exact SHAP values for additive models where the prediction
// is a sum of individual feature effects with no interactions:
//
//	f(x) = f₀ + f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ)
//
// For such models, SHAP values have a closed-form solution:
//
//	φᵢ = fᵢ(xᵢ) - E[fᵢ(Xᵢ)]
//
// The SHAP value for feature i equals its contribution at the instance value minus
// its expected contribution over the background distribution.
//
// # Complexity
//
// AdditiveExplainer has O(n × b) complexity where n is the number of features and
// b is the number of background samples. This is much faster than model-agnostic
// methods like KernelSHAP or PermutationSHAP.
//
// # Use Cases
//
//   - Generalized Additive Models (GAMs)
//   - Linear models (though LinearSHAP is more efficient)
//   - Spline-based models
//   - Any model with no feature interactions
//
// # Example
//
//	// Create a GAM-like model
//	model := NewGAMModel(...)
//
//	// Create explainer with background data
//	exp, err := additive.New(model, backgroundData)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Explain an instance
//	explanation, err := exp.Explain(ctx, instance)
//
// # How It Works
//
// The explainer computes individual feature effects by evaluating the model with
// only one feature varying from the reference point. For each feature i:
//
//  1. Compute effect_i(x) = f(ref₀, ..., xᵢ, ..., refₙ) where ref is the mean
//     of the background data
//  2. Compute the mean effect over background: E[effect_i] = mean(effect_i(bg_samples))
//  3. SHAP value: φᵢ = effect_i(instance) - E[effect_i]
//
// This approach works for any model that is truly additive. For non-additive models,
// the results will not be accurate SHAP values - use KernelSHAP or other methods instead.
package additive
