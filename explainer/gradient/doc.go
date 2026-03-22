// Package gradient provides GradientSHAP for explaining model predictions.
//
// GradientSHAP (Expected Gradients) computes SHAP values by combining
// ideas from Integrated Gradients and SHAP sampling. It computes gradients
// at interpolated points between the input and background samples.
//
// Algorithm:
//  1. Sample a background reference x' from the background dataset
//  2. Sample α uniformly from [0, 1]
//  3. Compute interpolated point: z = x' + α(x - x')
//  4. Compute gradient ∂f/∂z at the interpolated point
//  5. SHAP value for feature i = E[(x_i - x'_i) * ∂f/∂z_i]
//
// Key properties:
//   - Model-agnostic: works with any differentiable model
//   - Uses numerical gradients (finite differences)
//   - Satisfies local accuracy: sum(SHAP) = prediction - baseline
//   - Lower variance than path methods with sufficient samples
//
// Supported model types:
//   - Any model implementing the model.Model interface
//   - ONNX models via model/onnx.Session
//
// Usage:
//
//	exp, err := gradient.New(model, background,
//	    explainer.WithNumSamples(200),
//	    explainer.WithFeatureNames(names),
//	)
//	explanation, err := exp.Explain(ctx, instance)
//
// Configuration:
//   - NumSamples: Number of (background, alpha) pairs to sample (default: 100)
//   - NoiseStdev: Optional Gaussian noise stddev to add (default: 0)
//   - Epsilon: Step size for numerical gradients (default: 1e-7)
//
// For more information on Expected Gradients:
// https://arxiv.org/abs/1906.10670
package gradient
