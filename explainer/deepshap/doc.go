// Package deepshap provides DeepSHAP for explaining deep neural network predictions.
//
// DeepSHAP is an efficient method for computing SHAP (SHapley Additive exPlanations)
// values for neural networks. It combines the DeepLIFT algorithm with Shapley values
// to provide theoretically grounded feature attributions.
//
// # Overview
//
// DeepSHAP works by:
//  1. Running a forward pass to capture activations at each layer
//  2. Computing reference activations from a background dataset
//  3. Propagating attribution multipliers backward using the DeepLIFT rescale rule
//  4. Averaging attributions over multiple background samples
//
// # Key Properties
//
// DeepSHAP inherits SHAP's desirable properties:
//   - Local accuracy: sum(SHAP values) = prediction - baseline
//   - Consistency: if a feature's contribution increases, its SHAP value increases
//   - Missingness: features not present receive zero attribution
//
// # Supported Layers
//
// The implementation supports common neural network layer types:
//   - Dense/Gemm: fully connected layers
//   - ReLU: rectified linear unit activation
//   - Sigmoid: sigmoid activation
//   - Tanh: hyperbolic tangent activation
//   - Softmax: softmax output layer
//   - Add: element-wise addition (for residual connections)
//   - Identity: pass-through (Dropout, Flatten, etc.)
//
// # Usage
//
// Basic usage with an ONNX model:
//
//	// Parse the ONNX model graph
//	graphInfo, err := onnx.ParseGraph("model.onnx")
//
//	// Create activation session with intermediate outputs
//	config := onnx.ActivationConfig{
//	    Config: onnx.Config{
//	        ModelPath:   "model.onnx",
//	        NumFeatures: 10,
//	    },
//	    IntermediateOutputs: graphInfo.GetAllLayerOutputs(),
//	}
//	session, err := onnx.NewActivationSession(config)
//
//	// Create DeepSHAP explainer
//	explainer, err := deepshap.New(session, graphInfo, background)
//
//	// Explain a prediction
//	explanation, err := explainer.Explain(ctx, instance)
//
// # Simplified Usage
//
// For cases where you don't need full graph structure:
//
//	explainer, err := deepshap.NewSimple(session, background)
//	explanation, err := explainer.Explain(ctx, instance)
//
// This uses a simpler attribution method that doesn't require graph parsing.
//
// # Background Dataset
//
// The background dataset serves as the baseline for attribution. Good practices:
//   - Use representative samples from your training data
//   - 100-1000 samples typically provides good results
//   - More samples improve accuracy but increase computation time
//   - Consider using background.Dataset.KMeansSummary() for large datasets
//
// # Attribution Rules
//
// DeepSHAP uses the DeepLIFT rescale rule for propagating attributions:
//
//	mult_in = mult_out × (x - x_ref) / (y - y_ref)
//
// where:
//   - mult_in/mult_out are input/output multipliers
//   - x, x_ref are input activations for instance/reference
//   - y, y_ref are output activations for instance/reference
//
// When the denominator is near zero, the rule falls back to gradient computation.
//
// # References
//
//   - SHAP paper: https://arxiv.org/abs/1705.07874
//   - DeepLIFT paper: https://arxiv.org/abs/1704.02685
//   - Python SHAP library: https://github.com/slundberg/shap
package deepshap
