#!/usr/bin/env python3
"""
Generate KernelSHAP test cases using the official SHAP library.

This script creates test cases with known-correct SHAP values that can be used
to verify the Go KernelSHAP implementation. Output is JSON for easy parsing in Go.

Usage:
    python generate_kernelshap_test_cases.py > ../kernelshap_test_cases.json
"""

import json
import sys
import numpy as np
import shap


def create_linear_model(weights, bias):
    """Create a simple linear model function."""
    weights = np.array(weights)
    def predict(X):
        return np.dot(X, weights) + bias
    return predict


def create_quadratic_model():
    """Create a model with non-linear interactions: f(x) = x0^2 + 2*x1 + x0*x2"""
    def predict(X):
        return X[:, 0]**2 + 2*X[:, 1] + X[:, 0]*X[:, 2]
    return predict


def generate_linear_model_cases():
    """
    Generate test cases for a simple linear model.

    Model: f(x) = 2*x0 + 3*x1 + 1*x2

    For linear models, KernelSHAP should give exact results matching LinearSHAP.
    """
    weights = [2.0, 3.0, 1.0]
    bias = 0.0
    model = create_linear_model(weights, bias)

    # Background data: centered at [1, 1, 1]
    np.random.seed(42)
    background = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])

    # Create KernelExplainer
    explainer = shap.KernelExplainer(model, background)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.0, 0.0, 0.0], "origin"),
        ([1.0, 1.0, 1.0], "center"),
        ([2.0, 0.0, 1.0], "mixed"),
        ([3.0, 2.0, 1.0], "high"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model(x_arr)[0])
        # Use many samples for accurate estimates
        shap_values = explainer.shap_values(x_arr, nsamples=1000)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "linear_model",
        "description": "Linear model: f(x) = 2*x0 + 3*x1 + 1*x2",
        "model": {
            "type": "linear",
            "weights": weights,
            "bias": bias,
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_quadratic_model_cases():
    """
    Generate test cases for a quadratic model with interactions.

    Model: f(x) = x0^2 + 2*x1 + x0*x2
    """
    model = create_quadratic_model()

    # Background data
    np.random.seed(42)
    background = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.5, 0.5, 0.5],
        [0.5, 1.5, 0.5],
    ])

    explainer = shap.KernelExplainer(model, background)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.0, 0.0, 0.0], "origin"),
        ([1.0, 0.0, 0.0], "x0_only"),
        ([0.0, 1.0, 0.0], "x1_only"),
        ([1.0, 1.0, 1.0], "all_ones"),
        ([2.0, 0.5, 0.5], "high_x0"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=1000)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "quadratic_model",
        "description": "Quadratic model: f(x) = x0^2 + 2*x1 + x0*x2",
        "model": {
            "type": "quadratic",
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_two_feature_cases():
    """
    Generate test cases for a simple 2-feature model.

    Model: f(x) = x0 + x1

    This tests edge cases with minimal features.
    """
    model = create_linear_model([1.0, 1.0], 0.0)

    background = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    explainer = shap.KernelExplainer(model, background)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.0, 0.0], "origin"),
        ([1.0, 0.0], "x0_only"),
        ([0.0, 1.0], "x1_only"),
        ([2.0, 3.0], "mixed"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=500)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "two_features",
        "description": "Simple 2-feature linear model: f(x) = x0 + x1",
        "model": {
            "type": "linear",
            "weights": [1.0, 1.0],
            "bias": 0.0,
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_weighted_model_cases():
    """
    Generate test cases with weighted linear model and larger background.

    Model: f(x) = 0.5*x0 + 2.0*x1 + 0.3*x2 + 1.0*x3 + 5.0
    """
    weights = [0.5, 2.0, 0.3, 1.0]
    bias = 5.0
    model = create_linear_model(weights, bias)

    # Larger background dataset
    np.random.seed(42)
    background = np.random.rand(10, 4) * 2  # Random samples in [0, 2]

    explainer = shap.KernelExplainer(model, background)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.0, 0.0, 0.0, 0.0], "zeros"),
        ([1.0, 1.0, 1.0, 1.0], "ones"),
        ([2.0, 0.5, 1.5, 0.5], "mixed"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=1000)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "weighted_model",
        "description": "Weighted linear model: f(x) = 0.5*x0 + 2.0*x1 + 0.3*x2 + 1.0*x3 + 5.0",
        "model": {
            "type": "linear",
            "weights": weights,
            "bias": bias,
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def main():
    """Generate all test cases and output as JSON."""
    test_suites = [
        generate_linear_model_cases(),
        generate_quadratic_model_cases(),
        generate_two_feature_cases(),
        generate_weighted_model_cases(),
    ]

    output = {
        "version": "1.0",
        "description": "KernelSHAP test cases generated with Python SHAP library",
        "shap_version": shap.__version__,
        "generated_by": "testdata/python/generate_kernelshap_test_cases.py",
        "test_suites": test_suites,
    }

    json.dump(output, sys.stdout, indent=2)
    print()  # Newline at end


if __name__ == "__main__":
    main()
