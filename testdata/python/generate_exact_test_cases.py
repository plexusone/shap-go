#!/usr/bin/env python3
"""
Generate ExactSHAP test cases using the official SHAP library.

This script creates test cases with known-correct exact SHAP values that can be
used to verify the Go ExactSHAP implementation. Output is JSON for easy parsing in Go.

For ExactSHAP, we use Python SHAP's Exact explainer which computes true Shapley
values by enumerating all possible coalitions.

Usage:
    python generate_exact_test_cases.py > ../exactshap_test_cases.json
"""

import json
import sys
import numpy as np
import shap


def linear_model(x):
    """Simple linear model: y = 1*x0 + 2*x1 + 3*x2"""
    return np.sum(x * np.array([1.0, 2.0, 3.0]), axis=-1)


def weighted_model(x):
    """Weighted linear model with bias: y = 5 + 2*x0 + 3*x1"""
    return 5.0 + 2.0 * x[..., 0] + 3.0 * x[..., 1]


def quadratic_model(x):
    """Quadratic interaction model: y = x0 * x1"""
    return x[..., 0] * x[..., 1]


def generate_linear_model_cases():
    """
    Generate test cases for a simple linear model.

    Model: y = 1*x0 + 2*x1 + 3*x2
    Background: all zeros
    """
    background = np.array([[0.0, 0.0, 0.0]])

    explainer = shap.Explainer(linear_model, background)

    test_instances = [
        ([1.0, 1.0, 1.0], "all_ones"),
        ([1.0, 2.0, 3.0], "increasing"),
        ([0.5, 0.5, 0.5], "half"),
        ([2.0, 0.0, 1.0], "sparse"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(linear_model(x_arr)[0])
        shap_values = explainer(x_arr)
        base_value = float(shap_values.base_values[0])
        values = shap_values.values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": values,
        })

    # Get base value from first explanation
    base_value = float(explainer(np.array([[0.0, 0.0, 0.0]])).base_values[0])

    return {
        "name": "linear_model",
        "description": "Linear model: y = 1*x0 + 2*x1 + 3*x2 with zero background",
        "model": "linear",
        "n_features": 3,
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_weighted_model_cases():
    """
    Generate test cases for a weighted model with bias.

    Model: y = 5 + 2*x0 + 3*x1
    Background: mean at [1, 1]
    """
    background = np.array([
        [0.0, 0.0],
        [2.0, 2.0],
    ])
    # Mean = [1, 1], base_value = 5 + 2*1 + 3*1 = 10

    explainer = shap.Explainer(weighted_model, background)

    test_instances = [
        ([1.0, 1.0], "at_mean"),
        ([2.0, 3.0], "above_mean"),
        ([0.0, 0.0], "at_zero"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(weighted_model(x_arr)[0])
        shap_values = explainer(x_arr)
        values = shap_values.values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": values,
        })

    base_value = float(explainer(np.array([[1.0, 1.0]])).base_values[0])

    return {
        "name": "weighted_model",
        "description": "Weighted model: y = 5 + 2*x0 + 3*x1 with background mean at [1, 1]",
        "model": "weighted_linear",
        "n_features": 2,
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_quadratic_model_cases():
    """
    Generate test cases for a nonlinear (quadratic) model.

    Model: y = x0 * x1
    Background: all zeros
    """
    background = np.array([[0.0, 0.0]])

    explainer = shap.Explainer(quadratic_model, background)

    test_instances = [
        ([2.0, 3.0], "positive"),
        ([3.0, 4.0], "larger"),
        ([1.0, 1.0], "ones"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(quadratic_model(x_arr)[0])
        shap_values = explainer(x_arr)
        values = shap_values.values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": values,
        })

    base_value = float(explainer(np.array([[0.0, 0.0]])).base_values[0])

    return {
        "name": "quadratic_model",
        "description": "Quadratic model: y = x0 * x1 with zero background",
        "model": "quadratic",
        "n_features": 2,
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_four_feature_cases():
    """
    Generate test cases for a 4-feature linear model.

    Model: y = 1*x0 + 2*x1 + 3*x2 + 4*x3
    Background: all zeros
    """
    def four_feature_model(x):
        return np.sum(x * np.array([1.0, 2.0, 3.0, 4.0]), axis=-1)

    background = np.array([[0.0, 0.0, 0.0, 0.0]])

    explainer = shap.Explainer(four_feature_model, background)

    test_instances = [
        ([1.0, 1.0, 1.0, 1.0], "all_ones"),
        ([1.0, 2.0, 3.0, 4.0], "increasing"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(four_feature_model(x_arr)[0])
        shap_values = explainer(x_arr)
        values = shap_values.values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": values,
        })

    base_value = float(explainer(background).base_values[0])

    return {
        "name": "four_feature_model",
        "description": "Four-feature linear model: y = 1*x0 + 2*x1 + 3*x2 + 4*x3",
        "model": "linear",
        "n_features": 4,
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def main():
    """Generate all test cases and output as JSON."""
    test_suites = [
        generate_linear_model_cases(),
        generate_weighted_model_cases(),
        generate_quadratic_model_cases(),
        generate_four_feature_cases(),
    ]

    output = {
        "version": "1.0",
        "description": "ExactSHAP test cases generated with Python SHAP library",
        "shap_version": shap.__version__,
        "generated_by": "testdata/python/generate_exact_test_cases.py",
        "test_suites": test_suites,
    }

    json.dump(output, sys.stdout, indent=2)
    print()  # Newline at end


if __name__ == "__main__":
    main()
