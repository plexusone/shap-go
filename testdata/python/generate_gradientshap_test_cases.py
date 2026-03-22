#!/usr/bin/env python3
"""
Generate GradientSHAP test cases using the official SHAP library.

This script creates test cases with known-correct SHAP values that can be used
to verify the Go GradientSHAP implementation. Output is JSON for easy parsing in Go.

Note: Python SHAP's GradientExplainer requires a differentiable model (typically
a neural network). For testing, we use simple scikit-learn neural networks.

Usage:
    python generate_gradientshap_test_cases.py > ../gradientshap_test_cases.json
"""

import json
import sys
import numpy as np

# Check if required packages are available
try:
    import shap
    from sklearn.neural_network import MLPRegressor
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    IMPORT_ERROR = str(e)


def create_simple_mlp(hidden_layers=(10,), random_state=42):
    """Create a simple MLP regressor for testing."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=random_state,
    )


def generate_linear_behavior_cases():
    """
    Generate test cases using an MLP that approximates a linear function.

    We train on f(x) = 2*x0 + 3*x1 + x2 and verify GradientSHAP captures
    the linear relationship.
    """
    np.random.seed(42)

    # Training data for linear function
    X_train = np.random.rand(200, 3) * 2 - 1  # [-1, 1]
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + X_train[:, 2]

    # Train MLP
    model = create_simple_mlp(hidden_layers=(20, 10), random_state=42)
    model.fit(X_train, y_train)

    # Background data
    background = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
    ])

    # Create GradientExplainer
    explainer = shap.GradientExplainer(model.predict, background)
    base_value = float(np.mean(model.predict(background)))

    test_instances = [
        ([0.0, 0.0, 0.0], "origin"),
        ([1.0, 0.0, 0.0], "x0_only"),
        ([0.0, 1.0, 0.0], "x1_only"),
        ([0.0, 0.0, 1.0], "x2_only"),
        ([1.0, 1.0, 1.0], "all_ones"),
        ([0.5, 0.5, 0.5], "all_half"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        # Use more samples for stable estimates
        shap_values = explainer.shap_values(x_arr, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "linear_behavior_mlp",
        "description": "MLP trained on linear function: f(x) = 2*x0 + 3*x1 + x2",
        "model": {
            "type": "mlp",
            "target_function": "2*x0 + 3*x1 + x2",
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_quadratic_cases():
    """
    Generate test cases using an MLP trained on quadratic function.

    Model: f(x) = x0^2 + 2*x1 + x0*x2
    """
    np.random.seed(42)

    # Training data
    X_train = np.random.rand(300, 3) * 2  # [0, 2]
    y_train = X_train[:, 0]**2 + 2 * X_train[:, 1] + X_train[:, 0] * X_train[:, 2]

    # Train MLP
    model = create_simple_mlp(hidden_layers=(30, 20), random_state=42)
    model.fit(X_train, y_train)

    # Background data
    background = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

    explainer = shap.GradientExplainer(model.predict, background)
    base_value = float(np.mean(model.predict(background)))

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
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "quadratic_mlp",
        "description": "MLP trained on quadratic function: f(x) = x0^2 + 2*x1 + x0*x2",
        "model": {
            "type": "mlp",
            "target_function": "x0^2 + 2*x1 + x0*x2",
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_two_feature_cases():
    """
    Generate test cases with a simple 2-feature model.

    Model: f(x) = x0 + x1
    """
    np.random.seed(42)

    X_train = np.random.rand(100, 2) * 2 - 1
    y_train = X_train[:, 0] + X_train[:, 1]

    model = create_simple_mlp(hidden_layers=(10,), random_state=42)
    model.fit(X_train, y_train)

    background = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [-0.5, -0.5],
    ])

    explainer = shap.GradientExplainer(model.predict, background)
    base_value = float(np.mean(model.predict(background)))

    test_instances = [
        ([0.0, 0.0], "origin"),
        ([1.0, 0.0], "x0_only"),
        ([0.0, 1.0], "x1_only"),
        ([0.5, 0.5], "half_both"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "two_features",
        "description": "Simple 2-feature MLP: f(x) = x0 + x1",
        "model": {
            "type": "mlp",
            "target_function": "x0 + x1",
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def generate_sinusoidal_cases():
    """
    Generate test cases with sinusoidal components.

    Model: f(x) = sin(x0) + 2*x1
    """
    np.random.seed(42)

    X_train = np.random.rand(300, 2) * 4 - 2  # [-2, 2]
    y_train = np.sin(X_train[:, 0]) + 2 * X_train[:, 1]

    model = create_simple_mlp(hidden_layers=(30, 20), random_state=42)
    model.fit(X_train, y_train)

    background = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])

    explainer = shap.GradientExplainer(model.predict, background)
    base_value = float(np.mean(model.predict(background)))

    test_instances = [
        ([0.0, 0.0], "origin"),
        ([1.57, 0.0], "sin_peak"),  # pi/2
        ([0.0, 1.0], "x1_only"),
        ([1.0, 1.0], "mixed"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = shap_values[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    return {
        "name": "sinusoidal",
        "description": "MLP with sinusoidal component: f(x) = sin(x0) + 2*x1",
        "model": {
            "type": "mlp",
            "target_function": "sin(x0) + 2*x1",
        },
        "background": background.tolist(),
        "base_value": base_value,
        "cases": cases,
    }


def main():
    """Generate all test cases and output as JSON."""
    if not HAS_DEPS:
        print(f"Error: Missing required dependencies: {IMPORT_ERROR}", file=sys.stderr)
        print("Install with: pip install shap scikit-learn", file=sys.stderr)
        sys.exit(1)

    test_suites = [
        generate_linear_behavior_cases(),
        generate_quadratic_cases(),
        generate_two_feature_cases(),
        generate_sinusoidal_cases(),
    ]

    output = {
        "version": "1.0",
        "description": "GradientSHAP test cases generated with Python SHAP library",
        "shap_version": shap.__version__,
        "generated_by": "testdata/python/generate_gradientshap_test_cases.py",
        "note": "GradientSHAP values may vary between runs due to stochastic sampling. "
                "Use correlation or directional checks rather than exact value matching.",
        "test_suites": test_suites,
    }

    json.dump(output, sys.stdout, indent=2)
    print()  # Newline at end


if __name__ == "__main__":
    main()
