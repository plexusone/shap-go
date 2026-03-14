#!/usr/bin/env python3
"""
Generate TreeSHAP test cases using the official SHAP library.

This script creates test cases with known-correct SHAP values that can be used
to verify the Go TreeSHAP implementation. Output is JSON for easy parsing in Go.

The output format matches what the Go tests expect:
- tree.nodes with feature, threshold, yes, no, cover, is_leaf, value fields
- base_value at the suite level
- cases with name, instance, prediction, shap_values

Usage:
    python generate_test_cases.py > ../treeshap_test_cases.json
"""

import json
import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import shap


def sklearn_tree_to_nodes(tree):
    """
    Convert sklearn tree to our node format.

    Returns a list of nodes with:
    - feature: split feature index (omitted for leaves)
    - threshold: split threshold (omitted for leaves)
    - yes: left child index (omitted for leaves)
    - no: right child index (omitted for leaves)
    - cover: number of samples at this node
    - is_leaf: true for leaf nodes
    - value: prediction value (only for leaves)
    """
    tree_ = tree.tree_
    nodes = []

    for i in range(tree_.node_count):
        is_leaf = tree_.children_left[i] == -1

        if is_leaf:
            nodes.append({
                "is_leaf": True,
                "value": float(tree_.value[i].flatten()[0]),
                "cover": float(tree_.n_node_samples[i]),
            })
        else:
            nodes.append({
                "feature": int(tree_.feature[i]),
                "threshold": float(tree_.threshold[i]),
                "yes": int(tree_.children_left[i]),
                "no": int(tree_.children_right[i]),
                "cover": float(tree_.n_node_samples[i]),
            })

    return nodes


def generate_simple_tree_cases():
    """
    Generate test cases for a simple single-split tree.

    Tree structure:
        x0 < 0.5 -> 1.0
        x0 >= 0.5 -> 2.0
    """
    # Create balanced training data to get 50/50 cover split
    X = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]])
    y = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])

    model = DecisionTreeRegressor(max_depth=1, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.3], "goes_left"),
        ([0.7], "goes_right"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    # Normalize covers to sum to 100 for cleaner numbers
    nodes = sklearn_tree_to_nodes(model)
    root_cover = nodes[0]["cover"]
    for node in nodes:
        node["cover"] = node["cover"] / root_cover * 100

    return {
        "name": "simple_tree",
        "description": f"Single split tree: x0 < {model.tree_.threshold[0]:.1f} -> 1.0, else -> 2.0. Covers: left=50, right=50",
        "tree": {
            "n_features": model.n_features_in_,
            "nodes": nodes,
        },
        "base_value": base_value,
        "cases": cases,
    }


def generate_two_feature_tree_cases():
    """
    Generate test cases for a two-feature tree.

    Tree structure:
        x0 < 0.5 (root)
            |-- x1 < 0.5 (left child)
            |     |-- leaf=1.0 (left-left)
            |     |-- leaf=2.0 (left-right)
            |-- leaf=4.0 (right child)
    """
    # Create training data to get desired tree structure
    # 25% go to leaf 1.0, 25% go to leaf 2.0, 50% go to leaf 4.0
    X = np.array([
        [0.1, 0.1], [0.2, 0.2],  # -> 1.0 (25%)
        [0.1, 0.6], [0.2, 0.7],  # -> 2.0 (25%)
        [0.6, 0.3], [0.7, 0.4], [0.8, 0.5], [0.9, 0.6],  # -> 4.0 (50%)
    ])
    y = np.array([1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])

    model = DecisionTreeRegressor(max_depth=2, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.3, 0.3], "x0<0.5,x1<0.5"),
        ([0.3, 0.7], "x0<0.5,x1>=0.5"),
        ([0.7, 0.3], "x0>=0.5"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    # Normalize covers to sum to 100
    nodes = sklearn_tree_to_nodes(model)
    root_cover = nodes[0]["cover"]
    for node in nodes:
        node["cover"] = node["cover"] / root_cover * 100

    return {
        "name": "two_feature_tree",
        "description": "Two-feature tree: root splits on x0<0.5, left child splits on x1<0.5",
        "tree": {
            "n_features": model.n_features_in_,
            "nodes": nodes,
        },
        "base_value": base_value,
        "cases": cases,
    }


def generate_ensemble_simple_cases():
    """
    Generate test cases for a simple ensemble of two single-split trees.

    Tree 1: x0 < 0.5 -> 1.0, else -> 2.0 (base = 1.5)
    Tree 2: x0 < 0.5 -> 0.5, else -> 1.5 (base = 1.0)
    Combined base = 2.5
    """
    # We need to manually construct this since sklearn doesn't easily
    # create exactly this structure. Instead, we'll train a simple
    # GradientBoostingRegressor and verify the SHAP values.

    # Create balanced training data
    X = np.array([[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]])
    y = np.array([1.5, 1.5, 1.5, 1.5, 3.5, 3.5, 3.5, 3.5])

    model = GradientBoostingRegressor(
        n_estimators=2,
        max_depth=1,
        learning_rate=1.0,
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.3], "goes_left"),
        ([0.7], "goes_right"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    # Build tree structure for both trees in the ensemble
    trees = []
    for i, estimator in enumerate(model.estimators_.flatten()):
        nodes = sklearn_tree_to_nodes(estimator)
        # Normalize covers
        root_cover = nodes[0]["cover"]
        for node in nodes:
            node["cover"] = node["cover"] / root_cover * 100
        trees.append({"nodes": nodes})

    return {
        "name": "ensemble_simple",
        "description": "Ensemble of two single-split trees on same feature",
        "tree": {
            "n_features": model.n_features_in_,
            "n_trees": len(trees),
            "trees": trees,
        },
        "base_value": base_value,
        "cases": cases,
    }


def generate_deep_tree_cases():
    """
    Generate test cases for a deeper tree (depth 3) with more features.

    This tests the algorithm with more complex path combinations.
    """
    np.random.seed(42)

    # Create training data with 3 features
    n_samples = 100
    X = np.random.rand(n_samples, 3)
    # Target depends on all three features
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    base_value = float(explainer.expected_value)

    test_instances = [
        ([0.2, 0.2, 0.2], "low_all"),
        ([0.8, 0.8, 0.8], "high_all"),
        ([0.2, 0.8, 0.5], "mixed"),
        ([0.5, 0.5, 0.5], "middle"),
    ]

    cases = []
    for instance, name in test_instances:
        x_arr = np.array([instance])
        prediction = float(model.predict(x_arr)[0])
        shap_values = explainer.shap_values(x_arr)[0].tolist()

        cases.append({
            "name": name,
            "instance": instance,
            "prediction": prediction,
            "shap_values": shap_values,
        })

    # Normalize covers
    nodes = sklearn_tree_to_nodes(model)
    root_cover = nodes[0]["cover"]
    for node in nodes:
        node["cover"] = node["cover"] / root_cover * 100

    return {
        "name": "deep_tree",
        "description": "Depth-3 tree with 3 features for complex path testing",
        "tree": {
            "n_features": model.n_features_in_,
            "nodes": nodes,
        },
        "base_value": base_value,
        "cases": cases,
    }


def main():
    """Generate all test cases and output as JSON."""
    test_suites = [
        generate_simple_tree_cases(),
        generate_two_feature_tree_cases(),
        generate_ensemble_simple_cases(),
        generate_deep_tree_cases(),
    ]

    output = {
        "version": "1.0",
        "description": "TreeSHAP test cases generated with Python SHAP library",
        "shap_version": shap.__version__,
        "generated_by": "testdata/python/generate_test_cases.py",
        "test_suites": test_suites,
    }

    json.dump(output, sys.stdout, indent=2)
    print()  # Newline at end


if __name__ == "__main__":
    main()
